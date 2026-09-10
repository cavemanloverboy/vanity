#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "vanity_keypair.h"
#include "base58.h"
#include "sha256.h"

#include "ed25519/fe.cu"
#include "ed25519/ge.cu"
#include "ed25519/sha512.cu"

__device__ static int kp_done = 0;
__device__ static unsigned long long kp_count = 0;

#define KP_MAX_THREADS 128
#ifndef KP_BATCH
#define KP_BATCH KP_BATCH_MAX
#endif

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_keypair_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles,
                      const ge_niels *comb);

/* Device-wide nanosecond clock. clock64() is per-SM and cannot be compared
   across the grid; %globaltimer is what makes a 2s slice mean 2s wall. */
static __device__ __forceinline__ unsigned long long kp_wall_ns()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

/* Launch exactly one resident wave: enough blocks to fill every SM, not
   enough to queue a second wave (which would multiply wall time). Persistent
   threads already run for the whole slice, so extra queued blocks do not
   increase throughput. */
template <typename Kernel>
static int one_wave_blocks(Kernel kernel, int nthreads, int sms, const char *who)
{
    int bps = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &bps, kernel, nthreads, 0);
    if (e != cudaSuccess || bps < 1) {
        fprintf(stderr, "%s: occupancy query failed (%s); using 1 block/SM\n",
                who, e != cudaSuccess ? cudaGetErrorString(e) : "0 blocks");
        bps = 1;
    }
    return bps * sms;
}

static int build_comb_or_die(ge_niels **out, cudaStream_t stream, const char *who)
{
    cudaError_t err = cudaMalloc((void **)out, (size_t)COMB_TABLE_LEN * sizeof(ge_niels));
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaMalloc comb: %s\n", who, cudaGetErrorString(err));
        return -1;
    }
    build_comb_table<<<1, 1, 0, stream>>>(*out);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: build_comb_table launch: %s\n", who, cudaGetErrorString(err));
        return -1;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: build_comb_table sync: %s\n", who, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// ─── persistent context ─────────────────────────────────────────────────────

typedef struct {
    int device_id;
    cudaStream_t stream;
    uint8_t *d_buffer;
    ge_niels *d_comb;
    int num_blocks;
    int num_threads;
    unsigned long long target_cycles;
    uint64_t out_offset;
} GpuKeypairCtx;

extern "C" void* gpu_keypair_init(
    int id,
    uint8_t *prefix, uint64_t prefix_len,
    uint8_t *suffix, uint64_t suffix_len,
    bool case_insensitive)
{
    cudaSetDevice(id);

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, id);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_keypair_init(%d): cudaGetDeviceProperties: %s\n", id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int nthreads = KP_MAX_THREADS;
    int nblocks = one_wave_blocks(vanity_keypair_search, nthreads,
                                  prop.multiProcessorCount, "gpu_keypair_init");

    GpuKeypairCtx *ctx = (GpuKeypairCtx *)malloc(sizeof(GpuKeypairCtx));
    ctx->device_id   = id;
    ctx->num_blocks  = nblocks;
    ctx->num_threads = nthreads;
    /* 2s wall-clock slice, timed with %globaltimer (nanoseconds). */
    ctx->target_cycles = 2000000000ULL;

    cudaStreamCreate(&ctx->stream);

    // Buffer: [seed:32] [prefix_len:8] [prefix:N] [suffix_len:8] [suffix:M] [out:32]
    uint64_t buf_size = 32 + 8 + prefix_len + 8 + suffix_len + 32;
    ctx->out_offset = 32 + 8 + prefix_len + 8 + suffix_len;

    err = cudaMalloc((void**)&ctx->d_buffer, buf_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_keypair_init(%d): cudaMalloc: %s\n", id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Canonical base58 match indices + LUT (same scheme as gpu_grind_init). */
    {
        static const char alphabet_normal[59] =
            "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        static const char alphabet_ci[59] =
            "123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz";
        const char *alphabet = case_insensitive ? alphabet_ci : alphabet_normal;

        uint8_t host_match_lut[58];
        for (int i = 0; i < 58; ++i) {
            host_match_lut[i] = (uint8_t)i;
            for (int j = 0; j < i; ++j) {
                if (alphabet[j] == alphabet[i]) {
                    host_match_lut[i] = (uint8_t)j;
                    break;
                }
            }
        }

        uint8_t prefix_idx[64];
        uint8_t suffix_idx[64];
        for (uint64_t i = 0; i < prefix_len; ++i) {
            uint8_t v = 255;
            for (int k = 0; k < 58; ++k) {
                if ((uint8_t)alphabet[k] == prefix[i]) { v = host_match_lut[k]; break; }
            }
            prefix_idx[i] = v;
        }
        for (uint64_t i = 0; i < suffix_len; ++i) {
            uint8_t v = 255;
            for (int k = 0; k < 58; ++k) {
                if ((uint8_t)alphabet[k] == suffix[i]) { v = host_match_lut[k]; break; }
            }
            suffix_idx[i] = v;
        }

        uint64_t off = 32;
        cudaMemcpy(ctx->d_buffer + off, &prefix_len, 8, cudaMemcpyHostToDevice); off += 8;
        if (prefix_len > 0) cudaMemcpy(ctx->d_buffer + off, prefix_idx, prefix_len, cudaMemcpyHostToDevice);
        off += prefix_len;
        cudaMemcpy(ctx->d_buffer + off, &suffix_len, 8, cudaMemcpyHostToDevice); off += 8;
        if (suffix_len > 0) cudaMemcpy(ctx->d_buffer + off, suffix_idx, suffix_len, cudaMemcpyHostToDevice);

        cudaMemcpyToSymbol(d_match_lut, host_match_lut, sizeof(host_match_lut));
    }

    if (build_comb_or_die(&ctx->d_comb, ctx->stream, "gpu_keypair_init") != 0) {
        exit(EXIT_FAILURE);
    }

    return (void *)ctx;
}

extern "C" void gpu_keypair_launch(void *opaque, uint8_t *seed)
{
    GpuKeypairCtx *ctx = (GpuKeypairCtx *)opaque;
    cudaSetDevice(ctx->device_id);

    cudaMemcpy(ctx->d_buffer, seed, 32, cudaMemcpyHostToDevice);
    cudaMemset(ctx->d_buffer + ctx->out_offset, 0, 32);

    int zero = 0;
    unsigned long long zero_ull = 0;
    cudaMemcpyToSymbol(kp_done, &zero, sizeof(int));
    cudaMemcpyToSymbol(kp_count, &zero_ull, sizeof(unsigned long long));

    vanity_keypair_search<<<ctx->num_blocks, ctx->num_threads, 0, ctx->stream>>>(
        ctx->d_buffer,
        (uint64_t)ctx->num_blocks * ctx->num_threads,
        ctx->target_cycles,
        ctx->d_comb);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr,
                "gpu_keypair_launch: kernel launch failed: %s\n"
                "  if this is 'no kernel image is available', rebuild with your GPU's\n"
                "  compute capability, e.g. VANITY_CUDA_ARCH=86 for an RTX 3090.\n",
                cudaGetErrorString(launch_err));
        exit(EXIT_FAILURE);
    }
}

extern "C" int gpu_keypair_query(void *opaque)
{
    GpuKeypairCtx *ctx = (GpuKeypairCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    return cudaStreamQuery(ctx->stream) == cudaSuccess ? 1 : 0;
}

// out layout: [seed:32] [count:8]
extern "C" void gpu_keypair_read(void *opaque, uint8_t *out)
{
    GpuKeypairCtx *ctx = (GpuKeypairCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    cudaMemcpy(out, ctx->d_buffer + ctx->out_offset, 32, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(out + 32, kp_count, 8, 0, cudaMemcpyDeviceToHost);
}

extern "C" void gpu_keypair_destroy(void *opaque)
{
    GpuKeypairCtx *ctx = (GpuKeypairCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    cudaStreamSynchronize(ctx->stream);
    cudaStreamDestroy(ctx->stream);
    cudaFree(ctx->d_comb);
    cudaFree(ctx->d_buffer);
    free(ctx);
}

// ─── kernel ─────────────────────────────────────────────────────────────────

static __device__ __forceinline__ void kp_sha512_32(const unsigned char seed[32],
                                                   unsigned char out[64])
{
    sha512_context md;
    md.curlen = 0;
    md.length = 0;
    md.state[0] = UINT64_C(0x6a09e667f3bcc908);
    md.state[1] = UINT64_C(0xbb67ae8584caa73b);
    md.state[2] = UINT64_C(0x3c6ef372fe94f82b);
    md.state[3] = UINT64_C(0xa54ff53a5f1d36f1);
    md.state[4] = UINT64_C(0x510e527fade682d1);
    md.state[5] = UINT64_C(0x9b05688c2b3e6c1f);
    md.state[6] = UINT64_C(0x1f83d9abfb41bd6b);
    md.state[7] = UINT64_C(0x5be0cd19137e2179);
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        md.buf[i] = seed[i];
    }
    md.curlen = 32;
    sha512_final(&md, out);
}

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_keypair_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles,
                      const ge_niels *comb)
{
    (void)stride;
    uint8_t *host_seed = buffer;

    uint64_t prefix_len;
    memcpy(&prefix_len, buffer + 32, 8);
    uint8_t *prefix = buffer + 40;

    uint64_t suffix_len;
    memcpy(&suffix_len, buffer + 40 + prefix_len, 8);
    uint8_t *suffix = buffer + 40 + prefix_len + 8;

    uint8_t *out = suffix + suffix_len;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char seed[32];
    unsigned char privatek[64];
    unsigned char pubkey[32];
    unsigned char batch_seeds[KP_BATCH][32];
    fe Xs[KP_BATCH];
    fe Ys[KP_BATCH];
    fe Zs[KP_BATCH];
    ge_p3 A;

    CUDA_SHA256_CTX sha256_ctx;
    cuda_sha256_init(&sha256_ctx);
    cuda_sha256_update(&sha256_ctx, (BYTE *)host_seed, 32);
    cuda_sha256_update(&sha256_ctx, (BYTE *)(&idx), 8);
    cuda_sha256_final(&sha256_ctx, (BYTE *)seed);

    unsigned long long start_ns = kp_wall_ns();
    uint64_t iter = 0;
    uint32_t watchdog = 1u;

    for (;;)
    {
        if (--watchdog == 0) {
            watchdog = 16u; /* 16 * KP_BATCH = 128 keys, near the old % 100 */
            if (atomicMax(&kp_done, 0) == 1) {
                atomicAdd(&kp_count, iter);
                return;
            }
            if (kp_wall_ns() - start_ns >= max_cycles) {
                atomicAdd(&kp_count, iter);
                return;
            }
        }

        const int n = KP_BATCH;
        for (int j = 0; j < n; j++) {
            #pragma unroll
            for (int i = 0; i < 32; i++) batch_seeds[j][i] = seed[i];

            kp_sha512_32(seed, privatek);

            privatek[0]  &= 248;
            privatek[31] &= 63;
            privatek[31] |= 64;

            ge_scalarmult_base_comb(&A, privatek, comb);
            fe_copy(Xs[j], A.X);
            fe_copy(Ys[j], A.Y);
            fe_copy(Zs[j], A.Z);

            #pragma unroll
            for (int i = 0; i < 32; i++) seed[i] = privatek[32 + i];
        }

        fe_batch_invert(Zs, n);

        int matched = -1;
        for (int j = 0; j < n; j++) {
            ge_p3_tobytes_inv(pubkey, Xs[j], Ys[j], Zs[j]);

            uint pubkey_words[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                pubkey_words[k] = ((uint)pubkey[4*k    ] << 24)
                                | ((uint)pubkey[4*k + 1] << 16)
                                | ((uint)pubkey[4*k + 2] <<  8)
                                | ((uint)pubkey[4*k + 3]      );
            }

            if (fd_base58_check_match_32_words(pubkey_words, prefix, prefix_len, suffix, suffix_len))
            {
                if (atomicMax(&kp_done, 1) == 0) {
                    memcpy(out, batch_seeds[j], 32);
                }
                matched = j;
                break;
            }
        }

        if (matched >= 0) {
            atomicAdd(&kp_count, iter + (uint64_t)(matched + 1));
            return;
        }
        iter += (uint64_t)n;
    }
}

// ─── doppler ──────────────────────────────────────────────────────────────
//
// Shares this translation unit's ed25519 machinery (fe/ge/sha512). Matches a
// pubkey when at least dop_required of its four 8-byte segments are sign-
// extendable 32-bit values: low 4 bytes are an i32 and high 4 bytes are its
// sign extension (all 0x00 if bit 31 clear, all 0xFF if set). No base58.

__device__ static int dop_done = 0;
__device__ static unsigned long long dop_count = 0;
__device__ static uint32_t dop_required = 1;

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_doppler_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles,
                      const ge_niels *comb);

static __device__ __forceinline__ uint32_t doppler_count(const unsigned char *pk)
{
    uint32_t matched = 0;
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        int o = s * 8;
        unsigned char fill = (pk[o + 3] & 0x80) ? 0xFF : 0x00;
        if (pk[o + 4] == fill && pk[o + 5] == fill &&
            pk[o + 6] == fill && pk[o + 7] == fill)
            matched++;
    }
    return matched;
}

typedef struct {
    int device_id;
    cudaStream_t stream;
    uint8_t *d_buffer;
    ge_niels *d_comb;
    int num_blocks;
    int num_threads;
    unsigned long long target_cycles;
    uint64_t out_offset;
} GpuDopplerCtx;

extern "C" void* gpu_doppler_init(int id, uint32_t required_segments)
{
    cudaSetDevice(id);

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, id);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_doppler_init(%d): cudaGetDeviceProperties: %s\n", id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int nthreads = KP_MAX_THREADS;
    int nblocks = one_wave_blocks(vanity_doppler_search, nthreads,
                                  prop.multiProcessorCount, "gpu_doppler_init");

    GpuDopplerCtx *ctx = (GpuDopplerCtx *)malloc(sizeof(GpuDopplerCtx));
    ctx->device_id   = id;
    ctx->num_blocks  = nblocks;
    ctx->num_threads = nthreads;
    /* 2s wall-clock slice, timed with %globaltimer (nanoseconds). */
    ctx->target_cycles = 2000000000ULL;

    cudaStreamCreate(&ctx->stream);

    // Buffer: [seed:32] [out:32]
    uint64_t buf_size = 32 + 32;
    ctx->out_offset = 32;

    err = cudaMalloc((void**)&ctx->d_buffer, buf_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_doppler_init(%d): cudaMalloc: %s\n", id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMemcpyToSymbol(dop_required, &required_segments, sizeof(uint32_t), 0, cudaMemcpyHostToDevice);

    if (build_comb_or_die(&ctx->d_comb, ctx->stream, "gpu_doppler_init") != 0) {
        exit(EXIT_FAILURE);
    }

    return (void *)ctx;
}

extern "C" void gpu_doppler_launch(void *opaque, uint8_t *seed)
{
    GpuDopplerCtx *ctx = (GpuDopplerCtx *)opaque;
    cudaSetDevice(ctx->device_id);

    cudaMemcpy(ctx->d_buffer, seed, 32, cudaMemcpyHostToDevice);
    cudaMemset(ctx->d_buffer + ctx->out_offset, 0, 32);

    int zero = 0;
    unsigned long long zero_ull = 0;
    cudaMemcpyToSymbol(dop_done, &zero, sizeof(int));
    cudaMemcpyToSymbol(dop_count, &zero_ull, sizeof(unsigned long long));

    vanity_doppler_search<<<ctx->num_blocks, ctx->num_threads, 0, ctx->stream>>>(
        ctx->d_buffer,
        (uint64_t)ctx->num_blocks * ctx->num_threads,
        ctx->target_cycles,
        ctx->d_comb);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr,
                "gpu_doppler_launch: kernel launch failed: %s\n"
                "  if this is 'no kernel image is available', rebuild with your GPU's\n"
                "  compute capability, e.g. VANITY_CUDA_ARCH=86 for an RTX 3090.\n",
                cudaGetErrorString(launch_err));
        exit(EXIT_FAILURE);
    }
}

extern "C" int gpu_doppler_query(void *opaque)
{
    GpuDopplerCtx *ctx = (GpuDopplerCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    return cudaStreamQuery(ctx->stream) == cudaSuccess ? 1 : 0;
}

// out layout: [seed:32] [count:8]
extern "C" void gpu_doppler_read(void *opaque, uint8_t *out)
{
    GpuDopplerCtx *ctx = (GpuDopplerCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    cudaMemcpy(out, ctx->d_buffer + ctx->out_offset, 32, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(out + 32, dop_count, 8, 0, cudaMemcpyDeviceToHost);
}

extern "C" void gpu_doppler_destroy(void *opaque)
{
    GpuDopplerCtx *ctx = (GpuDopplerCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    cudaStreamSynchronize(ctx->stream);
    cudaStreamDestroy(ctx->stream);
    cudaFree(ctx->d_comb);
    cudaFree(ctx->d_buffer);
    free(ctx);
}

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_doppler_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles,
                      const ge_niels *comb)
{
    (void)stride;
    uint8_t *host_seed = buffer;
    uint8_t *out = buffer + 32;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char seed[32];
    unsigned char privatek[64];
    unsigned char pubkey[32];
    unsigned char batch_seeds[KP_BATCH][32];
    fe Xs[KP_BATCH];
    fe Ys[KP_BATCH];
    fe Zs[KP_BATCH];
    ge_p3 A;

    CUDA_SHA256_CTX sha256_ctx;
    cuda_sha256_init(&sha256_ctx);
    cuda_sha256_update(&sha256_ctx, (BYTE *)host_seed, 32);
    cuda_sha256_update(&sha256_ctx, (BYTE *)(&idx), 8);
    cuda_sha256_final(&sha256_ctx, (BYTE *)seed);

    unsigned long long start_ns = kp_wall_ns();
    uint64_t iter = 0;
    uint32_t watchdog = 1u;

    for (;;)
    {
        if (--watchdog == 0) {
            watchdog = 16u;
            if (atomicMax(&dop_done, 0) == 1) {
                atomicAdd(&dop_count, iter);
                return;
            }
            if (kp_wall_ns() - start_ns >= max_cycles) {
                atomicAdd(&dop_count, iter);
                return;
            }
        }

        const int n = KP_BATCH;
        for (int j = 0; j < n; j++) {
            #pragma unroll
            for (int i = 0; i < 32; i++) batch_seeds[j][i] = seed[i];

            kp_sha512_32(seed, privatek);

            privatek[0]  &= 248;
            privatek[31] &= 63;
            privatek[31] |= 64;

            ge_scalarmult_base_comb(&A, privatek, comb);
            fe_copy(Xs[j], A.X);
            fe_copy(Ys[j], A.Y);
            fe_copy(Zs[j], A.Z);

            #pragma unroll
            for (int i = 0; i < 32; i++) seed[i] = privatek[32 + i];
        }

        fe_batch_invert(Zs, n);

        int matched = -1;
        for (int j = 0; j < n; j++) {
            ge_p3_tobytes_inv(pubkey, Xs[j], Ys[j], Zs[j]);
            if (doppler_count(pubkey) >= dop_required) {
                if (atomicMax(&dop_done, 1) == 0) {
                    memcpy(out, batch_seeds[j], 32);
                }
                matched = j;
                break;
            }
        }

        if (matched >= 0) {
            atomicAdd(&dop_count, iter + (uint64_t)(matched + 1));
            return;
        }
        iter += (uint64_t)n;
    }
}
