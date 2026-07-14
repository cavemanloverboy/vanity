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

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_keypair_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles);

// ─── persistent context ─────────────────────────────────────────────────────

typedef struct {
    int device_id;
    cudaStream_t stream;
    uint8_t *d_buffer;
    int num_blocks;
    int num_threads;
    unsigned long long target_cycles;
    uint64_t out_offset;
} GpuKeypairCtx;

extern "C" void* gpu_keypair_init(
    int id,
    uint8_t *prefixes, uint64_t prefix_count,
    uint8_t *suffixes, uint64_t suffix_count,
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
    int blocks_per_sm = prop.maxThreadsPerMultiProcessor / nthreads;
    int nblocks = blocks_per_sm * prop.multiProcessorCount;

    int clock_khz = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, id) != cudaSuccess || clock_khz <= 0) {
        fprintf(stderr, "gpu_keypair_init(%d): clock rate query failed\n", id);
        exit(EXIT_FAILURE);
    }

    GpuKeypairCtx *ctx = (GpuKeypairCtx *)malloc(sizeof(GpuKeypairCtx));
    ctx->device_id   = id;
    ctx->num_blocks  = nblocks;
    ctx->num_threads = nthreads;
    ctx->target_cycles = (unsigned long long)clock_khz * 1000ULL * 5ULL;

    cudaStreamCreate(&ctx->stream);

    uint64_t prefix_bytes = VANITY_MATCH_PLAN_WORDS * sizeof(uint32_t);
    uint64_t suffix_bytes = VANITY_MATCH_PLAN_WORDS * sizeof(uint32_t);
    // Buffer: [seed:32] [prefix_count:8] [prefixes] [suffix_count:8] [suffixes] [out:32]
    uint64_t buf_size = 32 + 8 + prefix_bytes + 8 + suffix_bytes + 32;
    ctx->out_offset = 32 + 8 + prefix_bytes + 8 + suffix_bytes;

    err = cudaMalloc((void**)&ctx->d_buffer, buf_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_keypair_init(%d): cudaMalloc: %s\n", id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Canonical base58 LUT; match plans arrive precomputed by the Rust host. */
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

        uint64_t off = 32;
        cudaMemcpy(ctx->d_buffer + off, &prefix_count, 8, cudaMemcpyHostToDevice); off += 8;
        cudaMemcpy(ctx->d_buffer + off, prefixes, prefix_bytes, cudaMemcpyHostToDevice);
        off += prefix_bytes;
        cudaMemcpy(ctx->d_buffer + off, &suffix_count, 8, cudaMemcpyHostToDevice); off += 8;
        cudaMemcpy(ctx->d_buffer + off, suffixes, suffix_bytes, cudaMemcpyHostToDevice);

        cudaMemcpyToSymbol(d_match_lut, host_match_lut, sizeof(host_match_lut));
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
        ctx->target_cycles);

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
    cudaFree(ctx->d_buffer);
    free(ctx);
}

// ─── kernel ─────────────────────────────────────────────────────────────────

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_keypair_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles)
{
    uint8_t *host_seed = buffer;

    uint64_t prefix_count;
    memcpy(&prefix_count, buffer + 32, 8);
    uint8_t *prefixes = buffer + 40;
    uint64_t prefix_bytes = VANITY_MATCH_PLAN_WORDS * sizeof(uint32_t);

    uint64_t suffix_count;
    memcpy(&suffix_count, buffer + 40 + prefix_bytes, 8);
    uint8_t *suffixes = buffer + 40 + prefix_bytes + 8;
    uint64_t suffix_bytes = VANITY_MATCH_PLAN_WORDS * sizeof(uint32_t);

    uint8_t *out = suffixes + suffix_bytes;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char seed[32];
    unsigned char privatek[64];
    unsigned char pubkey[32];
    ge_p3 A;

    CUDA_SHA256_CTX sha256_ctx;
    cuda_sha256_init(&sha256_ctx);
    cuda_sha256_update(&sha256_ctx, (BYTE *)host_seed, 32);
    cuda_sha256_update(&sha256_ctx, (BYTE *)(&idx), 8);
    cuda_sha256_final(&sha256_ctx, (BYTE *)seed);

    unsigned long long start_clock = clock64();

    for (uint64_t iter = 0; iter < uint64_t(1000) * 1000 * 1000 * 1000; iter++)
    {
        if (iter % 100 == 0) {
            if (atomicMax(&kp_done, 0) == 1) {
                atomicAdd(&kp_count, iter);
                return;
            }
            if (clock64() - start_clock >= max_cycles) {
                atomicAdd(&kp_count, iter);
                return;
            }
        }

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

        sha512_final(&md, privatek);

        privatek[0]  &= 248;
        privatek[31] &= 63;
        privatek[31] |= 64;

        ge_scalarmult_base(&A, privatek);
        ge_p3_tobytes(pubkey, &A);

        uint pubkey_words[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            pubkey_words[k] = ((uint)pubkey[4*k    ] << 24)
                            | ((uint)pubkey[4*k + 1] << 16)
                            | ((uint)pubkey[4*k + 2] <<  8)
                            | ((uint)pubkey[4*k + 3]      );
        }

        if (fd_base58_check_match_32_words(pubkey_words, prefixes, prefix_count, suffixes, suffix_count))
        {
            if (atomicMax(&kp_done, 1) == 0) {
                memcpy(out, seed, 32);
            }
            atomicAdd(&kp_count, iter + 1);
            return;
        }

        memcpy(seed, privatek + 32, 32);
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
vanity_doppler_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles);

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
    int blocks_per_sm = prop.maxThreadsPerMultiProcessor / nthreads;
    int nblocks = blocks_per_sm * prop.multiProcessorCount;

    int clock_khz = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, id) != cudaSuccess || clock_khz <= 0) {
        fprintf(stderr, "gpu_doppler_init(%d): clock rate query failed\n", id);
        exit(EXIT_FAILURE);
    }

    GpuDopplerCtx *ctx = (GpuDopplerCtx *)malloc(sizeof(GpuDopplerCtx));
    ctx->device_id   = id;
    ctx->num_blocks  = nblocks;
    ctx->num_threads = nthreads;
    ctx->target_cycles = (unsigned long long)clock_khz * 1000ULL * 5ULL;

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
        ctx->target_cycles);

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
    cudaFree(ctx->d_buffer);
    free(ctx);
}

static __global__ void __launch_bounds__(KP_MAX_THREADS)
vanity_doppler_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles)
{
    uint8_t *host_seed = buffer;
    uint8_t *out = buffer + 32;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char seed[32];
    unsigned char privatek[64];
    unsigned char pubkey[32];
    ge_p3 A;

    CUDA_SHA256_CTX sha256_ctx;
    cuda_sha256_init(&sha256_ctx);
    cuda_sha256_update(&sha256_ctx, (BYTE *)host_seed, 32);
    cuda_sha256_update(&sha256_ctx, (BYTE *)(&idx), 8);
    cuda_sha256_final(&sha256_ctx, (BYTE *)seed);

    unsigned long long start_clock = clock64();

    for (uint64_t iter = 0; iter < uint64_t(1000) * 1000 * 1000 * 1000; iter++)
    {
        if (iter % 100 == 0) {
            if (atomicMax(&dop_done, 0) == 1) {
                atomicAdd(&dop_count, iter);
                return;
            }
            if (clock64() - start_clock >= max_cycles) {
                atomicAdd(&dop_count, iter);
                return;
            }
        }

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

        sha512_final(&md, privatek);

        privatek[0]  &= 248;
        privatek[31] &= 63;
        privatek[31] |= 64;

        ge_scalarmult_base(&A, privatek);
        ge_p3_tobytes(pubkey, &A);

        if (doppler_count(pubkey) >= dop_required)
        {
            if (atomicMax(&dop_done, 1) == 0) {
                memcpy(out, seed, 32);
            }
            atomicAdd(&dop_count, iter + 1);
            return;
        }

        memcpy(seed, privatek + 32, 32);
    }
}
