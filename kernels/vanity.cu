#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "base58.h"
#include "vanity.h"
#include "sha256.h"

// ─── device state ───────────────────────────────────────────────────────────

__device__ int done = 0;
__device__ unsigned long long count = 0;

/* Filled once in gpu_grind_init(): glyph_from_byte[i] maps hash byte → PDA seed char. */
__device__ uint8_t glyph_from_hash_byte[256];

/* Canonicalizing LUT lives in base58.cu; gpu_grind_init uploads it here. */
extern __constant__ uint8_t d_match_lut[58];

/* Loop-invariant SHA-256 message schedule for block 1 (depends only on
   owner). Precomputed once on the host in gpu_grind_init and broadcast
   via constant cache, eliminating ~256 bytes of per-thread stack. */
__constant__ WORD d_sha_W1[64];

/* Loop-invariant slots of SHA-256 block-0 message schedule:
     [0..7]   = byte-swapped `base`  (32 bytes)
     [12..15] = byte-swapped `owner[0..15]`
   Slots [8..11] (= seed16) are built per-iter and overwritten. */
__constant__ WORD d_sha_W0_fixed[16];

// SHA-256 pubkey (exact layout: base[32] || seed16[16] || owner[32] = 80 bytes) ─
// saves per-iteration ctx clone + 48 generic byte-sha256_update passes.

static __device__ __forceinline__ void vanity_emit_sha256_digest(const CUDA_SHA256_CTX *ctx,
                                                                 BYTE hash[32])
{
    hash[0] = (ctx->state[0] >> 24) & 0x000000ff;
    hash[1] = (ctx->state[0] >> 16) & 0x000000ff;
    hash[2] = (ctx->state[0] >> 8) & 0x000000ff;
    hash[3] = (ctx->state[0] >> 0) & 0x000000ff;
    hash[4] = (ctx->state[1] >> 24) & 0x000000ff;
    hash[5] = (ctx->state[1] >> 16) & 0x000000ff;
    hash[6] = (ctx->state[1] >> 8) & 0x000000ff;
    hash[7] = (ctx->state[1] >> 0) & 0x000000ff;
    hash[8] = (ctx->state[2] >> 24) & 0x000000ff;
    hash[9] = (ctx->state[2] >> 16) & 0x000000ff;
    hash[10] = (ctx->state[2] >> 8) & 0x000000ff;
    hash[11] = (ctx->state[2] >> 0) & 0x000000ff;
    hash[12] = (ctx->state[3] >> 24) & 0x000000ff;
    hash[13] = (ctx->state[3] >> 16) & 0x000000ff;
    hash[14] = (ctx->state[3] >> 8) & 0x000000ff;
    hash[15] = (ctx->state[3] >> 0) & 0x000000ff;
    hash[16] = (ctx->state[4] >> 24) & 0x000000ff;
    hash[17] = (ctx->state[4] >> 16) & 0x000000ff;
    hash[18] = (ctx->state[4] >> 8) & 0x000000ff;
    hash[19] = (ctx->state[4] >> 0) & 0x000000ff;
    hash[20] = (ctx->state[5] >> 24) & 0x000000ff;
    hash[21] = (ctx->state[5] >> 16) & 0x000000ff;
    hash[22] = (ctx->state[5] >> 8) & 0x000000ff;
    hash[23] = (ctx->state[5] >> 0) & 0x000000ff;
    hash[24] = (ctx->state[6] >> 24) & 0x000000ff;
    hash[25] = (ctx->state[6] >> 16) & 0x000000ff;
    hash[26] = (ctx->state[6] >> 8) & 0x000000ff;
    hash[27] = (ctx->state[6] >> 0) & 0x000000ff;
    hash[28] = (ctx->state[7] >> 24) & 0x000000ff;
    hash[29] = (ctx->state[7] >> 16) & 0x000000ff;
    hash[30] = (ctx->state[7] >> 8) & 0x000000ff;
    hash[31] = (ctx->state[7] >> 0) & 0x000000ff;
}

static __device__ __forceinline__ void vanity_pubkey_sha256(const BYTE *seed16, BYTE out[32])
{
    /* Build block-0 message schedule [0..15]: load fixed slots from
       constant memory, byte-swap seed16 into [8..11]. */
    WORD W0[64];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        W0[i] = d_sha_W0_fixed[i];
    }
    W0[ 8] = ((WORD)seed16[ 0] << 24) | ((WORD)seed16[ 1] << 16) | ((WORD)seed16[ 2] << 8) | (WORD)seed16[ 3];
    W0[ 9] = ((WORD)seed16[ 4] << 24) | ((WORD)seed16[ 5] << 16) | ((WORD)seed16[ 6] << 8) | (WORD)seed16[ 7];
    W0[10] = ((WORD)seed16[ 8] << 24) | ((WORD)seed16[ 9] << 16) | ((WORD)seed16[10] << 8) | (WORD)seed16[11];
    W0[11] = ((WORD)seed16[12] << 24) | ((WORD)seed16[13] << 16) | ((WORD)seed16[14] << 8) | (WORD)seed16[15];

    CUDA_SHA256_CTX ctx;
    cuda_sha256_init(&ctx);
    cuda_sha256_transform_from_w16(ctx.state, W0);
    cuda_sha256_transform_w(ctx.state, d_sha_W1);
    vanity_emit_sha256_digest(&ctx, out);
}

// ─── persistent context ─────────────────────────────────────────────────────

typedef struct {
    int device_id;
    cudaStream_t stream;
    uint8_t *d_buffer;
    int num_blocks;
    int num_threads;
    unsigned long long target_cycles;
    uint64_t out_offset;
} GpuGrindCtx;

extern "C" void* gpu_grind_init(
    int id,
    uint8_t *base,
    uint8_t *owner,
    uint8_t *target, uint64_t target_len,
    uint8_t *suffix, uint64_t suffix_len,
    bool case_insensitive)
{
    cudaSetDevice(id);

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, id);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_grind_init(%d): cudaGetDeviceProperties: %s\n", id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int max_tpb = prop.maxThreadsPerBlock;
    int max_tpm = prop.maxThreadsPerMultiProcessor;
    int block_size = max_tpm / gcd(max_tpm, max_tpb);
    int nthreads = 256;
    int nblocks  = block_size * prop.multiProcessorCount;

    int clock_khz = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, id) != cudaSuccess || clock_khz <= 0) {
        fprintf(stderr, "gpu_grind_init(%d): clock rate query failed\n", id);
        exit(EXIT_FAILURE);
    }

    GpuGrindCtx *ctx = (GpuGrindCtx *)malloc(sizeof(GpuGrindCtx));
    ctx->device_id     = id;
    ctx->num_blocks    = nblocks;
    ctx->num_threads   = nthreads;
    ctx->target_cycles = (unsigned long long)clock_khz * 1000ULL * 5ULL;

    cudaStreamCreate(&ctx->stream);

    // Buffer: [seed:32] [base:32] [owner:32] [target_len:8] [target:N] [suffix_len:8] [suffix:M] [out:16]
    uint64_t buf_size = 32 + 32 + 32 + 8 + target_len + 8 + suffix_len + 16;
    ctx->out_offset = 32 + 32 + 32 + 8 + target_len + 8 + suffix_len;

    err = cudaMalloc((void**)&ctx->d_buffer, buf_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpu_grind_init(%d): cudaMalloc: %s\n", id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Build canonicalization LUT (raw_base58 idx -> match key) and pre-
       translate target/suffix from ASCII to those same canonical indices.
       In normal mode the LUT is identity; in CI mode, raw indices that
       encode the same character (e.g. both 9 and 33 -> 'a') fold to the
       lower index. */
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

    /* Translate ASCII target/suffix to canonical indices in host buffers. */
    uint8_t target_idx[64];
    uint8_t suffix_idx[64];
    for (uint64_t i = 0; i < target_len; ++i) {
        uint8_t v = 255;
        for (int k = 0; k < 58; ++k) {
            if ((uint8_t)alphabet[k] == target[i]) { v = host_match_lut[k]; break; }
        }
        target_idx[i] = v;
    }
    for (uint64_t i = 0; i < suffix_len; ++i) {
        uint8_t v = 255;
        for (int k = 0; k < 58; ++k) {
            if ((uint8_t)alphabet[k] == suffix[i]) { v = host_match_lut[k]; break; }
        }
        suffix_idx[i] = v;
    }

    cudaMemcpy(ctx->d_buffer + 32, base, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_buffer + 64, owner, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_buffer + 96, &target_len, 8, cudaMemcpyHostToDevice);
    if (target_len > 0) cudaMemcpy(ctx->d_buffer + 104, target_idx, target_len, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_buffer + 104 + target_len, &suffix_len, 8, cudaMemcpyHostToDevice);
    if (suffix_len > 0) cudaMemcpy(ctx->d_buffer + 104 + target_len + 8, suffix_idx, suffix_len, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_match_lut, host_match_lut, sizeof(host_match_lut));

    /* Precompute SHA-256 block-1 message schedule from owner. */
    {
        BYTE b1[64];
        memset(b1, 0, sizeof(b1));
        memcpy(b1, owner + 16, 16);
        b1[16] = (BYTE)0x80;
        b1[62] = (BYTE)0x02;
        b1[63] = (BYTE)0x80;

        WORD host_W1[64];
        for (int i = 0; i < 16; ++i) {
            host_W1[i] = ((WORD)b1[4*i    ] << 24) | ((WORD)b1[4*i + 1] << 16)
                       | ((WORD)b1[4*i + 2] <<  8) | ((WORD)b1[4*i + 3]      );
        }
        for (int i = 16; i < 64; ++i) {
            host_W1[i] = SIG1(host_W1[i-2]) + host_W1[i-7]
                       + SIG0(host_W1[i-15]) + host_W1[i-16];
        }
        cudaMemcpyToSymbol(d_sha_W1, host_W1, sizeof(host_W1));
    }

    /* Precompute loop-invariant slots of SHA-256 block-0 message schedule:
       [0..7] from base, [12..15] from owner[0..15]. Slots [8..11] are
       built per-iter from seed16. */
    {
        WORD host_W0[16] = {0};
        for (int i = 0; i < 8; ++i) {
            host_W0[i] = ((WORD)base[4*i    ] << 24) | ((WORD)base[4*i + 1] << 16)
                       | ((WORD)base[4*i + 2] <<  8) | ((WORD)base[4*i + 3]      );
        }
        for (int i = 0; i < 4; ++i) {
            host_W0[12 + i] = ((WORD)owner[4*i    ] << 24) | ((WORD)owner[4*i + 1] << 16)
                            | ((WORD)owner[4*i + 2] <<  8) | ((WORD)owner[4*i + 3]      );
        }
        cudaMemcpyToSymbol(d_sha_W0_fixed, host_W0, sizeof(host_W0));
    }

    {
        uint8_t host_lut[256];
        static const char alnum[63] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        for (unsigned u = 0; u < 256; ++u) {
            host_lut[u] = (uint8_t)alnum[u % 62];
        }
        cudaMemcpyToSymbol(glyph_from_hash_byte, host_lut, sizeof(host_lut));
    }

    return (void *)ctx;
}

extern "C" void gpu_grind_launch(void *opaque, uint8_t *seed)
{
    GpuGrindCtx *ctx = (GpuGrindCtx *)opaque;
    cudaSetDevice(ctx->device_id);

    cudaMemcpy(ctx->d_buffer, seed, 32, cudaMemcpyHostToDevice);

    int zero = 0;
    unsigned long long zero_ull = 0;
    cudaMemcpyToSymbol(done, &zero, sizeof(int));
    cudaMemcpyToSymbol(count, &zero_ull, sizeof(unsigned long long));

    vanity_search<<<ctx->num_blocks, ctx->num_threads, 0, ctx->stream>>>(
        ctx->d_buffer,
        (uint64_t)ctx->num_blocks * ctx->num_threads,
        ctx->target_cycles);
}

extern "C" int gpu_grind_query(void *opaque)
{
    GpuGrindCtx *ctx = (GpuGrindCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    return cudaStreamQuery(ctx->stream) == cudaSuccess ? 1 : 0;
}

// out layout written by caller: [seed:16] [count:8]
extern "C" void gpu_grind_read(void *opaque, uint8_t *out)
{
    GpuGrindCtx *ctx = (GpuGrindCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    cudaMemcpy(out, ctx->d_buffer + ctx->out_offset, 16, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(out + 16, count, 8, 0, cudaMemcpyDeviceToHost);
}

extern "C" void gpu_grind_destroy(void *opaque)
{
    GpuGrindCtx *ctx = (GpuGrindCtx *)opaque;
    cudaSetDevice(ctx->device_id);
    cudaStreamSynchronize(ctx->stream);
    cudaStreamDestroy(ctx->stream);
    cudaFree(ctx->d_buffer);
    free(ctx);
}

// ─── kernel ─────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(256, 3)
vanity_search(uint8_t *buffer, uint64_t stride, unsigned long long max_cycles)
{
    (void)stride;
    uint8_t *seed = buffer;
    uint8_t *base = buffer + 32;
    uint8_t *owner = buffer + 64;
    uint64_t target_len;
    memcpy(&target_len, buffer + 96, 8);
    uint8_t *target = buffer + 104;
    uint64_t suffix_len;
    memcpy(&suffix_len, buffer + 104 + target_len, 8);
    uint8_t *suffix = buffer + 104 + target_len + 8;
    uint8_t *out = (buffer + 104 + target_len + suffix_len + 8);

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    __align__(8) unsigned char local_out[32];
    memcpy(local_out, seed, 32);
    uint64_t *lw = (uint64_t *)local_out;
    uint64_t const k = idx;
    lw[0] += k;
    lw[1] += k;
    lw[2] += k;
    lw[3] += k;

    unsigned long long start_clock = clock64();
    uint32_t watchdog = 1u;

    for (uint64_t iter = 0;; iter++)
    {
        if (--watchdog == 0)
        {
            watchdog = 10000u;
            if (atomicMax(&done, 0) == 1)
            {
                atomicAdd(&count, iter);
                return;
            }
            if (clock64() - start_clock >= max_cycles)
            {
                atomicAdd(&count, iter);
                return;
            }
        }

        uint8_t create_account_seed[16];
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            create_account_seed[b] = glyph_from_hash_byte[local_out[b]];
        }

        vanity_pubkey_sha256(create_account_seed, local_out);

        if (fd_base58_check_match_32(local_out, target, target_len, suffix, suffix_len))
        {
            if (atomicMax(&done, 1) == 0)
            {
                memcpy(out, create_account_seed, 16);
            }

            atomicAdd(&count, iter + 1);
            return;
        }
    }
}
