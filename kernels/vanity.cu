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
__device__ bool d_case_insensitive = false;

/* Filled once in gpu_grind_init(): glyph_from_byte[i] maps hash byte → PDA seed char. */
__device__ uint8_t glyph_from_hash_byte[256];

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

static __device__ __forceinline__ void vanity_pubkey_sha256(const BYTE *base,
                                                              const BYTE *seed16,
                                                              const BYTE *owner,
                                                              BYTE out[32])
{
    CUDA_SHA256_CTX ctx;
    BYTE b0[64];
    BYTE b1[64];
    cuda_sha256_init(&ctx);
    memcpy(b0, base, 32);
    memcpy(b0 + 32, seed16, 16);
    memcpy(b0 + 48, owner, 16);
    cuda_sha256_transform(&ctx, b0);
    memset(b1, 0, sizeof(b1));
    memcpy(b1, owner + 16, 16);
    b1[16] = (BYTE)0x80;
    {
        unsigned long long const bl = 80ULL * 8ULL;
        b1[63] = (BYTE)(bl);
        b1[62] = (BYTE)(bl >> 8);
        b1[61] = (BYTE)(bl >> 16);
        b1[60] = (BYTE)(bl >> 24);
        b1[59] = (BYTE)(bl >> 32);
        b1[58] = (BYTE)(bl >> 40);
        b1[57] = (BYTE)(bl >> 48);
        b1[56] = (BYTE)(bl >> 56);
    }
    cuda_sha256_transform(&ctx, b1);
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

    // Upload invariant data (everything after the 32-byte seed slot)
    cudaMemcpy(ctx->d_buffer + 32, base, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_buffer + 64, owner, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_buffer + 96, &target_len, 8, cudaMemcpyHostToDevice);
    if (target_len > 0) cudaMemcpy(ctx->d_buffer + 104, target, target_len, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_buffer + 104 + target_len, &suffix_len, 8, cudaMemcpyHostToDevice);
    if (suffix_len > 0) cudaMemcpy(ctx->d_buffer + 104 + target_len + 8, suffix, suffix_len, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_case_insensitive, &case_insensitive, 1, 0, cudaMemcpyHostToDevice);

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

__global__ void
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
    unsigned char local_encoded[44] = {0};

    unsigned long long start_clock = clock64();
    uint32_t watchdog = 1u;

    for (uint64_t iter = 0; iter < uint64_t(1000) * 1000 * 1000 * 1000; iter++)
    {
        if (--watchdog == 0)
        {
            watchdog = 100u;
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

        vanity_pubkey_sha256(base, create_account_seed, owner, local_out);
        ulong encoded_len = fd_base58_encode_32(local_out, (unsigned char *)(&local_encoded), d_case_insensitive);

        if (matches_target((unsigned char *)local_encoded, (unsigned char *)target, target_len, (unsigned char *)suffix, suffix_len, encoded_len))
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

__device__ bool matches_target(unsigned char *a, unsigned char *target, uint64_t n, unsigned char *suffix, uint64_t suffix_len, ulong encoded_len)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != target[i])
            return false;
    }
    for (int i = 0; i < suffix_len; i++)
    {
        if (a[encoded_len - suffix_len + i] != suffix[i])
            return false;
    }
    return true;
}
