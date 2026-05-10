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

/* SHA-256 working state (a..h) after rounds 0..7 of block-0, given
   that those rounds depend only on H0 (constants) and W[0..7] (= the
   `base` pubkey, fixed per kernel). Precomputed in gpu_grind_init. */
__constant__ WORD d_sha_state_r7[8];

/* Block-0 SHA-256 transform that skips rounds 0..7 by loading the
   precomputed working state from constant memory and starting at
   round 8. The full 48-step W expansion still runs because rounds
   8..15 use W[8..15] (W[8..11] are variable) and rounds 16+ depend
   on W[0..7] via the message schedule. */
static __device__ __forceinline__ void vanity_sha256_block0_skip8(WORD state_out[8], WORD W[64])
{
    /* Working state at entry to round 8. */
    WORD a = d_sha_state_r7[0], b = d_sha_state_r7[1], c = d_sha_state_r7[2], d = d_sha_state_r7[3];
    WORD e = d_sha_state_r7[4], f = d_sha_state_r7[5], g = d_sha_state_r7[6], h = d_sha_state_r7[7];

    VANITY_SHA_EXPAND48(W);

    VANITY_SHA_R(0xD807AA98U, W[ 8]); VANITY_SHA_R(0x12835B01U, W[ 9]);
    VANITY_SHA_R(0x243185BEU, W[10]); VANITY_SHA_R(0x550C7DC3U, W[11]);
    VANITY_SHA_R(0x72BE5D74U, W[12]); VANITY_SHA_R(0x80DEB1FEU, W[13]);
    VANITY_SHA_R(0x9BDC06A7U, W[14]); VANITY_SHA_R(0xC19BF174U, W[15]);
    VANITY_SHA_R(0xE49B69C1U, W[16]); VANITY_SHA_R(0xEFBE4786U, W[17]);
    VANITY_SHA_R(0x0FC19DC6U, W[18]); VANITY_SHA_R(0x240CA1CCU, W[19]);
    VANITY_SHA_R(0x2DE92C6FU, W[20]); VANITY_SHA_R(0x4A7484AAU, W[21]);
    VANITY_SHA_R(0x5CB0A9DCU, W[22]); VANITY_SHA_R(0x76F988DAU, W[23]);
    VANITY_SHA_R(0x983E5152U, W[24]); VANITY_SHA_R(0xA831C66DU, W[25]);
    VANITY_SHA_R(0xB00327C8U, W[26]); VANITY_SHA_R(0xBF597FC7U, W[27]);
    VANITY_SHA_R(0xC6E00BF3U, W[28]); VANITY_SHA_R(0xD5A79147U, W[29]);
    VANITY_SHA_R(0x06CA6351U, W[30]); VANITY_SHA_R(0x14292967U, W[31]);
    VANITY_SHA_R(0x27B70A85U, W[32]); VANITY_SHA_R(0x2E1B2138U, W[33]);
    VANITY_SHA_R(0x4D2C6DFCU, W[34]); VANITY_SHA_R(0x53380D13U, W[35]);
    VANITY_SHA_R(0x650A7354U, W[36]); VANITY_SHA_R(0x766A0ABBU, W[37]);
    VANITY_SHA_R(0x81C2C92EU, W[38]); VANITY_SHA_R(0x92722C85U, W[39]);
    VANITY_SHA_R(0xA2BFE8A1U, W[40]); VANITY_SHA_R(0xA81A664BU, W[41]);
    VANITY_SHA_R(0xC24B8B70U, W[42]); VANITY_SHA_R(0xC76C51A3U, W[43]);
    VANITY_SHA_R(0xD192E819U, W[44]); VANITY_SHA_R(0xD6990624U, W[45]);
    VANITY_SHA_R(0xF40E3585U, W[46]); VANITY_SHA_R(0x106AA070U, W[47]);
    VANITY_SHA_R(0x19A4C116U, W[48]); VANITY_SHA_R(0x1E376C08U, W[49]);
    VANITY_SHA_R(0x2748774CU, W[50]); VANITY_SHA_R(0x34B0BCB5U, W[51]);
    VANITY_SHA_R(0x391C0CB3U, W[52]); VANITY_SHA_R(0x4ED8AA4AU, W[53]);
    VANITY_SHA_R(0x5B9CCA4FU, W[54]); VANITY_SHA_R(0x682E6FF3U, W[55]);
    VANITY_SHA_R(0x748F82EEU, W[56]); VANITY_SHA_R(0x78A5636FU, W[57]);
    VANITY_SHA_R(0x84C87814U, W[58]); VANITY_SHA_R(0x8CC70208U, W[59]);
    VANITY_SHA_R(0x90BEFFFAU, W[60]); VANITY_SHA_R(0xA4506CEBU, W[61]);
    VANITY_SHA_R(0xBEF9A3F7U, W[62]); VANITY_SHA_R(0xC67178F2U, W[63]);

    /* Final state = H0 (literal) + working state after round 63. */
    state_out[0] = 0x6a09e667U + a;
    state_out[1] = 0xbb67ae85U + b;
    state_out[2] = 0x3c6ef372U + c;
    state_out[3] = 0xa54ff53aU + d;
    state_out[4] = 0x510e527fU + e;
    state_out[5] = 0x9b05688cU + f;
    state_out[6] = 0x1f83d9abU + g;
    state_out[7] = 0x5be0cd19U + h;
}

/* SHA-256 of base[32] || seed16[16] || owner[32] = 80 bytes, in pure
   word form: caller passes 4 words for seed16 (already big-endian
   packed, MSB-first), receives 8 words of digest in `state_out`.
   Avoids the byte-swap roundtrip that happens when the digest is
   serialized to bytes only to be re-loaded as words by the next
   iter's input or by base58. */
static __device__ __forceinline__ void vanity_pubkey_sha256_words(const WORD seed_words[4],
                                                                  WORD state_out[8])
{
    WORD W0[64];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        W0[i] = d_sha_W0_fixed[i];
    }
    W0[ 8] = seed_words[0];
    W0[ 9] = seed_words[1];
    W0[10] = seed_words[2];
    W0[11] = seed_words[3];

    vanity_sha256_block0_skip8(state_out, W0);
    cuda_sha256_transform_w(state_out, d_sha_W1);
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

    /* Precompute SHA-256 working state after rounds 0..7 of block-0.
       Those rounds depend only on H0 (literal constants) and W[0..7]
       (= byte-swapped `base`, fixed for the whole grind), so the result
       is invariant across every iteration. Skipping them in the kernel
       saves 7 main rounds + 8 H0 register writes per pubkey hash. */
    {
        WORD W[8];
        for (int i = 0; i < 8; ++i) {
            W[i] = ((WORD)base[4*i    ] << 24) | ((WORD)base[4*i + 1] << 16)
                 | ((WORD)base[4*i + 2] <<  8) | ((WORD)base[4*i + 3]      );
        }
        WORD a = 0x6a09e667U, b = 0xbb67ae85U, c = 0x3c6ef372U, d = 0xa54ff53aU;
        WORD e = 0x510e527fU, f = 0x9b05688cU, g = 0x1f83d9abU, h = 0x5be0cd19U;

        VANITY_SHA_R(0x428A2F98U, W[0]); VANITY_SHA_R(0x71374491U, W[1]);
        VANITY_SHA_R(0xB5C0FBCFU, W[2]); VANITY_SHA_R(0xE9B5DBA5U, W[3]);
        VANITY_SHA_R(0x3956C25BU, W[4]); VANITY_SHA_R(0x59F111F1U, W[5]);
        VANITY_SHA_R(0x923F82A4U, W[6]); VANITY_SHA_R(0xAB1C5ED5U, W[7]);

        WORD host_state_r7[8] = {a, b, c, d, e, f, g, h};
        cudaMemcpyToSymbol(d_sha_state_r7, host_state_r7, sizeof(host_state_r7));
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
    uint64_t target_len;
    memcpy(&target_len, buffer + 96, 8);
    uint8_t *target = buffer + 104;
    uint64_t suffix_len;
    memcpy(&suffix_len, buffer + 104 + target_len, 8);
    uint8_t *suffix = buffer + 104 + target_len + 8;
    uint8_t *out = (buffer + 104 + target_len + suffix_len + 8);

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Bootstrap per-thread digest state. We keep the original byte-form
       seed addition (idx mixed into each u64 lane) so semantics match
       prior commits, then convert once into 8 big-endian-packed words.
       Inside the loop we never go back to bytes. */
    WORD digest_words[8];
    {
        __align__(8) unsigned char local_seed[32];
        memcpy(local_seed, seed, 32);
        uint64_t *lw = (uint64_t *)local_seed;
        lw[0] += idx;
        lw[1] += idx;
        lw[2] += idx;
        lw[3] += idx;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            digest_words[i] = ((WORD)local_seed[4*i    ] << 24)
                            | ((WORD)local_seed[4*i + 1] << 16)
                            | ((WORD)local_seed[4*i + 2] <<  8)
                            | ((WORD)local_seed[4*i + 3]      );
        }
    }

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

        /* Glyph-translate the first 4 digest words (= first 16 bytes
           in big-endian byte order) and pack the resulting 16 glyph
           bytes back into 4 big-endian words. The packed words are
           exactly what vanity_pubkey_sha256_words wants for W0[8..11]. */
        WORD seed_words[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            WORD s = digest_words[k];
            seed_words[k] = ((WORD)glyph_from_hash_byte[(s >> 24) & 0xFFU] << 24)
                          | ((WORD)glyph_from_hash_byte[(s >> 16) & 0xFFU] << 16)
                          | ((WORD)glyph_from_hash_byte[(s >>  8) & 0xFFU] <<  8)
                          | ((WORD)glyph_from_hash_byte[(s      ) & 0xFFU]      );
        }

        vanity_pubkey_sha256_words(seed_words, digest_words);

        if (fd_base58_check_match_32_words(digest_words, target, target_len, suffix, suffix_len))
        {
            if (atomicMax(&done, 1) == 0)
            {
                /* Rare path: serialize the matched seed to big-endian
                   bytes for the host to consume. */
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    WORD w = seed_words[k];
                    out[4*k    ] = (uint8_t)(w >> 24);
                    out[4*k + 1] = (uint8_t)(w >> 16);
                    out[4*k + 2] = (uint8_t)(w >>  8);
                    out[4*k + 3] = (uint8_t)(w      );
                }
            }

            atomicAdd(&count, iter + 1);
            return;
        }
    }
}
