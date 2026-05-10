#ifndef SHA256_H
#define SHA256_H

typedef unsigned char BYTE;
typedef unsigned int WORD;
typedef unsigned long long LONG;

typedef struct
{
    BYTE data[64];
    WORD datalen;
    unsigned long long bitlen;
    WORD state[8];
} CUDA_SHA256_CTX;

__device__ void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const BYTE data[]);

/* variant of cuda_sha256_transform that takes a fully-expanded 64-word
   message schedule W[0..63] and runs only the 64 main rounds. lets the
   caller hoist W out of an inner loop when the block being hashed is
   loop-invariant. */
__device__ void cuda_sha256_transform_w(WORD state[8], const WORD W[64]);

/* build the 64-word SHA-256 message schedule from a 64-byte block:
   W[0..15] = big-endian word view of data, W[16..63] expanded. */
__device__ void cuda_sha256_expand_w(const BYTE data[64], WORD W[64]);

__device__ void cuda_sha256_init(CUDA_SHA256_CTX *ctx);

__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE hash[]);
__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[], size_t len);

#endif