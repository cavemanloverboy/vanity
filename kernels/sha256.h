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

__device__ __forceinline__ void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const BYTE data[]);

__device__ void cuda_sha256_init(CUDA_SHA256_CTX *ctx);

__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE hash[]);
__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[], size_t len);

#endif