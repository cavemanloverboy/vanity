/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>

#include "sha256.h"

/* cuda_sha256_transform / _w / _expand_w are defined inline in sha256.h
   so __launch_bounds__ register caps on the calling kernel apply
   transitively. */

__device__ void cuda_sha256_init(CUDA_SHA256_CTX *ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667;
    ctx->state[1] = 0xbb67ae85;
    ctx->state[2] = 0x3c6ef372;
    ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f;
    ctx->state[5] = 0x9b05688c;
    ctx->state[6] = 0x1f83d9ab;
    ctx->state[7] = 0x5be0cd19;
}

__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[], size_t len)
{
    WORD i;

    for (i = 0; i < len; ++i)
    {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64)
        {
            cuda_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE hash[])
{
    WORD i;

    i = ctx->datalen;

    if (ctx->datalen < 56)
    {
        ctx->data[i++] = 0x80;
        while (i < 56)
            ctx->data[i++] = 0x00;
    }
    else
    {
        ctx->data[i++] = 0x80;
        while (i < 64)
            ctx->data[i++] = 0x00;
        cuda_sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;
    ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16;
    ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32;
    ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48;
    ctx->data[56] = ctx->bitlen >> 56;
    cuda_sha256_transform(ctx, ctx->data);

    /* Big-endian emission of digest words */
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
