#ifndef ED25519_GE_H
#define ED25519_GE_H

#include "fe.h"

typedef struct { fe X; fe Y; fe Z; } ge_p2;
typedef struct { fe X; fe Y; fe Z; fe T; } ge_p3;
typedef struct { fe X; fe Y; fe Z; fe T; } ge_p1p1;
typedef struct { fe yplusx; fe yminusx; fe xy2d; } ge_precomp;
/* Extended Niels for radix-32 comb (matches CPU SIMD / OpenCL). */
typedef struct { fe yplusx; fe yminusx; fe z; fe t2d; } ge_niels;

#define COMB_W        5
#define COMB_WINDOWS  52
#define COMB_POS      16
#define COMB_TABLE_LEN (COMB_WINDOWS * COMB_POS)
#define KP_BATCH_MAX  32

__device__ void ge_p3_tobytes(unsigned char *s, const ge_p3 *h);
__device__ void ge_p3_tobytes_inv(unsigned char *s, const fe X, const fe Y, const fe zinv);
__device__ void fe_batch_invert(fe zs[KP_BATCH_MAX], int n);
__device__ void ge_scalarmult_base(ge_p3 *h, const unsigned char *a);
__device__ void ge_scalarmult_base_comb(ge_p3 *h, const unsigned char *a, const ge_niels *table);
__global__ void build_comb_table(ge_niels *table);

#endif
