#ifndef ED25519_GE_H
#define ED25519_GE_H

#include "fe.h"

typedef struct { fe X; fe Y; fe Z; } ge_p2;
typedef struct { fe X; fe Y; fe Z; fe T; } ge_p3;
typedef struct { fe X; fe Y; fe Z; fe T; } ge_p1p1;
typedef struct { fe yplusx; fe yminusx; fe xy2d; } ge_precomp;
/* Temporary coordinates used to build the table. */
typedef struct { fe yplusx; fe yminusx; fe z; fe t2d; } ge_niels;
typedef struct { fe yplusx; fe yminusx; fe t2d; } ge_affine;

#ifndef COMB_W
#define COMB_W 8
#endif
#if COMB_W < 5 || COMB_W > 12
#error "COMB_W must be between 5 and 12"
#endif
#define COMB_WINDOWS ((256 + COMB_W - 1) / COMB_W)
#define COMB_POS (1 << (COMB_W - 1))
#define COMB_TABLE_LEN (COMB_WINDOWS * COMB_POS)
#define KP_BATCH_MAX  32

/* Stored by window, coordinate, limb, then entry.
   Coordinates are y+x, y-x, and 2*d*t. No need to store z=1. */
typedef struct { int32_t limbs[COMB_TABLE_LEN * 3 * 10]; } ge_comb_table;

__device__ void ge_p3_tobytes(unsigned char *s, const ge_p3 *h);
__device__ void ge_p3_tobytes_inv(unsigned char *s, const fe X, const fe Y, const fe zinv);
__device__ void fe_batch_invert(fe zs[KP_BATCH_MAX], int n);
__device__ void ge_scalarmult_base(ge_p3 *h, const unsigned char *a);
/* Build and normalize the table before use. */
__device__ void ge_scalarmult_base_comb(ge_p3 *h, const unsigned char *a, const ge_comb_table *table);
__global__ void build_comb_table(ge_niels *table);
__global__ void normalize_comb_table(const ge_niels *table, ge_comb_table *out);

#endif
