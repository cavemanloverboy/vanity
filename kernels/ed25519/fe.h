#ifndef ED25519_FE_H
#define ED25519_FE_H

#include "fixedint.h"

typedef int32_t fe[10];

__device__ void fe_0(fe h);
__device__ void fe_1(fe h);
__device__ void fe_frombytes(fe h, const unsigned char *s);
__device__ void fe_tobytes(unsigned char *s, const fe h);
__device__ void fe_copy(fe h, const fe f);
__device__ int  fe_isnegative(const fe f);
__device__ int  fe_isnonzero(const fe f);
__device__ void fe_cmov(fe f, const fe g, unsigned int b);
__device__ void fe_neg(fe h, const fe f);
__device__ void fe_add(fe h, const fe f, const fe g);
__device__ void fe_sub(fe h, const fe f, const fe g);
__device__ void fe_mul(fe h, const fe f, const fe g);
__device__ void fe_sq(fe h, const fe f);
__device__ void fe_sq2(fe h, const fe f);
__device__ void fe_invert(fe out, const fe z);
__device__ void fe_pow22523(fe out, const fe z);

#endif
