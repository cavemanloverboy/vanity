#include "ge.h"
#include "precomp_data.h"

#ifndef ED25519_GE_CU
#define ED25519_GE_CU

static __device__ void ge_madd(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q) {
    fe t0;
    fe_add(r->X, p->Y, p->X);
    fe_sub(r->Y, p->Y, p->X);
    fe_mul(r->Z, r->X, q->yplusx);
    fe_mul(r->Y, r->Y, q->yminusx);
    fe_mul(r->T, q->xy2d, p->T);
    fe_add(t0, p->Z, p->Z);
    fe_sub(r->X, r->Z, r->Y);
    fe_add(r->Y, r->Z, r->Y);
    fe_add(r->Z, t0, r->T);
    fe_sub(r->T, t0, r->T);
}

static __device__ void ge_p1p1_to_p2(ge_p2 *r, const ge_p1p1 *p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
}

static __device__ void ge_p1p1_to_p3(ge_p3 *r, const ge_p1p1 *p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
    fe_mul(r->T, p->X, p->Y);
}

static __device__ void ge_p2_dbl(ge_p1p1 *r, const ge_p2 *p) {
    fe t0;
    fe_sq(r->X, p->X);
    fe_sq(r->Z, p->Y);
    fe_sq2(r->T, p->Z);
    fe_add(r->Y, p->X, p->Y);
    fe_sq(t0, r->Y);
    fe_add(r->Y, r->Z, r->X);
    fe_sub(r->Z, r->Z, r->X);
    fe_sub(r->X, t0, r->Y);
    fe_sub(r->T, r->T, r->Z);
}

static __device__ void ge_p3_0(ge_p3 *h) {
    fe_0(h->X);
    fe_1(h->Y);
    fe_1(h->Z);
    fe_0(h->T);
}

static __device__ void ge_p3_to_p2(ge_p2 *r, const ge_p3 *p) {
    fe_copy(r->X, p->X);
    fe_copy(r->Y, p->Y);
    fe_copy(r->Z, p->Z);
}

static __device__ void ge_p3_dbl(ge_p1p1 *r, const ge_p3 *p) {
    ge_p2 q;
    ge_p3_to_p2(&q, p);
    ge_p2_dbl(r, &q);
}

__device__ void ge_p3_tobytes(unsigned char *s, const ge_p3 *h) {
    fe recip;
    fe x;
    fe y;
    fe_invert(recip, h->Z);
    fe_mul(x, h->X, recip);
    fe_mul(y, h->Y, recip);
    fe_tobytes(s, y);
    s[31] ^= fe_isnegative(x) << 7;
}

static __device__ unsigned char negative(signed char b) {
    uint64_t x = b;
    x >>= 63;
    return (unsigned char) x;
}

static __device__ void ge_select(ge_precomp *t, int pos, signed char b) {
    /* Vanity grinding is not constant-time; direct table lookup beats the
       ref10 cmov ladder (8 fe_cmovs × 8 table slots per nibble). */
    unsigned char bnegative = negative(b);
    unsigned char babs = b - (((-bnegative) & b) << 1);

    if (babs == 0) {
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_0(t->xy2d);
        return;
    }

    const ge_precomp *u = &base[pos][babs - 1];
    fe_copy(t->yplusx, u->yplusx);
    fe_copy(t->yminusx, u->yminusx);
    fe_copy(t->xy2d, u->xy2d);

    if (bnegative) {
        fe tmp;
        fe_copy(tmp, t->yplusx);
        fe_copy(t->yplusx, t->yminusx);
        fe_copy(t->yminusx, tmp);
        fe_neg(t->xy2d, t->xy2d);
    }
}

__device__ void ge_scalarmult_base(ge_p3 *h, const unsigned char *a) {
    signed char e[64];
    signed char carry;
    ge_p1p1 r;
    ge_p2 s;
    ge_precomp t;
    int i;

    for (i = 0; i < 32; ++i) {
        e[2 * i + 0] = (a[i] >> 0) & 15;
        e[2 * i + 1] = (a[i] >> 4) & 15;
    }

    carry = 0;
    for (i = 0; i < 63; ++i) {
        e[i] += carry;
        carry = e[i] + 8;
        carry >>= 4;
        e[i] -= carry << 4;
    }
    e[63] += carry;

    ge_p3_0(h);

    for (i = 1; i < 64; i += 2) {
        ge_select(&t, i / 2, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }

    ge_p3_dbl(&r, h);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p2(&s, &r);
    ge_p2_dbl(&r, &s);
    ge_p1p1_to_p3(h, &r);

    for (i = 0; i < 64; i += 2) {
        ge_select(&t, i / 2, e[i]);
        ge_madd(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
}

/* Compress using a precomputed Z^{-1} (from fe_batch_invert). */
__device__ void ge_p3_tobytes_inv(unsigned char *s, const fe X, const fe Y, const fe zinv)
{
    fe x;
    fe y;
    fe_mul(x, X, zinv);
    fe_mul(y, Y, zinv);
    fe_tobytes(s, y);
    s[31] ^= fe_isnegative(x) << 7;
}

/* Montgomery batch invert: zs[i] <- 1/zs[i] for i in [0, n). n >= 1. */
__device__ void fe_batch_invert(fe zs[KP_BATCH_MAX], int n)
{
    fe scratch[KP_BATCH_MAX];
    fe acc;
    fe_1(acc);
    for (int i = 0; i < n; i++) {
        fe_copy(scratch[i], acc);
        fe_mul(acc, acc, zs[i]);
    }
    fe_invert(acc, acc);
    for (int i = n - 1; i >= 0; i--) {
        fe tmp;
        fe_mul(tmp, acc, zs[i]);
        fe_mul(zs[i], acc, scratch[i]);
        fe_copy(acc, tmp);
    }
}

static __device__ void ge_p3_to_niels(ge_niels *n, const ge_p3 *p, const fe d2)
{
    fe_add(n->yplusx, p->Y, p->X);
    fe_sub(n->yminusx, p->Y, p->X);
    fe_copy(n->z, p->Z);
    fe_mul(n->t2d, p->T, d2);
}

static __device__ void ge_add_niels(ge_p1p1 *r, const ge_p3 *p, const ge_niels *n)
{
    fe y_plus_x, y_minus_x, pp, mm, tt2d, zz, zz2;
    fe_add(y_plus_x, p->Y, p->X);
    fe_sub(y_minus_x, p->Y, p->X);
    fe_mul(pp, y_plus_x, n->yplusx);
    fe_mul(mm, y_minus_x, n->yminusx);
    fe_mul(tt2d, p->T, n->t2d);
    fe_mul(zz, p->Z, n->z);
    fe_add(zz2, zz, zz);
    fe_sub(r->X, pp, mm);
    fe_add(r->Y, pp, mm);
    fe_add(r->Z, zz2, tt2d);
    fe_sub(r->T, zz2, tt2d);
}

static __device__ void ge_niels_select(ge_niels *t, const ge_niels *window, signed char b)
{
    unsigned char bnegative = negative(b);
    unsigned char babs = b - (((-bnegative) & b) << 1);

    if (babs == 0) {
        fe_1(t->yplusx);
        fe_1(t->yminusx);
        fe_1(t->z);
        fe_0(t->t2d);
        return;
    }

    const ge_niels *u = &window[babs - 1];
    fe_copy(t->yplusx, u->yplusx);
    fe_copy(t->yminusx, u->yminusx);
    fe_copy(t->z, u->z);
    fe_copy(t->t2d, u->t2d);

    if (bnegative) {
        fe tmp;
        fe_copy(tmp, t->yplusx);
        fe_copy(t->yplusx, t->yminusx);
        fe_copy(t->yminusx, tmp);
        fe_neg(t->t2d, t->t2d);
    }
}

static __device__ void to_radix32(signed char e[COMB_WINDOWS], const unsigned char *a)
{
    const unsigned short mask = (unsigned short)((1u << COMB_W) - 1u);
    for (int i = 0; i < COMB_WINDOWS; i++) {
        int bit = COMB_W * i;
        int byte = bit / 8;
        int off = bit % 8;
        unsigned short lo = a[byte];
        unsigned short hi = (byte + 1 < 32) ? a[byte + 1] : 0;
        e[i] = (signed char)(((lo | (hi << 8)) >> off) & mask);
    }
    const signed char pos = (signed char)COMB_POS;
    signed char carry = 0;
    for (int i = 0; i < COMB_WINDOWS - 1; i++) {
        e[i] += carry;
        carry = (signed char)((e[i] + pos) >> COMB_W);
        e[i] -= (signed char)(carry << COMB_W);
    }
    e[COMB_WINDOWS - 1] += carry;
}

__device__ void ge_scalarmult_base_comb(ge_p3 *h, const unsigned char *a, const ge_niels *table)
{
    signed char e[COMB_WINDOWS];
    to_radix32(e, a);
    ge_p3_0(h);
    for (int i = 0; i < COMB_WINDOWS; i++) {
        ge_niels t;
        ge_p1p1 r;
        ge_niels_select(&t, &table[i * COMB_POS], e[i]);
        ge_add_niels(&r, h, &t);
        ge_p1p1_to_p3(h, &r);
    }
}

/* One-shot: table[window][1..16] = (k+1) * 32^window * B. */
__global__ void build_comb_table(ge_niels *table)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    fe d2;
    {
        fe t, inv, d;
        fe_0(t);
        t[0] = 121666;
        fe_invert(inv, t);
        fe_0(t);
        t[0] = 121665;
        fe_neg(t, t);
        fe_mul(d, t, inv);
        fe_add(d2, d, d);
    }

    unsigned char one[32];
    for (int i = 0; i < 32; i++) one[i] = 0;
    one[0] = 1;

    ge_p3 cur;
    ge_scalarmult_base(&cur, one); /* B */

    for (int w = 0; w < COMB_WINDOWS; w++) {
        ge_niels cur_n;
        ge_p3_to_niels(&cur_n, &cur, d2);

        ge_p3 mult;
        fe_copy(mult.X, cur.X);
        fe_copy(mult.Y, cur.Y);
        fe_copy(mult.Z, cur.Z);
        fe_copy(mult.T, cur.T);

        table[w * COMB_POS + 0] = cur_n;
        for (int k = 1; k < COMB_POS; k++) {
            ge_p1p1 r;
            ge_add_niels(&r, &mult, &cur_n);
            ge_p1p1_to_p3(&mult, &r);
            ge_p3_to_niels(&table[w * COMB_POS + k], &mult, d2);
        }

        for (int d = 0; d < COMB_W; d++) {
            ge_p1p1 r;
            ge_p3_dbl(&r, &cur);
            ge_p1p1_to_p3(&cur, &r);
        }
    }
}

#endif
