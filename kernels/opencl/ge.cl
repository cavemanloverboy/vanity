/* ge.cl — OpenCL port of kernels/ed25519/ge.cu (ref10 group operations).

   Mirrors the CUDA source; the only structural change is cmov_c/fe_cmov_c
   for reading the precomputed base table out of __constant memory, since
   OpenCL 1.2 lacks a generic address space (see fe.cl). */

static void ge_madd(ge_p1p1 *r, const ge_p3 *p, const ge_precomp *q) {
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

static void ge_p1p1_to_p2(ge_p2 *r, const ge_p1p1 *p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
}

static void ge_p1p1_to_p3(ge_p3 *r, const ge_p1p1 *p) {
    fe_mul(r->X, p->X, p->T);
    fe_mul(r->Y, p->Y, p->Z);
    fe_mul(r->Z, p->Z, p->T);
    fe_mul(r->T, p->X, p->Y);
}

static void ge_p2_dbl(ge_p1p1 *r, const ge_p2 *p) {
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

static void ge_p3_0(ge_p3 *h) {
    fe_0(h->X);
    fe_1(h->Y);
    fe_1(h->Z);
    fe_0(h->T);
}

static void ge_p3_to_p2(ge_p2 *r, const ge_p3 *p) {
    fe_copy(r->X, p->X);
    fe_copy(r->Y, p->Y);
    fe_copy(r->Z, p->Z);
}

static void ge_p3_dbl(ge_p1p1 *r, const ge_p3 *p) {
    ge_p2 q;
    ge_p3_to_p2(&q, p);
    ge_p2_dbl(r, &q);
}

void ge_p3_tobytes(uchar *s, const ge_p3 *h) {
    fe recip;
    fe x;
    fe y;
    fe_invert(recip, h->Z);
    fe_mul(x, h->X, recip);
    fe_mul(y, h->Y, recip);
    fe_tobytes(s, y);
    s[31] ^= fe_isnegative(x) << 7;
}

static uchar equal(signed char b, signed char c) {
    uchar ub = b;
    uchar uc = c;
    uchar x = ub ^ uc;
    uint64_t y = x;
    y -= 1;
    y >>= 63;
    return (uchar) y;
}

static uchar negative(signed char b) {
    uint64_t x = b;
    x >>= 63;
    return (uchar) x;
}

/* Conditional move from a private-memory precomp (minust). */
static void cmov(ge_precomp *t, const ge_precomp *u, uchar b) {
    fe_cmov(t->yplusx, u->yplusx, b);
    fe_cmov(t->yminusx, u->yminusx, b);
    fe_cmov(t->xy2d, u->xy2d, b);
}

/* Conditional move from a __constant precomp (the base table). */
static void cmov_c(ge_precomp *t, __constant const ge_precomp *u, uchar b) {
    fe_cmov_c(t->yplusx, u->yplusx, b);
    fe_cmov_c(t->yminusx, u->yminusx, b);
    fe_cmov_c(t->xy2d, u->xy2d, b);
}

static void ge_select(ge_precomp *t, int pos, signed char b) {
    ge_precomp minust;
    uchar bnegative = negative(b);
    uchar babs = b - (((-bnegative) & b) << 1);
    fe_1(t->yplusx);
    fe_1(t->yminusx);
    fe_0(t->xy2d);
    cmov_c(t, &base[pos][0], equal(babs, 1));
    cmov_c(t, &base[pos][1], equal(babs, 2));
    cmov_c(t, &base[pos][2], equal(babs, 3));
    cmov_c(t, &base[pos][3], equal(babs, 4));
    cmov_c(t, &base[pos][4], equal(babs, 5));
    cmov_c(t, &base[pos][5], equal(babs, 6));
    cmov_c(t, &base[pos][6], equal(babs, 7));
    cmov_c(t, &base[pos][7], equal(babs, 8));
    fe_copy(minust.yplusx, t->yminusx);
    fe_copy(minust.yminusx, t->yplusx);
    fe_neg(minust.xy2d, t->xy2d);
    cmov(t, &minust, bnegative);
}

void ge_scalarmult_base(ge_p3 *h, const uchar *a) {
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
