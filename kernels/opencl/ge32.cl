/* ge32.cl — Ed25519 group operations for the Apple GPU keypair path: a signed
   fixed-base comb over affine multiples, computed in fe32 (fe32.cl), and the
   kernels that build its table. The table builder does its arithmetic with
   ref10 (fe.cl, ge.cl) and stores canonical fe32 limbs.

   Against the radix-32 comb in ge.cl (52 windows at 8 multiplies each), the
   default 13-bit comb needs 20 windows at 7 multiplies (ref10's mixed
   addition), window 0 costs none because it stores (x, y, xy), and the last
   window skips T, which encoding never reads: about 132 multiplies per key
   instead of 416. */

/* Copy a field element out of global memory; OpenCL 1.2 has no generic
   address space, so the private-pointer fe helpers cannot read it. */
static void fe_load_global(fe h, __global const int32_t *src) {
    for (int i = 0; i < 10; i++) h[i] = src[i];
}

/* Copy a field element into global memory. */
static void fe_store_global(__global int32_t *dst, const fe h) {
    for (int i = 0; i < 10; i++) dst[i] = h[i];
}

/* The curve constant 2d, derived as ref10 does from d = -121665/121666. */
static void fe_d2(fe d2) {
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

/* Recode a clamped scalar into COMB32_WINDOWS signed digits: every digit but
   the top lies in [-COMB32_POS, COMB32_POS), and the top in [0, COMB32_POS]. */
static void comb32_digits(int e[COMB32_WINDOWS], const uchar *a) {
    const uint mask = (1u << COMB32_W) - 1u;
    for (int i = 0; i < COMB32_WINDOWS; i++) {
        int bit = COMB32_W * i;
        int byte = bit >> 3;
        uint v = a[byte];
        if (byte + 1 < 32) v |= (uint)a[byte + 1] << 8;
        if (byte + 2 < 32) v |= (uint)a[byte + 2] << 16;
        e[i] = (int)((v >> (bit & 7)) & mask);
    }
    int carry = 0;
    for (int i = 0; i < COMB32_WINDOWS - 1; i++) {
        e[i] += carry;
        carry = (e[i] + COMB32_POS) >> COMB32_W;
        e[i] -= carry << COMB32_W;
    }
    e[COMB32_WINDOWS - 1] += carry;
}

/* ─── search path over fe32 ────────────────────────────────────────────── */

/* Clamp a SHA-512 digest's low half into an Ed25519 secret scalar. */
static void clamp_scalar(uchar privatek[64]) {
    privatek[0]  &= 248;
    privatek[31] &= 63;
    privatek[31] |= 64;
}

/* Step `seed` `steps` times along the key chain, seed <- SHA-512(seed)[32..64]:
   how the search kernels recover a matched key from its batch's first seed. */
static void walk_chain(uchar seed[32], int steps) {
    uchar digest[64];
    for (int t = 0; t < steps; t++) {
        sha512_32(digest, seed);
        for (int i = 0; i < 32; i++) seed[i] = digest[32 + i];
    }
}

/* r = p + q for an extended point p and an affine precomputed point q:
   ref10's ge_madd over fe32. */
static void ge32_madd(ge32_p1p1 *r, const ge32_p3 *p, const ge32_precomp *q) {
    fe32 t0;
    fe32_add(r->X, p->Y, p->X);
    fe32_sub(r->Y, p->Y, p->X);
    fe32_mul(r->Z, r->X, q->yplusx);
    fe32_mul(r->Y, r->Y, q->yminusx);
    fe32_mul(r->T, q->xy2d, p->T);
    fe32_add(t0, p->Z, p->Z);
    fe32_sub(r->X, r->Z, r->Y);
    fe32_add(r->Y, r->Z, r->Y);
    fe32_add(r->Z, t0, r->T);
    fe32_sub(r->T, t0, r->T);
}

/* Extended coordinates of a completed point. */
static void ge32_p1p1_to_p3(ge32_p3 *r, const ge32_p1p1 *p) {
    fe32_mul(r->X, p->X, p->T);
    fe32_mul(r->Y, p->Y, p->Z);
    fe32_mul(r->Z, p->Z, p->T);
    fe32_mul(r->T, p->X, p->Y);
}

/* Projective coordinates of a completed point, without T. */
static void ge32_p1p1_to_p2(ge32_p2 *r, const ge32_p1p1 *p) {
    fe32_mul(r->X, p->X, p->T);
    fe32_mul(r->Y, p->Y, p->Z);
    fe32_mul(r->Z, p->Z, p->T);
}

/* Load the window entry for `digit` into t: the multiple |digit|, negated
   for a negative digit, or the identity for zero. */
static void ge32_precomp_select(ge32_precomp *t, __global const ge32_precomp *window,
                                int digit) {
    if (digit == 0) {
        fe32_1(t->yplusx);
        fe32_1(t->yminusx);
        fe32_0(t->xy2d);
        return;
    }
    __global const ge32_precomp *u = &window[(digit < 0 ? -digit : digit) - 1];
    if (digit < 0) {
        fe32_load_global(t->yplusx, u->yminusx);
        fe32_load_global(t->yminusx, u->yplusx);
        fe32_load_global(t->xy2d, u->xy2d);
        fe32_neg(t->xy2d, t->xy2d);
    } else {
        fe32_load_global(t->yplusx, u->yplusx);
        fe32_load_global(t->yminusx, u->yminusx);
        fe32_load_global(t->xy2d, u->xy2d);
    }
}

/* Start the accumulator at window 0's entry for `digit`, as the extended
   point (x : y : 1 : xy), negated for a negative digit. */
static void ge32_affine_start(ge32_p3 *h, __global const ge32_affine *window, int digit) {
    if (digit == 0) {
        fe32_0(h->X);
        fe32_1(h->Y);
        fe32_1(h->Z);
        fe32_0(h->T);
        return;
    }
    __global const ge32_affine *u = &window[(digit < 0 ? -digit : digit) - 1];
    fe32_load_global(h->X, u->x);
    fe32_load_global(h->Y, u->y);
    fe32_1(h->Z);
    fe32_load_global(h->T, u->xy);
    if (digit < 0) {
        fe32_neg(h->X, h->X);
        fe32_neg(h->T, h->T);
    }
}

/* One middle comb window, acc += the entry for `digit`, kept out of line
   for the same instruction-cache reason as fe32_mul. */
__attribute__((noinline)) static void ge32_window_step(ge32_p3 *acc, __global const ge32_precomp *window, int digit) {
    ge32_precomp t;
    ge32_p1p1 r;
    ge32_precomp_select(&t, window, digit);
    ge32_madd(&r, acc, &t);
    ge32_p1p1_to_p3(acc, &r);
}

/* h = a*B for a clamped scalar a. The last window stops at projective
   (X : Y : Z) because encoding never reads T. */
void ge32_scalarmult_base_comb(ge32_p2 *h, const uchar *a,
                               __global const ge32_precomp *table) {
    int e[COMB32_WINDOWS];
    comb32_digits(e, a);

    ge32_p3 acc;
    ge32_affine_start(&acc, (__global const ge32_affine *)table, e[0]);

    ge32_precomp t;
    ge32_p1p1 r;
    for (int i = 1; i < COMB32_WINDOWS - 1; i++) ge32_window_step(&acc, &table[i * COMB32_POS], e[i]);
    ge32_precomp_select(&t, &table[(COMB32_WINDOWS - 1) * COMB32_POS], e[COMB32_WINDOWS - 1]);
    ge32_madd(&r, &acc, &t);
    ge32_p1p1_to_p2(h, &r);
}

/* Canonical affine y = Y * zinv of (X : Y : Z), given zinv = 1/Z. */
static void ge32_affine_y(fe32 y, const fe32 Y, const fe32 zinv) {
    fe32_mul(y, Y, zinv);
    fe32_canonical(y);
}

/* Put the sign of x = X * zinv (its low bit) into bit 255 of a canonical y,
   completing the Ed25519 encoding as eight little-endian limbs. */
static void ge32_encode_sign(fe32 y, const fe32 X, const fe32 zinv) {
    fe32 x;
    fe32_mul(x, X, zinv);
    fe32_canonical(x);
    y[7] |= (x[0] & 1u) << 31;
}

/* Reverse a limb's bytes: word k of an encoding's big-endian view is limb k
   reversed. */
static uint bswap32(uint v) {
    return (v >> 24) | ((v >> 8) & 0xff00u) | ((v << 8) & 0xff0000u) | (v << 24);
}

/* The Ed25519 encoding of (X : Y : Z) as 32 bytes, given zinv = 1/Z. */
static void ge32_encode(uchar s[32], const fe32 X, const fe32 Y, const fe32 zinv) {
    fe32 y;
    ge32_affine_y(y, Y, zinv);
    ge32_encode_sign(y, X, zinv);
    for (int k = 0; k < 8; k++)
        for (int b = 0; b < 4; b++) s[4 * k + b] = (uchar)(y[k] >> (8 * b));
}

/* Store a ref10 field element into global memory as canonical fe32 limbs. */
static void fe_store_fe32_global(__global uint *dst, const fe h) {
    uchar s[32];
    fe_tobytes(s, h);
    for (int k = 0; k < 8; k++)
        dst[k] = (uint)s[4 * k] | ((uint)s[4 * k + 1] << 8)
               | ((uint)s[4 * k + 2] << 16) | ((uint)s[4 * k + 3] << 24);
}


/* Store each window's base 2^(COMB32_W*w)*B as an extended point, 40 limbs
   per window. Runs on one work-item: the bases form a doubling chain. */
__kernel void build_comb32_bases(__global int32_t *bases) {
    if (get_global_id(0) != 0) return;

    uchar one[32];
    for (int i = 0; i < 32; i++) one[i] = 0;
    one[0] = 1;

    ge_p3 cur;
    ge_scalarmult_base(&cur, one);

    for (int w = 0; w < COMB32_WINDOWS; w++) {
        __global int32_t *slot = bases + w * 40;
        fe_store_global(slot, cur.X);
        fe_store_global(slot + 10, cur.Y);
        fe_store_global(slot + 20, cur.Z);
        fe_store_global(slot + 30, cur.T);
        for (int d = 0; d < COMB32_W; d++) {
            ge_p1p1 r;
            ge_p3_dbl(&r, &cur);
            ge_p1p1_to_p3(&cur, &r);
        }
    }
}

/* Fill one table entry per work-item: k times its window's base, in affine
   form as canonical fe32 limbs. Window 0 stores (x, y, xy) for
   ge32_affine_start; every other window stores (y+x, y-x, 2dxy) for
   ge32_madd. The arithmetic here is ref10's; only the stored form is fe32. */
__kernel void build_comb32_table(__global const int32_t *bases,
                                 __global ge32_precomp *table) {
    uint gid = get_global_id(0);
    if (gid >= COMB32_TABLE_LEN) return;
    uint w = gid / COMB32_POS;
    uint k = gid % COMB32_POS + 1;

    fe d2;
    fe_d2(d2);

    __global const int32_t *slot = bases + w * 40;
    ge_p3 base;
    fe_load_global(base.X, slot);
    fe_load_global(base.Y, slot + 10);
    fe_load_global(base.Z, slot + 20);
    fe_load_global(base.T, slot + 30);
    ge_niels base_niels;
    ge_p3_to_niels(&base_niels, &base, d2);

    ge_p3 acc = base;
    for (int bit = 30 - (int)clz(k); bit >= 0; bit--) {
        ge_p1p1 r;
        ge_p3_dbl(&r, &acc);
        ge_p1p1_to_p3(&acc, &r);
        if ((k >> bit) & 1u) {
            ge_add_niels(&r, &acc, &base_niels);
            ge_p1p1_to_p3(&acc, &r);
        }
    }

    fe zinv, x, y, xy;
    fe_invert(zinv, acc.Z);
    fe_mul(x, acc.X, zinv);
    fe_mul(y, acc.Y, zinv);
    fe_mul(xy, x, y);

    if (w == 0) {
        __global ge32_affine *entry = (__global ge32_affine *)table + gid;
        fe_store_fe32_global(entry->x, x);
        fe_store_fe32_global(entry->y, y);
        fe_store_fe32_global(entry->xy, xy);
    } else {
        fe yplusx, yminusx, xy2d;
        fe_add(yplusx, y, x);
        fe_sub(yminusx, y, x);
        fe_mul(xy2d, xy, d2);
        fe_store_fe32_global(table[gid].yplusx, yplusx);
        fe_store_fe32_global(table[gid].yminusx, yminusx);
        fe_store_fe32_global(table[gid].xy2d, xy2d);
    }
}
