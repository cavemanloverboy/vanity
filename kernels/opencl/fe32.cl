/* fe32.cl — GF(2^255-19) arithmetic for the Apple GPU keypair path, as eight
   little-endian 32-bit limbs kept below 2^256 but not necessarily below p.
   A multiply costs 64 widening multiplies, against 100 for ref10's ten
   25.5-bit limbs (fe.cl); on Apple GPUs a widening multiply runs at 1/8 the
   rate of a float FMA, so it is the scarce instruction.

   The arithmetic is straight-line over named scalars, as ref10 writes it,
   and the multiply and square stay out of line: inlined at every call site
   they overflow the GPU's instruction cache and the search kernel runs at a
   third of the speed. */

#define FE32_LOAD(v, a) uint v##0 = a[0], v##1 = a[1], v##2 = a[2], v##3 = a[3], \
                        v##4 = a[4], v##5 = a[5], v##6 = a[6], v##7 = a[7]
/* Accumulate the 64-bit product x*y into columns lo and hi (lo + 2^32 hi). */
#define FE32_MAC(x, y, lo, hi) { ulong p_ = (ulong)(x) * (ulong)(y); lo += (uint)p_; hi += p_ >> 32; }

/* h = r mod p for a 512-bit r given as sixteen little-endian limbs, h < 2^256.
   2^256 = 38 (mod p): the high half folds onto the low half, leaving a
   carry below 39; folding that as 38*carry can carry out once more, and
   only when the low limbs are then below 1482, so the last carry adds 38
   to limb 0 without further carries. */
static void fe32_fold(fe32 h, uint r0, uint r1, uint r2, uint r3, uint r4, uint r5, uint r6, uint r7, uint r8, uint r9, uint r10, uint r11, uint r12, uint r13, uint r14, uint r15) {
    ulong c = 0;
    c += (ulong)r0 + (ulong)r8 * 38u; uint h0 = (uint)c; c >>= 32;
    c += (ulong)r1 + (ulong)r9 * 38u; uint h1 = (uint)c; c >>= 32;
    c += (ulong)r2 + (ulong)r10 * 38u; uint h2 = (uint)c; c >>= 32;
    c += (ulong)r3 + (ulong)r11 * 38u; uint h3 = (uint)c; c >>= 32;
    c += (ulong)r4 + (ulong)r12 * 38u; uint h4 = (uint)c; c >>= 32;
    c += (ulong)r5 + (ulong)r13 * 38u; uint h5 = (uint)c; c >>= 32;
    c += (ulong)r6 + (ulong)r14 * 38u; uint h6 = (uint)c; c >>= 32;
    c += (ulong)r7 + (ulong)r15 * 38u; uint h7 = (uint)c; c >>= 32;
    c *= 38u;
    c += h0; h0 = (uint)c; c >>= 32;
    c += h1; h1 = (uint)c; c >>= 32;
    c += h2; h2 = (uint)c; c >>= 32;
    c += h3; h3 = (uint)c; c >>= 32;
    c += h4; h4 = (uint)c; c >>= 32;
    c += h5; h5 = (uint)c; c >>= 32;
    c += h6; h6 = (uint)c; c >>= 32;
    c += h7; h7 = (uint)c; c >>= 32;
    h0 += (uint)c * 38u;
    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
    h[3] = h3;
    h[4] = h4;
    h[5] = h5;
    h[6] = h6;
    h[7] = h7;
}

#ifdef FE32_KARATSUBA
/* h = f * g (mod p) by one Karatsuba level over 128-bit halves, 48 limb
   products: L = f0 g0, H = f1 g1 and M = (f0 + f1)(g0 + g1), then
   f g = L + (M - L - H) 2^128 + H 2^256. Each half product runs by column as
   in the schoolbook version; the half sums carry one bit each, whose cross
   terms M picks up at limbs 4 and 8. Opt-in (VANITY_CL_OPTS=-DFE32_KARATSUBA):
   on an M3 Max it measured 1-2.6% slower in the search kernel than the
   schoolbook multiply, its extra signed additions and larger body costing
   more than the 16 multiplies it saves. */
__attribute__((noinline)) static void fe32_mul(fe32 h, const fe32 f, const fe32 g) {
    FE32_LOAD(f, f);
    FE32_LOAD(g, g);
    ulong t = 0;
    t += (ulong)f0 + f4; uint s0 = (uint)t; t >>= 32;
    t += (ulong)f1 + f5; uint s1 = (uint)t; t >>= 32;
    t += (ulong)f2 + f6; uint s2 = (uint)t; t >>= 32;
    t += (ulong)f3 + f7; uint s3 = (uint)t; t >>= 32;
    uint carry_f = (uint)t;
    t = 0;
    t += (ulong)g0 + g4; uint u0 = (uint)t; t >>= 32;
    t += (ulong)g1 + g5; uint u1 = (uint)t; t >>= 32;
    t += (ulong)g2 + g6; uint u2 = (uint)t; t >>= 32;
    t += (ulong)g3 + g7; uint u3 = (uint)t; t >>= 32;
    uint carry_g = (uint)t;
    uint mask_f = 0u - carry_f, mask_g = 0u - carry_g;
    ulong c, high, next_high, low;
    c = 0; high = 0;
    low = 0; next_high = 0;
    FE32_MAC(f0, g0, low, next_high)
    c += low + high; uint L0 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g1, low, next_high)
    FE32_MAC(f1, g0, low, next_high)
    c += low + high; uint L1 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g2, low, next_high)
    FE32_MAC(f1, g1, low, next_high)
    FE32_MAC(f2, g0, low, next_high)
    c += low + high; uint L2 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g3, low, next_high)
    FE32_MAC(f1, g2, low, next_high)
    FE32_MAC(f2, g1, low, next_high)
    FE32_MAC(f3, g0, low, next_high)
    c += low + high; uint L3 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f1, g3, low, next_high)
    FE32_MAC(f2, g2, low, next_high)
    FE32_MAC(f3, g1, low, next_high)
    c += low + high; uint L4 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f2, g3, low, next_high)
    FE32_MAC(f3, g2, low, next_high)
    c += low + high; uint L5 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f3, g3, low, next_high)
    c += low + high; uint L6 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    c += low + high; uint L7 = (uint)c; c >>= 32; high = next_high;
    c = 0; high = 0;
    low = 0; next_high = 0;
    FE32_MAC(f4, g4, low, next_high)
    c += low + high; uint H0 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f4, g5, low, next_high)
    FE32_MAC(f5, g4, low, next_high)
    c += low + high; uint H1 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f4, g6, low, next_high)
    FE32_MAC(f5, g5, low, next_high)
    FE32_MAC(f6, g4, low, next_high)
    c += low + high; uint H2 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f4, g7, low, next_high)
    FE32_MAC(f5, g6, low, next_high)
    FE32_MAC(f6, g5, low, next_high)
    FE32_MAC(f7, g4, low, next_high)
    c += low + high; uint H3 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f5, g7, low, next_high)
    FE32_MAC(f6, g6, low, next_high)
    FE32_MAC(f7, g5, low, next_high)
    c += low + high; uint H4 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f6, g7, low, next_high)
    FE32_MAC(f7, g6, low, next_high)
    c += low + high; uint H5 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f7, g7, low, next_high)
    c += low + high; uint H6 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    c += low + high; uint H7 = (uint)c; c >>= 32; high = next_high;
    c = 0; high = 0;
    low = 0; next_high = 0;
    FE32_MAC(s0, u0, low, next_high)
    c += low + high; uint M0 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(s0, u1, low, next_high)
    FE32_MAC(s1, u0, low, next_high)
    c += low + high; uint M1 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(s0, u2, low, next_high)
    FE32_MAC(s1, u1, low, next_high)
    FE32_MAC(s2, u0, low, next_high)
    c += low + high; uint M2 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(s0, u3, low, next_high)
    FE32_MAC(s1, u2, low, next_high)
    FE32_MAC(s2, u1, low, next_high)
    FE32_MAC(s3, u0, low, next_high)
    c += low + high; uint M3 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(s1, u3, low, next_high)
    FE32_MAC(s2, u2, low, next_high)
    FE32_MAC(s3, u1, low, next_high)
    low += (ulong)(u0 & mask_f) + (s0 & mask_g);
    c += low + high; uint M4 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(s2, u3, low, next_high)
    FE32_MAC(s3, u2, low, next_high)
    low += (ulong)(u1 & mask_f) + (s1 & mask_g);
    c += low + high; uint M5 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(s3, u3, low, next_high)
    low += (ulong)(u2 & mask_f) + (s2 & mask_g);
    c += low + high; uint M6 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    low += (ulong)(u3 & mask_f) + (s3 & mask_g);
    c += low + high; uint M7 = (uint)c; c >>= 32; high = next_high;
    c += carry_f & carry_g; uint M8 = (uint)c;
    long d = 0;
    d += (long)L0; uint r0 = (uint)d; d >>= 32;
    d += (long)L1; uint r1 = (uint)d; d >>= 32;
    d += (long)L2; uint r2 = (uint)d; d >>= 32;
    d += (long)L3; uint r3 = (uint)d; d >>= 32;
    d += (long)L4 + (long)M0 - (long)L0 - (long)H0; uint r4 = (uint)d; d >>= 32;
    d += (long)L5 + (long)M1 - (long)L1 - (long)H1; uint r5 = (uint)d; d >>= 32;
    d += (long)L6 + (long)M2 - (long)L2 - (long)H2; uint r6 = (uint)d; d >>= 32;
    d += (long)L7 + (long)M3 - (long)L3 - (long)H3; uint r7 = (uint)d; d >>= 32;
    d += (long)H0 + (long)M4 - (long)L4 - (long)H4; uint r8 = (uint)d; d >>= 32;
    d += (long)H1 + (long)M5 - (long)L5 - (long)H5; uint r9 = (uint)d; d >>= 32;
    d += (long)H2 + (long)M6 - (long)L6 - (long)H6; uint r10 = (uint)d; d >>= 32;
    d += (long)H3 + (long)M7 - (long)L7 - (long)H7; uint r11 = (uint)d; d >>= 32;
    d += (long)H4 + (long)M8; uint r12 = (uint)d; d >>= 32;
    d += (long)H5; uint r13 = (uint)d; d >>= 32;
    d += (long)H6; uint r14 = (uint)d; d >>= 32;
    d += (long)H7; uint r15 = (uint)d; d >>= 32;
    fe32_fold(h, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
}
#else
/* h = f * g (mod p), schoolbook over 64 limb products, one output column at
   a time: column k sums the low halves of products f_i g_j with i + j = k
   and the high halves of those with i + j = k - 1. Working by column keeps
   four accumulators live instead of sixteen, which is what lets the
   compiler keep the whole multiply in registers. */
__attribute__((noinline)) static void fe32_mul(fe32 h, const fe32 f, const fe32 g) {
    FE32_LOAD(f, f);
    FE32_LOAD(g, g);
    ulong c = 0, high = 0, next_high, low;
    low = 0; next_high = 0;
    FE32_MAC(f0, g0, low, next_high)
    c += low + high; uint r0 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g1, low, next_high)
    FE32_MAC(f1, g0, low, next_high)
    c += low + high; uint r1 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g2, low, next_high)
    FE32_MAC(f1, g1, low, next_high)
    FE32_MAC(f2, g0, low, next_high)
    c += low + high; uint r2 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g3, low, next_high)
    FE32_MAC(f1, g2, low, next_high)
    FE32_MAC(f2, g1, low, next_high)
    FE32_MAC(f3, g0, low, next_high)
    c += low + high; uint r3 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g4, low, next_high)
    FE32_MAC(f1, g3, low, next_high)
    FE32_MAC(f2, g2, low, next_high)
    FE32_MAC(f3, g1, low, next_high)
    FE32_MAC(f4, g0, low, next_high)
    c += low + high; uint r4 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g5, low, next_high)
    FE32_MAC(f1, g4, low, next_high)
    FE32_MAC(f2, g3, low, next_high)
    FE32_MAC(f3, g2, low, next_high)
    FE32_MAC(f4, g1, low, next_high)
    FE32_MAC(f5, g0, low, next_high)
    c += low + high; uint r5 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g6, low, next_high)
    FE32_MAC(f1, g5, low, next_high)
    FE32_MAC(f2, g4, low, next_high)
    FE32_MAC(f3, g3, low, next_high)
    FE32_MAC(f4, g2, low, next_high)
    FE32_MAC(f5, g1, low, next_high)
    FE32_MAC(f6, g0, low, next_high)
    c += low + high; uint r6 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, g7, low, next_high)
    FE32_MAC(f1, g6, low, next_high)
    FE32_MAC(f2, g5, low, next_high)
    FE32_MAC(f3, g4, low, next_high)
    FE32_MAC(f4, g3, low, next_high)
    FE32_MAC(f5, g2, low, next_high)
    FE32_MAC(f6, g1, low, next_high)
    FE32_MAC(f7, g0, low, next_high)
    c += low + high; uint r7 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f1, g7, low, next_high)
    FE32_MAC(f2, g6, low, next_high)
    FE32_MAC(f3, g5, low, next_high)
    FE32_MAC(f4, g4, low, next_high)
    FE32_MAC(f5, g3, low, next_high)
    FE32_MAC(f6, g2, low, next_high)
    FE32_MAC(f7, g1, low, next_high)
    c += low + high; uint r8 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f2, g7, low, next_high)
    FE32_MAC(f3, g6, low, next_high)
    FE32_MAC(f4, g5, low, next_high)
    FE32_MAC(f5, g4, low, next_high)
    FE32_MAC(f6, g3, low, next_high)
    FE32_MAC(f7, g2, low, next_high)
    c += low + high; uint r9 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f3, g7, low, next_high)
    FE32_MAC(f4, g6, low, next_high)
    FE32_MAC(f5, g5, low, next_high)
    FE32_MAC(f6, g4, low, next_high)
    FE32_MAC(f7, g3, low, next_high)
    c += low + high; uint r10 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f4, g7, low, next_high)
    FE32_MAC(f5, g6, low, next_high)
    FE32_MAC(f6, g5, low, next_high)
    FE32_MAC(f7, g4, low, next_high)
    c += low + high; uint r11 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f5, g7, low, next_high)
    FE32_MAC(f6, g6, low, next_high)
    FE32_MAC(f7, g5, low, next_high)
    c += low + high; uint r12 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f6, g7, low, next_high)
    FE32_MAC(f7, g6, low, next_high)
    c += low + high; uint r13 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f7, g7, low, next_high)
    c += low + high; uint r14 = (uint)c; c >>= 32; high = next_high;
    c += high; uint r15 = (uint)c;
    fe32_fold(h, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
}
#endif

/* h = f^2 (mod p) by column as in fe32_mul: each column's cross products
   f_i f_j (i < j) are summed once and doubled, then the square term of an
   even column is added. */
__attribute__((noinline)) static void fe32_sq(fe32 h, const fe32 f) {
    FE32_LOAD(f, f);
    ulong c = 0, high = 0, next_high, low;
    low = 0; next_high = 0;
    FE32_MAC(f0, f0, low, next_high)
    c += low + high; uint r0 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, f1, low, next_high)
    low <<= 1; next_high <<= 1;
    c += low + high; uint r1 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, f2, low, next_high)
    low <<= 1; next_high <<= 1;
    FE32_MAC(f1, f1, low, next_high)
    c += low + high; uint r2 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, f3, low, next_high)
    FE32_MAC(f1, f2, low, next_high)
    low <<= 1; next_high <<= 1;
    c += low + high; uint r3 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, f4, low, next_high)
    FE32_MAC(f1, f3, low, next_high)
    low <<= 1; next_high <<= 1;
    FE32_MAC(f2, f2, low, next_high)
    c += low + high; uint r4 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, f5, low, next_high)
    FE32_MAC(f1, f4, low, next_high)
    FE32_MAC(f2, f3, low, next_high)
    low <<= 1; next_high <<= 1;
    c += low + high; uint r5 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, f6, low, next_high)
    FE32_MAC(f1, f5, low, next_high)
    FE32_MAC(f2, f4, low, next_high)
    low <<= 1; next_high <<= 1;
    FE32_MAC(f3, f3, low, next_high)
    c += low + high; uint r6 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f0, f7, low, next_high)
    FE32_MAC(f1, f6, low, next_high)
    FE32_MAC(f2, f5, low, next_high)
    FE32_MAC(f3, f4, low, next_high)
    low <<= 1; next_high <<= 1;
    c += low + high; uint r7 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f1, f7, low, next_high)
    FE32_MAC(f2, f6, low, next_high)
    FE32_MAC(f3, f5, low, next_high)
    low <<= 1; next_high <<= 1;
    FE32_MAC(f4, f4, low, next_high)
    c += low + high; uint r8 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f2, f7, low, next_high)
    FE32_MAC(f3, f6, low, next_high)
    FE32_MAC(f4, f5, low, next_high)
    low <<= 1; next_high <<= 1;
    c += low + high; uint r9 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f3, f7, low, next_high)
    FE32_MAC(f4, f6, low, next_high)
    low <<= 1; next_high <<= 1;
    FE32_MAC(f5, f5, low, next_high)
    c += low + high; uint r10 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f4, f7, low, next_high)
    FE32_MAC(f5, f6, low, next_high)
    low <<= 1; next_high <<= 1;
    c += low + high; uint r11 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f5, f7, low, next_high)
    low <<= 1; next_high <<= 1;
    FE32_MAC(f6, f6, low, next_high)
    c += low + high; uint r12 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f6, f7, low, next_high)
    low <<= 1; next_high <<= 1;
    c += low + high; uint r13 = (uint)c; c >>= 32; high = next_high;
    low = 0; next_high = 0;
    FE32_MAC(f7, f7, low, next_high)
    c += low + high; uint r14 = (uint)c; c >>= 32; high = next_high;
    c += high; uint r15 = (uint)c;
    fe32_fold(h, r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);
}

/* h = f + g (mod p), h < 2^256. A carry out re-enters as 38; that can carry
   out again only when the limbs are then below 38, so a second carry adds
   38 to limb 0 alone. */
static void fe32_add(fe32 h, const fe32 f, const fe32 g) {
    ulong c = 0;
    c += (ulong)f[0] + g[0]; uint h0 = (uint)c; c >>= 32;
    c += (ulong)f[1] + g[1]; uint h1 = (uint)c; c >>= 32;
    c += (ulong)f[2] + g[2]; uint h2 = (uint)c; c >>= 32;
    c += (ulong)f[3] + g[3]; uint h3 = (uint)c; c >>= 32;
    c += (ulong)f[4] + g[4]; uint h4 = (uint)c; c >>= 32;
    c += (ulong)f[5] + g[5]; uint h5 = (uint)c; c >>= 32;
    c += (ulong)f[6] + g[6]; uint h6 = (uint)c; c >>= 32;
    c += (ulong)f[7] + g[7]; uint h7 = (uint)c; c >>= 32;
    c *= 38u;
    c += h0; h0 = (uint)c; c >>= 32;
    c += h1; h1 = (uint)c; c >>= 32;
    c += h2; h2 = (uint)c; c >>= 32;
    c += h3; h3 = (uint)c; c >>= 32;
    c += h4; h4 = (uint)c; c >>= 32;
    c += h5; h5 = (uint)c; c >>= 32;
    c += h6; h6 = (uint)c; c >>= 32;
    c += h7; h7 = (uint)c; c >>= 32;
    h0 += (uint)c * 38u;
    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
    h[3] = h3;
    h[4] = h4;
    h[5] = h5;
    h[6] = h6;
    h[7] = h7;
}

/* h = f - g (mod p), h < 2^256, as f + 4p - g = f + ~g + (2^256 - 75), since
   ~g = 2^256 - 1 - g and 4p = 2^257 - 76. Every term is non-negative, so the
   limbs carry but never borrow; the carry out (at most 2) folds back as 38
   per unit, and a second carry adds 38 to limb 0 alone as in fe32_add. */
static void fe32_sub(fe32 h, const fe32 f, const fe32 g) {
    ulong c = 0;
    c += (ulong)f[0] + (ulong)(~g[0]) + 0xffffffb5u; uint h0 = (uint)c; c >>= 32;
    c += (ulong)f[1] + (ulong)(~g[1]) + 0xffffffffu; uint h1 = (uint)c; c >>= 32;
    c += (ulong)f[2] + (ulong)(~g[2]) + 0xffffffffu; uint h2 = (uint)c; c >>= 32;
    c += (ulong)f[3] + (ulong)(~g[3]) + 0xffffffffu; uint h3 = (uint)c; c >>= 32;
    c += (ulong)f[4] + (ulong)(~g[4]) + 0xffffffffu; uint h4 = (uint)c; c >>= 32;
    c += (ulong)f[5] + (ulong)(~g[5]) + 0xffffffffu; uint h5 = (uint)c; c >>= 32;
    c += (ulong)f[6] + (ulong)(~g[6]) + 0xffffffffu; uint h6 = (uint)c; c >>= 32;
    c += (ulong)f[7] + (ulong)(~g[7]) + 0xffffffffu; uint h7 = (uint)c; c >>= 32;
    c *= 38u;
    c += h0; h0 = (uint)c; c >>= 32;
    c += h1; h1 = (uint)c; c >>= 32;
    c += h2; h2 = (uint)c; c >>= 32;
    c += h3; h3 = (uint)c; c >>= 32;
    c += h4; h4 = (uint)c; c >>= 32;
    c += h5; h5 = (uint)c; c >>= 32;
    c += h6; h6 = (uint)c; c >>= 32;
    c += h7; h7 = (uint)c; c >>= 32;
    h0 += (uint)c * 38u;
    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
    h[3] = h3;
    h[4] = h4;
    h[5] = h5;
    h[6] = h6;
    h[7] = h7;
}

/* Reduce h in place to its canonical value below p; h < 2^256 needs at most
   two subtractions of p. */
static void fe32_canonical(fe32 h) {
    FE32_LOAD(h, h);
    for (int pass = 0; pass < 2; pass++) {
        /* h - p = h + 19 - 2^255. With c the carry out of h + 19, h >= p iff c
           or bit 255 is set, and h - p is t with bit 255 set to c: when c is 1,
           t is below 19 and the subtracted 2^255 comes out of the carry. */
        ulong c = 19;
        c += h0; uint t0 = (uint)c; c >>= 32;
        c += h1; uint t1 = (uint)c; c >>= 32;
        c += h2; uint t2 = (uint)c; c >>= 32;
        c += h3; uint t3 = (uint)c; c >>= 32;
        c += h4; uint t4 = (uint)c; c >>= 32;
        c += h5; uint t5 = (uint)c; c >>= 32;
        c += h6; uint t6 = (uint)c; c >>= 32;
        c += h7; uint t7 = (uint)c; c >>= 32;
        uint ge = (t7 >> 31) | (uint)c;
        t7 = (t7 & 0x7fffffffu) | ((uint)c << 31);
        uint keep = ge - 1u;
        h0 = (h0 & keep) | (t0 & ~keep);
        h1 = (h1 & keep) | (t1 & ~keep);
        h2 = (h2 & keep) | (t2 & ~keep);
        h3 = (h3 & keep) | (t3 & ~keep);
        h4 = (h4 & keep) | (t4 & ~keep);
        h5 = (h5 & keep) | (t5 & ~keep);
        h6 = (h6 & keep) | (t6 & ~keep);
        h7 = (h7 & keep) | (t7 & ~keep);
    }
    h[0] = h0;
    h[1] = h1;
    h[2] = h2;
    h[3] = h3;
    h[4] = h4;
    h[5] = h5;
    h[6] = h6;
    h[7] = h7;
}

/* h = f. */
static void fe32_copy(fe32 h, const fe32 f) {
    for (int k = 0; k < 8; k++) h[k] = f[k];
}

/* h = 0. */
static void fe32_0(fe32 h) {
    for (int k = 0; k < 8; k++) h[k] = 0;
}

/* h = 1. */
static void fe32_1(fe32 h) {
    h[0] = 1;
    for (int k = 1; k < 8; k++) h[k] = 0;
}

/* h = -f (mod p). */
static void fe32_neg(fe32 h, const fe32 f) {
    fe32 zero;
    fe32_0(zero);
    fe32_sub(h, zero, f);
}

/* h = f^(2^n), for n >= 1. */
static void fe32_sqn(fe32 h, const fe32 f, int n) {
    fe32_sq(h, f);
    for (int i = 1; i < n; i++) fe32_sq(h, h);
}

/* out = 1/z = z^(p-2) (mod p), by ref10's addition chain: 254 squarings
   and 11 multiplications. */
static void fe32_invert(fe32 out, const fe32 z) {
    fe32 t0, t1, t2, t3;
    fe32_sq(t0, z);
    fe32_sqn(t1, t0, 2);
    fe32_mul(t1, z, t1);
    fe32_mul(t0, t0, t1);
    fe32_sq(t2, t0);
    fe32_mul(t1, t1, t2);
    fe32_sqn(t2, t1, 5);
    fe32_mul(t1, t2, t1);
    fe32_sqn(t2, t1, 10);
    fe32_mul(t2, t2, t1);
    fe32_sqn(t3, t2, 20);
    fe32_mul(t2, t3, t2);
    fe32_sqn(t2, t2, 10);
    fe32_mul(t1, t2, t1);
    fe32_sqn(t2, t1, 50);
    fe32_mul(t2, t2, t1);
    fe32_sqn(t3, t2, 100);
    fe32_mul(t2, t3, t2);
    fe32_sqn(t2, t2, 50);
    fe32_mul(t1, t2, t1);
    fe32_sqn(t1, t1, 5);
    fe32_mul(out, t1, t0);
}

/* Copy a field element out of global memory. */
static void fe32_load_global(fe32 h, __global const uint *src) {
    for (int k = 0; k < 8; k++) h[k] = src[k];
}

/* Forward half of Montgomery batch inversion: prefix[j] = zs[0] ... zs[j-1]
   for j < n (prefix[0] = 1), and total = zs[0] ... zs[n-1]. */
static void fe32_batch_prefix(fe32 prefix[KP32_BATCH], fe32 total,
                              const fe32 zs[KP32_BATCH], int n) {
    fe32_1(total);
    for (int i = 0; i < n; i++) {
        fe32_copy(prefix[i], total);
        fe32_mul(total, total, zs[i]);
    }
}

/* Backward half: zs[i] <- 1/zs[i] for i in [0, n), given the prefix products
   from fe32_batch_prefix and inv_total = 1/total, which it consumes. */
static void fe32_batch_finish(fe32 zs[KP32_BATCH], const fe32 prefix[KP32_BATCH],
                              fe32 inv_total, int n) {
    for (int i = n - 1; i >= 0; i--) {
        fe32 next;
        fe32_mul(next, inv_total, zs[i]);
        fe32_mul(zs[i], inv_total, prefix[i]);
        fe32_copy(inv_total, next);
    }
}

/* Montgomery batch inversion: zs[i] <- 1/zs[i] for i in [0, n), n >= 1,
   with one field inversion for the whole batch. */
static void fe32_batch_invert(fe32 zs[KP32_BATCH], int n) {
    fe32 prefix[KP32_BATCH];
    fe32 total;
    fe32_batch_prefix(prefix, total, zs, n);
    fe32_invert(total, total);
    fe32_batch_finish(zs, prefix, total, n);
}

/* Copy a field element into local memory. */
static void fe32_to_local(__local uint *dst, const fe32 f) {
    for (int k = 0; k < 8; k++) dst[k] = f[k];
}

/* Copy a field element out of local memory. */
static void fe32_from_local(fe32 h, __local const uint *src) {
    for (int k = 0; k < 8; k++) h[k] = src[k];
}

/* value <- 1/value for every work-item of the group with one field inversion
   for all of them: a product tree over the group in `tree` (local memory for
   2 * get_local_size(0) elements, heap-ordered with leaves from index size),
   work-item 0 inverting the root, and the tree walked back down, each node
   handing its children the inverse times the sibling. Every work-item of the
   group must call it, and the group size must be a power of two. */
static void fe32_group_invert(fe32 value, __local uint *tree) {
    uint lid = get_local_id(0);
    uint size = get_local_size(0);
    fe32 left, right, node;
    fe32_to_local(tree + 8 * (size + lid), value);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint span = size >> 1; span >= 1; span >>= 1) {
        if (lid < span) {
            uint i = span + lid;
            fe32_from_local(left, tree + 8 * (2 * i));
            fe32_from_local(right, tree + 8 * (2 * i + 1));
            fe32_mul(node, left, right);
            fe32_to_local(tree + 8 * i, node);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        fe32_from_local(node, tree + 8);
        fe32_invert(node, node);
        fe32_to_local(tree + 8, node);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint span = 1; span < size; span <<= 1) {
        if (lid < span) {
            uint i = span + lid;
            fe32_from_local(node, tree + 8 * i);
            fe32_from_local(left, tree + 8 * (2 * i));
            fe32_from_local(right, tree + 8 * (2 * i + 1));
            fe32 inverse;
            fe32_mul(inverse, node, right);
            fe32_to_local(tree + 8 * (2 * i), inverse);
            fe32_mul(inverse, node, left);
            fe32_to_local(tree + 8 * (2 * i + 1), inverse);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    fe32_from_local(value, tree + 8 * (size + lid));
}
