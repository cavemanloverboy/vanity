//! 4-wide ed25519 keygen using ARM NEON.
//!
//! Apple Silicon has NEON (128-bit) and no SSE/AVX. Four independent keys
//! share each `int32x4_t` limb of a ref10 field (ten signed 25.5-bit limbs).

#![allow(clippy::missing_safety_doc)]

use core::arch::aarch64::*;
use sha2::{Digest, Sha512};
use std::sync::OnceLock;

use super::field::Fe;
use super::group::{edwards_d2, Point};

#[derive(Clone, Copy)]
struct W4 {
    lo: int64x2_t,
    hi: int64x2_t,
}

#[derive(Clone, Copy)]
pub struct Fe4 {
    l: [int32x4_t; 10],
}

#[inline(always)]
unsafe fn wmul(a: int32x4_t, b: int32x4_t) -> W4 {
    W4 {
        lo: vmull_s32(vget_low_s32(a), vget_low_s32(b)),
        hi: vmull_high_s32(a, b),
    }
}

#[inline(always)]
unsafe fn wadd(a: W4, b: W4) -> W4 {
    W4 {
        lo: vaddq_s64(a.lo, b.lo),
        hi: vaddq_s64(a.hi, b.hi),
    }
}

#[inline(always)]
unsafe fn wmul19(a: W4) -> W4 {
    // 19 = 16 + 2 + 1. NEON has no 64x64 integer multiply.
    unsafe fn scale(v: int64x2_t) -> int64x2_t {
        vaddq_s64(
            vaddq_s64(vshlq_n_s64::<4>(v), vshlq_n_s64::<1>(v)),
            v,
        )
    }
    W4 {
        lo: scale(a.lo),
        hi: scale(a.hi),
    }
}

#[inline(always)]
unsafe fn narrow(a: W4) -> int32x4_t {
    vcombine_s32(vmovn_s64(a.lo), vmovn_s64(a.hi))
}

#[inline(always)]
unsafe fn carry<const SHIFT: i32>(h: &mut W4) -> W4 {
    let bias = vdupq_n_s64(1i64 << (SHIFT - 1));
    let clo = vshrq_n_s64::<SHIFT>(vaddq_s64(h.lo, bias));
    let chi = vshrq_n_s64::<SHIFT>(vaddq_s64(h.hi, bias));
    h.lo = vsubq_s64(h.lo, vshlq_n_s64::<SHIFT>(clo));
    h.hi = vsubq_s64(h.hi, vshlq_n_s64::<SHIFT>(chi));
    W4 { lo: clo, hi: chi }
}

#[inline(always)]
unsafe fn carry_chain(h: &mut [W4; 10]) {
    let c0 = carry::<26>(&mut h[0]);
    h[1] = wadd(h[1], c0);
    let c4 = carry::<26>(&mut h[4]);
    h[5] = wadd(h[5], c4);
    let c1 = carry::<25>(&mut h[1]);
    h[2] = wadd(h[2], c1);
    let c5 = carry::<25>(&mut h[5]);
    h[6] = wadd(h[6], c5);
    let c2 = carry::<26>(&mut h[2]);
    h[3] = wadd(h[3], c2);
    let c6 = carry::<26>(&mut h[6]);
    h[7] = wadd(h[7], c6);
    let c3 = carry::<25>(&mut h[3]);
    h[4] = wadd(h[4], c3);
    let c7 = carry::<25>(&mut h[7]);
    h[8] = wadd(h[8], c7);
    let c4 = carry::<26>(&mut h[4]);
    h[5] = wadd(h[5], c4);
    let c8 = carry::<26>(&mut h[8]);
    h[9] = wadd(h[9], c8);
    let c9 = carry::<25>(&mut h[9]);
    h[0] = wadd(h[0], wmul19(c9));
    let c0 = carry::<26>(&mut h[0]);
    h[1] = wadd(h[1], c0);
}

#[inline(always)]
unsafe fn pack(h: [W4; 10]) -> Fe4 {
    Fe4 {
        l: [
            narrow(h[0]),
            narrow(h[1]),
            narrow(h[2]),
            narrow(h[3]),
            narrow(h[4]),
            narrow(h[5]),
            narrow(h[6]),
            narrow(h[7]),
            narrow(h[8]),
            narrow(h[9]),
        ],
    }
}

#[inline(always)]
unsafe fn shl1(a: int32x4_t) -> int32x4_t {
    vshlq_n_s32::<1>(a)
}

#[inline(always)]
unsafe fn mul19_s(a: int32x4_t) -> int32x4_t {
    vmulq_n_s32(a, 19)
}

impl Fe4 {
    unsafe fn zero() -> Fe4 {
        Fe4 {
            l: [vdupq_n_s32(0); 10],
        }
    }

    unsafe fn one() -> Fe4 {
        let mut f = Fe4::zero();
        f.l[0] = vdupq_n_s32(1);
        f
    }

    unsafe fn add(self, o: Fe4) -> Fe4 {
        let mut r = Fe4::zero();
        for i in 0..10 {
            r.l[i] = vaddq_s32(self.l[i], o.l[i]);
        }
        r
    }

    unsafe fn sub(self, o: Fe4) -> Fe4 {
        let mut r = Fe4::zero();
        for i in 0..10 {
            r.l[i] = vsubq_s32(self.l[i], o.l[i]);
        }
        r
    }

    unsafe fn mul(self, o: Fe4) -> Fe4 {
        let (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9) = (
            self.l[0], self.l[1], self.l[2], self.l[3], self.l[4], self.l[5],
            self.l[6], self.l[7], self.l[8], self.l[9],
        );
        let (g0, g1, g2, g3, g4, g5, g6, g7, g8, g9) = (
            o.l[0], o.l[1], o.l[2], o.l[3], o.l[4], o.l[5], o.l[6], o.l[7],
            o.l[8], o.l[9],
        );
        let g1_19 = mul19_s(g1);
        let g2_19 = mul19_s(g2);
        let g3_19 = mul19_s(g3);
        let g4_19 = mul19_s(g4);
        let g5_19 = mul19_s(g5);
        let g6_19 = mul19_s(g6);
        let g7_19 = mul19_s(g7);
        let g8_19 = mul19_s(g8);
        let g9_19 = mul19_s(g9);
        let f1_2 = shl1(f1);
        let f3_2 = shl1(f3);
        let f5_2 = shl1(f5);
        let f7_2 = shl1(f7);
        let f9_2 = shl1(f9);
        let mut h = [wmul(f0, g0); 10];
        h[0] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g0), wmul(f1_2, g9_19)), wmul(f2, g8_19)),
                wadd(wmul(f3_2, g7_19), wmul(f4, g6_19)),
            ),
            wadd(
                wadd(wadd(wmul(f5_2, g5_19), wmul(f6, g4_19)), wmul(f7_2, g3_19)),
                wadd(wmul(f8, g2_19), wmul(f9_2, g1_19)),
            ),
        );
        h[1] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g1), wmul(f1, g0)), wmul(f2, g9_19)),
                wadd(wmul(f3, g8_19), wmul(f4, g7_19)),
            ),
            wadd(
                wadd(wadd(wmul(f5, g6_19), wmul(f6, g5_19)), wmul(f7, g4_19)),
                wadd(wmul(f8, g3_19), wmul(f9, g2_19)),
            ),
        );
        h[2] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g2), wmul(f1_2, g1)), wmul(f2, g0)),
                wadd(wmul(f3_2, g9_19), wmul(f4, g8_19)),
            ),
            wadd(
                wadd(wadd(wmul(f5_2, g7_19), wmul(f6, g6_19)), wmul(f7_2, g5_19)),
                wadd(wmul(f8, g4_19), wmul(f9_2, g3_19)),
            ),
        );
        h[3] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g3), wmul(f1, g2)), wmul(f2, g1)),
                wadd(wmul(f3, g0), wmul(f4, g9_19)),
            ),
            wadd(
                wadd(wadd(wmul(f5, g8_19), wmul(f6, g7_19)), wmul(f7, g6_19)),
                wadd(wmul(f8, g5_19), wmul(f9, g4_19)),
            ),
        );
        h[4] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g4), wmul(f1_2, g3)), wmul(f2, g2)),
                wadd(wmul(f3_2, g1), wmul(f4, g0)),
            ),
            wadd(
                wadd(wadd(wmul(f5_2, g9_19), wmul(f6, g8_19)), wmul(f7_2, g7_19)),
                wadd(wmul(f8, g6_19), wmul(f9_2, g5_19)),
            ),
        );
        h[5] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g5), wmul(f1, g4)), wmul(f2, g3)),
                wadd(wmul(f3, g2), wmul(f4, g1)),
            ),
            wadd(
                wadd(wadd(wmul(f5, g0), wmul(f6, g9_19)), wmul(f7, g8_19)),
                wadd(wmul(f8, g7_19), wmul(f9, g6_19)),
            ),
        );
        h[6] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g6), wmul(f1_2, g5)), wmul(f2, g4)),
                wadd(wmul(f3_2, g3), wmul(f4, g2)),
            ),
            wadd(
                wadd(wadd(wmul(f5_2, g1), wmul(f6, g0)), wmul(f7_2, g9_19)),
                wadd(wmul(f8, g8_19), wmul(f9_2, g7_19)),
            ),
        );
        h[7] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g7), wmul(f1, g6)), wmul(f2, g5)),
                wadd(wmul(f3, g4), wmul(f4, g3)),
            ),
            wadd(
                wadd(wadd(wmul(f5, g2), wmul(f6, g1)), wmul(f7, g0)),
                wadd(wmul(f8, g9_19), wmul(f9, g8_19)),
            ),
        );
        h[8] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g8), wmul(f1_2, g7)), wmul(f2, g6)),
                wadd(wmul(f3_2, g5), wmul(f4, g4)),
            ),
            wadd(
                wadd(wadd(wmul(f5_2, g3), wmul(f6, g2)), wmul(f7_2, g1)),
                wadd(wmul(f8, g0), wmul(f9_2, g9_19)),
            ),
        );
        h[9] = wadd(
            wadd(
                wadd(wadd(wmul(f0, g9), wmul(f1, g8)), wmul(f2, g7)),
                wadd(wmul(f3, g6), wmul(f4, g5)),
            ),
            wadd(
                wadd(wadd(wmul(f5, g4), wmul(f6, g3)), wmul(f7, g2)),
                wadd(wmul(f8, g1), wmul(f9, g0)),
            ),
        );
        carry_chain(&mut h);
        pack(h)
    }

    unsafe fn sq(self) -> Fe4 {
        let (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9) = (
            self.l[0], self.l[1], self.l[2], self.l[3], self.l[4], self.l[5],
            self.l[6], self.l[7], self.l[8], self.l[9],
        );
        let f0_2 = shl1(f0);
        let f1_2 = shl1(f1);
        let f2_2 = shl1(f2);
        let f3_2 = shl1(f3);
        let f4_2 = shl1(f4);
        let f5_2 = shl1(f5);
        let f6_2 = shl1(f6);
        let f7_2 = shl1(f7);
        let f5_38 = vmulq_n_s32(f5, 38);
        let f6_19 = mul19_s(f6);
        let f7_38 = vmulq_n_s32(f7, 38);
        let f8_19 = mul19_s(f8);
        let f9_38 = vmulq_n_s32(f9, 38);
        let mut h = [wmul(f0, f0); 10];
        h[0] = wadd(
            wadd(wadd(wmul(f0, f0), wmul(f1_2, f9_38)), wmul(f2_2, f8_19)),
            wadd(wadd(wmul(f3_2, f7_38), wmul(f4_2, f6_19)), wmul(f5, f5_38)),
        );
        h[1] = wadd(
            wadd(wmul(f0_2, f1), wmul(f2, f9_38)),
            wadd(wadd(wmul(f3_2, f8_19), wmul(f4, f7_38)), wmul(f5_2, f6_19)),
        );
        h[2] = wadd(
            wadd(wadd(wmul(f0_2, f2), wmul(f1_2, f1)), wmul(f3_2, f9_38)),
            wadd(wadd(wmul(f4_2, f8_19), wmul(f5_2, f7_38)), wmul(f6, f6_19)),
        );
        h[3] = wadd(
            wadd(wmul(f0_2, f3), wmul(f1_2, f2)),
            wadd(wadd(wmul(f4, f9_38), wmul(f5_2, f8_19)), wmul(f6, f7_38)),
        );
        h[4] = wadd(
            wadd(wadd(wmul(f0_2, f4), wmul(f1_2, f3_2)), wmul(f2, f2)),
            wadd(wadd(wmul(f5_2, f9_38), wmul(f6_2, f8_19)), wmul(f7, f7_38)),
        );
        h[5] = wadd(
            wadd(wmul(f0_2, f5), wmul(f1_2, f4)),
            wadd(wadd(wmul(f2_2, f3), wmul(f6, f9_38)), wmul(f7_2, f8_19)),
        );
        h[6] = wadd(
            wadd(wadd(wmul(f0_2, f6), wmul(f1_2, f5_2)), wmul(f2_2, f4)),
            wadd(wadd(wmul(f3_2, f3), wmul(f7_2, f9_38)), wmul(f8, f8_19)),
        );
        h[7] = wadd(
            wadd(wmul(f0_2, f7), wmul(f1_2, f6)),
            wadd(wadd(wmul(f2_2, f5), wmul(f3_2, f4)), wmul(f8, f9_38)),
        );
        h[8] = wadd(
            wadd(
                wadd(wmul(f0_2, f8), wmul(f1_2, f7_2)),
                wadd(wmul(f2_2, f6), wmul(f3_2, f5_2)),
            ),
            wadd(wmul(f4, f4), wmul(f9, f9_38)),
        );
        h[9] = wadd(
            wadd(wmul(f0_2, f9), wmul(f1_2, f8)),
            wadd(wadd(wmul(f2_2, f7), wmul(f3_2, f6)), wmul(f4_2, f5)),
        );
        carry_chain(&mut h);
        pack(h)
    }

    unsafe fn invert(self) -> Fe4 {
        let z = self;
        let mut t0 = z.sq();
        let mut t1 = t0.sq();
        t1 = t1.sq();
        t1 = z.mul(t1);
        t0 = t0.mul(t1);
        let mut t2 = t0.sq();
        t1 = t1.mul(t2);
        t2 = t1.sq();
        for _ in 1..5 {
            t2 = t2.sq();
        }
        t1 = t2.mul(t1);
        t2 = t1.sq();
        for _ in 1..10 {
            t2 = t2.sq();
        }
        t2 = t2.mul(t1);
        let mut t3 = t2.sq();
        for _ in 1..20 {
            t3 = t3.sq();
        }
        t2 = t3.mul(t2);
        t2 = t2.sq();
        for _ in 1..10 {
            t2 = t2.sq();
        }
        t1 = t2.mul(t1);
        t2 = t1.sq();
        for _ in 1..50 {
            t2 = t2.sq();
        }
        t2 = t2.mul(t1);
        t3 = t2.sq();
        for _ in 1..100 {
            t3 = t3.sq();
        }
        t2 = t3.mul(t2);
        t2 = t2.sq();
        for _ in 1..50 {
            t2 = t2.sq();
        }
        t1 = t2.mul(t1);
        t1 = t1.sq();
        for _ in 1..5 {
            t1 = t1.sq();
        }
        t1.mul(t0)
    }

    unsafe fn lane(self, lane: usize) -> [i32; 10] {
        let mut o = [0i32; 10];
        for i in 0..10 {
            o[i] = match lane {
                0 => vgetq_lane_s32::<0>(self.l[i]),
                1 => vgetq_lane_s32::<1>(self.l[i]),
                2 => vgetq_lane_s32::<2>(self.l[i]),
                _ => vgetq_lane_s32::<3>(self.l[i]),
            };
        }
        o
    }

    unsafe fn to_bytes4(self) -> [[u8; 32]; 4] {
        let mut out = [[0u8; 32]; 4];
        for lane in 0..4 {
            out[lane] = ref10_to_bytes(self.lane(lane));
        }
        out
    }
}

fn ref10_to_bytes(h: [i32; 10]) -> [u8; 32] {
    let mut h0 = h[0];
    let mut h1 = h[1];
    let mut h2 = h[2];
    let mut h3 = h[3];
    let mut h4 = h[4];
    let mut h5 = h[5];
    let mut h6 = h[6];
    let mut h7 = h[7];
    let mut h8 = h[8];
    let mut h9 = h[9];
    let mut q = (19 * h9 + (1 << 24)) >> 25;
    q = (h0 + q) >> 26;
    q = (h1 + q) >> 25;
    q = (h2 + q) >> 26;
    q = (h3 + q) >> 25;
    q = (h4 + q) >> 26;
    q = (h5 + q) >> 25;
    q = (h6 + q) >> 26;
    q = (h7 + q) >> 25;
    q = (h8 + q) >> 26;
    q = (h9 + q) >> 25;
    h0 += 19 * q;
    let c0 = h0 >> 26;
    h1 += c0;
    h0 -= c0 << 26;
    let c1 = h1 >> 25;
    h2 += c1;
    h1 -= c1 << 25;
    let c2 = h2 >> 26;
    h3 += c2;
    h2 -= c2 << 26;
    let c3 = h3 >> 25;
    h4 += c3;
    h3 -= c3 << 25;
    let c4 = h4 >> 26;
    h5 += c4;
    h4 -= c4 << 26;
    let c5 = h5 >> 25;
    h6 += c5;
    h5 -= c5 << 25;
    let c6 = h6 >> 26;
    h7 += c6;
    h6 -= c6 << 26;
    let c7 = h7 >> 25;
    h8 += c7;
    h7 -= c7 << 25;
    let c8 = h8 >> 26;
    h9 += c8;
    h8 -= c8 << 26;
    let c9 = h9 >> 25;
    h9 -= c9 << 25;
    let mut s = [0u8; 32];
    s[0] = h0 as u8;
    s[1] = (h0 >> 8) as u8;
    s[2] = (h0 >> 16) as u8;
    s[3] = ((h0 >> 24) | (h1 << 2)) as u8;
    s[4] = (h1 >> 6) as u8;
    s[5] = (h1 >> 14) as u8;
    s[6] = ((h1 >> 22) | (h2 << 3)) as u8;
    s[7] = (h2 >> 5) as u8;
    s[8] = (h2 >> 13) as u8;
    s[9] = ((h2 >> 21) | (h3 << 5)) as u8;
    s[10] = (h3 >> 3) as u8;
    s[11] = (h3 >> 11) as u8;
    s[12] = ((h3 >> 19) | (h4 << 6)) as u8;
    s[13] = (h4 >> 2) as u8;
    s[14] = (h4 >> 10) as u8;
    s[15] = (h4 >> 18) as u8;
    s[16] = h5 as u8;
    s[17] = (h5 >> 8) as u8;
    s[18] = (h5 >> 16) as u8;
    s[19] = ((h5 >> 24) | (h6 << 1)) as u8;
    s[20] = (h6 >> 7) as u8;
    s[21] = (h6 >> 15) as u8;
    s[22] = ((h6 >> 23) | (h7 << 3)) as u8;
    s[23] = (h7 >> 5) as u8;
    s[24] = (h7 >> 13) as u8;
    s[25] = ((h7 >> 21) | (h8 << 4)) as u8;
    s[26] = (h8 >> 4) as u8;
    s[27] = (h8 >> 12) as u8;
    s[28] = ((h8 >> 20) | (h9 << 6)) as u8;
    s[29] = (h9 >> 2) as u8;
    s[30] = (h9 >> 10) as u8;
    s[31] = (h9 >> 18) as u8;
    s
}

fn ref10_from_bytes(s: &[u8; 32]) -> [i32; 10] {
    let load4 = |i: usize| -> i64 {
        u32::from_le_bytes([s[i], s[i + 1], s[i + 2], s[i + 3]]) as i64
    };
    let load3 = |i: usize| -> i64 {
        (s[i] as i64) | ((s[i + 1] as i64) << 8) | ((s[i + 2] as i64) << 16)
    };
    let mut h = [0i64; 10];
    h[0] = load4(0);
    h[1] = load3(4) << 6;
    h[2] = load3(7) << 5;
    h[3] = load3(10) << 3;
    h[4] = load3(13) << 2;
    h[5] = load4(16);
    h[6] = load3(20) << 7;
    h[7] = load3(23) << 5;
    h[8] = load3(26) << 4;
    h[9] = (load3(29) & 8_388_607) << 2;
    let c9 = (h[9] + (1 << 24)) >> 25;
    h[0] += c9 * 19;
    h[9] -= c9 << 25;
    for (i, sh, nxt) in [(1, 25, 2), (3, 25, 4), (5, 25, 6), (7, 25, 8)] {
        let c = (h[i] + (1 << (sh - 1))) >> sh;
        h[nxt] += c;
        h[i] -= c << sh;
    }
    for (i, sh, nxt) in [(0, 26, 1), (2, 26, 3), (4, 26, 5), (6, 26, 7), (8, 26, 9)] {
        let c = (h[i] + (1 << (sh - 1))) >> sh;
        h[nxt] += c;
        h[i] -= c << sh;
    }
    let mut o = [0i32; 10];
    for i in 0..10 {
        o[i] = h[i] as i32;
    }
    o
}

fn fe_bytes(f: &Fe) -> [u8; 32] {
    f.to_bytes()
}

struct Affine {
    ypx: [i32; 10],
    ymx: [i32; 10],
    t2d: [i32; 10],
}

fn affine_of(p: &Point, d2: &Fe) -> Affine {
    let n = p.as_niels(d2);
    let zinv = n.z.invert();
    Affine {
        ypx: ref10_from_bytes(&fe_bytes(&n.y_plus_x.mul(&zinv))),
        ymx: ref10_from_bytes(&fe_bytes(&n.y_minus_x.mul(&zinv))),
        t2d: ref10_from_bytes(&fe_bytes(&n.t2d.mul(&zinv))),
    }
}

fn base_point() -> Point {
    Point {
        x: Fe([
            1738742601995546,
            1146398526822698,
            2070867633025821,
            562264141797630,
            587772402128613,
        ]),
        y: Fe([
            1801439850948184,
            1351079888211148,
            450359962737049,
            900719925474099,
            1801439850948198,
        ]),
        z: Fe::ONE,
        t: Fe([
            1841354044333475,
            16398895984059,
            755974180946558,
            900171276175154,
            1821297809914039,
        ]),
    }
}

struct Comb(Vec<Affine>);

const COMB_W: usize = 8;
const COMB_WINDOWS: usize = 256_usize.div_ceil(COMB_W);
const COMB_POS: usize = 1 << (COMB_W - 1);

fn comb() -> &'static Comb {
    static T: OnceLock<Comb> = OnceLock::new();
    T.get_or_init(|| {
        let d2 = edwards_d2();
        let mut rows = Vec::with_capacity(COMB_WINDOWS * COMB_POS);
        let mut cur = base_point();
        for _ in 0..COMB_WINDOWS {
            let cur_n = cur.as_niels(&d2);
            let mut multiple = cur;
            rows.push(affine_of(&cur, &d2));
            for _k in 1..COMB_POS {
                multiple = multiple.add_niels(&cur_n);
                rows.push(affine_of(&multiple, &d2));
            }
            for _ in 0..COMB_W {
                cur = cur.double();
            }
        }
        Comb(rows)
    })
}

fn to_comb_digits(bytes: &[u8; 32]) -> [i16; COMB_WINDOWS] {
    let mut e = [0i16; COMB_WINDOWS];
    let mask = (1u32 << COMB_W) - 1;
    for i in 0..COMB_WINDOWS {
        let bit = COMB_W * i;
        let byte = bit / 8;
        let off = bit % 8;
        let mut word = 0u32;
        for j in 0..3 {
            if byte + j < 32 {
                word |= (bytes[byte + j] as u32) << (8 * j);
            }
        }
        e[i] = ((word >> off) & mask) as i16;
    }
    let mut carry = 0i32;
    for i in 0..COMB_WINDOWS - 1 {
        let digit = e[i] as i32 + carry;
        carry = (digit + COMB_POS as i32) >> COMB_W;
        e[i] = (digit - (carry << COMB_W)) as i16;
    }
    e[COMB_WINDOWS - 1] =
        (e[COMB_WINDOWS - 1] as i32 + carry) as i16;
    e
}

unsafe fn gather4(rows: &[Affine], window: usize, digits: [i16; 4], coord: u8) -> Fe4 {
    let mut tmp = [[0i32; 10]; 4];
    for lane in 0..4 {
        let dig = digits[lane];
        let neg = dig < 0;
        let ad = if neg { -dig } else { dig } as usize;
        if ad == 0 {
            if coord != 2 {
                tmp[lane][0] = 1;
            }
            continue;
        }
        let e = &rows[window * COMB_POS + (ad - 1)];
        // Negative digits swap y+x with y-x and negate t2d.
        let src = match (coord, neg) {
            (0, false) | (1, true) => &e.ypx,
            (1, false) | (0, true) => &e.ymx,
            _ => &e.t2d,
        };
        tmp[lane] = *src;
        if neg && coord == 2 {
            for k in 0..10 {
                tmp[lane][k] = -tmp[lane][k];
            }
        }
    }
    let mut f = Fe4::zero();
    for k in 0..10 {
        let v = [tmp[0][k], tmp[1][k], tmp[2][k], tmp[3][k]];
        f.l[k] = vld1q_s32(v.as_ptr());
    }
    f
}

struct P4 {
    x: Fe4,
    y: Fe4,
    z: Fe4,
    t: Fe4,
}

unsafe fn add_affine(p: &mut P4, ypx: Fe4, ymx: Fe4, t2d: Fe4) {
    let a = p.y.add(p.x);
    let b = p.y.sub(p.x);
    let pp = a.mul(ypx);
    let mm = b.mul(ymx);
    let tt = p.t.mul(t2d);
    let zz2 = p.z.add(p.z);
    let rx = pp.sub(mm);
    let ry = pp.add(mm);
    let rz = zz2.add(tt);
    let rt = zz2.sub(tt);
    p.x = rx.mul(rt);
    p.y = ry.mul(rz);
    p.z = rz.mul(rt);
    p.t = rx.mul(ry);
}

unsafe fn scalarmult4(scalars: &[[u8; 32]; 4]) -> P4 {
    let table = &comb().0;
    let mut digits = [[0i16; COMB_WINDOWS]; 4];
    for lane in 0..4 {
        digits[lane] = to_comb_digits(&scalars[lane]);
    }
    let mut p = P4 {
        x: Fe4::zero(),
        y: Fe4::one(),
        z: Fe4::one(),
        t: Fe4::zero(),
    };
    for w in 0..COMB_WINDOWS {
        let ds = [digits[0][w], digits[1][w], digits[2][w], digits[3][w]];
        let ypx = gather4(table, w, ds, 0);
        let ymx = gather4(table, w, ds, 1);
        let t2d = gather4(table, w, ds, 2);
        add_affine(&mut p, ypx, ymx, t2d);
    }
    p
}

unsafe fn compress4(p: &P4, zinv: Fe4) -> [[u8; 32]; 4] {
    let x = p.x.mul(zinv).to_bytes4();
    let y = p.y.mul(zinv).to_bytes4();
    let mut out = y;
    for lane in 0..4 {
        out[lane][31] |= (x[lane][0] & 1) << 7;
    }
    out
}

fn clamp(h: &[u8; 64]) -> [u8; 32] {
    let mut s = [0u8; 32];
    s.copy_from_slice(&h[..32]);
    s[0] &= 248;
    s[31] &= 63;
    s[31] |= 64;
    s
}

/// `seeds` length must be a multiple of 4. Each seed is replaced by the next
/// chain value `SHA-512(seed)[32..64]`. `used` receives the preimage seed.
pub unsafe fn keygen_batch(
    seeds: &mut [[u8; 32]],
    used: &mut [[u8; 32]],
    pubs: &mut [[u8; 32]],
) {
    assert_eq!(seeds.len() % 4, 0);
    let mut off = 0;
    while off < seeds.len() {
        let mut sc = [[0u8; 32]; 4];
        for j in 0..4 {
            used[off + j] = seeds[off + j];
            let h: [u8; 64] = Sha512::digest(seeds[off + j]).into();
            sc[j] = clamp(&h);
            seeds[off + j].copy_from_slice(&h[32..64]);
        }
        let p = scalarmult4(&sc);
        let zinv = p.z.invert();
        let out = compress4(&p, zinv);
        pubs[off..off + 4].copy_from_slice(&out);
        off += 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    #[test]
    fn keygen4_matches_dalek() {
        let mut seeds = [[0u8; 32]; 8];
        for (i, s) in seeds.iter_mut().enumerate() {
            *s = Sha512::digest((i as u32).to_le_bytes()).as_slice()[..32]
                .try_into()
                .unwrap();
        }
        let original = seeds;
        let mut used = [[0u8; 32]; 8];
        let mut pubs = [[0u8; 32]; 8];
        unsafe { keygen_batch(&mut seeds, &mut used, &mut pubs) };
        for i in 0..8 {
            assert_eq!(used[i], original[i]);
            let expect = SigningKey::from_bytes(&original[i])
                .verifying_key()
                .to_bytes();
            assert_eq!(pubs[i], expect, "lane {i}");
        }
    }
}
