#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
use core::mem::MaybeUninit;

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct Fe8 {
    pub l: [__m512i; 5],
}

#[cfg(target_arch = "x86_64")]
const M52: u64 = (1u64 << 52) - 1;

pub fn available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512ifma")
            && is_x86_feature_detected!("avx512dq")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
impl Fe8 {
    #[cfg(test)]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn from_bytes8(inputs: &[[u8; 32]; 8]) -> Fe8 {
        let mut limbs = [[0u64; 8]; 5];
        for lane in 0..8 {
            let b = &inputs[lane];
            let load8 = |i: usize| -> u64 {
                u64::from_le_bytes([
                    b[i],
                    b[i + 1],
                    b[i + 2],
                    b[i + 3],
                    b[i + 4],
                    b[i + 5],
                    b[i + 6],
                    b[i + 7],
                ])
            };
            limbs[0][lane] = load8(0) & M52;
            limbs[1][lane] = (load8(6) >> 4) & M52;
            limbs[2][lane] = load8(13) & M52;
            limbs[3][lane] = (load8(19) >> 4) & M52;
            limbs[4][lane] = (load8(24) >> 16) & ((1u64 << 47) - 1);
        }
        Fe8 {
            l: [
                _mm512_loadu_si512(limbs[0].as_ptr() as *const __m512i),
                _mm512_loadu_si512(limbs[1].as_ptr() as *const __m512i),
                _mm512_loadu_si512(limbs[2].as_ptr() as *const __m512i),
                _mm512_loadu_si512(limbs[3].as_ptr() as *const __m512i),
                _mm512_loadu_si512(limbs[4].as_ptr() as *const __m512i),
            ],
        }
    }

    #[target_feature(enable = "avx512f")]
    pub unsafe fn to_bytes8(&self) -> [[u8; 32]; 8] {
        let mut limbs = [[0u64; 8]; 5];
        for i in 0..5 {
            _mm512_storeu_si512(
                limbs[i].as_mut_ptr() as *mut __m512i,
                self.l[i],
            );
        }
        let mut out = [[0u8; 32]; 8];
        for lane in 0..8 {
            let l = reduce_lane([
                limbs[0][lane],
                limbs[1][lane],
                limbs[2][lane],
                limbs[3][lane],
                limbs[4][lane],
            ]);
            out[lane] = serialize_lane(l);
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
fn reduce_lane(mut l: [u64; 5]) -> [u64; 5] {
    // Carry-normalize with the 2^260 == 608 top fold; twice is plenty.
    for _ in 0..2 {
        l[1] += l[0] >> 52;
        l[0] &= M52;
        l[2] += l[1] >> 52;
        l[1] &= M52;
        l[3] += l[2] >> 52;
        l[2] &= M52;
        l[4] += l[3] >> 52;
        l[3] &= M52;
        let top = l[4] >> 52;
        l[4] &= M52;
        l[0] += top * 608;
    }
    // Fold bits >= 255 back via 2^255 == 19 (bit 255 sits at limb4 bit 47).
    for _ in 0..2 {
        let hi = l[4] >> 47;
        l[4] &= (1u64 << 47) - 1;
        l[0] += 19 * hi;
        l[1] += l[0] >> 52;
        l[0] &= M52;
        l[2] += l[1] >> 52;
        l[1] &= M52;
        l[3] += l[2] >> 52;
        l[2] &= M52;
        l[4] += l[3] >> 52;
        l[3] &= M52;
    }
    // Now value < 2^255. Conditionally subtract p: add 19; if it overflows bit
    // 255 the value was >= p, so keep the reduced (masked) result.
    let mut t = [0u64; 5];
    t[0] = l[0] + 19;
    t[1] = l[1] + (t[0] >> 52);
    t[0] &= M52;
    t[2] = l[2] + (t[1] >> 52);
    t[1] &= M52;
    t[3] = l[3] + (t[2] >> 52);
    t[2] &= M52;
    t[4] = l[4] + (t[3] >> 52);
    t[3] &= M52;
    if (t[4] >> 47) & 1 == 1 {
        t[4] &= (1u64 << 47) - 1;
        t
    } else {
        l
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn serialize_lane(l: [u64; 5]) -> [u8; 32] {
    // 260 bits of limbs, 256-bit output: two u128 words, no carry loop.
    let w0 = (l[0] as u128)
        | ((l[1] as u128) << 52)
        | ((l[2] as u128) << 104);
    let w1 = ((l[2] >> 24) as u128)
        | ((l[3] as u128) << 28)
        | ((l[4] as u128) << 80);
    let mut s = [0u8; 32];
    s[0..8].copy_from_slice(&(w0 as u64).to_le_bytes());
    s[8..16].copy_from_slice(&((w0 >> 64) as u64).to_le_bytes());
    s[16..24].copy_from_slice(&(w1 as u64).to_le_bytes());
    s[24..32].copy_from_slice(&((w1 >> 64) as u64).to_le_bytes());
    s
}

// ─── arithmetic ──────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq")]
#[inline]
unsafe fn normalize(mut c: [__m512i; 5]) -> [__m512i; 5] {
    let mask = _mm512_set1_epi64(M52 as i64);
    let mask47 = _mm512_set1_epi64(((1u64 << 47) - 1) as i64);
    let c19 = _mm512_set1_epi64(19);

    // Pass 1.
    c[1] = _mm512_add_epi64(c[1], _mm512_srli_epi64(c[0], 52));
    c[0] = _mm512_and_si512(c[0], mask);
    c[2] = _mm512_add_epi64(c[2], _mm512_srli_epi64(c[1], 52));
    c[1] = _mm512_and_si512(c[1], mask);
    c[3] = _mm512_add_epi64(c[3], _mm512_srli_epi64(c[2], 52));
    c[2] = _mm512_and_si512(c[2], mask);
    c[4] = _mm512_add_epi64(c[4], _mm512_srli_epi64(c[3], 52));
    c[3] = _mm512_and_si512(c[3], mask);
    // Fold everything at/above bit 255 (limb4 bit 47) via 2^255 == 19.
    let top = _mm512_srli_epi64(c[4], 47);
    c[4] = _mm512_and_si512(c[4], mask47);
    c[0] = _mm512_add_epi64(c[0], _mm512_mullo_epi64(top, c19));

    // Pass 2: full ripple, mask every limb. limb4 has 5 bits of headroom so it
    // cannot overflow again.
    c[1] = _mm512_add_epi64(c[1], _mm512_srli_epi64(c[0], 52));
    c[0] = _mm512_and_si512(c[0], mask);
    c[2] = _mm512_add_epi64(c[2], _mm512_srli_epi64(c[1], 52));
    c[1] = _mm512_and_si512(c[1], mask);
    c[3] = _mm512_add_epi64(c[3], _mm512_srli_epi64(c[2], 52));
    c[2] = _mm512_and_si512(c[2], mask);
    c[4] = _mm512_add_epi64(c[4], _mm512_srli_epi64(c[3], 52));
    c[3] = _mm512_and_si512(c[3], mask);
    c
}

#[cfg(target_arch = "x86_64")]
impl Fe8 {
    #[target_feature(enable = "avx512f,avx512dq")]
    #[inline]
    pub unsafe fn add(&self, o: &Fe8) -> Fe8 {
        let c = [
            _mm512_add_epi64(self.l[0], o.l[0]),
            _mm512_add_epi64(self.l[1], o.l[1]),
            _mm512_add_epi64(self.l[2], o.l[2]),
            _mm512_add_epi64(self.l[3], o.l[3]),
            _mm512_add_epi64(self.l[4], o.l[4]),
        ];
        Fe8 { l: normalize(c) }
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    #[inline]
    pub unsafe fn sub(&self, o: &Fe8) -> Fe8 {
        // Add 2*p (limbwise) before subtracting so no lane underflows.
        let two_p = [
            _mm512_set1_epi64(((1u64 << 53) - 38) as i64),
            _mm512_set1_epi64(((1u64 << 53) - 2) as i64),
            _mm512_set1_epi64(((1u64 << 53) - 2) as i64),
            _mm512_set1_epi64(((1u64 << 53) - 2) as i64),
            _mm512_set1_epi64(((1u64 << 48) - 2) as i64),
        ];
        let c = [
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[0], two_p[0]),
                o.l[0],
            ),
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[1], two_p[1]),
                o.l[1],
            ),
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[2], two_p[2]),
                o.l[2],
            ),
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[3], two_p[3]),
                o.l[3],
            ),
            _mm512_sub_epi64(
                _mm512_add_epi64(self.l[4], two_p[4]),
                o.l[4],
            ),
        ];
        Fe8 { l: normalize(c) }
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    #[inline]
    pub unsafe fn mul(&self, o: &Fe8) -> Fe8 {
        let a = &self.l;
        let b = &o.l;
        let zero = _mm512_setzero_si512();
        // Full product accumulators c[0..9], each split low/high by IFMA.
        let mut c = [zero; 10];
        for i in 0..5 {
            for j in 0..5 {
                c[i + j] = _mm512_madd52lo_epu64(c[i + j], a[i], b[j]);
                c[i + j + 1] =
                    _mm512_madd52hi_epu64(c[i + j + 1], a[i], b[j]);
            }
        }
        Self::reduce_product(c)
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    #[inline]
    pub unsafe fn square(&self) -> Fe8 {
        self.mul(self)
    }

    #[target_feature(enable = "avx512f,avx512dq")]
    #[inline]
    unsafe fn reduce_product(mut c: [__m512i; 10]) -> Fe8 {
        let mask = _mm512_set1_epi64(M52 as i64);
        // Carry-propagate the 10 accumulators so each < 2^52 (c[9] a bit more).
        for m in 0..9 {
            c[m + 1] =
                _mm512_add_epi64(c[m + 1], _mm512_srli_epi64(c[m], 52));
            c[m] = _mm512_and_si512(c[m], mask);
        }
        let extra = _mm512_srli_epi64(c[9], 52);
        c[9] = _mm512_and_si512(c[9], mask);

        // Fold high half: r[k] = c[k] + 608*c[k+5]; the position-10 overflow
        // (extra) folds into r[0] via 2^520 == 19^2 * 2^10 == 369664.
        let c608 = _mm512_set1_epi64(608);
        let c369664 = _mm512_set1_epi64(369664);
        let mut r = [
            _mm512_add_epi64(c[0], _mm512_mullo_epi64(c[5], c608)),
            _mm512_add_epi64(c[1], _mm512_mullo_epi64(c[6], c608)),
            _mm512_add_epi64(c[2], _mm512_mullo_epi64(c[7], c608)),
            _mm512_add_epi64(c[3], _mm512_mullo_epi64(c[8], c608)),
            _mm512_add_epi64(c[4], _mm512_mullo_epi64(c[9], c608)),
        ];
        r[0] =
            _mm512_add_epi64(r[0], _mm512_mullo_epi64(extra, c369664));
        Fe8 { l: normalize(r) }
    }
}

// ─── field constants / invert (8-lane) ───────────────────────────────────────

#[cfg(target_arch = "x86_64")]
impl Fe8 {
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn zero8() -> Fe8 {
        Fe8 {
            l: [_mm512_setzero_si512(); 5],
        }
    }

    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn one8() -> Fe8 {
        Fe8 {
            l: [
                _mm512_set1_epi64(1),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            ],
        }
    }

    #[cfg(test)]
    #[target_feature(enable = "avx512f")]
    #[inline]
    pub unsafe fn splat_limbs(l: &[u64; 5]) -> Fe8 {
        Fe8 {
            l: [
                _mm512_set1_epi64(l[0] as i64),
                _mm512_set1_epi64(l[1] as i64),
                _mm512_set1_epi64(l[2] as i64),
                _mm512_set1_epi64(l[3] as i64),
                _mm512_set1_epi64(l[4] as i64),
            ],
        }
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    #[inline]
    unsafe fn pow2k(&self, k: u32) -> Fe8 {
        let mut r = *self;
        for _ in 0..k {
            r = r.square();
        }
        r
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    pub unsafe fn invert(&self) -> Fe8 {
        let t0 = self.square();
        let t1 = t0.square().square();
        let t2 = self.mul(&t1);
        let t3 = t0.mul(&t2);
        let t4 = t3.square();
        let t5 = t2.mul(&t4);
        let t6 = t5.pow2k(5);
        let t7 = t6.mul(&t5);
        let t8 = t7.pow2k(10);
        let t9 = t8.mul(&t7);
        let t10 = t9.pow2k(20);
        let t11 = t10.mul(&t9);
        let t12 = t11.pow2k(10);
        let t13 = t12.mul(&t7);
        let t14 = t13.pow2k(50);
        let t15 = t14.mul(&t13);
        let t16 = t15.pow2k(100);
        let t17 = t16.mul(&t15);
        let t18 = t17.pow2k(50);
        let t19 = t18.mul(&t13);
        t19.pow2k(5).mul(&t3)
    }
}

#[cfg(target_arch = "x86_64")]
pub fn bytes_to_limbs52(b: &[u8; 32]) -> [u64; 5] {
    let load8 = |i: usize| -> u64 {
        u64::from_le_bytes([
            b[i],
            b[i + 1],
            b[i + 2],
            b[i + 3],
            b[i + 4],
            b[i + 5],
            b[i + 6],
            b[i + 7],
        ])
    };
    [
        load8(0) & M52_CONST,
        (load8(6) >> 4) & M52_CONST,
        load8(13) & M52_CONST,
        (load8(19) >> 4) & M52_CONST,
        (load8(24) >> 16) & ((1u64 << 47) - 1),
    ]
}

#[cfg(target_arch = "x86_64")]
pub const M52_CONST: u64 = (1u64 << 52) - 1;

// ─── 8-lane Edwards points ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct Point8 {
    pub x: Fe8,
    pub y: Fe8,
    pub z: Fe8,
    pub t: Fe8,
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub struct Niels8 {
    pub y_plus_x: Fe8,
    pub y_minus_x: Fe8,
    pub z: Fe8,
    pub t2d: Fe8,
}

#[cfg(target_arch = "x86_64")]
impl Point8 {
    #[target_feature(enable = "avx512f")]
    pub unsafe fn identity() -> Point8 {
        Point8 {
            x: Fe8::zero8(),
            y: Fe8::one8(),
            z: Fe8::one8(),
            t: Fe8::zero8(),
        }
    }

    #[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
    #[inline]
    unsafe fn add_niels(&self, n: &Niels8) -> Point8 {
        let y_plus_x = self.y.add(&self.x);
        let y_minus_x = self.y.sub(&self.x);
        let pp = y_plus_x.mul(&n.y_plus_x);
        let mm = y_minus_x.mul(&n.y_minus_x);
        let tt2d = self.t.mul(&n.t2d);
        let zz = self.z.mul(&n.z);
        let zz2 = zz.add(&zz);
        let cx = pp.sub(&mm);
        let cy = pp.add(&mm);
        let cz = zz2.add(&tt2d);
        let ct = zz2.sub(&tt2d);
        // completed -> extended
        Point8 {
            x: cx.mul(&ct),
            y: cy.mul(&cz),
            z: cz.mul(&ct),
            t: cx.mul(&cy),
        }
    }
}

#[cfg(target_arch = "x86_64")]
const SIMD_W: usize = 5; // comb radix = 2^SIMD_W
#[cfg(target_arch = "x86_64")]
const SIMD_WINDOWS: usize = 256usize.div_ceil(SIMD_W);
#[cfg(target_arch = "x86_64")]
const SIMD_POS: usize = 1 << (SIMD_W - 1); // positive multiples per window
#[cfg(target_arch = "x86_64")]
const SIMD_ENTRIES: usize = 2 * SIMD_POS + 1; // signed digits -POS..POS
#[cfg(target_arch = "x86_64")]
const SIMD_FIELDS: usize = 4;
#[cfg(target_arch = "x86_64")]
const SIMD_ENTRY_U64: usize = SIMD_FIELDS * 5; // 20
#[cfg(target_arch = "x86_64")]
const SIMD_WINDOW_U64: usize = SIMD_ENTRIES * SIMD_ENTRY_U64; // 660
#[cfg(target_arch = "x86_64")]
pub const SIMD_TABLE_LEN: usize = SIMD_WINDOWS * SIMD_WINDOW_U64;

#[cfg(target_arch = "x86_64")]
pub fn build_simd_table() -> [u64; SIMD_TABLE_LEN] {
    use crate::fast::group::{edwards_d2, Niels};
    use crate::fast::BASE;

    let d2 = edwards_d2();
    let ident = Niels {
        y_plus_x: crate::fast::field::Fe::ONE,
        y_minus_x: crate::fast::field::Fe::ONE,
        z: crate::fast::field::Fe::ONE,
        t2d: crate::fast::field::Fe::ZERO,
    };
    let neg = |n: &Niels| Niels {
        y_plus_x: n.y_minus_x,
        y_minus_x: n.y_plus_x,
        z: n.z,
        t2d: n.t2d.negate(),
    };

    let mut table = [0u64; SIMD_TABLE_LEN];
    let mut off = 0usize;
    let mut cur = BASE; // 32^i * B
    for _ in 0..SIMD_WINDOWS {
        let cur_niels = cur.as_niels(&d2);
        // positive multiples 1..=SIMD_POS of cur
        let mut mult = cur;
        let mut pos = [cur_niels; SIMD_POS];
        pos[0] = cur.as_niels(&d2);
        for k in 1..SIMD_POS {
            mult = mult.add_niels(&cur_niels);
            pos[k] = mult.as_niels(&d2);
        }
        for e in 0..SIMD_ENTRIES {
            let digit = e as i32 - SIMD_POS as i32;
            let niels = if digit == 0 {
                ident
            } else if digit > 0 {
                pos[(digit - 1) as usize]
            } else {
                neg(&pos[(-digit - 1) as usize])
            };
            for f in
                [niels.y_plus_x, niels.y_minus_x, niels.z, niels.t2d]
            {
                let limbs = bytes_to_limbs52(&f.to_bytes());
                table[off..off + 5].copy_from_slice(&limbs);
                off += 5;
            }
        }
        // cur *= 2^SIMD_W
        for _ in 0..SIMD_W {
            cur = cur.double();
        }
    }
    debug_assert_eq!(off, SIMD_TABLE_LEN);
    table
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn to_radix(bytes: &[u8; 32]) -> [i8; SIMD_WINDOWS] {
    let mask = (1u16 << SIMD_W) - 1;
    let mut d = [0i8; SIMD_WINDOWS];
    for (i, di) in d.iter_mut().enumerate() {
        let bit = SIMD_W * i;
        let byte = bit / 8;
        let off = bit % 8;
        let lo = bytes[byte] as u16;
        let hi = if byte + 1 < 32 {
            bytes[byte + 1] as u16
        } else {
            0
        };
        *di = (((lo | (hi << 8)) >> off) & mask) as i8;
    }
    // Signed-convert: fold digits > 2^(W-1) down, carrying up.
    let pos = SIMD_POS as i8;
    let mut carry = 0i8;
    for i in 0..SIMD_WINDOWS - 1 {
        d[i] += carry;
        carry = (d[i] + pos) >> SIMD_W;
        d[i] -= carry << SIMD_W;
    }
    d[SIMD_WINDOWS - 1] += carry;
    d
}

#[cfg(target_arch = "x86_64")]
impl Niels8 {
    #[target_feature(enable = "avx512f")]
    unsafe fn gather(
        table: &[u64],
        i: usize,
        idx: &[usize; 8],
    ) -> Niels8 {
        let wbase = i * SIMD_WINDOW_U64;
        // fields[field][limb][lane] — uninit: every slot is written before read.
        let mut fields =
            MaybeUninit::<[[[u64; 8]; 5]; SIMD_FIELDS]>::uninit();
        let fields = &mut *fields.as_mut_ptr();
        for lane in 0..8 {
            let ebase = wbase + idx[lane] * SIMD_ENTRY_U64;
            for f in 0..SIMD_FIELDS {
                for m in 0..5 {
                    fields[f][m][lane] = table[ebase + f * 5 + m];
                }
            }
        }
        let mk = |f: usize| Fe8 {
            l: [
                _mm512_loadu_si512(
                    fields[f][0].as_ptr() as *const __m512i
                ),
                _mm512_loadu_si512(
                    fields[f][1].as_ptr() as *const __m512i
                ),
                _mm512_loadu_si512(
                    fields[f][2].as_ptr() as *const __m512i
                ),
                _mm512_loadu_si512(
                    fields[f][3].as_ptr() as *const __m512i
                ),
                _mm512_loadu_si512(
                    fields[f][4].as_ptr() as *const __m512i
                ),
            ],
        };
        Niels8 {
            y_plus_x: mk(0),
            y_minus_x: mk(1),
            z: mk(2),
            t2d: mk(3),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn prefetch_entries(
    table: &[u64],
    window: usize,
    idx: &[usize; 8],
) {
    let wbase = window * SIMD_WINDOW_U64;
    for lane in 0..8 {
        let ebase = wbase + idx[lane] * SIMD_ENTRY_U64;
        let ptr = table.as_ptr().add(ebase) as *const i8;
        // Each Niels entry is 160 bytes; two lines covers the scalar gather loads.
        _mm_prefetch(ptr, _MM_HINT_T1);
        _mm_prefetch(ptr.add(64), _MM_HINT_T1);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
pub unsafe fn scalarmult8(
    scalars: &[[u8; 32]; 8],
    table: &[u64],
) -> Point8 {
    let mut digits = [[0i8; SIMD_WINDOWS]; 8];
    for lane in 0..8 {
        digits[lane] = to_radix(&scalars[lane]);
    }
    let mut acc = Point8::identity();
    for i in 0..SIMD_WINDOWS {
        let mut idx = [0usize; 8];
        for lane in 0..8 {
            idx[lane] =
                (digits[lane][i] as i32 + SIMD_POS as i32) as usize;
        }
        if i + 1 < SIMD_WINDOWS {
            let mut next_idx = [0usize; 8];
            for lane in 0..8 {
                next_idx[lane] = (digits[lane][i + 1] as i32
                    + SIMD_POS as i32)
                    as usize;
            }
            prefetch_entries(table, i + 1, &next_idx);
        }
        let n = Niels8::gather(table, i, &idx);
        acc = acc.add_niels(&n);
    }
    acc
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
pub unsafe fn compress8(p: &Point8) -> [[u8; 32]; 8] {
    let recip = p.z.invert();
    write_compressed(p, &recip)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
#[inline]
unsafe fn write_compressed(p: &Point8, zinv: &Fe8) -> [[u8; 32]; 8] {
    let x = p.x.mul(zinv);
    let y = p.y.mul(zinv);
    let ybytes = y.to_bytes8();
    let xbytes = x.to_bytes8();
    let mut out = [[0u8; 32]; 8];
    for lane in 0..8 {
        out[lane] = ybytes[lane];
        out[lane][31] |= (xbytes[lane][0] & 1) << 7;
    }
    out
}

#[cfg(target_arch = "x86_64")]
pub const MAX_GROUPS: usize = 64;

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f,avx512ifma,avx512dq")]
pub unsafe fn batch_compress(points: &[Point8], out: &mut [[u8; 32]]) {
    let n = points.len();
    assert!(n <= MAX_GROUPS && out.len() >= n * 8);
    if n == 0 {
        return;
    }
    // Prefix products of Z: tmp[i] = Z[0]*...*Z[i].
    let mut tmp = [Fe8::one8(); MAX_GROUPS];
    tmp[0] = points[0].z;
    for i in 1..n {
        tmp[i] = tmp[i - 1].mul(&points[i].z);
    }
    // One inversion for the whole batch.
    let mut inv = tmp[n - 1].invert();
    // Back-substitute: zinv[i] = inv * tmp[i-1]; peel inv *= Z[i].
    let mut i = n - 1;
    while i > 0 {
        let zinv = inv.mul(&tmp[i - 1]);
        let outs = write_compressed(&points[i], &zinv);
        out[i * 8..i * 8 + 8].copy_from_slice(&outs);
        inv = inv.mul(&points[i].z);
        i -= 1;
    }
    let outs = write_compressed(&points[0], &inv);
    out[0..8].copy_from_slice(&outs);
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::fast::field::Fe;

    // Simple deterministic PRNG for test inputs.
    fn rng_bytes(state: &mut u64) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            chunk.copy_from_slice(&state.to_le_bytes());
        }
        out
    }

    fn fill8(state: &mut u64) -> [[u8; 32]; 8] {
        let mut a = [[0u8; 32]; 8];
        for x in a.iter_mut() {
            *x = rng_bytes(state);
        }
        a
    }

    #[test]
    fn roundtrip_bytes() {
        if !available() {
            eprintln!("skip: no IFMA");
            return;
        }
        let mut st = 0x1234_5678_9abc_def1u64;
        for _ in 0..500 {
            let ins = fill8(&mut st);
            // Reduce inputs to canonical form so comparison is well-defined.
            let expect: Vec<[u8; 32]> = ins
                .iter()
                .map(|b| Fe::from_bytes(b).to_bytes())
                .collect();
            let got = unsafe { Fe8::from_bytes8(&ins).to_bytes8() };
            for lane in 0..8 {
                assert_eq!(
                    got[lane],
                    expect[lane].as_slice(),
                    "roundtrip lane {lane}"
                );
            }
        }
    }

    fn check_binop(
        scalar: impl Fn(&Fe, &Fe) -> Fe,
        simd: unsafe fn(&Fe8, &Fe8) -> Fe8,
        name: &str,
    ) {
        if !available() {
            eprintln!("skip {name}: no IFMA");
            return;
        }
        let mut st = 0xdead_beef_0000_0001u64;
        for _ in 0..2000 {
            let a = fill8(&mut st);
            let b = fill8(&mut st);
            let mut expect = [[0u8; 32]; 8];
            for lane in 0..8 {
                let fa = Fe::from_bytes(&a[lane]);
                let fb = Fe::from_bytes(&b[lane]);
                expect[lane] = scalar(&fa, &fb).to_bytes();
            }
            let got = unsafe {
                let fa = Fe8::from_bytes8(&a);
                let fb = Fe8::from_bytes8(&b);
                simd(&fa, &fb).to_bytes8()
            };
            for lane in 0..8 {
                assert_eq!(
                    got[lane], expect[lane],
                    "{name} lane {lane}"
                );
            }
        }
    }

    #[test]
    fn mul_matches_scalar() {
        check_binop(|a, b| a.mul(b), |a, b| unsafe { a.mul(b) }, "mul");
    }

    #[test]
    fn add_matches_scalar() {
        check_binop(|a, b| a.add(b), |a, b| unsafe { a.add(b) }, "add");
    }

    // Chained multiply: feeds a mul's output back into another mul, which is
    // where an out-of-range limb (>= 2^52) would surface as truncation.
    #[test]
    fn mul_chained_matches_scalar() {
        use crate::fast::field::Fe;
        if !available() {
            return;
        }
        let mut st = 0x1357_9bdf_2468_ace0u64;
        for _ in 0..1000 {
            let a = fill8(&mut st);
            let b = fill8(&mut st);
            let c = fill8(&mut st);
            let got = unsafe {
                Fe8::from_bytes8(&a)
                    .mul(&Fe8::from_bytes8(&b))
                    .mul(&Fe8::from_bytes8(&c))
                    .to_bytes8()
            };
            for lane in 0..8 {
                let expect = Fe::from_bytes(&a[lane])
                    .mul(&Fe::from_bytes(&b[lane]))
                    .mul(&Fe::from_bytes(&c[lane]))
                    .to_bytes();
                assert_eq!(got[lane], expect, "lane {lane}");
            }
        }
    }

    #[test]
    fn sub_matches_scalar() {
        check_binop(|a, b| a.sub(b), |a, b| unsafe { a.sub(b) }, "sub");
    }

    #[test]
    fn square_matches_scalar() {
        if !available() {
            return;
        }
        let mut st = 0x0f0f_0f0f_1234_5678u64;
        for _ in 0..2000 {
            let a = fill8(&mut st);
            let mut expect = [[0u8; 32]; 8];
            for lane in 0..8 {
                expect[lane] = Fe::from_bytes(&a[lane])
                    .square()
                    .to_bytes();
            }
            let got = unsafe {
                Fe8::from_bytes8(&a)
                    .square()
                    .to_bytes8()
            };
            for lane in 0..8 {
                assert_eq!(
                    got[lane], expect[lane],
                    "square lane {lane}"
                );
            }
        }
    }

    #[test]
    fn invert_matches_scalar() {
        if !available() {
            return;
        }
        let mut st = 0xabcd_1234_5678_9999u64;
        for _ in 0..500 {
            let a = fill8(&mut st);
            let mut expect = [[0u8; 32]; 8];
            for lane in 0..8 {
                expect[lane] = Fe::from_bytes(&a[lane])
                    .invert()
                    .to_bytes();
            }
            let got = unsafe {
                Fe8::from_bytes8(&a)
                    .invert()
                    .to_bytes8()
            };
            for lane in 0..8 {
                assert_eq!(
                    got[lane], expect[lane],
                    "invert lane {lane}"
                );
            }
        }
    }

    #[test]
    fn add_niels8_identity_plus_base() {
        use crate::fast::group::edwards_d2;
        if !available() {
            return;
        }
        let d2 = edwards_d2();
        let n = crate::fast::BASE.as_niels(&d2); // 1*B in Niels form
        let to8 = |f: &Fe| unsafe {
            Fe8::splat_limbs(&bytes_to_limbs52(&f.to_bytes()))
        };
        let n8 = Niels8 {
            y_plus_x: to8(&n.y_plus_x),
            y_minus_x: to8(&n.y_minus_x),
            z: to8(&n.z),
            t2d: to8(&n.t2d),
        };
        let got = unsafe {
            let acc = Point8::identity().add_niels(&n8);
            compress8(&acc)
        };
        let mut expect = [0x66u8; 32];
        expect[0] = 0x58;
        assert_eq!(got[0], expect, "identity + 1*B");
    }

    #[test]
    fn compress8_of_basepoint() {
        if !available() {
            return;
        }
        // Broadcast the scalar BASE point into all 8 lanes, compress.
        let b = crate::fast::BASE;
        let to8 = |f: &Fe| unsafe {
            Fe8::splat_limbs(&bytes_to_limbs52(&f.to_bytes()))
        };
        let p = Point8 {
            x: to8(&b.x),
            y: to8(&b.y),
            z: to8(&b.z),
            t: to8(&b.t),
        };
        let got = unsafe { compress8(&p) };
        let mut expect = [0x66u8; 32];
        expect[0] = 0x58;
        assert_eq!(got[0], expect, "compress8 basepoint");
    }

    #[test]
    fn compress8_of_identity() {
        if !available() {
            return;
        }
        let p = unsafe { Point8::identity() };
        let got = unsafe { compress8(&p) };
        let mut expect = [0u8; 32];
        expect[0] = 1;
        assert_eq!(got[0], expect, "compress8 identity");
    }

    #[test]
    fn scalarmult8_matches_scalar_comb() {
        if !available() {
            return;
        }
        let table = build_simd_table();
        // Smallest-first probe: single small scalar in every lane.
        for k in [1u8, 2, 3, 5, 16, 17] {
            let mut sc = [[0u8; 32]; 8];
            for s in sc.iter_mut() {
                s[0] = k;
            }
            let got = unsafe { compress8(&scalarmult8(&sc, &table)) };
            let expect =
                crate::fast::scalarmult_compress_scalar(&sc[0]);
            assert_eq!(got[0], expect, "single-window scalar k={k}");
        }
        let mut st = 0x9999_7777_3333_1111u64;
        for _ in 0..100 {
            let mut scalars = fill8(&mut st);
            // Clamp each so they are valid scalars (matches grind usage).
            for s in scalars.iter_mut() {
                s[0] &= 248;
                s[31] &= 63;
                s[31] |= 64;
            }
            let got =
                unsafe { compress8(&scalarmult8(&scalars, &table)) };
            for lane in 0..8 {
                let expect = crate::fast::scalarmult_compress_scalar(
                    &scalars[lane],
                );
                assert_eq!(got[lane], expect, "lane {lane}");
            }
        }
    }

    #[test]
    fn batch_compress_matches_compress8() {
        if !available() {
            return;
        }
        let table = build_simd_table();
        let mut st = 0x0bad_c0de_1234_5678u64;
        let n = 10usize;
        let mut points = Vec::new();
        let mut per_group = Vec::new();
        for _ in 0..n {
            let mut scalars = fill8(&mut st);
            for s in scalars.iter_mut() {
                s[0] &= 248;
                s[31] &= 63;
                s[31] |= 64;
            }
            let p = unsafe { scalarmult8(&scalars, &table) };
            per_group.extend_from_slice(&unsafe { compress8(&p) });
            points.push(p);
        }
        let mut batched = vec![[0u8; 32]; n * 8];
        unsafe { batch_compress(&points, &mut batched) };
        assert_eq!(batched, per_group);
    }

    #[test]
    fn keygen8_matches_dalek() {
        use ed25519_dalek::SigningKey;
        use sha2::{Digest, Sha512};
        if !available() {
            eprintln!("skip: no IFMA");
            return;
        }
        let table = build_simd_table();
        let mut st = 0x5555_1111_aaaa_2222u64;
        for _ in 0..200 {
            // 8 random seeds.
            let seeds = fill8(&mut st);
            let mut scalars = [[0u8; 32]; 8];
            for lane in 0..8 {
                let h: [u8; 64] = Sha512::digest(seeds[lane]).into();
                let mut s = [0u8; 32];
                s.copy_from_slice(&h[..32]);
                s[0] &= 248;
                s[31] &= 63;
                s[31] |= 64;
                scalars[lane] = s;
            }
            let got = unsafe {
                let p = scalarmult8(&scalars, &table);
                compress8(&p)
            };
            for lane in 0..8 {
                let theirs = SigningKey::from_bytes(&seeds[lane])
                    .verifying_key()
                    .to_bytes();
                assert_eq!(got[lane], theirs, "keygen8 lane {lane}");
            }
        }
    }
}
