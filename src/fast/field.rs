#[derive(Clone, Copy, Debug)]
pub struct Fe(pub [u64; 5]);

const MASK51: u64 = (1u64 << 51) - 1;

impl Fe {
    pub const ZERO: Fe = Fe([0, 0, 0, 0, 0]);
    pub const ONE: Fe = Fe([1, 0, 0, 0, 0]);

    #[inline(always)]
    fn reduce(mut l: [u64; 5]) -> Fe {
        let c0 = l[0] >> 51;
        let c1 = l[1] >> 51;
        let c2 = l[2] >> 51;
        let c3 = l[3] >> 51;
        let c4 = l[4] >> 51;
        l[0] &= MASK51;
        l[1] &= MASK51;
        l[2] &= MASK51;
        l[3] &= MASK51;
        l[4] &= MASK51;
        l[0] += c4 * 19;
        l[1] += c0;
        l[2] += c1;
        l[3] += c2;
        l[4] += c3;
        Fe(l)
    }

    #[cfg(test)]
    pub fn from_bytes(b: &[u8; 32]) -> Fe {
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
        let l0 = load8(0) & MASK51;
        let l1 = (load8(6) >> 3) & MASK51;
        let l2 = (load8(12) >> 6) & MASK51;
        let l3 = (load8(19) >> 1) & MASK51;
        let l4 = (load8(24) >> 12) & MASK51;
        Fe([l0, l1, l2, l3, l4])
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        // First bring limbs into [0, 2^51).
        let mut l = Fe::reduce(self.0).0;
        // Compute q = floor((self + 19) / 2^255): whether self >= p.
        let mut q = (l[0] + 19) >> 51;
        q = (l[1] + q) >> 51;
        q = (l[2] + q) >> 51;
        q = (l[3] + q) >> 51;
        q = (l[4] + q) >> 51;
        // If self >= p, adding 19 overflowed bit 255 => q==1: subtract p by
        // adding 19 and dropping the top bit via the mask chain below.
        l[0] += 19 * q;
        l[1] += l[0] >> 51;
        l[0] &= MASK51;
        l[2] += l[1] >> 51;
        l[1] &= MASK51;
        l[3] += l[2] >> 51;
        l[2] &= MASK51;
        l[4] += l[3] >> 51;
        l[3] &= MASK51;
        l[4] &= MASK51;

        let mut s = [0u8; 32];
        // Pack 5x51 bits into 32 bytes.
        s[0] = l[0] as u8;
        s[1] = (l[0] >> 8) as u8;
        s[2] = (l[0] >> 16) as u8;
        s[3] = (l[0] >> 24) as u8;
        s[4] = (l[0] >> 32) as u8;
        s[5] = (l[0] >> 40) as u8;
        s[6] = ((l[0] >> 48) | (l[1] << 3)) as u8;
        s[7] = (l[1] >> 5) as u8;
        s[8] = (l[1] >> 13) as u8;
        s[9] = (l[1] >> 21) as u8;
        s[10] = (l[1] >> 29) as u8;
        s[11] = (l[1] >> 37) as u8;
        s[12] = ((l[1] >> 45) | (l[2] << 6)) as u8;
        s[13] = (l[2] >> 2) as u8;
        s[14] = (l[2] >> 10) as u8;
        s[15] = (l[2] >> 18) as u8;
        s[16] = (l[2] >> 26) as u8;
        s[17] = (l[2] >> 34) as u8;
        s[18] = (l[2] >> 42) as u8;
        s[19] = ((l[2] >> 50) | (l[3] << 1)) as u8;
        s[20] = (l[3] >> 7) as u8;
        s[21] = (l[3] >> 15) as u8;
        s[22] = (l[3] >> 23) as u8;
        s[23] = (l[3] >> 31) as u8;
        s[24] = (l[3] >> 39) as u8;
        s[25] = ((l[3] >> 47) | (l[4] << 4)) as u8;
        s[26] = (l[4] >> 4) as u8;
        s[27] = (l[4] >> 12) as u8;
        s[28] = (l[4] >> 20) as u8;
        s[29] = (l[4] >> 28) as u8;
        s[30] = (l[4] >> 36) as u8;
        s[31] = (l[4] >> 44) as u8;
        s
    }

    #[inline(always)]
    pub fn add(&self, o: &Fe) -> Fe {
        Fe([
            self.0[0] + o.0[0],
            self.0[1] + o.0[1],
            self.0[2] + o.0[2],
            self.0[3] + o.0[3],
            self.0[4] + o.0[4],
        ])
    }

    #[inline(always)]
    pub fn sub(&self, o: &Fe) -> Fe {
        Fe::reduce([
            (self.0[0] + 36028797018963664) - o.0[0], // 2^55 - 16*19
            (self.0[1] + 36028797018963952) - o.0[1], // 2^55 - 16
            (self.0[2] + 36028797018963952) - o.0[2],
            (self.0[3] + 36028797018963952) - o.0[3],
            (self.0[4] + 36028797018963952) - o.0[4],
        ])
    }

    #[inline(always)]
    pub fn negate(&self) -> Fe {
        Fe::reduce([
            36028797018963664 - self.0[0],
            36028797018963952 - self.0[1],
            36028797018963952 - self.0[2],
            36028797018963952 - self.0[3],
            36028797018963952 - self.0[4],
        ])
    }

    #[inline(always)]
    pub fn mul(&self, o: &Fe) -> Fe {
        let a = &self.0;
        let b = &o.0;
        let b1_19 = b[1] * 19;
        let b2_19 = b[2] * 19;
        let b3_19 = b[3] * 19;
        let b4_19 = b[4] * 19;

        let m = |x: u64, y: u64| -> u128 { (x as u128) * (y as u128) };

        let c0 = m(a[0], b[0])
            + m(a[4], b1_19)
            + m(a[3], b2_19)
            + m(a[2], b3_19)
            + m(a[1], b4_19);
        let c1 = m(a[1], b[0])
            + m(a[0], b[1])
            + m(a[4], b2_19)
            + m(a[3], b3_19)
            + m(a[2], b4_19);
        let c2 = m(a[2], b[0])
            + m(a[1], b[1])
            + m(a[0], b[2])
            + m(a[4], b3_19)
            + m(a[3], b4_19);
        let c3 = m(a[3], b[0])
            + m(a[2], b[1])
            + m(a[1], b[2])
            + m(a[0], b[3])
            + m(a[4], b4_19);
        let c4 = m(a[4], b[0])
            + m(a[3], b[1])
            + m(a[2], b[2])
            + m(a[1], b[3])
            + m(a[0], b[4]);

        Fe::carry_reduce([c0, c1, c2, c3, c4])
    }

    #[inline(always)]
    pub fn square(&self) -> Fe {
        let a = &self.0;
        let a3_19 = 19 * a[3];
        let a4_19 = 19 * a[4];

        let m = |x: u64, y: u64| -> u128 { (x as u128) * (y as u128) };

        let c0 = m(a[0], a[0]) + 2 * (m(a[1], a4_19) + m(a[2], a3_19));
        let c1 =
            2 * m(a[0], a[1]) + 2 * m(a[2], a4_19) + m(a[3], a3_19);
        let c2 = 2 * m(a[0], a[2]) + m(a[1], a[1]) + 2 * m(a[3], a4_19);
        let c3 = 2 * (m(a[0], a[3]) + m(a[1], a[2])) + m(a[4], a4_19);
        let c4 = 2 * (m(a[0], a[4]) + m(a[1], a[3])) + m(a[2], a[2]);

        Fe::carry_reduce([c0, c1, c2, c3, c4])
    }

    #[inline(always)]
    fn carry_reduce(mut c: [u128; 5]) -> Fe {
        const MASK: u128 = (1u128 << 51) - 1;
        c[1] += c[0] >> 51;
        let o0 = (c[0] & MASK) as u64;
        c[2] += c[1] >> 51;
        let o1 = (c[1] & MASK) as u64;
        c[3] += c[2] >> 51;
        let o2 = (c[2] & MASK) as u64;
        c[4] += c[3] >> 51;
        let o3 = (c[3] & MASK) as u64;
        let carry = (c[4] >> 51) as u64;
        let o4 = (c[4] & MASK) as u64;

        // Fold the top carry back in via *19, then ripple the one carry it can
        // produce from o0 into o1.
        let mut o0 = o0 + carry * 19;
        let o1 = o1 + (o0 >> 51);
        o0 &= MASK51;
        Fe([o0, o1, o2, o3, o4])
    }

    #[inline(always)]
    pub fn pow2k(&self, k: u32) -> Fe {
        let mut r = *self;
        for _ in 0..k {
            r = r.square();
        }
        r
    }

    pub fn invert(&self) -> Fe {
        let (t19, t3) = self.pow22501();
        // self^(2^255 - 21) = self^(p-2)
        t19.pow2k(5).mul(&t3)
    }

    fn pow22501(&self) -> (Fe, Fe) {
        let t0 = self.square(); // ^2
        let t1 = t0.square().square(); // ^8
        let t2 = self.mul(&t1); // ^9
        let t3 = t0.mul(&t2); // ^11
        let t4 = t3.square(); // ^22
        let t5 = t2.mul(&t4); // ^31 = ^(2^5-1)
        let t6 = t5.pow2k(5);
        let t7 = t6.mul(&t5); // ^(2^10-1)
        let t8 = t7.pow2k(10);
        let t9 = t8.mul(&t7); // ^(2^20-1)
        let t10 = t9.pow2k(20);
        let t11 = t10.mul(&t9); // ^(2^40-1)
        let t12 = t11.pow2k(10);
        let t13 = t12.mul(&t7); // ^(2^50-1)
        let t14 = t13.pow2k(50);
        let t15 = t14.mul(&t13); // ^(2^100-1)
        let t16 = t15.pow2k(100);
        let t17 = t16.mul(&t15); // ^(2^200-1)
        let t18 = t17.pow2k(50);
        let t19 = t18.mul(&t13); // ^(2^250-1)
        (t19, t3)
    }

    #[inline(always)]
    pub fn is_odd(&self) -> bool {
        (self.to_bytes()[0] & 1) == 1
    }
}

pub fn batch_invert(zs: &mut [Fe]) {
    let n = zs.len();
    if n == 0 {
        return;
    }
    let mut scratch = vec![Fe::ONE; n];
    let mut acc = Fe::ONE;
    for i in 0..n {
        scratch[i] = acc;
        acc = acc.mul(&zs[i]);
    }
    acc = acc.invert();
    for i in (0..n).rev() {
        let tmp = acc.mul(&zs[i]);
        zs[i] = acc.mul(&scratch[i]);
        acc = tmp;
    }
}
