use super::field::Fe;

#[derive(Clone, Copy)]
pub struct Point {
    pub x: Fe,
    pub y: Fe,
    pub z: Fe,
    pub t: Fe,
}

#[derive(Clone, Copy)]
struct Projective {
    x: Fe,
    y: Fe,
    z: Fe,
}

#[derive(Clone, Copy)]
struct Completed {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

#[derive(Clone, Copy)]
pub struct Niels {
    pub y_plus_x: Fe,
    pub y_minus_x: Fe,
    pub z: Fe,
    pub t2d: Fe,
}

pub fn edwards_d2() -> Fe {
    let a = Fe([121666, 0, 0, 0, 0]).invert();
    let d = Fe([121665, 0, 0, 0, 0])
        .negate()
        .mul(&a); // -121665/121666
    d.add(&d)
}

impl Point {
    pub const IDENTITY: Point = Point {
        x: Fe::ZERO,
        y: Fe::ONE,
        z: Fe::ONE,
        t: Fe::ZERO,
    };

    #[inline(always)]
    fn as_projective(&self) -> Projective {
        Projective {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }

    pub fn as_niels(&self, d2: &Fe) -> Niels {
        Niels {
            y_plus_x: self.y.add(&self.x),
            y_minus_x: self.y.sub(&self.x),
            z: self.z,
            t2d: self.t.mul(d2),
        }
    }

    #[inline(always)]
    pub fn double(&self) -> Point {
        self.as_projective()
            .double()
            .as_extended()
    }

    #[inline(always)]
    pub fn add_niels(&self, n: &Niels) -> Point {
        let y_plus_x = self.y.add(&self.x);
        let y_minus_x = self.y.sub(&self.x);
        let pp = y_plus_x.mul(&n.y_plus_x);
        let mm = y_minus_x.mul(&n.y_minus_x);
        let tt2d = self.t.mul(&n.t2d);
        let zz = self.z.mul(&n.z);
        let zz2 = zz.add(&zz);
        Completed {
            x: pp.sub(&mm),
            y: pp.add(&mm),
            z: zz2.add(&tt2d),
            t: zz2.sub(&tt2d),
        }
        .as_extended()
    }

    #[inline(always)]
    pub fn sub_niels(&self, n: &Niels) -> Point {
        let y_plus_x = self.y.add(&self.x);
        let y_minus_x = self.y.sub(&self.x);
        // Adding the negation: -niels = (y_minus_x, y_plus_x, z, -t2d).
        let pp = y_plus_x.mul(&n.y_minus_x);
        let mm = y_minus_x.mul(&n.y_plus_x);
        let tt2d = self.t.mul(&n.t2d);
        let zz = self.z.mul(&n.z);
        let zz2 = zz.add(&zz);
        Completed {
            x: pp.sub(&mm),
            y: pp.add(&mm),
            z: zz2.sub(&tt2d),
            t: zz2.add(&tt2d),
        }
        .as_extended()
    }
}

impl Projective {
    #[inline(always)]
    fn double(&self) -> Completed {
        let xx = self.x.square();
        let yy = self.y.square();
        let zz = self.z.square();
        let zz2 = zz.add(&zz);
        let x_plus_y = self.x.add(&self.y);
        let x_plus_y_sq = x_plus_y.square();
        let yy_plus_xx = yy.add(&xx);
        let yy_minus_xx = yy.sub(&xx);
        Completed {
            x: x_plus_y_sq.sub(&yy_plus_xx),
            y: yy_plus_xx,
            z: yy_minus_xx,
            t: zz2.sub(&yy_minus_xx),
        }
    }
}

impl Completed {
    #[inline(always)]
    fn as_extended(&self) -> Point {
        Point {
            x: self.x.mul(&self.t),
            y: self.y.mul(&self.z),
            z: self.z.mul(&self.t),
            t: self.x.mul(&self.y),
        }
    }
}

/// Affine-Niels add that consumes a completed point and emits a completed
/// point. This is the p1p1→p1p1 spike: it is algebraically the same 7 field
/// multiplications as `as_extended` (4) plus an affine mixed add (3), so it
/// does not reduce the mul count and is not used by the Metal kernel.
#[cfg(test)]
fn add_affine_niels_completed(
    c: &Completed,
    y_plus_x: &Fe,
    y_minus_x: &Fe,
    t2d: &Fe,
) -> (Completed, u32) {
    let mut muls = 0u32;
    let ye = c.y.mul(&c.z);
    muls += 1;
    let xe = c.x.mul(&c.t);
    muls += 1;
    let ze = c.z.mul(&c.t);
    muls += 1;
    let te = c.x.mul(&c.y);
    muls += 1;
    let pp = ye.add(&xe).mul(y_plus_x);
    muls += 1;
    let mm = ye.sub(&xe).mul(y_minus_x);
    muls += 1;
    let tt2d = te.mul(t2d);
    muls += 1;
    let zz2 = ze.add(&ze);
    (
        Completed {
            x: pp.sub(&mm),
            y: pp.add(&mm),
            z: zz2.add(&tt2d),
            t: zz2.sub(&tt2d),
        },
        muls,
    )
}

#[cfg(test)]
mod p1p1_spike {
    use super::*;

    fn base_point() -> Point {
        // Same fixed base as `fast::BASE` (z = 1, t = x*y).
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

    fn compress(p: &Point) -> [u8; 32] {
        let zinv = p.z.invert();
        let x = p.x.mul(&zinv);
        let y = p.y.mul(&zinv);
        let mut bytes = y.to_bytes();
        bytes[31] |= (x.to_bytes()[0] & 1) << 7;
        bytes
    }

    #[test]
    fn completed_affine_add_matches_extended_and_costs_seven_muls() {
        let base = base_point();
        let d2 = edwards_d2();
        let n = base.as_niels(&d2);
        // Completed form of `base`: (X:Y:Z:T) = (x:y:1:1) so x=X/T, y=Y/Z.
        let mut c = Completed {
            x: base.x,
            y: base.y,
            z: Fe::ONE,
            t: Fe::ONE,
        };
        let mut acc = base;
        for _ in 0..4 {
            let (next, muls) = add_affine_niels_completed(
                &c,
                &n.y_plus_x,
                &n.y_minus_x,
                &n.t2d,
            );
            assert_eq!(muls, 7, "p1p1 chain does not beat the 7-mul affine add");
            acc = acc.add_niels(&n);
            let got = next.as_extended();
            assert_eq!(compress(&got), compress(&acc));
            c = next;
        }
    }
}
