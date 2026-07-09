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
