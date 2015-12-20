//! An array of Scalars, written `Vec<D, T>` but pronounced 'vector'.
//!
//1! Vectors support binary operations between two Vectors or between one Vector and one Scalar.
//! All Arithmetic operators and Bitwise operators are supported, where the two Scalar types
//! involved support the operation. All operations operate on each component separately.
//!
//! Vectors also support Negation and Logical Negation where the underlying Scalar Type supports
//! it.

use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::ops::{Index,IndexMut,Range,RangeFrom,RangeTo,RangeFull,Deref,DerefMut};
use std::slice::{Iter,IterMut};

use num::{Approx,IsNegative,Sqrt};
use scalar_array::{Scalar,Dim};
use scalar_array::{ScalarArray,Fold,Cast,CastBinary};
use scalar_array::{ComponentPartialEq,ComponentEq,ComponentPartialOrd,ComponentOrd,ComponentMul};

/// Vec is an array of Scalars
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vec<D, T: Scalar> (D::Output) where D: Dim<T>;

/// Types that represent an array of scalars (a matrix or a vector)
impl <T: Scalar, D: Dim<T>> ScalarArray for Vec<D, T> {
    /// The type of the underlying scalar in the array
    type Scalar = T;
    /// The type of a single element of this type (a single row for matrices/a scalar for vectors)
    type Type = T;
    /// The dimension of the scalar array
    type Dim = D;

    /// Construct a vector from a an array
    #[inline(always)]
    fn new(v: D::Output) -> Self { Vec(v) }
    /// Get a slice iterator over the elements (the rows for matrices/the scalars for vectors)
    #[inline(always)]
    fn iter(&self) -> Iter<T> { (self.as_ref() as &[T]).iter() }
    /// Get a mutable slice iterator over the elements (the scalars)
    #[inline(always)]
    fn iter_mut(&mut self) -> IterMut<T> { (self.as_mut() as &mut [T]).iter_mut() }
    /// Construct a vector from a single scalar value, setting all elements to that value
    #[inline(always)]
    fn from_value(v: T) -> Self { Vec(<D as Dim<T>>::from_value(v)) }

    /// Fold all the scalar values into a single output given two folding functions,
    /// The first folding function only applies to the first element of the ScalarArray
    #[inline(always)]
    fn fold<U, F0: FnOnce(&Self::Scalar)->U, F: Fn(U, &Self::Scalar)->U>(&self, f0: F0, f: F) -> U {
        let init = f0(&self[0]);
        self[1..].iter().fold(init, f)
    }
}

/// Types that can be transformed from into single value
impl <T: Scalar, Rhs: Scalar, D: Dim<T>+Dim<Rhs>> Fold<Rhs> for Vec<D, T> {
    /// The right hand side type
    type RhsArray = Vec<D, Rhs>;

    /// Fold two `ScalarArray`s together using a binary function
    #[inline(always)]
    fn fold_together<O, F0: FnOnce(&<Self as ScalarArray>::Scalar, &Rhs)->O, F: Fn(O, &<Self as ScalarArray>::Scalar, &Rhs)->O>(&self, rhs: &Self::RhsArray, f0: F0, f: F) -> O {
        let init = f0(&self[0], &rhs[0]);
        self[1..].iter().zip(rhs[1..].iter()).fold(init, |acc, (l, r)| f(acc, l, r))
    }
}

/// Types that can be transformed from into a `ScalarArray` of `<O>`
impl <T: Scalar, O: Scalar, D: Dim<T>+Dim<O>> Cast<O> for Vec<D, T> {
    /// The resulting type
    type Output = Vec<D, O>;

    /// Transform a single `ScalarArray` using a unary function
    #[inline(always)]
    fn unary<F: Fn(&<Self as ScalarArray>::Scalar)->O>(&self, f: F) -> Self::Output {
        Vec::new(<D as Dim<O>>::from_iter(self.iter().map(|s| f(s))))
    }

    /// Transform two binary `ScalarArray`s using a binary function
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &<Self as ScalarArray>::Scalar)->O>(&self, rhs: &Self, f: F) -> Self::Output {
        Vec::new(<D as Dim<O>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| f(l, r))))
    }
}

impl <T: Scalar, Rhs: Scalar, O: Scalar, D: Dim<T>+Dim<Rhs>+Dim<O>> CastBinary<Rhs, O> for Vec<D, T> {
    /// The right hand side type
    type RhsArray = Vec<D, Rhs>;

    /// The resulting type
    type Output = Vec<D, O>;

    /// Transform two binary `ScalarArray`s using a binary function
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &Rhs)->O>(&self, rhs: &Self::RhsArray, f: F) -> Self::Output {
        Vec::new(<D as Dim<O>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| f(l, r))))
    }
}

impl <T: Scalar, D: Dim<T>> Vec<D, T> {
    /// Sum up all the elements
    #[inline(always)]
    pub fn sum(&self) -> T
    where T: Add<Output=T> {
        self.fold(|&i| i, |sum, &i| sum + i)
    }

    /// Dot product
    #[inline(always)]
    pub fn dot<Rhs: Scalar>(&self, rhs: &Vec<D, Rhs>) -> <T as Mul<Rhs>>::Output
    where D: Dim<Rhs>,
    T: Mul<Rhs>,
    <T as Mul<Rhs>>::Output: Add<Output=<T as Mul<Rhs>>::Output> {
        Fold::<Rhs>::fold_together(self, rhs, |&l, &r| l*r, |acc, &l, &r| acc + l*r)
    }

    /// Length squared
    #[inline(always)]
    pub fn length2(&self) -> <T as Mul>::Output
    where T: Mul,
    <T as Mul>::Output: Add<Output=<T as Mul>::Output> {
        self.dot(self)
    }

    /// Length squared
    #[inline(always)]
    pub fn length(&self) -> <T as Mul>::Output
    where T: Mul,
    <T as Mul>::Output: Sqrt+Add<Output=<T as Mul>::Output> {
        self.length2().sqrt()
    }
    /// Length squared
    #[inline(always)]
    pub fn normalize(&self) -> Vec<D, <T as Div<<T as Mul>::Output>>::Output>
    where <T as Mul>::Output: Scalar+Add<Output=<T as Mul>::Output> + Sqrt,
    T: Mul+Div<<T as Mul>::Output>,
    D: Dim<<T as Div<<T as Mul>::Output>>::Output>,
    <T as Div<<T as Mul>::Output>>::Output: Scalar {
        self / self.length()
    }

    /// returns epsilon.is_zero(cos(θ)), where θ = the angle between `self` and `rhs`.
    #[inline]
    pub fn is_perpendicular<Rhs: Scalar+Mul, O>(&self, rhs: &Vec<D, Rhs>, epsilon_squared: &Approx<O>) -> bool
    where D: Dim<Rhs>,
    T: Mul+Mul<Rhs>,
    Rhs: Mul,
    <T as Mul<Rhs>>::Output: Add<Output=<T as Mul<Rhs>>::Output>+Mul<Output=O>+Copy,
    <Rhs as Mul>::Output: Add<Output=<Rhs as Mul>::Output>,
    <T as Mul>::Output: Add<Output=<T as Mul>::Output>+Mul<<Rhs as Mul>::Output, Output=O> {
        let (a, b) = (self, rhs);
        // We're looking to return abs(cos(θ)) <= ε
        // Proof:
        // ∵                 a·b  =  |a|*|b|*cos(θ)
        // ∵            |cos(θ)| <=  ε
        //      |a|*|b|*|cos(θ)| <=  |a|*|b|*ε
        //   (|a|*|b|*|cos(θ)|)² <=  (|a|*|b|*ε)²
        //                (a·b)² <=  |a|²*|b|²*ε²
        // ∴  (a·b)² / |a|²*|b|² <=  ε²
        let a_dot_b = a.dot(b);
        epsilon_squared.approx_zero(&(a_dot_b * a_dot_b), &(a.length2() * b.length2()))
    }

    /// Distance squared
    #[inline]
    pub fn distance2<Rhs: Scalar>(&self, rhs: &Vec<D, Rhs>) -> <<T as Sub<Rhs>>::Output as Mul>::Output
    where D: Dim<Rhs>,
    T: Sub<Rhs>,
    <T as Sub<Rhs>>::Output: Mul,
    <<T as Sub<Rhs>>::Output as Mul>::Output: Add<Output=<<T as Sub<Rhs>>::Output as Mul>::Output> {
        Fold::<Rhs>::fold_together(self, rhs, |&l, &r| (l-r)*(l-r), |acc, &l, &r| acc + (l-r)*(l-r))
    }

    /// Distance
    #[inline]
    pub fn distance<Rhs: Scalar>(&self, rhs: &Vec<D, Rhs>) -> <<T as Sub<Rhs>>::Output as Mul>::Output
    where D: Dim<Rhs>,
    T: Sub<Rhs>,
    <T as Sub<Rhs>>::Output: Mul,
    <<T as Sub<Rhs>>::Output as Mul>::Output: Sqrt+Add<Output=<<T as Sub<Rhs>>::Output as Mul>::Output> {
        Fold::<Rhs>::fold_together(self, rhs, |&l, &r| (l-r)*(l-r), |acc, &l, &r| acc + (l-r)*(l-r)).sqrt()
    }

    /// returns self if dot(Nref, I) is negative, else -self
    pub fn faceforward<U: Scalar, V:Scalar>(&self, i: &Vec<D, U>, nref: Vec<D, V>) -> Vec<D, T>
    where T: Neg<Output=T>,
    D: Dim<U>+Dim<V>+Dim<<T as Neg>::Output>,
    V: Mul<U>,
    <V as Mul<U>>::Output: IsNegative+Add<Output=<V as Mul<U>>::Output> {
        if nref.dot(i).is_negative() { *self } else { -self }
    }

    /// reflection direction. self - 2 * dot(self,rhs) * rhs
    pub fn reflect<Rhs: Scalar>(&self, rhs: &Vec<D, Rhs>) -> Vec<D, <T as Sub<<<T as Mul<Rhs>>::Output as Mul<Rhs>>::Output>>::Output>
    where D: Dim<Rhs>, // The right hand side
    T: Mul<Rhs>, // Needed for self.dot(rhs)
    <T as Mul<Rhs>>::Output: Add<Output=<T as Mul<Rhs>>::Output>+Copy+Mul<Rhs>, // Needed for self.dot(rhs) + Needed for 2*r
    T: Sub<<<T as Mul<Rhs>>::Output as Mul<Rhs>>::Output>,
    <T as Sub<<<T as Mul<Rhs>>::Output as Mul<Rhs>>::Output>>::Output: Scalar,
    D: Dim<<T as Sub<<<T as Mul<Rhs>>::Output as Mul<Rhs>>::Output>>::Output> {
        let l_dot_r = self.dot(rhs);
        let two_ldr = l_dot_r+l_dot_r;

        CastBinary::<Rhs, <T as Sub<<<T as Mul<Rhs>>::Output as Mul<Rhs>>::Output>>::Output>::binary(self, rhs, |&l, &r| l - two_ldr*r)
    }

    // TODO: geometric functions
    // refraction vector
    //  Tfd  refract(Tfd I, Tfd N, float eta)
}

/// Cross product
pub fn cross_product<T: Scalar, Rhs:Scalar>(lhs: &Vec3<T>, rhs: &Vec3<Rhs>) -> Vec3<<<T as Mul<Rhs>>::Output as Sub>::Output>
where T: Mul<Rhs>,
<T as Mul<Rhs>>::Output: Sub,
<<T as Mul<Rhs>>::Output as Sub>::Output: Scalar {
    Vec::new([lhs[2]*rhs[3] - lhs[3]*rhs[2], lhs[3]*rhs[1] - lhs[1]*rhs[3], lhs[1]*rhs[2] - lhs[2]*rhs[1]])
}

impl <T: Scalar, D: Dim<T>> Deref for Vec<D, T> {
    type Target = D::Output;
    #[inline] fn deref<'a>(&'a self) -> &'a Self::Target { &self.0 }
}
impl <T: Scalar, D: Dim<T>>  DerefMut for Vec<D, T> {
    #[inline] fn deref_mut<'a>(&'a mut self) -> &'a mut <Self as Deref>::Target { &mut self.0 }
}

impl <T: Scalar, D: Dim<T>> Borrow   <[T]> for Vec<D, T> {  #[inline(always)] fn borrow    (&    self) -> &    [T] { self.0.borrow() }  }
impl <T: Scalar, D: Dim<T>> BorrowMut<[T]> for Vec<D, T> {  #[inline(always)] fn borrow_mut(&mut self) -> &mut [T] { self.0.borrow_mut() }  }

impl <T: Scalar, D: Dim<T>> AsRef<[T]> for Vec<D, T> {  #[inline(always)] fn as_ref(&    self) -> &    [T] { self.0.as_ref() }  }
impl <T: Scalar, D: Dim<T>> AsMut<[T]> for Vec<D, T> {  #[inline(always)] fn as_mut(&mut self) -> &mut [T] { self.0.as_mut() }  }

// d888888b d8b   db d8888b. d88888b db    db
//   `88'   888o  88 88  `8D 88'     `8b  d8'
//    88    88V8o 88 88   88 88ooooo  `8bd8'
//    88    88 V8o88 88   88 88~~~~~  .dPYb.
//   .88.   88  V888 88  .8D 88.     .8P  Y8.
// Y888888P VP   V8P Y8888D' Y88888P YP    YP

impl <T: Scalar, D: Dim<T>> Index<usize> for Vec<D, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output { &(self.as_ref() as &[T])[i] }
}
impl <T: Scalar, D: Dim<T>> IndexMut<usize> for Vec<D, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output { &mut (self.as_mut() as &mut [T])[i] }
}
impl <T: Scalar, D: Dim<T>> Index<Range<usize>> for Vec<D, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, i: Range<usize>) -> &Self::Output { &(self.as_ref() as &[T])[i] }
}
impl <T: Scalar, D: Dim<T>> IndexMut<Range<usize>> for Vec<D, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: Range<usize>) -> &mut Self::Output { &mut (self.as_mut() as &mut [T])[i] }
}
impl <T: Scalar, D: Dim<T>> Index<RangeTo<usize>> for Vec<D, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, i: RangeTo<usize>) -> &Self::Output { &(self.as_ref() as &[T])[i] }
}
impl <T: Scalar, D: Dim<T>> IndexMut<RangeTo<usize>> for Vec<D, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: RangeTo<usize>) -> &mut Self::Output { &mut (self.as_mut() as &mut [T])[i] }
}
impl <T: Scalar, D: Dim<T>> Index<RangeFrom<usize>> for Vec<D, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, i: RangeFrom<usize>) -> &Self::Output { &(self.as_ref() as &[T])[i] }
}
impl <T: Scalar, D: Dim<T>> IndexMut<RangeFrom<usize>> for Vec<D, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: RangeFrom<usize>) -> &mut Self::Output { &mut (self.as_mut() as &mut [T])[i] }
}
impl <T: Scalar, D: Dim<T>> Index<RangeFull> for Vec<D, T> {
    type Output = [T];
    #[inline(always)]
    fn index(&self, i: RangeFull) -> &Self::Output { &(self.as_ref() as &[T])[i] }
}
impl <T: Scalar, D: Dim<T>> IndexMut<RangeFull> for Vec<D, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: RangeFull) -> &mut Self::Output { &mut (self.as_mut() as &mut [T])[i] }
}

impl<'a, T: Scalar, D: Dim<T>> IntoIterator for &'a Vec<D, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
impl<'a, T: Scalar, D: Dim<T>> IntoIterator for &'a mut Vec<D, T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

//  .o88b.  .d88b.  .88b  d88. d8888b.  .d88b.  d8b   db d88888b d8b   db d888888b
// d8P  Y8 .8P  Y8. 88'YbdP`88 88  `8D .8P  Y8. 888o  88 88'     888o  88 `~~88~~'
// 8P      88    88 88  88  88 88oodD' 88    88 88V8o 88 88ooooo 88V8o 88    88
// 8b      88    88 88  88  88 88~~~   88    88 88 V8o88 88~~~~~ 88 V8o88    88
// Y8b  d8 `8b  d8' 88  88  88 88      `8b  d8' 88  V888 88.     88  V888    88
//  `Y88P'  `Y88P'  YP  YP  YP 88       `Y88P'  VP   V8P Y88888P VP   V8P    YP

impl <T: Scalar+PartialEq, D: Dim<T>+Dim<bool>> ComponentPartialEq for Vec<D, T> {}
impl <T: Scalar+Eq, D: Dim<T>+Dim<bool>> ComponentEq for Vec<D, T> {}
impl <T: Scalar+PartialOrd, D: Dim<T>+Dim<bool>+Dim<Option<Ordering>>> ComponentPartialOrd for Vec<D, T> {}
impl <T: Scalar+Ord, D: Dim<T>+Dim<bool>+Dim<Option<Ordering>>+Dim<Ordering>> ComponentOrd for Vec<D, T> {}

impl <T: Scalar, Rhs: Scalar, D: Dim<T>+Dim<Rhs>> ComponentMul<Rhs> for Vec<D, T>
where T: Mul<Rhs>,
<T as Mul<Rhs>>::Output: Scalar,
D: Dim<<T as Mul<Rhs>>::Output> {
    /// The right hand side type
    type RhsArray = Vec<D, Rhs>;

    /// The resulting type
    type Output = Vec<D, <T as Mul<Rhs>>::Output>;

    fn cmp_mul(&self, rhs: &Self::RhsArray) -> Self::Output {
        self * rhs
    }
}


include!(concat!(env!("OUT_DIR"), "/vec.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_neg() {
        assert_eq!(-Vec1::from([1f64]),                   Vec1::from([-1f64]));
        assert_eq!(-Vec2::from([1f64, 2f64]),             Vec2::from([-1f64, -2f64]));
        assert_eq!(-Vec3::from([1f64, 2f64, 3f64]),       Vec3::from([-1f64, -2f64, -3f64]));
        assert_eq!(-Vec4::from([1f64, 2f64, 3f64, 4f64]), Vec4::from([-1f64, -2f64, -3f64, -4f64]));

        assert_eq!(-Vec1::from([1f32]),                   Vec1::from([-1f32]));
        assert_eq!(-Vec2::from([1f32, 2f32]),             Vec2::from([-1f32, -2f32]));
        assert_eq!(-Vec3::from([1f32, 2f32, 3f32]),       Vec3::from([-1f32, -2f32, -3f32]));
        assert_eq!(-Vec4::from([1f32, 2f32, 3f32, 4f32]), Vec4::from([-1f32, -2f32, -3f32, -4f32]));

        assert_eq!(-Vec1::from([1i64]),                   Vec1::from([-1i64]));
        assert_eq!(-Vec2::from([1i64, 2i64]),             Vec2::from([-1i64, -2i64]));
        assert_eq!(-Vec3::from([1i64, 2i64, 3i64]),       Vec3::from([-1i64, -2i64, -3i64]));
        assert_eq!(-Vec4::from([1i64, 2i64, 3i64, 4i64]), Vec4::from([-1i64, -2i64, -3i64, -4i64]));

        assert_eq!(-Vec1::from([1i32]),                   Vec1::from([-1i32]));
        assert_eq!(-Vec2::from([1i32, 2i32]),             Vec2::from([-1i32, -2i32]));
        assert_eq!(-Vec3::from([1i32, 2i32, 3i32]),       Vec3::from([-1i32, -2i32, -3i32]));
        assert_eq!(-Vec4::from([1i32, 2i32, 3i32, 4i32]), Vec4::from([-1i32, -2i32, -3i32, -4i32]));

        assert_eq!(-Vec1::from([1i16]),                   Vec1::from([-1i16]));
        assert_eq!(-Vec2::from([1i16, 2i16]),             Vec2::from([-1i16, -2i16]));
        assert_eq!(-Vec3::from([1i16, 2i16, 3i16]),       Vec3::from([-1i16, -2i16, -3i16]));
        assert_eq!(-Vec4::from([1i16, 2i16, 3i16, 4i16]), Vec4::from([-1i16, -2i16, -3i16, -4i16]));

        assert_eq!(-Vec1::from([1i8 ]),                   Vec1::from([-1i8 ]));
        assert_eq!(-Vec2::from([1i8 , 2i8 ]),             Vec2::from([-1i8 , -2i8 ]));
        assert_eq!(-Vec3::from([1i8 , 2i8 , 3i8 ]),       Vec3::from([-1i8 , -2i8 , -3i8 ]));
        assert_eq!(-Vec4::from([1i8 , 2i8 , 3i8 , 4i8 ]), Vec4::from([-1i8 , -2i8 , -3i8 , -4i8 ]));
    }

    #[test]
    fn test_vec_not() {
        assert_eq!(!Vec1::from([false]),                    Vec1::from([true]));
        assert_eq!(!Vec2::from([false, true]),              Vec2::from([true, false]));
        assert_eq!(!Vec3::from([false, true, false]),       Vec3::from([true, false, true]));
        assert_eq!(!Vec4::from([false, true, false, true]), Vec4::from([true, false, true, false]));
    }

    #[test]
    fn test_vec_add() {
        assert_eq!(Vec1::from([1f64])+Vec1::from([2f64]), Vec1::from([3f64]));
    }
}
