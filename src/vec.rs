//! An array of Scalars, written `Vec<D, T>` but pronounced 'vector'.
//!
//! Vectors support binary operations between two Vectors or between one Vector and one Scalar.
//! All Arithmetic operators and Bitwise operators are supported, where the two Scalar types
//! involved support the operation. All operations operate on each component separately.
//!
//! Vectors also support Negation and Logical Negation where the underlying Scalar Type supports
//! it.
//!
//! # Examples
//!
//! ```
//! use beagle::vec::{Vec3};
//!
//! let v1 = Vec3::from([2f32, 3f32, 5f32]);
//! let v2 = Vec3::from([7f32, 11f32, 13f32]);
//! assert_eq!(v1+v2, Vec3::from([9f32, 14f32, 18f32]));
//! ```

use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::ops::{Index,IndexMut,Range,RangeFrom,RangeTo,RangeFull,Deref,DerefMut};
use std::slice::{Iter,IterMut};

use num::{ApproxZero,Sign,Sqrt};
use scalar_array::{Scalar,Dim};
use scalar_array::{ScalarArray,Fold,Cast,CastBinary};
use scalar_array::{ComponentPartialEq,ComponentEq,ComponentPartialOrd,ComponentOrd,ComponentMul};

/// An array of Scalars, written `Vec<D, T>` but pronounced 'vector'.
///
/// Vectors support binary operations between two Vectors or between one Vector and one Scalar.
/// All Arithmetic operators and Bitwise operators are supported, where the two Scalar types
/// involved support the operation. All operations operate on each component separately.
///
/// Vectors also support Negation and Logical Negation where the underlying Scalar Type supports
/// it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vec<D, T: Scalar> (D::Output) where D: Dim<T>;

/// Types that represent an array of scalars.
impl <T: Scalar, D: Dim<T>> ScalarArray for Vec<D, T> {
    /// The type of the underlying scalar in the array.
    type Scalar = T;
    /// The type of a single scalar.
    type Type = T;
    /// The dimension of the scalar array.
    type Dim = D;

    /// Construct a vector from a an array of scalars `v`. Most useful in conjunction
    /// with `Dim::from_iter`.
    #[inline(always)]
    fn new(v: D::Output) -> Self { Vec(v) }
    /// Returns a slice iterator over the scalars of `self`.
    #[inline(always)]
    fn iter(&self) -> Iter<T> { (self.as_ref() as &[T]).iter() }
    /// Returns a mutable slice iterator over the scalars of `self`.
    #[inline(always)]
    fn iter_mut(&mut self) -> IterMut<T> { (self.as_mut() as &mut [T]).iter_mut() }
    /// Construct a vector from a single scalar value, setting all elements to that value.
    #[inline(always)]
    fn from_value(v: T) -> Self { Vec(<D as Dim<T>>::from_value(v)) }

    /// Fold all the scalar values into a single output given two folding functions,
    /// The first folding function only applies to the first element of the ScalarArray.
    #[inline(always)]
    fn fold<U, F0: FnOnce(&Self::Scalar)->U, F: Fn(U, &Self::Scalar)->U>(&self, f0: F0, f: F) -> U {
        let init = f0(&self[0]);
        self[1..].iter().fold(init, f)
    }
}

/// Types that can be fold with another `ScalarArray` of the same dimension into single value.
impl <T: Scalar, Rhs: Scalar, D: Dim<T>+Dim<Rhs>> Fold<Rhs> for Vec<D, T> {
    /// The right hand side type.
    type RhsArray = Vec<D, Rhs>;

    /// Fold two `ScalarArray`s together using a binary function.
    #[inline(always)]
    fn fold_together<O, F0: FnOnce(&<Self as ScalarArray>::Scalar, &Rhs)->O, F: Fn(O, &<Self as ScalarArray>::Scalar, &Rhs)->O>(&self, rhs: &Self::RhsArray, f0: F0, f: F) -> O {
        let init = f0(&self[0], &rhs[0]);
        self[1..].iter().zip(rhs[1..].iter()).fold(init, |acc, (l, r)| f(acc, l, r))
    }
}

/// Types that can be transformed from into a `ScalarArray` with Scalar type `O`.
impl <T: Scalar, O: Scalar, D: Dim<T>+Dim<O>> Cast<O> for Vec<D, T> {
    /// The resulting type
    type Output = Vec<D, O>;

    /// Transform a single `ScalarArray` using a unary function.
    #[inline(always)]
    fn unary<F: Fn(&<Self as ScalarArray>::Scalar)->O>(&self, f: F) -> Self::Output {
        Vec::new(<D as Dim<O>>::from_iter(self.iter().map(|s| f(s))))
    }

    /// Transform two binary `ScalarArray`s using a binary function.
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &<Self as ScalarArray>::Scalar)->O>(&self, rhs: &Self, f: F) -> Self::Output {
        Vec::new(<D as Dim<O>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| f(l, r))))
    }
}

/// Types that can be transformed from into a `ScalarArray` with Scalar type `O`.
impl <T: Scalar, Rhs: Scalar, O: Scalar, D: Dim<T>+Dim<Rhs>+Dim<O>> CastBinary<Rhs, O> for Vec<D, T> {
    /// The right hand side type.
    type RhsArray = Vec<D, Rhs>;

    /// The resulting type.
    type Output = Vec<D, O>;

    /// Transform two binary `ScalarArray`s using a binary function.
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &Rhs)->O>(&self, rhs: &Self::RhsArray, f: F) -> Self::Output {
        Vec::new(<D as Dim<O>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| f(l, r))))
    }
}

impl <T: Scalar, D: Dim<T>> Vec<D, T> {
    /// Returns the sum of all the elements of `self`.
    #[inline(always)]
    pub fn sum(&self) -> T
    where T: Add<Output=T> {
        self.fold(|&i| i, |sum, &i| sum + i)
    }

    // TODO: geometric functions
    // refraction vector
    //  Tfd  refract(Tfd I, Tfd N, float eta)
}

/// Returns the sum of all the elements of `v`.
#[inline(always)]
pub fn sum<T: Scalar, D: Dim<T>>(v: &Vec<D, T>) -> T
where T: Add<Output=T> { v.sum() }

/// Types that can be Dot producted
pub trait Dot<Rhs=Self> {
    /// The resulting type.
    type Output: Scalar;
    /// Returns the dot product of `self` and `rhs`.
    #[inline(always)]
    fn dot(self, rhs: Rhs) -> Self::Output;
}

impl <Rhs: Scalar, T: Scalar+Mul<Rhs>, D: Dim<T>+Dim<Rhs>> Dot<Vec<D, Rhs>> for Vec<D, T>
where <T as Mul<Rhs>>::Output: Scalar+Add<Output=<T as Mul<Rhs>>::Output> {
    /// The resulting type.
    type Output = <T as Mul<Rhs>>::Output;
    /// Returns the dot product of `self` and `rhs`.
    #[inline(always)]
    fn dot(self, rhs: Vec<D, Rhs>) -> Self::Output {
        Fold::<Rhs>::fold_together(&self, &rhs, |&l, &r| l*r, |acc, &l, &r| acc + l*r)
    }
}

impl <'l, Rhs: Scalar, T: Scalar+Mul<Rhs>, D: Dim<T>+Dim<Rhs>> Dot<Vec<D, Rhs>> for &'l Vec<D, T>
where <T as Mul<Rhs>>::Output: Scalar+Add<Output=<T as Mul<Rhs>>::Output> {
    /// The resulting type.
    type Output = <T as Mul<Rhs>>::Output;
    /// Returns the dot product of `self` and `rhs`.
    #[inline(always)]
    fn dot(self, rhs: Vec<D, Rhs>) -> Self::Output {
        Fold::<Rhs>::fold_together(self, &rhs, |&l, &r| l*r, |acc, &l, &r| acc + l*r)
    }
}

impl <'r, Rhs: Scalar, T: Scalar+Mul<Rhs>, D: Dim<T>+Dim<Rhs>> Dot<&'r Vec<D, Rhs>> for Vec<D, T>
where <T as Mul<Rhs>>::Output: Scalar+Add<Output=<T as Mul<Rhs>>::Output> {
    /// The resulting type.
    type Output = <T as Mul<Rhs>>::Output;
    /// Returns the dot product of `self` and `rhs`.
    #[inline(always)]
    fn dot(self, rhs: &'r Vec<D, Rhs>) -> Self::Output {
        Fold::<Rhs>::fold_together(&self, rhs, |&l, &r| l*r, |acc, &l, &r| acc + l*r)
    }
}

impl <'l, 'r, Rhs: Scalar, T: Scalar+Mul<Rhs>, D: Dim<T>+Dim<Rhs>> Dot<&'r Vec<D, Rhs>> for &'l Vec<D, T>
where <T as Mul<Rhs>>::Output: Scalar+Add<Output=<T as Mul<Rhs>>::Output> {
    /// The resulting type.
    type Output = <T as Mul<Rhs>>::Output;
    /// Returns the dot product of `self` and `rhs`.
    #[inline(always)]
    fn dot(self, rhs: &'r Vec<D, Rhs>) -> Self::Output {
        Fold::<Rhs>::fold_together(self, rhs, |&l, &r| l*r, |acc, &l, &r| acc + l*r)
    }
}

/// Returns the dot product of `lhs` and `rhs`.
#[inline(always)]
pub fn dot<Rhs, Lhs: Dot<Rhs>>(lhs:Lhs, rhs: Rhs) -> <Lhs as Dot<Rhs>>::Output {
    lhs.dot(rhs)
}

/// Returns the length squared of `v`.
#[inline(always)]
pub fn length2<V: Copy+Dot>(v: V) -> <V as Dot>::Output {
    v.dot(v)
}

/// Returns the length of `v`.
#[inline(always)]
pub fn length<V: Copy+Dot>(v: V) -> <V as Dot>::Output
where <V as Dot>::Output: Sqrt {
    length2(v).sqrt()
}

/// Returns the normalized version of `v`.
#[inline(always)]
pub fn normalize<V: Copy+Dot>(v: V) -> <V as Div<<V as Dot>::Output>>::Output
where <V as Dot>::Output: Sqrt,
V: Div<<V as Dot>::Output> {
    v / length(v)
}

/// Returns `n` if `nref.dot(i)` is negative, else `-n`.
pub fn faceforward<N: Neg<Output=N>, I, NRef: Dot<I>>(n: N, i: I, nref: NRef) -> N
where <NRef as Dot<I>>::Output: Sign<Output=bool>,
N: Mul<<NRef as Dot<I>>::Output, Output=N> {
    if nref.dot(i).is_negative() { n } else { -n }
}

/// Returns `cos²(θ) < ε²`, where `θ` = the angle between `a` and `b`.
#[inline]
pub fn is_perpendicular<O:Scalar, B: Copy+Dot, A: Copy+Dot+Dot<B>>(a: A, b: B, epsilon_squared: &ApproxZero<O>) -> bool
where <A as Dot<B>>::Output: Mul<Output=O>,
<A as Dot>::Output: Mul<<B as Dot>::Output, Output=O> {
    // We're looking to return abs(cos(θ)) <= ε
    // Proof:
    // ∵                 a·b  =  |a|*|b|*cos(θ)
    // ∵            cos²(θ)  <=  ε²
    //     |a|²*|b|²*cos²(θ) <=  |a|²*|b|²*ε²
    //   (|a|*|b|*|cos(θ)|)² <=  |a|²*|b|²*ε²
    //                (a·b)² <=  |a|²*|b|²*ε²
    // ∴  (a·b)² / |a|²*|b|² <=  ε²
    let a_dot_b = a.dot(b);
    epsilon_squared.approx_zero_ratio(a_dot_b * a_dot_b, length2(a) * length2(b))
}

/// Reflects `i` against `n` as `i - n * dot(i,n) * 2`.
pub fn reflect<N: Copy, I: Copy+Dot<N>>(i: I, n: N) -> <I as Sub<<N as Mul<<<I as Dot<N>>::Output as Add>::Output>>::Output>>::Output
where <I as Dot<N>>::Output: Add,
N: Mul<<<I as Dot<N>>::Output as Add>::Output>,
I: Sub<<N as Mul<<<I as Dot<N>>::Output as Add>::Output>>::Output> {
    let i_n = i.dot(n);
    i - n * (i_n+i_n)
}

/// Returns the distance squared between `lhs` and `rhs`.
#[inline]
pub fn distance2<Rhs, Lhs>(lhs: Lhs, rhs: Rhs) -> <<Lhs as Sub<Rhs>>::Output as Dot>::Output
where Lhs: Sub<Rhs>,
<Lhs as Sub<Rhs>>::Output: Copy+Dot {
    length2(lhs - rhs)
}

/// Returns the distance between `lhs` and `rhs`.
#[inline]
pub fn distance<Rhs, Lhs>(lhs: Lhs, rhs: Rhs) -> <<Lhs as Sub<Rhs>>::Output as Dot>::Output
where Lhs: Sub<Rhs>,
<Lhs as Sub<Rhs>>::Output: Copy+Dot,
<<Lhs as Sub<Rhs>>::Output as Dot>::Output: Sqrt {
    length(lhs - rhs)
}

/// Returns the cross product of `lhs` and `rhs`.
#[inline(always)]
pub fn cross<Rhs: Scalar, Lhs: Scalar+Mul<Rhs>>(lhs: &Vec3<Lhs>, rhs: &Vec3<Rhs>) -> Vec3<<<Lhs as Mul<Rhs>>::Output as Sub>::Output>
where <Lhs as Mul<Rhs>>::Output: Sub,
<<Lhs as Mul<Rhs>>::Output as Sub>::Output: Scalar {
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

/// Types that can be component-wise multiplied.
impl <T: Scalar, Rhs: Scalar, D: Dim<T>+Dim<Rhs>> ComponentMul<Rhs> for Vec<D, T>
where T: Mul<Rhs>,
<T as Mul<Rhs>>::Output: Scalar,
D: Dim<<T as Mul<Rhs>>::Output> {
    /// The right hand side type.
    type RhsArray = Vec<D, Rhs>;

    /// The resulting type.
    type Output = Vec<D, <T as Mul<Rhs>>::Output>;

    /// Multiplies the components of `self` and `rhs`.
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
