//! An array of Scalars, written `Vec<D, V>` but pronounced 'vector'.
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
use std::mem::transmute;

use Value;
use index::Apply;
use num::{ApproxZero,Sign,Sqrt};
use scalar_array::{Scalar,Construct,Dim,Array};
use scalar_array::{ScalarArray,ScalarIterator,ScalarIteratorMut,Fold,Cast,CastBinary};
use scalar_array::{ComponentPartialEq,ComponentEq,ComponentPartialOrd,ComponentOrd,ComponentMul};
use index::{VecRef};

/// An array of Scalars, written `Vec<D, V>` but pronounced 'vector'.
///
/// Vectors support binary operations between two Vectors or between one Vector and one Scalar.
/// All Arithmetic operators and Bitwise operators are supported, where the two Scalar types
/// involved support the operation. All operations operate on each component separately.
///
/// Vectors also support Negation and Logical Negation where the underlying Scalar Type supports
/// it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vec<D, V: Scalar> (<D as Array<T>>::Type)
where D: Dim<V>, <D as Array<T>>::Type: Copy; // TODO: Remove Copy where clause. Blocked on rust/issues#32722

/// Returns the normalized version of `v`.
#[inline(always)]
pub fn normalize<V: Copy+Dot>(v: V) -> <V as Div<<V as Dot>::Output>>::Output
where <V as Dot>::Output: Sqrt,
V: Div<<V as Dot>::Output> {
    v / length(v)
}

/// Returns an approximated normalized version of `v`.
#[inline(always)]
pub fn normalize_approx<V: Copy+Dot>(v: V) -> <V as Mul<<V as Dot>::Output>>::Output
where <V as Dot>::Output: Sqrt,
V: Mul<<V as Dot>::Output> {
    v * length2(v).inverse_sqrt()
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


// d888888b d8b   db d8888b. d88888b db    db
//   `88'   888o  88 88  `8D 88'     `8b  d8'
//    88    88V8o 88 88   88 88ooooo  `8bd8'
//    88    88 V8o88 88   88 88~~~~~  .dPYb.
//   .88.   88  V888 88  .8D 88.     .8P  Y8.
// Y888888P VP   V8P Y8888D' Y88888P YP    YP

impl <V: Scalar, D: Dim<V>> Index<usize> for Vec<D, V> {
    type Output = V;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output { &(self.as_ref() as &[V])[i] }
}
impl <V: Scalar, D: Dim<V>> IndexMut<usize> for Vec<D, V> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output { &mut (self.as_mut() as &mut [V])[i] }
}
impl <V: Scalar, D: Dim<V>> Index<Range<usize>> for Vec<D, V> {
    type Output = [V];
    #[inline(always)]
    fn index(&self, i: Range<usize>) -> &Self::Output { &(self.as_ref() as &[V])[i] }
}
impl <V: Scalar, D: Dim<V>> IndexMut<Range<usize>> for Vec<D, V> {
    #[inline(always)]
    fn index_mut(&mut self, i: Range<usize>) -> &mut Self::Output { &mut (self.as_mut() as &mut [V])[i] }
}
impl <V: Scalar, D: Dim<V>> Index<RangeTo<usize>> for Vec<D, V> {
    type Output = [V];
    #[inline(always)]
    fn index(&self, i: RangeTo<usize>) -> &Self::Output { &(self.as_ref() as &[V])[i] }
}
impl <V: Scalar, D: Dim<V>> IndexMut<RangeTo<usize>> for Vec<D, V> {
    #[inline(always)]
    fn index_mut(&mut self, i: RangeTo<usize>) -> &mut Self::Output { &mut (self.as_mut() as &mut [V])[i] }
}
impl <V: Scalar, D: Dim<V>> Index<RangeFrom<usize>> for Vec<D, V> {
    type Output = [V];
    #[inline(always)]
    fn index(&self, i: RangeFrom<usize>) -> &Self::Output { &(self.as_ref() as &[V])[i] }
}
impl <V: Scalar, D: Dim<V>> IndexMut<RangeFrom<usize>> for Vec<D, V> {
    #[inline(always)]
    fn index_mut(&mut self, i: RangeFrom<usize>) -> &mut Self::Output { &mut (self.as_mut() as &mut [V])[i] }
}
impl <V: Scalar, D: Dim<V>> Index<RangeFull> for Vec<D, V> {
    type Output = [V];
    #[inline(always)]
    fn index(&self, i: RangeFull) -> &Self::Output { &(self.as_ref() as &[V])[i] }
}
impl <V: Scalar, D: Dim<V>> IndexMut<RangeFull> for Vec<D, V> {
    #[inline(always)]
    fn index_mut(&mut self, i: RangeFull) -> &mut Self::Output { &mut (self.as_mut() as &mut [V])[i] }
}
