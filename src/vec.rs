//! An array of Scalars, written `Vec<D, T>` but pronounced 'vector.'
//!
//! Vectors support binary operations between two Vectors or between one Vector and one Scalar. All operations operate on each component
//! separately. All Arithmetic operators and Bitwise operators are supported, where the two Scalar types involved support the operation.
//!
//! Vectors also support Negation and Logical negation where the underlying Scalar type(s) support(s) it.

use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::ops::{Index,IndexMut};
use std::slice::{Iter,IterMut};

use scalar_array::{Scalar,Dim, ScalarArray,Cast, ComponentPartialEq,ComponentEq,ComponentPartialOrd,ComponentOrd};


/// Vec is an array of Scalars
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vec<D, T: Scalar> (D::Output) where D: Dim<T>;


impl <T: Scalar, D: Dim<T>> ScalarArray for Vec<D, T> {
    type Scalar = T;
    type Type = T;
    type Dim = D;

    #[inline(always)]
    fn new(v: D::Output) -> Self { Vec(v) }
    #[inline(always)]
    fn iter(&self) -> Iter<T> { (self.as_ref() as &[T]).iter() }
    #[inline(always)]
    fn iter_mut(&mut self) -> IterMut<T> { (self.as_mut() as &mut [T]).iter_mut() }
    #[inline(always)]
    fn from_value(v: T) -> Self { Vec(<D as Dim<T>>::from_value(v)) }

    #[inline(always)]
    fn fold<U, F: Fn(U, &T)->U>(&self, init: U, f: F) -> U { self.iter().fold(init, f) }
}


impl <T: Scalar, U: Scalar, D: Dim<T>+Dim<U>> Cast<U> for Vec<D, T> {
    type Output = Vec<D, U>;

    #[inline(always)]
    fn unary<F: Fn(&<Self as ScalarArray>::Scalar)->U>(&self, f: F) -> Self::Output {
        Vec::new(<D as Dim<U>>::from_iter(self.iter().map(|s| f(s))))
    }

    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &<Self as ScalarArray>::Scalar)->U>(&self, rhs: &Self, f: F) -> Self::Output {
        Vec::new(<D as Dim<U>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| f(l, r))))
    }
}

impl <T: Scalar+Default+Add<Output=T>, D: Dim<T>> Vec<D, T> {
    pub fn sum(&self) -> T {
        self.fold(Default::default(), |sum, &i| sum + i)
    }
}

impl <T: Scalar+Default+Add<Output=T>+Mul<Output=T>, D: Dim<T>> Vec<D, T> {
    pub fn dot(&self, rhs: &Self) -> T {
        (self*rhs).sum()
    }
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

impl <T: Scalar, D: Dim<T>> Index<usize> for Vec<D, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output { &(self.as_ref() as &[T])[i] }
}
impl <T: Scalar, D: Dim<T>> IndexMut<usize> for Vec<D, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output { &mut (self.as_mut() as &mut [T])[i] }
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

impl <T: Scalar+PartialEq, D: Dim<T>+Dim<bool>> ComponentPartialEq for Vec<D, T> {}
impl <T: Scalar+Eq, D: Dim<T>+Dim<bool>> ComponentEq for Vec<D, T> {}
impl <T: Scalar+PartialOrd, D: Dim<T>+Dim<bool>+Dim<Option<Ordering>>> ComponentPartialOrd for Vec<D, T> {}
impl <T: Scalar+Ord, D: Dim<T>+Dim<bool>+Dim<Option<Ordering>>+Dim<Ordering>> ComponentOrd for Vec<D, T> {}


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
