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
use std::ops::{Index,IndexMut,Range,RangeFrom,RangeTo,RangeFull};
use std::slice::{Iter,IterMut};

use scalar_array::{Scalar,Dim, ScalarArray,Cast, ComponentPartialEq,ComponentEq,ComponentPartialOrd,ComponentOrd};

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

    /// Construct a matrix/vector from a an array
    #[inline(always)]
    fn new(v: D::Output) -> Self { Vec(v) }
    /// Get a slice iterator over the elements (the rows for matrices/the scalars for vectors)
    #[inline(always)]
    fn iter(&self) -> Iter<T> { (self.as_ref() as &[T]).iter() }
    /// Get a mutable slice iterator over the elements (the rows for matrices/the scalars for
    /// vectors)
    #[inline(always)]
    fn iter_mut(&mut self) -> IterMut<T> { (self.as_mut() as &mut [T]).iter_mut() }
    /// Construct a matrix/vector from a single scalar value, setting all elements to that value
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

/// Types that can be transformed from into a `ScalarArray` of `<T>`
impl <T: Scalar, U: Scalar, D: Dim<T>+Dim<U>> Cast<U> for Vec<D, T> {
    /// The resulting type
    type Output = Vec<D, U>;

    // Fold two `ScalarArray`s together using a binary function
    #[inline(always)]
    fn fold_together<O, F0: FnOnce(&<Self as ScalarArray>::Scalar, &U)->O, F: Fn(O, &<Self as ScalarArray>::Scalar, &U)->O>(&self, rhs: &Self::Output, f0: F0, f: F) -> O {
        let init = f0(&self[0], &rhs[0]);
        self[1..].iter().zip(rhs[1..].iter()).fold(init, |acc, (l, r)| f(acc, l, r))
    }

    /// Transform a single `ScalarArray` using a unary function
    #[inline(always)]
    fn unary<F: Fn(&<Self as ScalarArray>::Scalar)->U>(&self, f: F) -> Self::Output {
        Vec::new(<D as Dim<U>>::from_iter(self.iter().map(|s| f(s))))
    }

    /// Transform two binary `ScalarArray`s using a binary function
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &<Self as ScalarArray>::Scalar)->U>(&self, rhs: &Self, f: F) -> Self::Output {
        Vec::new(<D as Dim<U>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| f(l, r))))
    }
}

impl <T: Scalar, D: Dim<T>> Vec<D, T> {
    /// Sum up all the elements
    #[inline(always)]
    pub fn sum<U: Add<T, Output=U>>(&self) -> U
    where T: Into<U> {
        self.fold(|&i| i.into(), |sum, &i| sum + i)
    }

    /// Dot product
    #[inline(always)]
    pub fn dot<U: Scalar, V>(&self, rhs: &Vec<D, U>) -> V
    where D: Dim<U>,
    T: Mul<U>,
    <T as Mul<U>>::Output: Into<V>,
    V: Add<<T as Mul<U>>::Output, Output=V> {
        Cast::<U>::fold_together(self, rhs, |&l, &r| (l*r).into(), |acc, &l, &r| acc + l*r)
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
