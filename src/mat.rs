//! An row-major array of vectors, written `Mat<R, C, T>` but pronounced 'matrix'.
//!
//! Matrices support binary operations between two Matrices or between one Matrix and one Scalar.
//! All Arithmetic operators and Bitwise operators are supported, where the two Scalar types
//! involved support the operation. Most operations operate on each component separately.
//!
//! The exceptions is matrix multiplied by matrix, which requires performs a linear algebraic
//! multiplication and requires the underlying Scalar types to support Add and Mul.
//!
//! Matrices can be multiplied by a vector, vector multiplied by matrix, both of which also
//! require the underlying Scalar types to support Add and Mul.
//!
//! Matrices also support Negation and Logical Negation where the underlying Scalar Type supports
//! it.
//!
//! `Mat1x1`, `Mat1x2`, etc. aliases are provided to simplify typing, as well as `Mat1`, `Mat2`,
//! etc. for square matrices.
//!
//! # Examples
//!
//! ```
//! use beagle::mat::{Mat3};
//! use beagle::vec::{Vec3};
//!
//! let m = Mat3::from([
//!     [ 2f32,  3f32,  5f32],
//!     [ 7f32, 11f32, 13f32],
//!     [17f32, 19f32, 23f32]]);
//! let v = Vec3::from([29f32, 31f32, 37f32]);
//! assert_eq!(m*v, Vec3::from([336f32, 1025f32, 1933f32]));
//! ```

use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::ops::{Index,IndexMut,Range,RangeFrom,RangeTo,RangeFull,Deref,DerefMut};
use std::slice::{Iter,IterMut};

use scalar_array::{Scalar,Dim};
use scalar_array::{ScalarArray,Fold,Cast,CastBinary};
use scalar_array::{ComponentPartialEq,ComponentEq,ComponentPartialOrd,ComponentOrd,ComponentMul};
use vec::{Vec, Dot};

/// An row-major array of vectors, written `Mat<R, C, T>` but pronounced 'matrix'.
///
/// Matrices support binary operations between two Matrices or between one Matrix and one Scalar.
/// All Arithmetic operators and Bitwise operators are supported, where the two Scalar types
/// involved support the operation. Most operations operate on each component separately.
///
/// The exceptions is matrix multiplied by matrix, which requires performs a linear algebraic
/// multiplication and requires the underlying Scalar types to support Add and Mul.
///
/// Matrices can be multiplied by a vector, vector multiplied by matrix, both of which also
/// require the underlying Scalar types to support Add and Mul.
///
/// Matrices also support Negation and Logical Negation where the underlying Scalar Type supports
/// it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mat<R, C, T: Scalar> (R::Output) where C: Dim<T>, R: Dim<Vec<C, T>>;

/// Types that represent an array of scalars.
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> ScalarArray for Mat<R, C, T> {
    /// The type of the underlying scalar in the array.
    type Scalar = T;
    /// The type of a single row.
    type Type = Vec<C, T>;
    /// The dimension of the scalar array.
    type Dim = R;

    /// Constructs a matrix from a an array of vectors `v`. Most useful in conjunction with
    /// `Dim::from_iter`. Code not using `Dim::from_iter` should prefer `Mat::from`.
    #[inline(always)]
    fn new(v: R::Output) -> Self { Mat(v) }
    /// Returns a slice iterator over the rows of `self`.
    #[inline(always)]
    fn iter(&self) -> Iter<Vec<C, T>> { (self.as_ref() as &[Vec<C, T>]).iter() }
    /// Returns a mutable slice iterator over the rows of `self`.
    #[inline(always)]
    fn iter_mut(&mut self) -> IterMut<Vec<C, T>> { (self.as_mut() as &mut [Vec<C, T>]).iter_mut() }
    /// Constructs a matrix from a single scalar value, setting all elements to that value.
    #[inline(always)]
    fn from_value(v: T) -> Self { Mat(<R as Dim<Vec<C, T>>>::from_value(Vec::from_value(v))) }

    /// Folds all the scalar values into a single output given two folding functions,
    /// The first folding function only applies to the first element of the ScalarArray.
    #[inline(always)]
    fn fold<U, F0: FnOnce(&Self::Scalar)->U, F: Fn(U, &Self::Scalar)->U>(&self, f0: F0, f: F) -> U {
        let init = self[0].fold(f0, &f);
        self[1..].iter().fold(init, |acc, row| row.fold(|v| f(acc, v), &f))
    }

    /// Map all the scalar values, keeping the same underlying type.
    #[inline(always)]
    fn map<F: Fn(Self::Scalar)->Self::Scalar>(mut self, f: F) -> Self {
        for v in self.iter_mut() { *v = v.map(&f); }
        self
    }
}

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Mat<R, C, T> {
    #[inline(always)]
    fn from_2d_array<V: Clone+Into<Vec<C, T>>>(v: &[V]) -> Self {
        Mat::new(<R as Dim<Vec<C, T>>>::from_iter(v.into_iter().map(|v| (*v).clone().into())))
    }
}

/// Types that can be fold with another `ScalarArray` of the same dimension into single value.
impl <T: Scalar, Rhs: Scalar, C: Dim<T>+Dim<Rhs>, R: Dim<Vec<C, T>>+Dim<Vec<C, Rhs>>> Fold<Rhs> for Mat<R, C, T> {
    /// The right hand side type.
    type RhsArray = Mat<R, C, Rhs>;

    /// Folds two `ScalarArray`s together using a binary function.
    #[inline(always)]
    fn fold_together<O, F0: FnOnce(&<Self as ScalarArray>::Scalar, &Rhs)->O, F: Fn(O, &<Self as ScalarArray>::Scalar, &Rhs)->O>(&self, rhs: &Self::RhsArray, f0: F0, f: F) -> O {
        let init = Fold::<Rhs>::fold_together(&self[0], &rhs[0], f0, &f);
        self[1..].iter().zip(rhs[1..].iter()).fold(init, |acc, (l, r)| Fold::<Rhs>::fold_together(l, r, |l2, r2| f(acc, l2, r2), &f))
    }
}

/// Types that can be transformed from into a `ScalarArray` with Scalar type `O`.
impl <T: Scalar, O: Scalar, C: Dim<T>+Dim<O>, R: Dim<Vec<C, T>>+Dim<Vec<C, O>>> Cast<O> for Mat<R, C, T> {
    /// The resulting type.
    type Output = Mat<R, C, O>;

    /// Transforms a single `ScalarArray` using a unary function.
    #[inline(always)]
    fn unary<F: Fn(&<Self as ScalarArray>::Scalar)->O>(&self, f: F) -> Self::Output {
        Mat::new(<R as Dim<Vec<C, O>>>::from_iter(self.iter().map(|s| Cast::<O>::unary(s, &f))))
    }

    /// Transforms two binary `ScalarArray`s using a binary function.
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &<Self as ScalarArray>::Scalar)->O>(&self, rhs: &Self, f: F) -> Self::Output {
        Mat::new(<R as Dim<Vec<C, O>>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| Cast::<O>::binary(l, r, &f))))
    }
}

/// Types that can be transformed from into a `ScalarArray` with Scalar type `O`.
impl <T: Scalar, Rhs: Scalar, O: Scalar, C: Dim<T>+Dim<Rhs>+Dim<O>, R: Dim<Vec<C, T>>+Dim<Vec<C, Rhs>>+Dim<Vec<C, O>>> CastBinary<Rhs, O> for Mat<R, C, T> {
    /// The right hand side type.
    type RhsArray = Mat<R, C, Rhs>;

    /// The resulting type.
    type Output = Mat<R, C, O>;

    /// Transforms two binary `ScalarArray`s using a binary function.
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &Rhs)->O>(&self, rhs: &Self::RhsArray, f: F) -> Self::Output {
        Mat::new(<R as Dim<Vec<C, O>>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| CastBinary::<Rhs, O>::binary(l, r, &f))))
    }
}


impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Mat<R, C, T>
where R: Dim<T>,
C: Dim<Vec<R,T>> {
    /// Transposes an RxC matrix into a CxR matrix.
    pub fn transpose(&self) -> Mat<C, R, T> {
        Mat(<C as Dim<Vec<R, T>>>::from_iter(self[0].iter().enumerate().map(
            |(c, _)|
            Vec::new(<R as Dim<T>>::from_iter(self.iter().map(|row| row[c])))
        )))
    }
}

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Mat<R, C, T> {
    #[inline(always)]
    fn mul_vector<U: Scalar>(&self, rhs: &Vec<C, U>) -> Vec<R, <T as Mul<U>>::Output>
    where C: Dim<U>,
    R: Dim<<T as Mul<U>>::Output>,
    T: Mul<U>,
    <T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output> {
        Vec::new(<R as Dim<<T as Mul<U>>::Output>>::from_iter(self.iter().map(|lhs_row| lhs_row.dot(rhs))))
    }

    #[inline(always)]
    fn mul_vector_transpose<U: Scalar>(&self, lhs: &Vec<R, U>) -> Vec<C, <U as Mul<T>>::Output>
    where C: Dim<<U as Mul<T>>::Output>,
    R: Dim<U>,
    U: Mul<T>,
    <U as Mul<T>>::Output: Scalar+Add<Output=<U as Mul<T>>::Output> {
        Vec::new(<C as Dim<<U as Mul<T>>::Output>>::from_iter(self[0].iter().enumerate().map(
            |(c, _)| {
                let init = lhs[0] * self[0][c];
                lhs[1..].iter().zip(self[1..].iter()).fold(init, |sum, (&lhs_value, rhs_row)| sum + lhs_value * rhs_row[c])
            }
        )))
    }

    #[inline(always)]
    fn mul_matrix<U: Scalar, C2: Dim<U>>(&self, rhs: &Mat<C, C2, U>) -> Mat<R, C2, <T as Mul<U>>::Output>
    where R: Dim<Vec<C2,<T as Mul<U>>::Output>>,
    C: Dim<Vec<C2,U>>,
    C2: Dim<<T as Mul<U>>::Output>,
    T: Mul<U>,
    <T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output> {
        Mat(<R as Dim<Vec<C2, <T as Mul<U>>::Output>>>::from_iter(self.iter().map(|lhs_row: &Vec<C, T>| rhs.mul_vector_transpose(lhs_row))))
    }

    /// Constructs a matrix via the outer product of `lhs` and `rhs`.
    #[inline]
    pub fn outer_product<Rhs: Scalar, Lhs: Scalar+Mul<Rhs, Output=T>>(lhs: &Vec<C, Lhs>, rhs: &Vec<R, Rhs>) -> Self
    where C: Dim<Lhs>,
    R: Dim<Rhs> {
        Mat::new(<R as Dim<Vec<C, T>>>::from_iter(rhs.iter().map(|&r| lhs * r)))
    }

    // TODO: Matrix functions
    // inverse
    //  matN inverse(matN m)
}

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Deref for Mat<R, C, T> {
    type Target = R::Output;
    #[inline(always)] fn deref<'a>(&'a self) -> &'a Self::Target { &self.0 }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> DerefMut for Mat<R, C, T> {
    #[inline(always)] fn deref_mut<'a>(&'a mut self) -> &'a mut <Self as Deref>::Target { &mut self.0 }
}

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Borrow   <[Vec<C, T>]> for Mat<R, C, T> {  #[inline(always)] fn borrow    (&    self) -> &    [Vec<C, T>] { self.0.borrow() }  }
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> BorrowMut<[Vec<C, T>]> for Mat<R, C, T> {  #[inline(always)] fn borrow_mut(&mut self) -> &mut [Vec<C, T>] { self.0.borrow_mut() }  }

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> AsRef<[Vec<C, T>]> for Mat<R, C, T> {  #[inline(always)] fn as_ref(&    self) -> &    [Vec<C, T>] { self.0.as_ref() }  }
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> AsMut<[Vec<C, T>]> for Mat<R, C, T> {  #[inline(always)] fn as_mut(&mut self) -> &mut [Vec<C, T>] { self.0.as_mut() }  }

// d888888b d8b   db d8888b. d88888b db    db
//   `88'   888o  88 88  `8D 88'     `8b  d8'
//    88    88V8o 88 88   88 88ooooo  `8bd8'
//    88    88 V8o88 88   88 88~~~~~  .dPYb.
//   .88.   88  V888 88  .8D 88.     .8P  Y8.
// Y888888P VP   V8P Y8888D' Y88888P YP    YP

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Index<usize> for Mat<R, C, T> {
    type Output = Vec<C, T>;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output { &(self.as_ref() as &[Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IndexMut<usize> for Mat<R, C, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output { &mut (self.as_mut() as &mut [Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Index<Range<usize>> for Mat<R, C, T> {
    type Output = [Vec<C, T>];
    #[inline(always)]
    fn index(&self, i: Range<usize>) -> &Self::Output { &(self.as_ref() as &[Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IndexMut<Range<usize>> for Mat<R, C, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: Range<usize>) -> &mut Self::Output { &mut (self.as_mut() as &mut [Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Index<RangeTo<usize>> for Mat<R, C, T> {
    type Output = [Vec<C, T>];
    #[inline(always)]
    fn index(&self, i: RangeTo<usize>) -> &Self::Output { &(self.as_ref() as &[Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IndexMut<RangeTo<usize>> for Mat<R, C, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: RangeTo<usize>) -> &mut Self::Output { &mut (self.as_mut() as &mut [Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Index<RangeFrom<usize>> for Mat<R, C, T> {
    type Output = [Vec<C, T>];
    #[inline(always)]
    fn index(&self, i: RangeFrom<usize>) -> &Self::Output { &(self.as_ref() as &[Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IndexMut<RangeFrom<usize>> for Mat<R, C, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: RangeFrom<usize>) -> &mut Self::Output { &mut (self.as_mut() as &mut [Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Index<RangeFull> for Mat<R, C, T> {
    type Output = [Vec<C, T>];
    #[inline(always)]
    fn index(&self, i: RangeFull) -> &Self::Output { &(self.as_ref() as &[Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IndexMut<RangeFull> for Mat<R, C, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: RangeFull) -> &mut Self::Output { &mut (self.as_mut() as &mut [Vec<C, T>])[i] }
}



impl<'a, T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IntoIterator for &'a Mat<R, C, T> {
    type Item = &'a Vec<C, T>;
    type IntoIter = Iter<'a, Vec<C, T>>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
impl<'a, T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IntoIterator for &'a mut Mat<R, C, T> {
    type Item = &'a mut Vec<C, T>;
    type IntoIter = IterMut<'a, Vec<C, T>>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

// .88b  d88.  .d8b.  d888888b                   .88b  d88.  .d8b.  d888888b
// 88'YbdP`88 d8' `8b `~~88~~'      8. A .8      88'YbdP`88 d8' `8b `~~88~~'
// 88  88  88 88ooo88    88         `8.8.8'      88  88  88 88ooo88    88
// 88  88  88 88~~~88    88           888        88  88  88 88~~~88    88
// 88  88  88 88   88    88         .d'8`b.      88  88  88 88   88    88
// YP  YP  YP YP   YP    YP         8' V `8      YP  YP  YP YP   YP    YP

impl <T: Scalar, U: Scalar, R, C, C2> Mul<Mat<C, C2, U>> for Mat<R, C, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C2: Dim<U>+Dim<<T as Mul<U>>::Output>,
C: Dim<T>+Dim<Vec<C2,U>>,
R: Dim<Vec<C, T>>+Dim<Vec<C2,<T as Mul<U>>::Output>> {
    type Output = Mat<R, C2, <T as Mul<U>>::Output>;
    fn mul(self, rhs: Mat<C, C2, U>) -> Self::Output { self.mul_matrix(&rhs) }
}

impl <'t, T: Scalar, U: Scalar, R, C, C2> Mul<Mat<C, C2, U>> for &'t Mat<R, C, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C2: Dim<U>+Dim<<T as Mul<U>>::Output>,
C: Dim<T>+Dim<Vec<C2,U>>,
R: Dim<Vec<C, T>>+Dim<Vec<C2,<T as Mul<U>>::Output>> {
    type Output = Mat<R, C2, <T as Mul<U>>::Output>;
    fn mul(self, rhs: Mat<C, C2, U>) -> Self::Output { self.mul_matrix(&rhs) }
}

impl <'r, T: Scalar, U: Scalar, R, C, C2> Mul<&'r Mat<C, C2, U>> for Mat<R, C, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C2: Dim<U>+Dim<<T as Mul<U>>::Output>,
C: Dim<T>+Dim<Vec<C2,U>>,
R: Dim<Vec<C, T>>+Dim<Vec<C2,<T as Mul<U>>::Output>> {
    type Output = Mat<R, C2, <T as Mul<U>>::Output>;
    fn mul(self, rhs: &'r Mat<C, C2, U>) -> Self::Output { self.mul_matrix(rhs) }
}

impl <'t, 'r, T: Scalar, U: Scalar, R, C, C2> Mul<&'r Mat<C, C2, U>> for &'t Mat<R, C, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C2: Dim<U>+Dim<<T as Mul<U>>::Output>,
C: Dim<T>+Dim<Vec<C2,U>>,
R: Dim<Vec<C, T>>+Dim<Vec<C2,<T as Mul<U>>::Output>> {
    type Output = Mat<R, C2, <T as Mul<U>>::Output>;
    fn mul(self, rhs: &'r Mat<C, C2, U>) -> Self::Output { self.mul_matrix(rhs) }
}

// .88b  d88.  .d8b.  d888888b                   db    db d88888b  .o88b.
// 88'YbdP`88 d8' `8b `~~88~~'      8. A .8      88    88 88'     d8P  Y8
// 88  88  88 88ooo88    88         `8.8.8'      Y8    8P 88ooooo 8P
// 88  88  88 88~~~88    88           888        `8b  d8' 88~~~~~ 8b
// 88  88  88 88   88    88         .d'8`b.       `8bd8'  88.     Y8b  d8
// YP  YP  YP YP   YP    YP         8' V `8         YP    Y88888P  `Y88P'

impl <T: Scalar, U: Scalar, R, C> Mul<Vec<C, U>> for Mat<R, C, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C: Dim<T>+Dim<U>,
R: Dim<Vec<C, T>>+Dim<<T as Mul<U>>::Output> {
    type Output = Vec<R, <T as Mul<U>>::Output>;
    fn mul(self, rhs: Vec<C, U>) -> Self::Output { self.mul_vector(&rhs) }
}

impl <'t, T: Scalar, U: Scalar, R, C> Mul<Vec<C, U>> for &'t Mat<R, C, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C: Dim<T>+Dim<U>,
R: Dim<Vec<C, T>>+Dim<<T as Mul<U>>::Output> {
    type Output = Vec<R, <T as Mul<U>>::Output>;
    fn mul(self, rhs: Vec<C, U>) -> Self::Output { self.mul_vector(&rhs) }
}

impl <'r, T: Scalar, U: Scalar, R, C> Mul<&'r Vec<C, U>> for Mat<R, C, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C: Dim<T>+Dim<U>,
R: Dim<Vec<C, T>>+Dim<<T as Mul<U>>::Output> {
    type Output = Vec<R, <T as Mul<U>>::Output>;
    fn mul(self, rhs: &'r Vec<C, U>) -> Self::Output { self.mul_vector(rhs) }
}

impl <'t, 'r, T: Scalar, U: Scalar, R, C> Mul<&'r Vec<C, U>> for &'t Mat<R, C, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C: Dim<T>+Dim<U>,
R: Dim<Vec<C, T>>+Dim<<T as Mul<U>>::Output> {
    type Output = Vec<R, <T as Mul<U>>::Output>;
    fn mul(self, rhs: &'r Vec<C, U>) -> Self::Output { self.mul_vector(rhs) }
}

// db    db d88888b  .o88b.                   .88b  d88.  .d8b.  d888888b
// 88    88 88'     d8P  Y8      8. A .8      88'YbdP`88 d8' `8b `~~88~~'
// Y8    8P 88ooooo 8P           `8.8.8'      88  88  88 88ooo88    88
// `8b  d8' 88~~~~~ 8b             888        88  88  88 88~~~88    88
//  `8bd8'  88.     Y8b  d8      .d'8`b.      88  88  88 88   88    88
//    YP    Y88888P  `Y88P'      8' V `8      YP  YP  YP YP   YP    YP

impl <T: Scalar, U: Scalar, R, C> Mul<Mat<R, C, U>> for Vec<R, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C: Dim<U>+Dim<<T as Mul<U>>::Output>,
R: Dim<Vec<C, U>>+Dim<T> {
    type Output = Vec<C, <T as Mul<U>>::Output>;
    fn mul(self, rhs: Mat<R, C, U>) -> Self::Output { rhs.mul_vector_transpose(&self) }
}

impl <'t, T: Scalar, U: Scalar, R, C> Mul<Mat<R, C, U>> for &'t Vec<R, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C: Dim<U>+Dim<<T as Mul<U>>::Output>,
R: Dim<Vec<C, U>>+Dim<T> {
    type Output = Vec<C, <T as Mul<U>>::Output>;
    fn mul(self, rhs: Mat<R, C, U>) -> Self::Output { rhs.mul_vector_transpose(self) }
}

impl <'r, T: Scalar, U: Scalar, R, C> Mul<&'r Mat<R, C, U>> for Vec<R, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C: Dim<U>+Dim<<T as Mul<U>>::Output>,
R: Dim<Vec<C, U>>+Dim<T> {
    type Output = Vec<C, <T as Mul<U>>::Output>;
    fn mul(self, rhs: &'r Mat<R, C, U>) -> Self::Output { rhs.mul_vector_transpose(&self) }
}

impl <'t, 'r, T: Scalar, U: Scalar, R, C> Mul<&'r Mat<R, C, U>> for &'t Vec<R, T>
where T: Mul<U>,
<T as Mul<U>>::Output: Scalar+Add<Output=<T as Mul<U>>::Output>,
C: Dim<U>+Dim<<T as Mul<U>>::Output>,
R: Dim<Vec<C, U>>+Dim<T> {
    type Output = Vec<C, <T as Mul<U>>::Output>;
    fn mul(self, rhs: &'r Mat<R, C, U>) -> Self::Output { rhs.mul_vector_transpose(self) }
}

//  .o88b.  .d88b.  .88b  d88. d8888b.  .d88b.  d8b   db d88888b d8b   db d888888b
// d8P  Y8 .8P  Y8. 88'YbdP`88 88  `8D .8P  Y8. 888o  88 88'     888o  88 `~~88~~'
// 8P      88    88 88  88  88 88oodD' 88    88 88V8o 88 88ooooo 88V8o 88    88
// 8b      88    88 88  88  88 88~~~   88    88 88 V8o88 88~~~~~ 88 V8o88    88
// Y8b  d8 `8b  d8' 88  88  88 88      `8b  d8' 88  V888 88.     88  V888    88
//  `Y88P'  `Y88P'  YP  YP  YP 88       `Y88P'  VP   V8P Y88888P VP   V8P    YP

impl <T: Scalar+PartialEq, C: Dim<T>+Dim<bool>, R: Dim<Vec<C, T>>+Dim<Vec<C, bool>>> ComponentPartialEq for Mat<R, C, T> {}
impl <T: Scalar+Eq, C: Dim<T>+Dim<bool>, R: Dim<Vec<C, T>>+Dim<Vec<C, bool>>> ComponentEq for Mat<R, C, T> {}
impl <T: Scalar+PartialOrd, C: Dim<T>+Dim<bool>+Dim<Option<Ordering>>, R: Dim<Vec<C, T>>+Dim<Vec<C, bool>>+Dim<Vec<C, Option<Ordering>>>> ComponentPartialOrd for Mat<R, C, T> {}
impl <T: Scalar+Ord, C: Dim<T>+Dim<bool>+Dim<Option<Ordering>>+Dim<Ordering>, R: Dim<Vec<C, T>>+Dim<Vec<C, bool>>+Dim<Vec<C, Option<Ordering>>>+Dim<Vec<C, Ordering>>> ComponentOrd for Mat<R, C, T> {}

/// Types that can be component-wise multiplied.
impl <T: Scalar, Rhs: Scalar, C: Dim<T>+Dim<Rhs>,R: Dim<Vec<C, T>>+Dim<Vec<C, Rhs>>> ComponentMul<Rhs> for Mat<R, C, T>
where T: Mul<Rhs>,
<T as Mul<Rhs>>::Output: Scalar,
C: Dim<<T as Mul<Rhs>>::Output>,
R: Dim<Vec<C, <T as Mul<Rhs>>::Output>> {
    /// The right hand side type.
    type RhsArray = Mat<R, C, Rhs>;

    /// The resulting type.
    type Output = Mat<R, C, <T as Mul<Rhs>>::Output>;

    /// Multiplies the components of `self` and `rhs`.
    fn cmp_mul(&self, rhs: &Self::RhsArray) -> Self::Output {
        CastBinary::<Rhs, <T as Mul<Rhs>>::Output>::binary(self, rhs, |&l, &r| l*r)
    }
}


include!(concat!(env!("OUT_DIR"), "/mat.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use scalar_array::ScalarArray;

    #[test]
    fn test_neg() {
        assert_eq!(-Mat1x1::from([[1f64]]),                   Mat1x1::from([[-1f64]]));
        assert_eq!(-Mat1x2::from([[1f64, 2f64]]),             Mat1x2::from([[-1f64, -2f64]]));
        assert_eq!(-Mat1x3::from([[1f64, 2f64, 3f64]]),       Mat1x3::from([[-1f64, -2f64, -3f64]]));
        assert_eq!(-Mat1x4::from([[1f64, 2f64, 3f64, 4f64]]), Mat1x4::from([[-1f64, -2f64, -3f64, -4f64]]));
    }

    #[test]
    fn test_multiply_zero() {
        assert_eq!(Mat1x1::from_value(0f64)*Mat1x1::from_value(0f64), Mat1x1::from_value(0f64));
        assert_eq!(Mat1x2::from_value(0f64)*Mat2x1::from_value(0f64), Mat1x1::from_value(0f64));
        assert_eq!(Mat1x3::from_value(0f64)*Mat3x1::from_value(0f64), Mat1x1::from_value(0f64));
        assert_eq!(Mat1x4::from_value(0f64)*Mat4x1::from_value(0f64), Mat1x1::from_value(0f64));

        assert_eq!(Mat1x1::from_value(0f64)*Mat1x2::from_value(0f64), Mat1x2::from_value(0f64));
        assert_eq!(Mat1x2::from_value(0f64)*Mat2x2::from_value(0f64), Mat1x2::from_value(0f64));
        assert_eq!(Mat1x3::from_value(0f64)*Mat3x2::from_value(0f64), Mat1x2::from_value(0f64));
        assert_eq!(Mat1x4::from_value(0f64)*Mat4x2::from_value(0f64), Mat1x2::from_value(0f64));

        assert_eq!(Mat1x1::from_value(0f64)*Mat1x3::from_value(0f64), Mat1x3::from_value(0f64));
        assert_eq!(Mat1x2::from_value(0f64)*Mat2x3::from_value(0f64), Mat1x3::from_value(0f64));
        assert_eq!(Mat1x3::from_value(0f64)*Mat3x3::from_value(0f64), Mat1x3::from_value(0f64));
        assert_eq!(Mat1x4::from_value(0f64)*Mat4x3::from_value(0f64), Mat1x3::from_value(0f64));

        assert_eq!(Mat1x1::from_value(0f64)*Mat1x4::from_value(0f64), Mat1x4::from_value(0f64));
        assert_eq!(Mat1x2::from_value(0f64)*Mat2x4::from_value(0f64), Mat1x4::from_value(0f64));
        assert_eq!(Mat1x3::from_value(0f64)*Mat3x4::from_value(0f64), Mat1x4::from_value(0f64));
        assert_eq!(Mat1x4::from_value(0f64)*Mat4x4::from_value(0f64), Mat1x4::from_value(0f64));




        assert_eq!(Mat2x1::from_value(0f64)*Mat1x1::from_value(0f64), Mat2x1::from_value(0f64));
        assert_eq!(Mat2x2::from_value(0f64)*Mat2x1::from_value(0f64), Mat2x1::from_value(0f64));
        assert_eq!(Mat2x3::from_value(0f64)*Mat3x1::from_value(0f64), Mat2x1::from_value(0f64));
        assert_eq!(Mat2x4::from_value(0f64)*Mat4x1::from_value(0f64), Mat2x1::from_value(0f64));

        assert_eq!(Mat2x1::from_value(0f64)*Mat1x2::from_value(0f64), Mat2x2::from_value(0f64));
        assert_eq!(Mat2x2::from_value(0f64)*Mat2x2::from_value(0f64), Mat2x2::from_value(0f64));
        assert_eq!(Mat2x3::from_value(0f64)*Mat3x2::from_value(0f64), Mat2x2::from_value(0f64));
        assert_eq!(Mat2x4::from_value(0f64)*Mat4x2::from_value(0f64), Mat2x2::from_value(0f64));

        assert_eq!(Mat2x1::from_value(0f64)*Mat1x3::from_value(0f64), Mat2x3::from_value(0f64));
        assert_eq!(Mat2x2::from_value(0f64)*Mat2x3::from_value(0f64), Mat2x3::from_value(0f64));
        assert_eq!(Mat2x3::from_value(0f64)*Mat3x3::from_value(0f64), Mat2x3::from_value(0f64));
        assert_eq!(Mat2x4::from_value(0f64)*Mat4x3::from_value(0f64), Mat2x3::from_value(0f64));

        assert_eq!(Mat2x1::from_value(0f64)*Mat1x4::from_value(0f64), Mat2x4::from_value(0f64));
        assert_eq!(Mat2x2::from_value(0f64)*Mat2x4::from_value(0f64), Mat2x4::from_value(0f64));
        assert_eq!(Mat2x3::from_value(0f64)*Mat3x4::from_value(0f64), Mat2x4::from_value(0f64));
        assert_eq!(Mat2x4::from_value(0f64)*Mat4x4::from_value(0f64), Mat2x4::from_value(0f64));




        assert_eq!(Mat3x1::from_value(0f64)*Mat1x1::from_value(0f64), Mat3x1::from_value(0f64));
        assert_eq!(Mat3x2::from_value(0f64)*Mat2x1::from_value(0f64), Mat3x1::from_value(0f64));
        assert_eq!(Mat3x3::from_value(0f64)*Mat3x1::from_value(0f64), Mat3x1::from_value(0f64));
        assert_eq!(Mat3x4::from_value(0f64)*Mat4x1::from_value(0f64), Mat3x1::from_value(0f64));

        assert_eq!(Mat3x1::from_value(0f64)*Mat1x2::from_value(0f64), Mat3x2::from_value(0f64));
        assert_eq!(Mat3x2::from_value(0f64)*Mat2x2::from_value(0f64), Mat3x2::from_value(0f64));
        assert_eq!(Mat3x3::from_value(0f64)*Mat3x2::from_value(0f64), Mat3x2::from_value(0f64));
        assert_eq!(Mat3x4::from_value(0f64)*Mat4x2::from_value(0f64), Mat3x2::from_value(0f64));

        assert_eq!(Mat3x1::from_value(0f64)*Mat1x3::from_value(0f64), Mat3x3::from_value(0f64));
        assert_eq!(Mat3x2::from_value(0f64)*Mat2x3::from_value(0f64), Mat3x3::from_value(0f64));
        assert_eq!(Mat3x3::from_value(0f64)*Mat3x3::from_value(0f64), Mat3x3::from_value(0f64));
        assert_eq!(Mat3x4::from_value(0f64)*Mat4x3::from_value(0f64), Mat3x3::from_value(0f64));

        assert_eq!(Mat3x1::from_value(0f64)*Mat1x4::from_value(0f64), Mat3x4::from_value(0f64));
        assert_eq!(Mat3x2::from_value(0f64)*Mat2x4::from_value(0f64), Mat3x4::from_value(0f64));
        assert_eq!(Mat3x3::from_value(0f64)*Mat3x4::from_value(0f64), Mat3x4::from_value(0f64));
        assert_eq!(Mat3x4::from_value(0f64)*Mat4x4::from_value(0f64), Mat3x4::from_value(0f64));




        assert_eq!(Mat4x1::from_value(0f64)*Mat1x1::from_value(0f64), Mat4x1::from_value(0f64));
        assert_eq!(Mat4x2::from_value(0f64)*Mat2x1::from_value(0f64), Mat4x1::from_value(0f64));
        assert_eq!(Mat4x3::from_value(0f64)*Mat3x1::from_value(0f64), Mat4x1::from_value(0f64));
        assert_eq!(Mat4x4::from_value(0f64)*Mat4x1::from_value(0f64), Mat4x1::from_value(0f64));

        assert_eq!(Mat4x1::from_value(0f64)*Mat1x2::from_value(0f64), Mat4x2::from_value(0f64));
        assert_eq!(Mat4x2::from_value(0f64)*Mat2x2::from_value(0f64), Mat4x2::from_value(0f64));
        assert_eq!(Mat4x3::from_value(0f64)*Mat3x2::from_value(0f64), Mat4x2::from_value(0f64));
        assert_eq!(Mat4x4::from_value(0f64)*Mat4x2::from_value(0f64), Mat4x2::from_value(0f64));

        assert_eq!(Mat4x1::from_value(0f64)*Mat1x3::from_value(0f64), Mat4x3::from_value(0f64));
        assert_eq!(Mat4x2::from_value(0f64)*Mat2x3::from_value(0f64), Mat4x3::from_value(0f64));
        assert_eq!(Mat4x3::from_value(0f64)*Mat3x3::from_value(0f64), Mat4x3::from_value(0f64));
        assert_eq!(Mat4x4::from_value(0f64)*Mat4x3::from_value(0f64), Mat4x3::from_value(0f64));

        assert_eq!(Mat4x1::from_value(0f64)*Mat1x4::from_value(0f64), Mat4x4::from_value(0f64));
        assert_eq!(Mat4x2::from_value(0f64)*Mat2x4::from_value(0f64), Mat4x4::from_value(0f64));
        assert_eq!(Mat4x3::from_value(0f64)*Mat3x4::from_value(0f64), Mat4x4::from_value(0f64));
        assert_eq!(Mat4x4::from_value(0f64)*Mat4x4::from_value(0f64), Mat4x4::from_value(0f64));
    }

    #[test]
    fn test_multiply() {
        let a = Mat3x3::from([[   2f64,    3f64,    5f64], [   7f64,   11f64,   13f64], [  17f64,   19f64,   23f64]]);
        let b = Mat3x3::from([[  29f64,   31f64,   37f64], [  41f64,   43f64,   47f64], [  53f64,   59f64,   61f64]]);
        let c = Mat3x3::from([[ 446f64,  486f64,  520f64], [1343f64, 1457f64, 1569f64], [2491f64, 2701f64, 2925f64]]);
        assert_eq!(a*b, c);
    }
}
