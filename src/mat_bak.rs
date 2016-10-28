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
use std::iter::FlatMap;
use std::slice::{Iter,IterMut};
use std::mem::transmute;

use scalar_array::{Scalar,Construct,Dim};
use scalar_array::{ScalarArray,ScalarIterator,ScalarIteratorMut,Fold,Cast,CastBinary};
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


impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Mat<R, C, T> {
    /// Constructs a matrix via the outer product of `lhs` and `rhs`.
    #[inline]
    pub fn outer_product<Rhs: Scalar, Lhs: Scalar+Mul<Rhs, Output=T>>(lhs: &Vec<C, Lhs>, rhs: &Vec<R, Rhs>) -> Self
    where C: Dim<Lhs>,
    R: Dim<Rhs> {
        Mat::new(<R as Dim<Vec<C, T>>>::from_iter(rhs.into_iter().map(|&r| lhs * r)))
    }

    // TODO: Matrix functions
    // inverse
    //  matN inverse(matN m)
}

// d888888b d8b   db d8888b. d88888b db    db
//   `88'   888o  88 88  `8D 88'     `8b  d8'
//    88    88V8o 88 88   88 88ooooo  `8bd8'
//    88    88 V8o88 88   88 88~~~~~  .dPYb.
//   .88.   88  V888 88  .8D 88.     .8P  Y8.
// Y888888P VP   V8P Y8888D' Y88888P YP    YP
