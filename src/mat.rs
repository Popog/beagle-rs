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
//! let m = Mat3::new([
//!     [ 2f32,  3f32,  5f32],
//!     [ 7f32, 11f32, 13f32],
//!     [17f32, 19f32, 23f32]]);
//! let v = Vec3::new([29f32, 31f32, 37f32]);
//! assert_eq!(m*v, Vec3::new([336f32, 1025f32, 1933f32]));
//! ```

use std::ops::{
    Neg,Not,
    BitAnd,BitOr,BitXor,
    Shl,Shr,
    Add,Div,Mul,Rem,Sub,
    BitAndAssign,BitOrAssign,BitXorAssign,
    ShlAssign,ShrAssign,
    AddAssign,DivAssign,MulAssign,RemAssign,SubAssign,
    Deref,DerefMut,
};

use super::Value;
use consts::{
    One,Two,Three,Four,
    Array,
    Dim,DimRef,DimMut,
    TwoDim,TwoDimRef,TwoDimMut,
};
use scalar_array::{
    ScalarArray,ScalarArrayVal,ScalarArrayRef,ScalarArrayMut,
    ConcreteScalarArray,HasConcreteScalarArray,HasConcreteVecArray,ConcreteVecArray,
    VecArrayVal,
    apply2_mut_val,map,map2,mul_vector,mul_matrix,
};
use scalar_array::array;
use utils::ArrayRefCast;
use vec::Vec;

/// An row-major array of vectors, written `Mat<R, C, T>` but pronounced 'matrix'.
///
/// `R` represents the number of rows. `C` represents the number of columns.
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
pub struct Mat<R, C, S> (R::Type)
where C: Dim<S>,
R: TwoDim<S, C>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type>;

/// An alias for Mat&lt;One, One, T&gt;
pub type Mat1<T> = Mat<One, One, T>;
/// An alias for Mat&lt;One, One, T&gt;
pub type Mat1x1<T> = Mat<One, One, T>;
/// An alias for Mat&lt;One, Two, T&gt;
pub type Mat1x2<T> = Mat<One, Two, T>;
/// An alias for Mat&lt;One, Three, T&gt;
pub type Mat1x3<T> = Mat<One, Three, T>;
/// An alias for Mat&lt;One, Four, T&gt;
pub type Mat1x4<T> = Mat<One, Four, T>;

/// An alias for Mat&lt;Two, Two, T&gt;
pub type Mat2<T> = Mat<Two, Two, T>;
/// An alias for Mat&lt;Two, One, T&gt;
pub type Mat2x1<T> = Mat<Two, One, T>;
/// An alias for Mat&lt;Two, Two, T&gt;
pub type Mat2x2<T> = Mat<Two, Two, T>;
/// An alias for Mat&lt;Two, Three, T&gt;
pub type Mat2x3<T> = Mat<Two, Three, T>;
/// An alias for Mat&lt;Two, Four, T&gt;
pub type Mat2x4<T> = Mat<Two, Four, T>;

/// An alias for Mat&lt;Three, Three, T&gt;
pub type Mat3<T> = Mat<Three, Three, T>;
/// An alias for Mat&lt;Three, One, T&gt;
pub type Mat3x1<T> = Mat<Three, One, T>;
/// An alias for Mat&lt;Three, Two, T&gt;
pub type Mat3x2<T> = Mat<Three, Two, T>;
/// An alias for Mat&lt;Three, Three, T&gt;
pub type Mat3x3<T> = Mat<Three, Three, T>;
/// An alias for Mat&lt;Three, Four, T&gt;
pub type Mat3x4<T> = Mat<Three, Four, T>;

/// An alias for Mat&lt;Four, Four, T&gt;
pub type Mat4<T> = Mat<Four, Four, T>;
/// An alias for Mat&lt;Four, One, T&gt;
pub type Mat4x1<T> = Mat<Four, One, T>;
/// An alias for Mat&lt;Four, Two, T&gt;
pub type Mat4x2<T> = Mat<Four, Two, T>;
/// An alias for Mat&lt;Four, Three, T&gt;
pub type Mat4x3<T> = Mat<Four, Three, T>;
/// An alias for Mat&lt;Four, Four, T&gt;
pub type Mat4x4<T> = Mat<Four, Four, T>;

impl<S, C: Dim<S>, R: TwoDim<S, C>> Mat<R, C, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type>,
{
    /// Construct a matrix from an array of arrays
    pub fn new(m: <R as Array<C::RawType>>::RawType) -> Self
    where R: Array<C::RawType>,
    <R as Array<C::RawType>>::RawType: Into<<R as Array<C::RawType>>::Type>,
    C::RawType: Into<C::Type> {
        Mat(<R as Array<C::RawType>>::map(m.into(), |m| m.into()))
    }

    /// Construct a matrix from a single value
    pub fn from_value(s: S) -> Self
    where S: Clone, C::Type: Clone {
        Mat(R::from_value(C::from_value(s)))
    }
}

impl<S, C: Dim<S>, R: TwoDim<S, C>> Deref for Mat<R, C, S>
where R: Dim<Vec<C, S>>,
<R as Array<Vec<C, S>>>::Type: Deref<Target=<R as Array<Vec<C, S>>>::RawType>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type> + Array<Vec<C, S>>,
{
    type Target = <R as Array<Vec<C, S>>>::RawType;

    #[inline(always)]
    fn deref(&self) -> &<R as Array<Vec<C, S>>>::RawType {
        <Vec<C, S> as ArrayRefCast<<C as Array<S>>::Type>>::from_array_ref::<R>(&self.0).deref()
    }
}

impl<S, C: Dim<S>, R: TwoDim<S, C>> DerefMut for Mat<R, C, S>
where R: Dim<Vec<C, S>>,
<R as Array<Vec<C, S>>>::Type: DerefMut<Target=<R as Array<Vec<C, S>>>::RawType>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type> + Array<Vec<C, S>>,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut <R as Array<Vec<C, S>>>::RawType {
        <Vec<C, S> as ArrayRefCast<<C as Array<S>>::Type>>::from_array_mut::<R>(&mut self.0).deref_mut()
    }
}

impl<S, C: Dim<S>, R: TwoDim<S, C>> ScalarArray for Mat<R, C, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type>,
{
    type Scalar = S;
    type Row = C;
    type Dim = R;
}

impl<S, C: Dim<S>, R: TwoDim<S, C>> ScalarArrayVal for Mat<R, C, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type>,
{
    #[inline(always)]
    fn get_val(self) -> R::Type { self.0 }
}

impl<S, C: DimRef<S>, R: TwoDimRef<S, C>> ScalarArrayRef for Mat<R, C, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type>,
{
    #[inline(always)]
    fn get_ref(&self) -> <R as Array<<C as Array<&S>>::Type>>::Type {
        R::map(R::get_ref(&self.0), C::get_ref)
    }
}

impl<S, C: DimMut<S>, R: TwoDimMut<S, C>> ScalarArrayMut for Mat<R, C, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type>,
{
    #[inline(always)]
    fn get_mut(&mut self) -> <R as Array<<C as Array<&mut S>>::Type>>::Type {
        R::map(R::get_mut(&mut self.0), C::get_mut)
    }
}

impl<S, T, C: Dim<S>, C2: Dim<T>, R: TwoDim<S, C>, R2: TwoDim<T, C2>> HasConcreteScalarArray<T, C2, R2> for Mat<R, C, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where C::Smaller: Array<S>,
C2::Smaller: Array<T>,
R::Smaller: Array<<C as Array<S>>::Type>,
R2::Smaller: Array<<C2 as Array<T>>::Type>,
{
    type Concrete = Mat<R2, C2, T>;
}

impl<S, C: Dim<S>, R: TwoDim<S, C>> ConcreteScalarArray for Mat<R, C, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type>,
{
    #[inline(always)]
    fn from_val(v: R::Type) -> Self { Mat(v) }
}


impl<S, C: Dim<S>, R: TwoDim<S, C>> Mat<R, C, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where C::Smaller: Array<S>,
R::Smaller: Array<<C as Array<S>>::Type>,
{
    /// Constructs a matrix via the outer product of `lhs` and `rhs`.
    #[inline]
    pub fn outer_product<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Self
    where C: Dim<Lhs::Scalar> + Array<Rhs::Scalar>,
    R: Dim<Rhs::Scalar> + Array<<C as Array<Lhs::Scalar>>::Type> + Array<<C as Array<Rhs::Scalar>>::Type>,
    <C as Array<Lhs::Scalar>>::Type: Clone,
    Lhs: VecArrayVal<Row=C>,
    Rhs: VecArrayVal<Row=R>,
    Lhs::Scalar: Mul<Rhs::Scalar, Output=S>,
    Rhs::Scalar: Clone,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    C::Smaller: Array<Lhs::Scalar>,
    R::Smaller: Array<Rhs::Scalar>,
    {
        let lhs = <R as Array<<C as Array<Lhs::Scalar>>::Type>>::from_value(lhs.get_vec_val());
        let rhs = <R as Array<Rhs::Scalar>>::map(rhs.get_vec_val(), C::from_value);

        Mat::from_val(array::map2::<Lhs::Scalar, C, R, _, _, _>(lhs, rhs, |lhs, rhs| lhs * rhs))
    }

    // TODO: Matrix functions
    // inverse
    //  matN inverse(matN m)
}

macro_rules! impl_mat_unop {
    ($($trait_name:ident::$method_name:ident)+) => {$(
impl<S, C: Dim<S>, R: TwoDim<S, C>> $trait_name for Mat<R, C, S>
where C: Dim<S::Output>,
R: TwoDim<S::Output, C>,
S: $trait_name,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
C::Smaller: Array<S> + Array<S::Output>,
R::Smaller: Array<<C as Array<S>>::Type> + Array<<C as Array<S::Output>>::Type>,
{
    type Output = Mat<R, C, S::Output>;
    fn $method_name(self) -> Self::Output { map(self, $trait_name::$method_name) }
}
    )+};
}

macro_rules! impl_mat_binop {
    ($($trait_name:ident::$method_name:ident)+) => {$(
        impl_mat_binop!{$trait_name::$method_name for Mat}
        impl_mat_binop!{$trait_name::$method_name for Value}
    )+};

    ($trait_name:ident::$method_name:ident for Mat) => {
impl<S, C: Dim<S>, R: TwoDim<S, C>, Rhs> $trait_name<Rhs> for Mat<R, C, S>
where Rhs: ScalarArrayVal<Row=C, Dim=R>,
C: Dim<Rhs::Scalar> + Dim<S::Output>,
R: TwoDim<Rhs::Scalar, C> + TwoDim<S::Output, C>,
S: $trait_name<Rhs::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
C::Smaller: Array<S> + Array<Rhs::Scalar> + Array<S::Output>,
R::Smaller: Array<<C as Array<S>>::Type> + Array<<C as Array<Rhs::Scalar>>::Type> + Array<<C as Array<S::Output>>::Type>,
{
    type Output = Mat<R, C, S::Output>;
    fn $method_name(self, rhs: Rhs) -> Self::Output { map2(self, rhs, $trait_name::$method_name) }
}
    };

    ($trait_name:ident::$method_name:ident for Value) => {
impl<S, C: Dim<S>, R: TwoDim<S, C>, Rhs> $trait_name<Value<Rhs>> for Mat<R, C, S>
where Rhs: Clone,
<C as Array<Rhs>>::Type: Clone,
C: Dim<Rhs> + Dim<S::Output>,
R: TwoDim<Rhs, C> + TwoDim<S::Output, C>,
S: $trait_name<Rhs>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
C::Smaller: Array<S> + Array<Rhs> + Array<S::Output>,
R::Smaller: Array<<C as Array<S>>::Type> + Array<<C as Array<Rhs>>::Type> + Array<<C as Array<S::Output>>::Type>,
{
    type Output = Mat<R, C, S::Output>;
    fn $method_name(self, rhs: Value<Rhs>) -> Self::Output {
        map2(self, Mat::from_val(R::from_value(C::from_value(rhs.0))), $trait_name::$method_name)
    }
}
impl<S, C: Dim<S>, R: TwoDim<S, C>, Lhs> $trait_name<Mat<R, C, S>> for Value<Lhs>
where Lhs: Clone,
<C as Array<Lhs>>::Type: Clone,
C: Dim<Lhs> + Dim<Lhs::Output>,
R: TwoDim<Lhs, C> + TwoDim<Lhs::Output, C>,
Lhs: $trait_name<S>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
C::Smaller: Array<S> + Array<Lhs> + Array<Lhs::Output>,
R::Smaller: Array<<C as Array<S>>::Type> + Array<<C as Array<Lhs>>::Type> + Array<<C as Array<Lhs::Output>>::Type>,
{
    type Output = Mat<R, C, Lhs::Output>;
    fn $method_name(self, rhs: Mat<R, C, S>) -> Self::Output {
        map2(Mat::<_, _, Lhs>::from_val(R::from_value(C::from_value(self.0))), rhs, $trait_name::$method_name)
    }
}
    };
}

macro_rules! impl_mat_binop_assign {
    ($($trait_name:ident::$method_name:ident)+) => {$(
        impl_mat_binop_assign!{$trait_name::$method_name for Mat}
        impl_mat_binop_assign!{$trait_name::$method_name for Value}
    )+};

    ($trait_name:ident::$method_name:ident for Mat) => {
impl<S, C: DimMut<S>, R: TwoDimMut<S, C>, Rhs> $trait_name<Rhs> for Mat<R, C, S>
where Rhs: ScalarArrayVal<Row=C, Dim=R>,
C: Dim<Rhs::Scalar>,
R: TwoDim<Rhs::Scalar, C>,
S: $trait_name<Rhs::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
for<'a> C::Smaller: Array<S> + Array<Rhs::Scalar> + Array<&'a S> + Array<&'a mut S>,
for<'a> R::Smaller: Array<<C as Array<S>>::Type> + Array<<C as Array<Rhs::Scalar>>::Type> + Array<<C as Array<&'a S>>::Type> + Array<<C as Array<&'a mut S>>::Type>,
{
    fn $method_name(&mut self, rhs: Rhs) { apply2_mut_val(self, rhs, $trait_name::$method_name) }
}
    };

    ($trait_name:ident::$method_name:ident for Value) => {
impl<S, C: DimMut<S>, R: TwoDimMut<S, C>, Rhs> $trait_name<Value<Rhs>> for Mat<R, C, S>
where Rhs: Clone,
<C as Array<Rhs>>::Type: Clone,
C: Dim<Rhs>,
R: TwoDim<Rhs, C>,
S: $trait_name<Rhs>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
for<'a> C::Smaller: Array<S> + Array<Rhs> + Array<&'a S> + Array<&'a mut S>,
for<'a> R::Smaller: Array<<C as Array<S>>::Type> + Array<<C as Array<Rhs>>::Type> + Array<<C as Array<&'a S>>::Type> + Array<<C as Array<&'a mut S>>::Type>,
{
    fn $method_name(&mut self, rhs: Value<Rhs>) {
        apply2_mut_val(self, Mat::from_val(R::from_value(C::from_value(rhs.0))), $trait_name::$method_name)
    }
}
    };
}

impl_mat_unop!{Neg::neg Not::not}

impl_mat_binop!{
    BitAnd::bitand BitOr::bitor BitXor::bitxor
    Shl::shl Shr::shr
    Add::add Div::div Rem::rem Sub::sub
}
impl_mat_binop!{Mul::mul for Value}
impl_mat_binop_assign!{
    BitAndAssign::bitand_assign BitOrAssign::bitor_assign BitXorAssign::bitxor_assign
    ShlAssign::shl_assign ShrAssign::shr_assign
    AddAssign::add_assign DivAssign::div_assign RemAssign::rem_assign SubAssign::sub_assign
}
impl_mat_binop_assign!{MulAssign::mul_assign for Value}

impl<S, C, R, V> Mul<V> for Mat<R, C, S>
where C: Dim<S> + Dim<V::Scalar>,
R: TwoDim<S, C> + TwoDim<V::Scalar, V::Row> + Dim<<S as Mul<V::Scalar>>::Output>,
V: VecArrayVal<Row=C> + HasConcreteVecArray<<S as Mul<<V as ScalarArray>::Scalar>>::Output, R>,
V::Row: Dim<V::Scalar>,
<V::Row as Array<V::Scalar>>::Type: Clone,
S: Mul<V::Scalar>,
<S as Mul<V::Scalar>>::Output: Add<Output=<S as Mul<V::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<V as HasConcreteScalarArray<<S as Mul<V::Scalar>>::Output, R, One>>::Concrete: ConcreteVecArray,
C::Smaller: Array<S> + Array<V::Scalar>,
R::Smaller: Array<<C as Array<S>>::Type> + Array<<V::Row as Array<V::Scalar>>::Type> + Array<<S as Mul<V::Scalar>>::Output>,
{
    type Output = V::Concrete;
    fn mul(self, rhs: V) -> Self::Output { mul_vector(self, rhs) }
}

impl<S, C, R, C2, T> Mul<Mat<C, C2, T>> for Mat<R, C, S>
where C: DimMut<S> + TwoDim<T, C2> + Dim<<C2 as Array<S>>::Type>,
R: TwoDim<S, C> + TwoDim<S::Output, C2> + Array<<C as Array<<C2 as Array<T>>::Type>>::Type>,
C2: Dim<T> + Dim<S::Output> + Array<S>,
S: Mul<T> + Clone,
S::Output: Add<Output=S::Output>,
<C as Array<<C2 as Array<T>>::Type>>::Type: Clone,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
C::Smaller: Array<S> + Array<<C2 as Array<T>>::Type> + Array<<C2 as Array<S>>::Type>,
R::Smaller: Array<<C as Array<S>>::Type> + Array<<C2 as Array<S::Output>>::Type>,
C2::Smaller: Array<T> + Array<S::Output>,
{
    type Output = Mat<R, C2, S::Output>;
    fn mul(self, rhs: Mat<C, C2, T>) -> Self::Output {
        mul_matrix(self, rhs)
    }
}

include!(concat!(env!("OUT_DIR"), "/mat.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Test that Vec::new should deduce the type based on what is passed
    fn test_new() {
        assert_eq!(Mat1x1::new([[1f64]]),                   Mat::new([[1f64]]));
        assert_eq!(Mat1x2::new([[1f64, 2f64]]),             Mat::new([[1f64, 2f64]]));
        assert_eq!(Mat1x3::new([[1f64, 2f64, 3f64]]),       Mat::new([[1f64, 2f64, 3f64]]));
        assert_eq!(Mat1x4::new([[1f64, 2f64, 3f64, 4f64]]), Mat::new([[1f64, 2f64, 3f64, 4f64]]));

        assert_eq!(Mat2x2::new([[1f64, 2f64],[3f64, 4f64]]), Mat2::new([[1f64, 2f64],[3f64, 4f64]]));
        assert_eq!(Mat2x2::new([[1f64, 2f64],[3f64, 4f64]]),  Mat::new([[1f64, 2f64],[3f64, 4f64]]));
    }

    #[test]
    /// Test the unary `-` operator
    fn test_neg() {
        assert_eq!(-Mat1x1::new([[1f64]]),                   Mat1x1::new([[-1f64]]));
        assert_eq!(-Mat1x2::new([[1f64, 2f64]]),             Mat1x2::new([[-1f64, -2f64]]));
        assert_eq!(-Mat1x3::new([[1f64, 2f64, 3f64]]),       Mat1x3::new([[-1f64, -2f64, -3f64]]));
        assert_eq!(-Mat1x4::new([[1f64, 2f64, 3f64, 4f64]]), Mat1x4::new([[-1f64, -2f64, -3f64, -4f64]]));
    }


    #[test]
    /// Test a bunch of 0 multiplications
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
    /// Test the binary `*` operator
    fn test_multiply() {
        let a = Mat3x3::new([[   2f64,    3f64,    5f64], [   7f64,   11f64,   13f64], [  17f64,   19f64,   23f64]]);
        let b = Mat3x3::new([[  29f64,   31f64,   37f64], [  41f64,   43f64,   47f64], [  53f64,   59f64,   61f64]]);
        let c = Mat3x3::new([[ 446f64,  486f64,  520f64], [1343f64, 1457f64, 1569f64], [2491f64, 2701f64, 2925f64]]);
        assert_eq!(a*b, c);
    }

    /// Test the `[]` operator
    #[test]
    fn test_index() {
        use vec::Vec3;
        let m = Mat4x3::new([
            [   2f64,    3f64,    5f64],
            [   7f64,   11f64,   13f64],
            [  17f64,   19f64,   23f64],
            [  29f64,   31f64,   37f64],
        ]);

        assert_eq!(m[0], Vec3::new([   2f64,    3f64,    5f64]));
        assert_eq!(m[1], Vec3::new([   7f64,   11f64,   13f64]));
        assert_eq!(m[2], Vec3::new([  17f64,   19f64,   23f64]));
        assert_eq!(m[3], Vec3::new([  29f64,   31f64,   37f64]));
    }
}
