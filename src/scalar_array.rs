//! Traits and functions that operate on ScalarArrays.

// TODO: more commments

use std::borrow::{BorrowMut};
use std::cmp::Ordering;
use std::ops::{Mul};
use std::slice::{Iter,IterMut};

use angle;
use num::{Abs,Hyperbolic,Pow,Recip,Sqrt};

/// Types that can be held in a Matrix/Vector.
pub trait Scalar: Copy {}

/// Types that represent a dimension.
pub trait Dim<T:Copy>: Copy {
    /// An array of the size equal to the dimension this type represents.
    type Output: Copy+AsMut<[T]>+AsRef<[T]>+BorrowMut<[T]>;

    /// Construct an array from a single value `v`, replicating it to all positions in the array.
    #[inline(always)]
    fn from_value(v: T) -> Self::Output;

    /// Construct an array from an ExactSizeIterator with len() == the dimension this type
    /// represents.
    #[inline(always)]
    fn from_iter<U>(iterator: U) -> Self::Output
    where U: IntoIterator<Item=T>,
    U::IntoIter: ExactSizeIterator;
}

/// Types that represent an array of scalars (a matrix or a vector).
pub trait ScalarArray {
    /// The type of the underlying scalar in the array.
    type Scalar: Scalar;
    /// The type of a single element of this type (a single row for matrices/a scalar for vectors).
    type Type: Copy;
    /// The dimension of the scalar array.
    type Dim: Dim<Self::Type>;

    /// Constructs a matrix/vector from a an array (`v`) of the underlying type. Most useful in
    /// conjuction with `Dim::from_iter`.
    #[inline(always)]
    fn new(v: <Self::Dim as Dim<Self::Type>>::Output) -> Self;
    /// Returns a slice iterator over the elements of `self` (the rows for matrices/the scalars
    /// for vectors).
    #[inline(always)]
    fn iter(&self) -> Iter<Self::Type>;
    /// Returns a mutable slice iterator over the elements of `self` (the rows for matrices/the
    /// scalars for vectors).
    #[inline(always)]
    fn iter_mut(&mut self) -> IterMut<Self::Type>;
    /// Constructs a matrix/vector from a single scalar value, setting all elements to that value.
    #[inline(always)]
    fn from_value(v: Self::Scalar) -> Self;

    /// Fold all the scalar values into a single output given two folding functions,
    /// The first folding function only applies to the first element of the ScalarArray.
    #[inline(always)]
    fn fold<T, F0: FnOnce(&Self::Scalar)->T, F: Fn(T, &Self::Scalar)->T>(&self, f0: F0, f: F) -> T;

    /// Map all the scalar values, keeping the same underlying type.
    #[inline(always)]
    fn map<F: Fn(Self::Scalar)->Self::Scalar>(self, f: F) -> Self;
}

/// Types that can be fold with another `ScalarArray` of the same dimension into single value.
pub trait Fold<Rhs: Scalar>: ScalarArray
where <Self as ScalarArray>::Dim: Dim<<Self::RhsArray as ScalarArray>::Type> {
    /// The right hand side type.
    type RhsArray: ScalarArray<Scalar=Rhs, Dim=<Self as ScalarArray>::Dim>;

    /// Fold two `ScalarArray`s together using a binary function.
    #[inline(always)]
    fn fold_together<O, F0: FnOnce(&<Self as ScalarArray>::Scalar, &Rhs)->O, F: Fn(O, &<Self as ScalarArray>::Scalar, &Rhs)->O>(&self, rhs: &Self::RhsArray, f0: F0, f: F) -> O;
}

/// Types that can be transformed from into a `ScalarArray` with Scalar type `O`.
pub trait Cast<O: Scalar>: ScalarArray
where <Self as ScalarArray>::Dim: Dim<<Self::Output as ScalarArray>::Type> {
    /// The resulting type.
    type Output: ScalarArray<Scalar=O, Dim=<Self as ScalarArray>::Dim>;

    /// Transform a single `ScalarArray` using a unary function.
    #[inline(always)]
    fn unary<F: Fn(&<Self as ScalarArray>::Scalar)->O>(&self, f: F) -> Self::Output;

    /// Transform two binary `ScalarArray`s using a binary function.
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &<Self as ScalarArray>::Scalar)->O>(&self, rhs: &Self, f: F) -> Self::Output;
}

/// Types that can be transformed from into a `ScalarArray` with Scalar type `O`.
pub trait CastBinary<Rhs: Scalar, O: Scalar>: ScalarArray
where <Self as ScalarArray>::Dim: Dim<<Self::RhsArray as ScalarArray>::Type>+Dim<<Self::Output as ScalarArray>::Type> {
    /// The right hand side type.
    type RhsArray: ScalarArray<Scalar=Rhs, Dim=<Self as ScalarArray>::Dim>;

    /// The resulting type.
    type Output: ScalarArray<Scalar=O, Dim=<Self as ScalarArray>::Dim>;

    /// Transform two binary `ScalarArray`s using a binary function.
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &Rhs)->O>(&self, rhs: &Self::RhsArray, f: F) -> Self::Output;
}


/// Types that can be component-wise compared using PartialEq.
pub trait ComponentPartialEq: ScalarArray + Cast<bool>
where <Self as ScalarArray>::Scalar: PartialEq,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> {
    /// Tests for the components of `self` and `rhs` values to be equal.
    fn cpt_eq(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialEq::eq)
    }

    /// Tests for the components of `self` and `rhs` values to be unequal.
    fn cpt_ne(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialEq::ne)
    }
}

/// Types that can be component-wise compared using Eq.
pub trait ComponentEq : ComponentPartialEq
where <Self as ScalarArray>::Scalar: Eq,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> {}


/// Types that can be component-wise using PartialOrd.
pub trait ComponentPartialOrd: ComponentPartialEq + Cast<Option<Ordering>>
where <Self as ScalarArray>::Scalar: PartialOrd,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> + Dim<<<Self as Cast<Option<Ordering>>>::Output as ScalarArray>::Type> {
    /// Returns an ordering between the components of `self` and `rhs` values if one exists.
    fn cpt_partial_cmp(&self, rhs: &Self) -> <Self as Cast<Option<Ordering>>>::Output {
        Cast::<Option<Ordering>>::binary(self, rhs, PartialOrd::partial_cmp)
    }

    /// Tests less than between the components of `self` and `rhs` values.
    fn cpt_lt(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::lt)
    }
    /// Tests less than or equal to between the components of `self` and `rhs` values.
    fn cpt_le(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::le)
    }
    /// Tests greater than between the components of `self` and `rhs` values.
    fn cpt_gt(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::gt)
    }
    /// Tests greater than or equal to between the components of `self` and `rhs` values.
    fn cpt_ge(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::ge)
    }
}

/// Types that can be component-wise using Ord.
pub trait ComponentOrd: ComponentEq + ComponentPartialOrd + Cast<Ordering>
where <Self as ScalarArray>::Scalar: Ord,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> + Dim<<<Self as Cast<Option<Ordering>>>::Output as ScalarArray>::Type> + Dim<<<Self as Cast<Ordering>>::Output as ScalarArray>::Type> {
    /// Returns an ordering between the components of `self` and `rhs` values.
    fn cpt_cmp(&self, rhs: &Self) -> <Self as Cast<Ordering>>::Output {
        Cast::<Ordering>::binary(self, rhs, Ord::cmp)
    }
}

/// Types that can be component-wise multiplied.
pub trait ComponentMul<Rhs: Scalar>: ScalarArray
where <Self as ScalarArray>::Scalar: Mul<Rhs>,
<<Self as ScalarArray>::Scalar as Mul<Rhs>>::Output: Scalar,
<Self as ScalarArray>::Dim: Dim<<Self::RhsArray as ScalarArray>::Type>+Dim<<Self::Output as ScalarArray>::Type> {
    /// The right hand side type.
    type RhsArray: ScalarArray<Scalar=Rhs, Dim=<Self as ScalarArray>::Dim>;

    /// The resulting type.
    type Output: ScalarArray<Scalar=<<Self as ScalarArray>::Scalar as Mul<Rhs>>::Output, Dim=<Self as ScalarArray>::Dim>;

    /// Multiplies the components of `self` with those of `rhs`.
    fn cmp_mul(&self, rhs: &Self::RhsArray) -> Self::Output;
}


/// Types that can be square-rooted.
impl <S: ScalarArray> Sqrt for S
where <S as ScalarArray>::Scalar: Sqrt {
    /// Takes the square root of a number.
    ///
    /// Returns NaN if self is a negative number.
    fn sqrt(self) -> Self { self.map(Sqrt::sqrt) }

    /// Returns the inverse of the square root of a number. i.e. the value `1/√x`.
    ///
    /// Returns NaN if `self` is a negative number.
    fn inverse_sqrt(self) -> Self { self.map(Sqrt::inverse_sqrt) }
}

/// Types that can have the reciprocal taken.
impl <S: ScalarArray> Recip for S
where <S as ScalarArray>::Scalar: Recip {
    /// Takes the reciprocal (inverse) of a number, `1/x`.
    fn recip(self) -> Self { self.map(Recip::recip) }
}

// Types that implment hyperbolic angle functions.
impl <S: ScalarArray> Hyperbolic for S
where <S as ScalarArray>::Scalar: Hyperbolic {
    // Hyperbolic sine function.
    fn sinh(self) -> Self { self.map(Hyperbolic::sinh) }
    // Hyperbolic cosine function.
    fn cosh(self) -> Self { self.map(Hyperbolic::cosh) }
    // Hyperbolic tangent function.
    fn tanh(self) -> Self { self.map(Hyperbolic::tanh) }
    // Hyperbolic sine function.
    fn asinh(self) -> Self { self.map(Hyperbolic::asinh) }
    // Hyperbolic cosine function.
    fn acosh(self) -> Self { self.map(Hyperbolic::acosh) }
    // Hyperbolic tangent function.
    fn atanh(self) -> Self { self.map(Hyperbolic::atanh) }
}

impl <S: ScalarArray> Abs for S
where <S as ScalarArray>::Scalar: Abs,
<<S as ScalarArray>::Scalar as Abs>::Output: Scalar,
S: Cast<<<S as ScalarArray>::Scalar as Abs>::Output>,
<S as ScalarArray>::Dim: Dim<<<S as Cast<<<S as ScalarArray>::Scalar as Abs>::Output>>::Output as ScalarArray>::Type> {
    /// The resulting type.
    type Output = <S as Cast<<<S as ScalarArray>::Scalar as Abs>::Output>>::Output;

    /// Computes the absolute value of self. Returns NaN if the number is NaN.
    fn abs(self) -> Self::Output {
        Cast::<<<S as ScalarArray>::Scalar as Abs>::Output>::unary(&self, |&v| Abs::abs(v))
    }

    /// Returns |self-rhs| without modulo overflow.
    fn abs_diff(self, rhs: Self) -> Self::Output {
        Cast::<<<S as ScalarArray>::Scalar as Abs>::Output>::binary(&self, &rhs, |&l, &r| Abs::abs_diff(l, r))
    }
}

/// Types that implement the Pow function
impl <Lhs: ScalarArray, Rhs: ScalarArray<Dim=<Lhs as ScalarArray>::Dim>> Pow<Rhs> for Lhs
where <Lhs as ScalarArray>::Scalar: Pow<<Rhs as ScalarArray>::Scalar>,
Lhs: CastBinary<<Rhs as ScalarArray>::Scalar, <Lhs as ScalarArray>::Scalar, RhsArray=Rhs, Output=Lhs>,
<Lhs as ScalarArray>::Dim: Dim<<Rhs as ScalarArray>::Type> {
    /// Returns `self` raised to the power `rhs`
    fn pow(self, rhs: Rhs) -> Self {
        CastBinary::<<Rhs as ScalarArray>::Scalar, <Lhs as ScalarArray>::Scalar>::binary(&self, &rhs, |&l, &r| Pow::pow(l, r))
    }
}

// TODO: exponential functions
// e^x
//  Tf  exp(Tf x)
// ln
//  Tf  log(Tf x)
// 2^x
//  Tf  exp2(Tf x)
// log2
//  Tf  log2(Tf x)

// TODO: common functions
// Returns nearest integer <= x:
//  Tfd floor(Tfd x)
// Returns nearest integer with absolute value <= absolute value of x:
//  Tfd trunc(Tfd x)
// Returns nearest integer, implementation-dependent rounding mode:
//  Tfd round(Tfd x)
// Returns nearest integer, 0.5 rounds to nearest even integer:
//  Tfd roundEven(Tfd x)
// Returns nearest integer >= x:
//  Tfd ceil(Tfd x)
// Returns x - floor(x):
//  Tfd fract(Tfd x)
// Returns modulus:
//  Tfd mod(Tfd x, Tfd y)
//  Tf  mod(Tf x, float y)
//  Td  mod(Td x, double y)
// Returns separate integer and fractional parts:
//  Tfd modf(Tfd x, out Tfd i)
// Returns minimum value:
//  Tfd min(Tfd x, Tfd y)
//  Tf  min(Tf x, float y)
//  Td  min(Td x, double y)
//  Tiu min(Tiu x, Tiu y)
//  Ti  min(Ti x, int y)
//  Tu  min(Tu x, uint y)
// Returns maximum value:
//  Tfd max(Tfd x, Tfd y)
//  Tf  max(Tf x, float y)
//  Td  max(Td x, double y)
//  Tiu max(Tiu x, Tiu y)
//  Ti  max(Ti x, int y)
//  Tu  max(Tu x, uint y)
// Returns min(max(x, minVal), maxVal):
//  Tfd clamp(Tfd x, Tfd minVal, Tfd maxVal)
//  Tf  clamp(Tf x, float minVal, float maxVal)
//  Td  clamp(Td x, double minVal, double maxVal)
//  Tiu clamp(Tiu x, Tiu minVal, Tiu maxVal)
//  Ti  clamp(Ti x, int minVal, int maxVal)
//  Tu  clamp(Tu x, uint minVal, uint maxVal)
// Returns linear blend of x and y:
//  Tfd mix(Tfd x, Tfd y, Tfd a)
//  Tf  mix(Tf x, Tf y, float a)
//  Td  mix(Td x, Td y, double a)
//  Ti  mix(Ti x, Ti y, Ti a)
//  Tu  mix(Tu x, Tu y, Tu a)
// Components returned come from x when a components are true, from y when a components are false:
//  Tfd mix(Tfd x, Tfd y, Tb a)
//  Tb  mix(Tb x, Tb y, Tb a)
//  Tiu mix(Tiu x, Tiu y, Tb a)
// Returns 0.0 if x < edge, else 1.0:
//  Tfd step(Tfd edge, Tfd x)
//  Tf  step(float edge, Tf x)
//  Td  step(double edge, Td x)
// Clamps and smoothes:
//  Tfd smoothstep(Tfd edge0, Tfd edge1, Tfd x)
//  Tf  smoothstep(float edge0, float edge1, Tf x)
//  Td  smoothstep(double edge0, double edge1, Td x)
// Returns true if x is NaN:
//  Tb  isnan(Tfd x)
// Returns true if x is positive or negative infinity:
//  Tb  isinf(Tfd x)
// Returns signed int or uint value of the encoding of a float :
//  Ti  floatBitsToint (Tf value)
//  Tu  floatBitsToUint (Tf value)
// Returns float value of a signed int  or uint encoding of a float :
//  Tf  intBitsTofloat (Ti value)
//  Tf uintBitsTofloat (Tu value)
// Computes and returns a*b + c.Treated as a single operation when using precise:
//  Tfd  fma(Tfd a, Tfd b, Tfd c)
// Splits x into a floating-point significand in the range[0.5, 1.0) and an integer exponent of 2:
//  Tfd  frexp(Tfd x, out Ti exp)
// Builds a floating-point number from x and the corresponding integral exponent of 2 in exp:
//  Tfd  ldexp(Tfd x, in Ti exp)

impl Scalar for Option<Ordering> {}
impl Scalar for Ordering {}

include!(concat!(env!("OUT_DIR"), "/scalar_array.rs"));
