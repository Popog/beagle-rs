//! A collection of numeric traits and functions.

use std::ops::Mul;
use std::cmp;
use std::cmp::Ordering;
use std::num::Wrapping;

pub trait Abs {
    /// The resulting type.
    type Output;

    /// Computes the absolute value of self. Returns NaN if the number is NaN.
    fn abs(self) -> Self::Output;

    /// Returns |self-rhs| without modulo overflow.
    fn abs_diff(self, rhs: Self) -> Self::Output;
}

impl Abs for f32 {
    /// The resulting type.
    type Output = f32;

    /// Computes the absolute value of self. Returns NaN if the number is NaN.
    fn abs(self) -> Self::Output { f32::abs(self) }

    /// Returns |self-rhs| without modulo overflow.
    fn abs_diff(self, rhs: Self) -> Self::Output { f32::abs(self - rhs) }
}

impl Abs for f64 {
    /// The resulting type.
    type Output = f64;

    /// Computes the absolute value of self. Returns NaN if the number is NaN.
    fn abs(self) -> Self::Output { f64::abs(self) }

    /// Returns |self-rhs| without modulo overflow.
    fn abs_diff(self, rhs: Self) -> Self::Output { f64::abs(self - rhs) }
}

/// Types that can be square-rooted.
pub trait Sqrt {
    /// The output type
    type Output;

    /// Takes the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number.
    fn sqrt(self) -> Self::Output;

    /// Returns the inverse of the square root of a number. i.e. the value `1/√x`.
    ///
    /// Returns NaN if `self` is a negative number.
    ///
    /// ```
    /// use beagle::num::Sqrt;
    ///
    /// let positive = 4.0_f32;
    /// let negative = -4.0_f32;
    /// let inf = std::f32::INFINITY;
    /// let nan = std::f32::NAN;
    /// let zero = 0f32;
    /// let negzero = -0f32;
    ///
    /// assert!((positive.inverse_sqrt() - 0.5).abs() < 1e-3);
    /// assert!(negative.inverse_sqrt().is_nan());
    /// assert!(nan.inverse_sqrt().is_nan());
    /// assert_eq!(inf.inverse_sqrt(), 0f32);
    /// assert_eq!(zero.inverse_sqrt(), std::f32::INFINITY);
    /// assert_eq!(negzero.inverse_sqrt(), std::f32::INFINITY);
    /// ```
    fn inverse_sqrt(self) -> Self::Output;
}

/// Types that can be square-rooted.
impl Sqrt for f32 {
    /// The output type
    type Output = f32;

    /// Takes the square root of a number.
    ///
    /// Returns NaN if self is a negative number.
    fn sqrt(self) -> Self { f32::sqrt(self) }

    /// Returns the inverse of the square root of a number. i.e. the value `1/√x`.
    ///
    /// Returns NaN if `self` is a negative number.
    fn inverse_sqrt(self) -> Self {
        use std::mem;
        use std::f32;

        // Turns out, this already handles NaN input.
        if self < 0.0 { return f32::NAN; }
        if self == 0.0 { return f32::INFINITY; }
        if self == f32::INFINITY { return 0.0; }

        let y: Wrapping<u32> = unsafe { mem::transmute(self) }; // evil floating point bit level hacking
        let y  = Wrapping(0x5f375a86) - ( y >> 1 );             // what the fuck?
        let y: f32 = unsafe { mem::transmute(y) };
        let x2 = self * 0.5;
        let y  = y * ( 1.5 - ( x2 * y * y ) );     // 1st iteration
        //let y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed
        y
    }
}
/// Types that can be square-rooted.
impl Sqrt for f64 {
    /// The output type
    type Output = f64;

    /// Takes the square root of a number.
    ///
    /// Returns NaN if self is a negative number.
    fn sqrt(self) -> Self { f64::sqrt(self) }

    /// Returns the inverse of the square root of a number. i.e. the value `1/√x`.
    ///
    /// Returns NaN if `self` is a negative number.
    ///
    fn inverse_sqrt(self) -> Self {
        use std::mem;
        use std::f64;

        // Turns out, this already handles NaN input.
        if self < 0.0 { return f64::NAN; }
        if self == 0.0 { return f64::INFINITY; }
        if self == f64::INFINITY { return 0.0; }

        let y: Wrapping<u64> = unsafe { mem::transmute(self) }; // evil floating point bit level hacking
        let y  = Wrapping(0x5fe6eb50c7b537a9) - ( y >> 1 );     // what the fuck?
        let y: f64 = unsafe { mem::transmute(y) };
        let x2 = self * 0.5;
        let y  = y * ( 1.5 - ( x2 * y * y ) );     // 1st iteration
        //let y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed
        y
    }
}

/// Types that implment hyperbolic angle functions.
pub trait Hyperbolic {
    /// The output type
    type Output;
    /// Hyperbolic sine function.
    fn sinh(self) -> Self::Output;
    /// Hyperbolic cosine function.
    fn cosh(self) -> Self::Output;
    /// Hyperbolic tangent function.
    fn tanh(self) -> Self::Output;
    /// Hyperbolic sine function.
    fn asinh(self) -> Self::Output;
    /// Hyperbolic cosine function.
    fn acosh(self) -> Self::Output;
    /// Hyperbolic tangent function.
    fn atanh(self) -> Self::Output;
}

/// Types that implment hyperbolic angle functions.
impl Hyperbolic for f32 {
    /// The output type
    type Output = f32;
    /// Hyperbolic sine function.
    fn sinh(self) -> f32 { f32::sinh(self) }
    /// Hyperbolic cosine function.
    fn cosh(self) -> f32 { f32::cosh(self) }
    /// Hyperbolic tangent function.
    fn tanh(self) -> f32 { f32::tanh(self) }
    /// Hyperbolic sine function.
    fn asinh(self) -> f32 { f32::asinh(self) }
    /// Hyperbolic cosine function.
    fn acosh(self) -> f32 { f32::acosh(self) }
    /// Hyperbolic tangent function.
    fn atanh(self) -> f32 { f32::atanh(self) }
}

/// Types that implment hyperbolic angle functions.
impl Hyperbolic for f64 {
    /// The output type
    type Output = f64;
    /// Hyperbolic sine function.
    fn sinh(self) -> f64 { f64::sinh(self) }
    /// Hyperbolic cosine function.
    fn cosh(self) -> f64 { f64::cosh(self) }
    /// Hyperbolic tangent function.
    fn tanh(self) -> f64 { f64::tanh(self) }
    /// Hyperbolic sine function.
    fn asinh(self) -> f64 { f64::asinh(self) }
    /// Hyperbolic cosine function.
    fn acosh(self) -> f64 { f64::acosh(self) }
    /// Hyperbolic tangent function.
    fn atanh(self) -> f64 { f64::atanh(self) }
}

/// Types that implement the Pow function
pub trait Pow<Rhs> {
    /// The output type
    type Output;
    /// Returns `self` raised to the power `rhs`
    fn pow(self, rhs: Rhs) -> Self::Output;
}

/// Types that implement the Pow function
impl Pow<f32> for f32 {
    /// The output type
    type Output = f32;
    /// Returns `self` raised to the power `rhs`
    fn pow(self, rhs: f32) -> f32 { f32::powf(self, rhs) }
}

/// Types that implement the Pow function
impl Pow<i32> for f32 {
    /// The output type
    type Output = f32;
    /// Returns `self` raised to the power `rhs`
    fn pow(self, rhs: i32) -> f32 { f32::powi(self, rhs) }
}

/// Types that implement the Pow function
impl Pow<f64> for f64 {
    /// The output type
    type Output = f64;
    /// Returns `self` raised to the power `rhs`
    fn pow(self, rhs: f64) -> f64 { f64::powf(self, rhs) }
}

/// Types that implement the Pow function
impl Pow<i32> for f64 {
    /// The output type
    type Output = f64;
    /// Returns `self` raised to the power `rhs`
    fn pow(self, rhs: i32) -> f64 { f64::powi(self, rhs) }
}

/// Types that implement exponential functions
pub trait Exp {
    /// The output type
    type Output;
    /// Returns `e^(self)`, (the exponential function).
    fn exp(self) -> Self::Output;
    /// Returns `2^(self)`.
    fn exp2(self) -> Self::Output;
    /// Returns the natural logarithm of the number.
    fn ln(self) -> Self::Output;
    /// Returns the base 2 logarithm of the number.
    fn log2(self) -> Self::Output;
    /// Returns the base 10 logarithm of the number.
    fn log10(self) -> Self::Output;
}

/// Types that implement exponential functions
impl Exp for f32 {
    /// The output type
    type Output = f32;
    /// Returns `e^(self)`, (the exponential function).
    fn exp(self) -> f32 { f32::exp(self) }
    /// Returns `2^(self)`.
    fn exp2(self) -> f32 { f32::exp2(self) }
    /// Returns the natural logarithm of the number.
    fn ln(self) -> f32 { f32::ln(self) }
    /// Returns the base 2 logarithm of the number.
    fn log2(self) -> f32 { f32::log2(self) }
    /// Returns the base 10 logarithm of the number.
    fn log10(self) -> f32 { f32::log10(self) }
}

/// Types that implement exponential functions
impl Exp for f64 {
    /// The output type
    type Output = f64;
    /// Returns `e^(self)`, (the exponential function).
    fn exp(self) -> f64 { f64::exp(self) }
    /// Returns `2^(self)`.
    fn exp2(self) -> f64 { f64::exp2(self) }
    /// Returns the natural logarithm of the number.
    fn ln(self) -> f64 { f64::ln(self) }
    /// Returns the base 2 logarithm of the number.
    fn log2(self) -> f64 { f64::log2(self) }
    /// Returns the base 10 logarithm of the number.
    fn log10(self) -> f64 { f64::log10(self) }
}

/// Types that can be rounded.
pub trait Round {
    /// The output type
    type Output;
    /// Returns the largest integer less than or equal to a number.
    fn floor(self) -> Self::Output;
    /// Returns the smallest integer greater than or equal to a number.
    fn ceil(self) -> Self::Output;
    /// Returns the integer part of a number.
    fn trunc(self) -> Self::Output;
    /// Returns the fractional part of a number.
    fn fract(self) -> Self::Output;
    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    fn round(self) -> Self::Output;
}

/// Types that can be rounded.
impl Round for f32 {
    /// The output type
    type Output = f32;
    /// Returns the largest integer less than or equal to a number.
    fn floor(self) -> f32 { f32::floor(self) }
    /// Returns the smallest integer greater than or equal to a number.
    fn ceil(self) -> f32 { f32::ceil(self) }
    /// Returns the integer part of a number.
    fn trunc(self) -> f32 { f32::trunc(self) }
    /// Returns the fractional part of a number.
    fn fract(self) -> f32 { f32::fract(self) }
    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    fn round(self) -> f32 { f32::round(self) }
}

/// Types that can be rounded.
impl Round for f64 {
    /// The output type
    type Output = f64;
    /// Returns the largest integer less than or equal to a number.
    fn floor(self) -> f64 { f64::floor(self) }
    /// Returns the smallest integer greater than or equal to a number.
    fn ceil(self) -> f64 { f64::ceil(self) }
    /// Returns the integer part of a number.
    fn trunc(self) -> f64 { f64::trunc(self) }
    /// Returns the fractional part of a number.
    fn fract(self) -> f64 { f64::fract(self) }
    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    fn round(self) -> f64 { f64::round(self) }
}

/// Types that can be clamped
pub trait Clamp<Rhs = Self>: Sized {
    /// The output type
    type Output;
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn min(self, rhs: Rhs) -> Self::Output;
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn max(self, rhs: Rhs) -> Self::Output;
}

/// Returns the value of `self` constrained to the range [min_val, max_val]
/// The returned value is computed as min(max(self, min_val), mal_val).
//fn clamp<V, L, H> (v: V, min_val: L, max_val: H) -> Self::Output {
//    Self::min(Self::max(self, min_val), max_val)
//}

/// Types that can be clamped
impl Clamp for f32 {
    /// The output type
    type Output = f32;
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn min(self, rhs: f32) -> f32 { f32::min(self, rhs) }
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn max(self, rhs: f32) -> f32 { f32::max(self, rhs) }
}

/// Types that can be clamped
impl Clamp for f64 {
    /// The output type
    type Output = f64;
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn min(self, rhs: f64) -> f64 { f64::min(self, rhs) }
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn max(self, rhs: f64) -> f64 { f64::max(self, rhs) }
}

macro_rules! clamp_impl {
    ($($x:ident)*) => {
$(/// Types that can be clamped
impl Clamp for $x {
    /// The output type
    type Output = $x;
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn min(self, rhs: $x) -> $x { cmp::min(self, rhs) }
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn max(self, rhs: $x) -> $x { cmp::max(self, rhs) }
})*
    };
}

clamp_impl!{i8 i16 i32 i64 u8 u16 u32 u64}

/// Types which can be mixed
pub trait Mix<A> {
    /// performs a linear interpolation between a number and y using `a` to weight between them.
    /// For non-boolean `a`, the return value is computed as `x×(1−a)+y×a×x×(1−a)+y×a`.
    /// For boolean `a`, the return value is composed of selected components of `x` and `y`.
    /// False components of `a` select the corresponding component of `x`. True componenets
    /// of `a` select the corresponding components of 'y'.
    fn mix(x: Self, y: Self, a: A);
}


/// Types that can have the reciprocal taken.
pub trait Recip {
    /// The output type
    type Output;
    /// Takes the reciprocal (inverse) of a number, `1/x`.
    fn recip(self) -> Self::Output;
}

/// Types that can have the reciprocal taken.
impl Recip for f32 {
    /// The output type
    type Output = f32;
    /// Takes the reciprocal (inverse) of a number, `1/x`.
    fn recip(self) -> f32 { f32::recip(self) }
}

/// Types that can have the reciprocal taken.
impl Recip for f64 {
    /// The output type
    type Output = f64;
    /// Takes the reciprocal (inverse) of a number, `1/x`.
    fn recip(self) -> f64 { f64::recip(self) }
}

/// Types that can be sign functions.
pub trait Sign {
    /// The resulting boolean type.
    type Output;

    /// Returns `true` if `self` is strictly less than zero.
    fn is_negative(&self) -> Self::Output;
    /// Returns `true` if `self` is strictly greater than zero.
    fn is_positive(&self) -> Self::Output;

    /// Returns `true` if `self`'s sign is negative (including `-0`).
    fn is_sign_negative(&self) -> Self::Output;
    /// Returns `true` if `self`'s sign is positive (including `+0`).
    fn is_sign_positive(&self) -> Self::Output;

    /// Returns a number that represents the sign of self.
    /// NaN if `self` is NaN
    /// One if the `self.is_sign_positive()`
    /// Negative one if `self.is_sign_negative()`
    fn signum(self) -> Self;

    /// Returns a number that represents the sign of self.
    /// NaN if `self` is NaN
    /// One if the `self.is_positive()`
    /// Negative one if `self.is_negative()`
    /// else Zero
    fn sign(self) -> Self;
}

impl Sign for f32 {
    /// The resulting type
    type Output = bool;

    /// Returns `true` if `self` is strictly less than zero.
    fn is_negative(&self) -> Self::Output { self.lt(&0f32) }
    /// Returns `true` if `self` is strictly greater than zero.
    fn is_positive(&self) -> Self::Output { self.gt(&0f32) }

    /// Returns `true` if `self`'s sign is negative (including `-0`).
    fn is_sign_negative(&self) -> Self::Output { f32::is_sign_negative(*self) }
    /// Returns `true` if `self`'s sign is positive (including `+0`).
    fn is_sign_positive(&self) -> Self::Output { f32::is_sign_positive(*self) }

    /// Returns a number that represents the sign of self.
    /// NaN if `self` is NaN
    /// One if the `self.is_sign_positive()`
    /// Negative one if `self.is_sign_negative()`
    fn signum(self) -> Self { f32::signum(self) }

    /// Returns a number that represents the sign of self.
    /// NaN if `self` is NaN
    /// One if the `self.is_positive()`
    /// Negative one if `self.is_negative()`
    /// else Zero
    fn sign(self) -> Self {
        match self.partial_cmp(&0f32) {
            Some(Ordering::Less) => -1f32,
            Some(Ordering::Greater) => 1f32,
            _ => self,
        }
    }
}
impl Sign for f64 {
    /// The resulting type
    type Output = bool;

    /// Returns `true` if `self` is strictly less than zero.
    fn is_negative(&self) -> Self::Output { self.lt(&0f64) }
    /// Returns `true` if `self` is strictly greater than zero.
    fn is_positive(&self) -> Self::Output { self.gt(&0f64) }

    /// Returns `true` if `self`'s sign is negative (including `-0`).
    fn is_sign_negative(&self) -> Self::Output { f64::is_sign_negative(*self) }
    /// Returns `true` if `self`'s sign is positive (including `+0`).
    fn is_sign_positive(&self) -> Self::Output { f64::is_sign_positive(*self) }

    /// Returns a number that represents the sign of self.
    /// NaN if `self` is NaN
    /// One if the `self.is_sign_positive()`
    /// Negative one if `self.is_sign_negative()`
    fn signum(self) -> Self { f64::signum(self) }

    /// Returns a number that represents the sign of self.
    /// NaN if `self` is NaN
    /// One if the `self.is_positive()`
    /// Negative one if `self.is_negative()`
    /// else Zero
    fn sign(self) -> Self {
        match self.partial_cmp(&0f64) {
            Some(Ordering::Less) => -1f64,
            Some(Ordering::Greater) => 1f64,
            _ => self,
        }
    }
}

/// Types that can be used with compared to be approximate equal.
pub trait Approx<T> {
    /// Returns true if `lhs` is approximately `rhs`.
    fn approx(&self, lhs: T, rhs: T) -> bool;
}

/// Types that can be used to compare to zero efficiently.
pub trait ApproxZero<T> {
    /// Returns true if `a` is approximately zero.
    fn approx_zero(&self, a: T) -> bool;

    /// Returns true if `a/b` is approximately zero.
    fn approx_zero_ratio(&self, a: T, b: T) -> bool;
}

/// An approximator using simple episilon.
pub struct AbsoluteEpsilon<F>(pub F);

impl <T: Copy+Mul<Output=T>> AbsoluteEpsilon<T> {
    /// Tighten approximation so that `self.approx_zero(x) == self.squared().approx_zero(x*x)`.
    pub fn squared(&self) -> Self { AbsoluteEpsilon(self.0*self.0) }
}

/// Types that can be used with compared to be approximate equal.
impl <T: Abs> Approx<T> for AbsoluteEpsilon<<T as Abs>::Output>
where <T as Abs>::Output: Copy+PartialOrd {
    /// Returns true if `lhs` is approximately `rhs`.
    fn approx(&self, lhs: T, rhs: T) -> bool {
        Abs::abs_diff(lhs, rhs) <= self.0
    }
}

/// Types that can be used to compare to zero efficiently.
impl <T: Copy+Abs> ApproxZero<T> for AbsoluteEpsilon<<T as Abs>::Output>
where <T as Abs>::Output: Copy+PartialOrd+Mul<Output=<T as Abs>::Output> {
    /// Returns true if `a` is approximately zero.
    fn approx_zero(&self, a: T) -> bool {
        Abs::abs(a) <= self.0
    }
    /// Returns true if `a/b` is approximately zero without doing a division.
    fn approx_zero_ratio(&self, a: T, b: T) -> bool {
        Abs::abs(a) <= Abs::abs(b) * self.0
    }
}

#[cfg(test)]
mod tests {
    use std::num::Wrapping;
    use super::*;

    #[test]
    #[ignore]
    fn test_inverse_sqrt_nan() {
        use std::mem;
        use std::f32;

        // Whitebox test all NaN and negative numbers
        let mut nanu: Wrapping<u32> = unsafe { mem::transmute(f32::INFINITY) };
        loop {
             nanu = nanu+Wrapping(1);
             if nanu == Wrapping(0) { break; }
             let nan: f32 = unsafe { mem::transmute(nanu) };
             if nan == 0.0 { continue; }
             assert!(nan.inverse_sqrt().is_nan(), "inverse_sqrt was not nan for {} ({})", nan, nanu.0);
        }
    }
}
