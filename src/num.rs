//! A collection of numeric traits and functions.

use std::ops::Mul;
use std::cmp::Ordering;

/// Types that can be square-rooted.
pub trait Sqrt {
    /// Takes the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number.
    fn sqrt(self) -> Self;

    /// Returns the inverse of the square root of a number. i.e. the value `1/√x`.
    ///
    /// Returns NaN if `self` is a negative number.
    fn inversesqrt(self) -> Self;
}

/// Types that can be square-rooted.
impl Sqrt for f32 {
    /// Takes the square root of a number.
    ///
    /// Returns NaN if self is a negative number.
    fn sqrt(self) -> Self { f32::sqrt(self) }

    /// Returns the inverse of the square root of a number. i.e. the value `1/√x`.
    ///
    /// Returns NaN if `self` is a negative number.
    fn inversesqrt(self) -> Self { self.sqrt().recip() }
}
/// Types that can be square-rooted.
impl Sqrt for f64 {
    /// Takes the square root of a number.
    ///
    /// Returns NaN if self is a negative number.
    fn sqrt(self) -> Self { f64::sqrt(self) }

    /// Returns the inverse of the square root of a number. i.e. the value `1/√x`.
    ///
    /// Returns NaN if `self` is a negative number.
    fn inversesqrt(self) -> Self { self.sqrt().recip() }
}

// Types that implment hyperbolic angle functions
pub trait Hyperbolic {
    // Hyperbolic sine function.
    fn sinh(self) -> Self;
    // Hyperbolic cosine function.
    fn cosh(self) -> Self;
    // Hyperbolic tangent function.
    fn tanh(self) -> Self;
    // Hyperbolic sine function.
    fn asinh(self) -> Self;
    // Hyperbolic cosine function.
    fn acosh(self) -> Self;
    // Hyperbolic tangent function.
    fn atanh(self) -> Self;
}

// Types that implment hyperbolic angle functions
impl Hyperbolic for f32 {
    // Hyperbolic sine function.
    fn sinh(self) -> Self { f32::sinh(self) }
    // Hyperbolic cosine function.
    fn cosh(self) -> Self { f32::cosh(self) }
    // Hyperbolic tangent function.
    fn tanh(self) -> Self { f32::tanh(self) }
    // Hyperbolic sine function.
    fn asinh(self) -> Self { f32::asinh(self) }
    // Hyperbolic cosine function.
    fn acosh(self) -> Self { f32::acosh(self) }
    // Hyperbolic tangent function.
    fn atanh(self) -> Self { f32::atanh(self) }
}

// Types that implment hyperbolic angle functions
impl Hyperbolic for f64 {
    // Hyperbolic sine function.
    fn sinh(self) -> Self { f64::sinh(self) }
    // Hyperbolic cosine function.
    fn cosh(self) -> Self { f64::cosh(self) }
    // Hyperbolic tangent function.
    fn tanh(self) -> Self { f64::tanh(self) }
    // Hyperbolic sine function.
    fn asinh(self) -> Self { f64::asinh(self) }
    // Hyperbolic cosine function.
    fn acosh(self) -> Self { f64::acosh(self) }
    // Hyperbolic tangent function.
    fn atanh(self) -> Self { f64::atanh(self) }
}

/// Types that can be sign functions.
pub trait Sign {
    /// The resulting type
    type Output;

    /// Returns `true` if `self` is strictly less than zero
    fn is_negative(&self) -> Self::Output;
    /// Returns `true` if `self` is strictly greater than zero
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

    /// Returns `true` if `self` is strictly less than zero
    fn is_negative(&self) -> Self::Output { self.lt(&0f32) }
    /// Returns `true` if `self` is strictly greater than zero
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

    /// Returns `true` if `self` is strictly less than zero
    fn is_negative(&self) -> Self::Output { self.lt(&0f64) }
    /// Returns `true` if `self` is strictly greater than zero
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
impl Approx<f32> for AbsoluteEpsilon<f32> {
    /// Returns true if `lhs` is approximately `rhs`.
    fn approx(&self, lhs: f32, rhs: f32) -> bool {
        f32::abs(lhs - rhs) <= self.0
    }
}

/// Types that can be used to compare to zero efficiently.
impl ApproxZero<f32> for AbsoluteEpsilon<f32> {
    /// Returns true if `a` is approximately zero.
    fn approx_zero(&self, a: f32) -> bool {
        f32::abs(a) <= self.0
    }
    /// Returns true if `a/b` is approximately zero without doing a division.
    fn approx_zero_ratio(&self, a: f32, b: f32) -> bool {
        f32::abs(a) <= f32::abs(b) * self.0
    }
}

/// Types that can be used with compared to be approximate equal.
impl Approx<f64> for AbsoluteEpsilon<f64> {
    fn approx(&self, lhs: f64, rhs: f64) -> bool {
        f64::abs(lhs - rhs) <= self.0
    }
}

/// Types that can be used to compare to zero efficiently.
impl ApproxZero<f64> for AbsoluteEpsilon<f64> {
    /// Returns true if `a` is approximately zero.
    fn approx_zero(&self, a: f64) -> bool {
        f64::abs(a) <= self.0
    }
    /// Returns true if `a/b` is approximately zero without doing a division.
    fn approx_zero_ratio(&self, a: f64, b: f64) -> bool {
        f64::abs(a) <= f64::abs(b) * self.0
    }
}
