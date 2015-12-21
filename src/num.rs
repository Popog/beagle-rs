//! A collection of numeric traits and functions.

use std::ops::Mul;

/// Types that can be square-rooted.
pub trait Sqrt {
    /// Takes the square root of a number.
    ///
    /// Returns NaN if self is a negative number.
    fn sqrt(&self) -> Self;
}

impl Sqrt for f32 {  fn sqrt(&self) -> Self { f32::sqrt(*self) }  }
impl Sqrt for f64 {  fn sqrt(&self) -> Self { f64::sqrt(*self) }  }

/// Types that can be negative.
pub trait IsNegative {
    fn is_negative(&self) -> bool;
}

impl IsNegative for f32 {  fn is_negative(&self) -> bool { self.is_sign_negative() }  }
impl IsNegative for f64 {  fn is_negative(&self) -> bool { self.is_sign_negative() }  }

/// Types that can be used with compared to be approximate equal.
pub trait Approx<T> {
    /// Returns true if `lhs` is approximately `rhs`.
    fn approx(&self, lhs: &T, rhs: &T) -> bool;
}

/// Types that can be used to compare to zero efficiently.
pub trait ApproxZero<T> {
    /// Returns true if `a` is approximately zero.
    fn approx_zero(&self, a: &T) -> bool;

    /// Returns true if `a/b` is approximately zero.
    fn approx_zero_ratio(&self, a: &T, b: &T) -> bool;
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
    fn approx(&self, lhs: &f32, rhs: &f32) -> bool {
        f32::abs(lhs - rhs) <= self.0
    }
}

/// Types that can be used to compare to zero efficiently.
impl ApproxZero<f32> for AbsoluteEpsilon<f32> {
    /// Returns true if `a` is approximately zero.
    fn approx_zero(&self, &a: &f32) -> bool {
        f32::abs(a) <= self.0
    }
    /// Returns true if `a/b` is approximately zero without doing a division.
    fn approx_zero_ratio(&self, &a: &f32, &b: &f32) -> bool {
        f32::abs(a) <= f32::abs(b) * self.0
    }
}

/// Types that can be used with compared to be approximate equal.
impl Approx<f64> for AbsoluteEpsilon<f64> {
    fn approx(&self, lhs: &f64, rhs: &f64) -> bool {
        f64::abs(lhs - rhs) <= self.0
    }
}

/// Types that can be used to compare to zero efficiently.
impl ApproxZero<f64> for AbsoluteEpsilon<f64> {
    /// Returns true if `a` is approximately zero.
    fn approx_zero(&self, &a: &f64) -> bool {
        f64::abs(a) <= self.0
    }
    /// Returns true if `a/b` is approximately zero without doing a division.
    fn approx_zero_ratio(&self, &a: &f64, &b: &f64) -> bool {
        f64::abs(a) <= f64::abs(b) * self.0
    }
}
