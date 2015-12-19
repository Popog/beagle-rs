// Types that can be square-rooted
pub trait Sqrt {
    fn sqrt(&self) -> Self;
}

impl Sqrt for f32 {  fn sqrt(&self) -> Self { f32::sqrt(*self) }  }
impl Sqrt for f64 {  fn sqrt(&self) -> Self { f64::sqrt(*self) }  }

pub trait IsNegative {
    fn is_negative(&self) -> bool;
}

impl IsNegative for f32 {  fn is_negative(&self) -> bool { self.is_sign_negative() }  }
impl IsNegative for f64 {  fn is_negative(&self) -> bool { self.is_sign_negative() }  }

/// Types that can be used with compared to be approximate equal
pub trait Approx<T> {
    /// Returns true if `a/b` is approximately zero without doing a division
    fn approx_zero(&self, a: &T, b: &T) -> bool;

    /// Returns true if `lhs` is approximately `rhs`
    fn approx(&self, lhs: &T, rhs: &T) -> bool;
}

pub trait ApproxSquared<T> : Approx<T> {
    type Output: Approx<T>;
    /// Tighten approximation so that `self.approx_zero(x) == self.squared().approx_zero(x*x)`
    fn squared(&self) -> Self::Output;
}

pub struct AbsoluteEpsilon<F>(pub F);

impl Approx<f32> for AbsoluteEpsilon<f32> {
    fn approx_zero(&self, &a: &f32, &b: &f32) -> bool {
        f32::abs(a) <= f32::abs(b) * self.0
    }
    fn approx(&self, lhs: &f32, rhs: &f32) -> bool {
        f32::abs(lhs - rhs) <= self.0
    }
}

impl Approx<f64> for AbsoluteEpsilon<f64> {
    fn approx_zero(&self, &a: &f64, &b: &f64) -> bool {
        f64::abs(a) <= f64::abs(b) * self.0
    }
    fn approx(&self, lhs: &f64, rhs: &f64) -> bool {
        f64::abs(lhs - rhs) <= self.0
    }
}
