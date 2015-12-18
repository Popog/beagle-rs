
//! 32-bit and 64-bit type-safe angle units

use std::{f64,f32};

/// Types that can be used with trigonometric functions
pub trait Angle: Sized {
    /// The underlying type of the angle
    type Type : Into<Self>;

    // TODO: Convert to associated constant. Blocked by rust-lang/rust#30396
    // const TURN: Self::Type;
    /// The numerical value representing full turn
    #[inline] fn turn() -> Self::Type;

    /// Computes the sine of a number.
    #[inline] fn sin(self) -> Self::Type;
    /// Computes the cosine of a number
    #[inline] fn cos(self) -> Self::Type;
    /// Computes the tangent of a number
    #[inline] fn tan(self) -> Self::Type;
    /// Simultaneously computes the sine and cosine of the number, `x`. Returns `(sin(x), cos(x))`.
    #[inline] fn sin_cos(self) -> (Self::Type, Self::Type);

    /// Computes the arcsine of a number. Return value is in the range [-TURN/4, TURN/4] or NaN if
    /// the number is outside the range [-1, 1].
    #[inline] fn asin(s: Self::Type) -> Self;
    /// Computes the arccosine of a number. Return value is in the range [0, TURN/2] or NaN if the
    /// number is outside the range [-1, 1].
    #[inline] fn acos(s: Self::Type) -> Self;
    /// Computes the arctangent of a number. Return value is in the range [-TURN/4, TURN/4];
    #[inline] fn atan(s: Self::Type) -> Self;
    /// Computes the four quadrant arctangent of `y` and `x`.
    #[inline] fn atan2(y: Self::Type, x: Self::Type) -> Self;
}

impl From<Rad64> for Rad32 {  fn from(v: Rad64) -> Self { Rad32(v.into()) }  }
impl From<Deg32> for Rad32 {  fn from(v: Deg32) -> Self { Rad32(v.0.to_radians()) }  }
impl From<Deg64> for Rad32 {  fn from(v: Deg64) -> Self { Rad32(v.0.to_radians() as f32) }  }

impl From<Rad32> for Rad64 {  fn from(v: Rad32) -> Self { Rad64(v.into()) }  }
impl From<Deg32> for Rad64 {  fn from(v: Deg32) -> Self { Rad64(f64::from(v).to_radians()) }  }
impl From<Deg64> for Rad64 {  fn from(v: Deg64) -> Self { Rad64(v.0.to_radians()) }  }

impl From<Deg64> for Deg32 {  fn from(v: Deg64) -> Self { Deg32(v.into()) }  }
impl From<Rad32> for Deg32 {  fn from(v: Rad32) -> Self { Deg32(v.0.to_degrees()) }  }
impl From<Rad64> for Deg32 {  fn from(v: Rad64) -> Self { Deg32(v.0.to_degrees() as f32) }  }

impl From<Deg32> for Deg64 {  fn from(v: Deg32) -> Self { Deg64(v.into()) }  }
impl From<Rad32> for Deg64 {  fn from(v: Rad32) -> Self { Deg64(f64::from(v).to_degrees()) }  }
impl From<Rad64> for Deg64 {  fn from(v: Rad64) -> Self { Deg64(v.0.to_degrees()) }  }

include!(concat!(env!("OUT_DIR"), "/angle.rs"));
