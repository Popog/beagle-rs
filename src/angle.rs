//! 32-bit and 64-bit type-safe angle units

use std::{f64,f32};
use std::ops::{
    Neg,
    Add,Div,Mul,Rem,Sub,
    AddAssign,DivAssign,MulAssign,RemAssign,SubAssign,
};

/// Types that can be used with trigonometric functions.
pub trait Angle: Sized {
    /// The underlying type of the angle.
    type Type : Into<Self>;

    /// The numerical value representing full turn.
    const TURN: Self::Type;

    /// Construct an angle from a numerical type
    fn from_value<V: Into<Self::Type>>(v: V) -> Self;

    /// Deconstruct an angle into a numerical value
    fn into_value<V>(self) -> V
    where Self::Type: Into<V>;

    /// Computes the sine of a number.
    #[inline]
    fn sin(self) -> Self::Type;
    /// Computes the cosine of a number.
    #[inline]
    fn cos(self) -> Self::Type;
    /// Computes the tangent of a number.
    #[inline]
    fn tan(self) -> Self::Type;
    /// Simultaneously computes the sine and cosine of the number, `x`. Returns `(sin(x), cos(x))`.
    #[inline]
    fn sin_cos(self) -> (Self::Type, Self::Type);

    /// Computes the arcsine of a number. Return value is in the range [-TURN/4, TURN/4] or NaN if
    /// the number is outside the range [-1, 1].
    #[inline]
    fn asin(s: Self::Type) -> Self;
    /// Computes the arccosine of a number. Return value is in the range [0, TURN/2] or NaN if the
    /// number is outside the range [-1, 1].
    #[inline]
    fn acos(s: Self::Type) -> Self;
    /// Computes the arctangent of a number. Return value is in the range [-TURN/4, TURN/4].
    #[inline]
    fn atan(s: Self::Type) -> Self;
    /// Computes the four quadrant arctangent of `y` and `x`.
    #[inline]
    fn atan2(y: Self::Type, x: Self::Type) -> Self;
}

macro_rules! impl_angle_binop {
    ($float:ty, $trait_name:ident::$method_name:ident for $id:ty = $output:ty ) => {
        impl_angle_binop!{$float, $trait_name::$method_name<$id> for $id = $output}
    };
    ($float:ty, $trait_name:ident::$method_name:ident<$rhs:ty> for $id:ty = $output:ty) => {
        impl $trait_name<$rhs> for $id {
            type Output = $output;
            fn $method_name(self, rhs: $rhs) -> $output {
                let lhs: $float = self.into();
                let rhs: $float = rhs.into();
                $trait_name::$method_name(lhs, rhs).into()
            }
        }
    }
}

macro_rules! impl_angle_binop_assign {
    ($trait_name:ident::$method_name:ident for $id:ident) => {
        impl_angle_binop_assign!{$trait_name::$method_name<$id> for $id}
    };
    ($trait_name:ident::$method_name:ident<$rhs:ty> for $id:ident) => {
        impl $trait_name<$rhs> for $id {
            fn $method_name(&mut self, rhs: $rhs) {
                let rhs: <$id as Angle>::Type = rhs.into();
                $trait_name::$method_name(&mut self.0, rhs)
            }
        }
    };
}

macro_rules! impl_angle {
    ($id:ident: $float:ty, $size:tt, $rad:ident, $pi:expr) => {
impl From<$float> for $id {  fn from(v: $float) -> Self { $id(v) }  }
impl From<$id> for $float {  fn from(v: $id) -> Self { v.0 }  }

impl Angle for $id {
    type Type = $float;

    const TURN: $float = $pi;

    #[inline]
    fn from_value<V: Into<Self::Type>>(v: V) -> Self {
        $id(v.into())
    }

    #[inline]
    fn into_value<V>(self) -> V
    where Self::Type: Into<V> {
        self.0.into()
    }

    #[inline]
    fn sin(self) -> Self::Type { $rad::from(self).0.sin() }
    #[inline]
    fn cos(self) -> Self::Type { $rad::from(self).0.cos() }
    #[inline]
    fn tan(self) -> Self::Type { $rad::from(self).0.tan() }
    #[inline]
    fn sin_cos(self) -> (Self::Type, Self::Type) { $rad::from(self).0.sin_cos() }

    #[inline]
    fn asin(s: Self::Type) -> Self { $rad(s.asin()).into() }
    #[inline]
    fn acos(s: Self::Type) -> Self { $rad(s.acos()).into() }
    #[inline]
    fn atan(s: Self::Type) -> Self { $rad(s.atan()).into() }
    #[inline]
    fn atan2(y: Self::Type, x: Self::Type) -> Self { $rad(y.atan2(x)).into() }
}

impl Neg for $id {
    type Output = $id;
    fn neg(self) -> Self::Output { $id(Neg::neg(self.0)) }
}

impl_angle_binop!{$float, Add::add for $id = $id} // A + A = A

impl_angle_binop!{$float, Div::div for $id = $float} // A / A = Float
impl_angle_binop!{$float, Div::div<$float> for $id = $id} // A / Float = A

impl_angle_binop!{$float, Mul::mul<$float> for $id = $id} // A * Float = A
impl_angle_binop!{$float, Mul::mul<$id> for $float = $id} // Float * A = A

impl_angle_binop!{$float, Rem::rem for $id = $id} // A % A = A

impl_angle_binop!{$float, Sub::sub for $id = $id} // A - A = A

impl_angle_binop_assign!{ AddAssign::add_assign for $id }
impl_angle_binop_assign!{ MulAssign::mul_assign<$float> for $id }
impl_angle_binop_assign!{ DivAssign::div_assign<$float> for $id }
impl_angle_binop_assign!{ RemAssign::rem_assign for $id }
impl_angle_binop_assign!{ SubAssign::sub_assign for $id }
    };
}

/// A 32-bit floating point angle in radians
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Rad32(f32);
impl_angle!{Rad32: f32, 32, Rad32, f32::consts::PI}

/// A 64-bit floating point angle in radians
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Rad64(f64);
impl_angle!{Rad64: f64, 64, Rad64, f64::consts::PI}

/// A 32-bit floating point angle in degrees
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Deg32(f32);
impl_angle!{Deg32: f32, 32, Rad32, 360f32}

/// A 64-bit floating point angle in degrees
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Deg64(f64);
impl_angle!{Deg64: f64, 64, Rad64, 360f64}

impl From<Deg32> for Rad32 {  fn from(v: Deg32) -> Self { Rad32(v.0.to_radians()) }  }

impl From<Rad32> for Rad64 {  fn from(v: Rad32) -> Self { Rad64(v.0.into()) }  }
impl From<Deg32> for Rad64 {  fn from(v: Deg32) -> Self { Rad64((v.0 as f64).to_radians()) }  }
impl From<Deg64> for Rad64 {  fn from(v: Deg64) -> Self { Rad64(v.0.to_radians()) }  }

impl From<Rad32> for Deg32 {  fn from(v: Rad32) -> Self { Deg32(v.0.to_degrees()) }  }

impl From<Deg32> for Deg64 {  fn from(v: Deg32) -> Self { Deg64(v.0.into()) }  }
impl From<Rad32> for Deg64 {  fn from(v: Rad32) -> Self { Deg64((v.0 as f64).to_degrees()) }  }
impl From<Rad64> for Deg64 {  fn from(v: Rad64) -> Self { Deg64(v.0.to_degrees()) }  }
