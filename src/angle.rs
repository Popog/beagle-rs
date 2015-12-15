
pub trait Angle {
    type Type;
    #[inline] fn sin(self) -> Self::Type;
    #[inline] fn cos(self) -> Self::Type;
    #[inline] fn tan(self) -> Self::Type;
    #[inline] fn sin_cos(self) -> (Self::Type, Self::Type);

    #[inline] fn asin(s: Self::Type) -> Self;
    #[inline] fn acos(s: Self::Type) -> Self;
    #[inline] fn atan(s: Self::Type) -> Self;
    #[inline] fn atan2(a: Self::Type, b: Self::Type) -> Self;
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
