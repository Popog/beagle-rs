//! Indexing and Swizzling for Vec
//!
//! # Examples
//!
//! ```
//! use beagle::vec::{Vec3,Vec2};
//! use beagle::index::xyzw::*;
//!
//! let mut v = Vec3::from([8f32, 12f32, 10f32]);
//! v[YZ] = v[XY] + 1f32;
//! assert_eq!(v, Vec3::from([8f32, 9f32, 13f32]));
//! ```
// TODO: More examples

use vec::{Vec};
use scalar_array::{Scalar,Dim};
use std::mem::transmute;
use std::marker::PhantomData;

// TODO: Make safe. Blocked on rust-lang/rfcs#997
/// This is a temporary structure until something is done with rust-lang/rfcs#997
/// Do not rely on this type, just rely on the overloaded operators defined for the cases where
// it is returned right now.
pub struct VecRef<D, V: Scalar, I>(Vec<D, V>, PhantomData<I>)
where D: Dim<V>;

impl<D, V: Scalar, I> VecRef<D, V, I>
where D: Dim<V> {
    // TODO: Make safe. Blocked on rust-lang/rfcs#997
    /// Constructs a reference from a vector reference.
    #[inline(always)]
    fn from_ref(v: &Vec<D, V>) -> &Self { unsafe { transmute(v) } }
    /// Constructs a mutable reference from a mutable vector reference.
    #[inline(always)]
    fn from_mut(v: &mut Vec<D, V>) -> &mut Self { unsafe { transmute(v) } }

    /// Constructs a vector reference from a reference.
    #[inline(always)]
    fn to_ref(&self) -> &Vec<D, V> { unsafe { transmute(self) } }
    /// Constructs a mutable vector reference from a mutable reference.
    #[inline(always)]
    fn to_mut(&mut self) -> &mut Vec<D, V> { unsafe { transmute(self) } }
}



pub trait Apply<D, V: Scalar>
where D: Dim<V> {
    #[inline(always)]
    fn apply_rhs<U: Scalar, F:FnMut(&mut V, &U)>(&mut self, mut f: F, rhs: &Vec<D, U>)
    where D: Dim<U>;
    #[inline(always)]
    fn apply_lhs<U: Scalar, F:FnMut(&V, &mut U)>(&self, mut f: F, lhs: &mut Vec<D, U>)
    where D: Dim<U>;
}

include!(concat!(env!("OUT_DIR"), "/index.rs"));

mod test {
    pub use super::s0123::{S0 as adf, S1 as ad};
}

#[cfg(test)]
mod tests {
    use super::xyzw::*;
    use vec::Vec4;
    use scalar_array::Construct;

    #[test]
    fn test_vec_single() {
        let v = Vec4::from([1i8 , 2i8 , 3i8 , 4i8 ]);
        assert_eq!(v[X], 1);
    }

    #[test]
    fn test_vec_swizzle() {
        let mut v = Vec4::from([1i8 , 2i8 , 3i8 , 4i8 ]);
        v[ZWYX] += Vec4::from_value(1i8);
        assert_eq!(v, Vec4::from([2i8 , 3i8 , 4i8 , 5i8 ]));
    }


}
