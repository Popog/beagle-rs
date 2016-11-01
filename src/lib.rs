//! A basic linear algebra library for computer graphics.
//!
//! It uses generics to provide extensible square and rectangular Matrix types of sizes 1x1 through
//! 4x4 as well as Vector types of size 1 through 4.
#![feature(associated_consts)]
#![feature(slice_patterns)]
#![feature(advanced_slice_patterns)]

// TODO: More comments

pub mod angle;
pub mod consts;
pub mod index;
pub mod mat;
pub mod scalar_array;
pub mod vec;
pub mod num;

mod utils;

/// A simple wrapper to allow generic scalar binary ops
///
/// There's generally no reason to store this type, just create it with `v` for use in expressions.
pub struct Value<S>(pub S);

/// An function to produce Value which is short to type
pub fn v<S>(s: S) -> Value<S> { Value(s) }
