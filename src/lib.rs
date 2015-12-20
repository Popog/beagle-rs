//! A basic linear algebra library for computer graphics.
//!
//! It uses generics to provide extensible square and rectangular Matrix types of sizes 1x1 through
//! 4x4 as well as Vector types of size 1 through 4.

#![feature(float_extras)]

pub mod angle;
pub mod index;
pub mod mat;
pub mod scalar_array;
pub mod vec;
pub mod num;
