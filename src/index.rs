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

include!(concat!(env!("OUT_DIR"), "/s0123.rs"));
include!(concat!(env!("OUT_DIR"), "/xyzw.rs"));
include!(concat!(env!("OUT_DIR"), "/stpq.rs"));
include!(concat!(env!("OUT_DIR"), "/rgba.rs"));
