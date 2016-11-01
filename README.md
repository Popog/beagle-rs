# beagle-rs

[![Build Status](https://travis-ci.org/Popog/beagle-rs.svg?branch=master)](https://travis-ci.org/Popog/beagle-rs)
[![Version](https://img.shields.io/crates/v/beagle.svg)](https://crates.io/crates/beagle)
[![License](https://img.shields.io/crates/l/beagle.svg)](https://github.com/Popog/beagle-rs/blob/gh-pages/LICENSE)
[![Downloads](https://img.shields.io/crates/d/beagle.svg)](https://crates.io/crates/beagle)

[Documentation](http://Popog.github.io/beagle-rs)

A basic linear algebra library for computer graphics. 🐶

Beagle is mostly inspired by GLSL (however Beagle is row-major) and attempts to recreate the majority of its functionality in Rust.

## FAQ

### Why should I use beagle?

Beagle provides generic matrix and vector types up to 4x4 size. It provides all the operators you'd expect, including between vectors/matrices and scalars (though scalars must be wrapped with the `v` function).

Beagle also provides operations that function generically on both matrices and vectors (e.g component-wise comparisons, component-wise square roots, component summation).

Beagle makes it very easy to design your own custom component-wise functions.

Beagle also provides swizzles via the `Index` operator.

### This crate requires `nightly`, why?

Sadly yes, for right now `nightly` is required. The things used are:
* `associated_consts`
* `advanced_slice_patterns`

I'd be willing to give up `associated_consts`, but `advanced_slice_patterns` are needed, so in for a penny, in for a pound.

### Seems like you've got a lot of `unsafe` in there, Should I be concerned?

Yes, please voice your concern on rust-lang/rust#37302 so I can remove most of the `unsafe` code.

After that, the next thing blocking `unsafe` removal is the lack of `IndexGet`/`IndexAssign` for swizzles. `unsafe` is used to get slightly around this for some cases, but it's a bit of a hack until rust-lang/rfcs#997 goes anywhere.

There are a few `unsafe` things that won't be going away, however.
* The fast inverse square root floating point hacking.
* Cast back and forth between `&[T; 4]` and `&Vec4<T>`.

### What's next?

* Swizzles on swizzles
* More component-wise functions.
* More tests.

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
