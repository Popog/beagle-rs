# beagle-rs

[![Build Status](https://travis-ci.org/Popog/beagle-rs.svg?branch=master)](https://travis-ci.org/Popog/beagle-rs)
[![Version](https://img.shields.io/crates/v/beagle.svg)](https://crates.io/crates/beagle)
![License](https://img.shields.io/crates/l/beagle.svg)
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

### Swizzles? What are those?

Swizzles are a nice way of rearranging the elements of a vector. For example, given the vector `a`:
```rust
use beagle::vec::*;
use beagle::index::swizzle:xyzw::*;

let a = Vec4::new([3, 5, 7, 11]);
```

We can swizzle it to produce a subvector of just the `X` and `Y` components via `a[XY]`. We can even modify `a` through this.
```rust
a[XY] += Vec2::new(1, 3);
assert_eq!(a, Vec4::new([4, 8, 7, 11]));
```

Swizzles can even be used to duplicate components:
```rust
let r = a[ZZZZ] + Vec4::new([1, 2, 3, 4]);
assert_eq!(r, Vec4::new([12, 13, 14, 15]));
```

Like glsl, swizzles with multiple copies of a sinlg ecomponent cannot be modified:
```rust
a[ZZ] += Vec2::new(1, 3); // error
```

Also, rust's lack of `IndexGet`/`IndexAssign` traits impose some limitations. Only sequential swizzles like `XY` or `YZW` result in actual `Vec` objects, and thus only they can be directly assigned to. Other swizzles are currently relegated to returning reference objects.

This means you can't never assign to non-sequential swizzles (except through use of `unsafe`, but just don't do that). Also when assigning to `Vec` objects (either simple objects or the result of sequential swizzles), you cannot directly use non-sequential swizzles, as they are not direct Vec objects.
```rust
a[YX] = Vec2::new(1, 2); // error: non-sequential LHS
a[XY] = b[YX];           // error: non-sequential RHS
a[XY] = b[YX].into();    // works
a[XY] = &b[YX] + v(0);   // works
```

Note the `&` in the last example. This is a another limitation of non-sequential swizzles. As they are reference objects, and the `Index` operator automatically derefernces in most situations, we must explicitly add the reference back.

As in glsl, you are free to swizzle the result of a swizzle. Note that this can result in what amounts to a sequential swizzle, and as such the result will be a `Vec` object.
```rust
a[YX][YX] = Vec2::new(1, 2);
```
I literally have no idea why you would ever do this, but you can.

Also as in glsl, since the result of indexing a matrix is a `Vec` object, you are free to swizzle that:
```rust
m[0][XY] = Vec2::new(1, 2);
```

### This crate requires `nightly`, why?

Sadly yes, for right now `nightly` is required. The things used are:
* `associated_consts`
* `advanced_slice_patterns`

I'd be willing to give up `associated_consts` in favor of static functions for the moment, but `advanced_slice_patterns` are absolutely necessary, so in for a penny, in for a pound.

### Seems like you've got a lot of `unsafe` in there, Should I be concerned?

Yes, please voice your concern on [rust-lang/rust#37302](https://github.com/rust-lang/rust/issues/37302) so I can remove most of the `unsafe` code.

After that, the next thing blocking `unsafe` removal is the lack of `IndexGet`/`IndexAssign` for swizzles. `unsafe` is used to get slightly around this for some cases, but it's a bit of a hack until [rust-lang/rfcs#997](https://github.com/rust-lang/rfcs/issues/997) goes anywhere.

There are a few `unsafe` things that won't be going away, however.
* The `FloatTransmute` implementations.
* Casting back and forth between references to `[T; 4]`, `CustomArrayFour<T>`, and `Vec4<T>`.

### What's next?

* More tests.
* More documentation.

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
