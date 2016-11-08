#![feature(float_extras)]
#![feature(test)]
#![allow(deprecated)]

extern crate beagle;
extern crate test;


use beagle::num::{FloatTransmute, FractionExponent, LoadExponent};
use test::Bencher;

const NUMBER_32: f32 = 14.0;
const NUMBER_64: f64 = 14.0;
const SHIFT: isize = 10;


//    dD     j88D
//   d8'    j8~88
//  d8'    j8' 88
// d8888b. V88888D
// 88' `8D     88
// `8888P      VP

#[bench]
fn bench_frexp64_a(b: &mut Bencher) {
    let a = test::black_box(NUMBER_64);
    b.iter(|| {
        let r = FractionExponent::frexp(a);
        r.0 * r.1 as f64
    });
}
#[bench]
fn bench_frexp64_b(b: &mut Bencher) {
    let a = test::black_box(NUMBER_64);
    b.iter(|| FractionExponent::frexp(a));
}
#[bench]
fn bench_frexp64_c(b: &mut Bencher) {
    let a = test::black_box(1u64.float_transmute());
    b.iter(|| {
        let r = FractionExponent::frexp(a);
        r.0 * r.1 as f64
    });
}
#[bench]
fn bench_frexp64_d(b: &mut Bencher) {
    let a = test::black_box(1u64.float_transmute());
    b.iter(|| FractionExponent::frexp(a));
}

#[bench]
fn bench_ldexp64_a(b: &mut Bencher) {
    let a = test::black_box(NUMBER_64);
    b.iter(|| a + LoadExponent::ldexp(a, SHIFT as i64));
}
#[bench]
fn bench_ldexp64_b(b: &mut Bencher) {
    let a = test::black_box(NUMBER_64);
    b.iter(|| LoadExponent::ldexp(a, SHIFT as i64));
}
#[bench]
fn bench_ldexp64_c(b: &mut Bencher) {
    let a = test::black_box(1u64.float_transmute());
    b.iter(|| a + LoadExponent::ldexp(a, SHIFT as i64));
}
#[bench]
fn bench_ldexp64_d(b: &mut Bencher) {
    let a = test::black_box(1u64.float_transmute());
    b.iter(|| LoadExponent::ldexp(a, SHIFT as i64));
}

// d8888b. .d888b.
// VP  `8D VP  `8D
//   oooY'    odD'
//   ~~~b.  .88'
// db   8D j88.
// Y8888P' 888888D

#[bench]
fn bench_frexp32_a(b: &mut Bencher) {
    let a = test::black_box(NUMBER_32);
    b.iter(|| {
        let r = FractionExponent::frexp(a);
        r.0 * r.1 as f32
    });
}
#[bench]
fn bench_frexp32_b(b: &mut Bencher) {
    let a = test::black_box(NUMBER_32);
    b.iter(|| FractionExponent::frexp(a));
}
#[bench]
fn bench_frexp32_c(b: &mut Bencher) {
    let a = test::black_box(1u32.float_transmute());
    b.iter(|| {
        let r = FractionExponent::frexp(a);
        r.0 * r.1 as f32
    });
}
#[bench]
fn bench_frexp32_d(b: &mut Bencher) {
    let a = test::black_box(1u32.float_transmute());
    b.iter(|| FractionExponent::frexp(a));
}

#[bench]
fn bench_ldexp32_a(b: &mut Bencher) {
    let a = test::black_box(NUMBER_32);
    b.iter(|| a + LoadExponent::ldexp(a, SHIFT as i32));
}
#[bench]
fn bench_ldexp32_b(b: &mut Bencher) {
    let a = test::black_box(NUMBER_32);
    b.iter(|| LoadExponent::ldexp(a, SHIFT as i32));
}
#[bench]
fn bench_ldexp32_c(b: &mut Bencher) {
    let a = test::black_box(1u32.float_transmute());
    b.iter(|| a + LoadExponent::ldexp(a, SHIFT as i32));
}
#[bench]
fn bench_ldexp32_d(b: &mut Bencher) {
    let a = test::black_box(1u32.float_transmute());
    b.iter(|| LoadExponent::ldexp(a, SHIFT as i32));
}

//  .o88b. .88b  d88.  .d8b.  d888888b db   db         dD     j88D
// d8P  Y8 88'YbdP`88 d8' `8b `~~88~~' 88   88        d8'    j8~88
// 8P      88  88  88 88ooo88    88    88ooo88       d8'    j8' 88
// 8b      88  88  88 88~~~88    88    88~~~88      d8888b. V88888D
// Y8b  d8 88  88  88 88   88    88    88   88      88' `8D     88
//  `Y88P' YP  YP  YP YP   YP    YP    YP   YP      `8888P      VP

#[bench]
fn bench_std_frexp64_a(b: &mut Bencher) {
    use std::f64;
    let a = test::black_box(NUMBER_64);
    b.iter(|| {
        let r = f64::frexp(a);
        r.0 * r.1 as f64
    });
}
#[bench]
fn bench_std_frexp64_b(b: &mut Bencher) {
    use std::f64;
    let a = test::black_box(NUMBER_64);
    b.iter(|| f64::frexp(a));
}
#[bench]
fn bench_std_frexp64_c(b: &mut Bencher) {
    use std::f64;
    let a = test::black_box(1u64.float_transmute());
    b.iter(|| {
        let r = f64::frexp(a);
        r.0 * r.1 as f64
    });
}
#[bench]
fn bench_std_frexp64_d(b: &mut Bencher) {
    use std::f64;
    let a = test::black_box(1u64.float_transmute());
    b.iter(|| f64::frexp(a));
}

#[bench]
fn bench_std_ldexp64_a(b: &mut Bencher) {
    use std::f64;
    let a = test::black_box(NUMBER_64);
    b.iter(|| a + f64::ldexp(a, SHIFT as isize));
}
#[bench]
fn bench_std_ldexp64_b(b: &mut Bencher) {
    use std::f64;
    let a = test::black_box(NUMBER_64);
    b.iter(|| f64::ldexp(a, SHIFT as isize));
}
#[bench]
fn bench_std_ldexp64_c(b: &mut Bencher) {
    use std::f64;
    let a = test::black_box(1u64.float_transmute());
    b.iter(|| a + f64::ldexp(a, SHIFT as isize));
}
#[bench]
fn bench_std_ldexp64_d(b: &mut Bencher) {
    use std::f64;
    let a = test::black_box(1u64.float_transmute());
    b.iter(|| f64::ldexp(a, SHIFT as isize));
}


//  .o88b. .88b  d88.  .d8b.  d888888b db   db      d8888b. .d888b.
// d8P  Y8 88'YbdP`88 d8' `8b `~~88~~' 88   88      VP  `8D VP  `8D
// 8P      88  88  88 88ooo88    88    88ooo88        oooY'    odD'
// 8b      88  88  88 88~~~88    88    88~~~88        ~~~b.  .88'
// Y8b  d8 88  88  88 88   88    88    88   88      db   8D j88.
//  `Y88P' YP  YP  YP YP   YP    YP    YP   YP      Y8888P' 888888D

#[bench]
fn bench_std_frexp32_a(b: &mut Bencher) {
    use std::f32;
    let a = test::black_box(NUMBER_32);
    b.iter(|| {
        let r = f32::frexp(a);
        r.0 * r.1 as f32
    });
}
#[bench]
fn bench_std_frexp32_b(b: &mut Bencher) {
    use std::f32;
    let a = test::black_box(NUMBER_32);
    b.iter(|| f32::frexp(a));
}

#[bench]
fn bench_std_frexp32_c(b: &mut Bencher) {
    use std::f32;
    let a = test::black_box(1u32.float_transmute());
    b.iter(|| {
        let r = f32::frexp(a);
        r.0 * r.1 as f32
    });
}
#[bench]
fn bench_std_frexp32_d(b: &mut Bencher) {
    use std::f32;
    let a = test::black_box(1u32.float_transmute());
    b.iter(|| f32::frexp(a));
}

#[bench]
fn bench_std_ldexp32_a(b: &mut Bencher) {
    use std::f32;
    let a = test::black_box(NUMBER_32);
    b.iter(|| a + f32::ldexp(a, SHIFT as isize));
}
#[bench]
fn bench_std_ldexp32_b(b: &mut Bencher) {
    use std::f32;
    let a = test::black_box(NUMBER_32);
    b.iter(|| f32::ldexp(a, SHIFT as isize));
}

#[bench]
fn bench_std_ldexp32_c(b: &mut Bencher) {
    use std::f32;
    let a = test::black_box(1u32.float_transmute());
    b.iter(|| a + f32::ldexp(a, SHIFT as isize));
}
#[bench]
fn bench_std_ldexp32_d(b: &mut Bencher) {
    use std::f32;
    let a = test::black_box(1u32.float_transmute());
    b.iter(|| f32::ldexp(a, SHIFT as isize));
}
