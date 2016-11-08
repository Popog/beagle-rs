#![feature(test)]

extern crate beagle;
extern crate test;


use beagle::vec::Vec3;
use beagle::mat::Mat3x3;
//use beagle::vec::{faceforward, distance2};
use beagle::num::{Sqrt, Recip};
use test::Bencher;

#[bench]
fn bench_mul_vec_mat3s(b: &mut Bencher) {
    let m: Mat3x3<f64> = test::black_box(Mat3x3::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]));
    let mut v: Vec3<f64> = test::black_box(Vec3::new([1.0, 2.0, 3.0]));
    b.iter(|| v = m*v );
    test::black_box(v);
}

#[bench]
fn bench_rsqrt1x32_a(b: &mut Bencher) {
    let mut a: f32 = test::black_box(2.0);
    b.iter(|| a += a.sqrt().recip() );
    test::black_box(a);
}

#[bench]
fn bench_rsqrt1x32_b(b: &mut Bencher) {
    let mut a: f32 = test::black_box(2.0);
    b.iter(|| a += a.inverse_sqrt() );
    test::black_box(a);
}

#[bench]
fn bench_rsqrt3x32_a(b: &mut Bencher) {
    let mut a: Vec3<f32> = test::black_box(Vec3::new([1.0, 2.0, 3.0]));
    b.iter(|| a = a+a.sqrt().recip() );
    test::black_box(a);
}

#[bench]
fn bench_rsqrt3x32_b(b: &mut Bencher) {
    let mut a: Vec3<f32> = test::black_box(Vec3::new([1.0, 2.0, 3.0]));
    b.iter(|| a = a+a.inverse_sqrt() );
    test::black_box(a);
}



#[bench]
fn bench_rsqrt1x64_a(b: &mut Bencher) {
    let mut a: f64 = test::black_box(2.0);
    b.iter(|| a += a.sqrt().recip() );
    test::black_box(a);
}

#[bench]
fn bench_rsqrt1x64_b(b: &mut Bencher) {
    let mut a: f64 = test::black_box(2.0);
    b.iter(|| a += a.inverse_sqrt() );
    test::black_box(a);
}


#[bench]
fn bench_rsqrt3x64_a(b: &mut Bencher) {
    let mut a: Vec3<f64> = test::black_box(Vec3::new([1.0, 2.0, 3.0]));
    b.iter(|| a = a+a.sqrt().recip() );
    test::black_box(a);
}

#[bench]
fn bench_rsqrt3x64_b(b: &mut Bencher) {
    let mut a: Vec3<f64> = test::black_box(Vec3::new([1.0, 2.0, 3.0]));
    b.iter(|| a = a+a.inverse_sqrt() );
    test::black_box(a);
}

/*
#[bench]
fn bench_faceforward(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);
        let b: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(ROUNDS);
        for _ in 0..n {
            a = faceforward(a, &a, &b);
        }
        a
    });
}

#[bench]
fn bench_distance2(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f64> = Vec3::new([1.0, 2.0, 5.0]);
        let b: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(ROUNDS);
        for _ in 0..n {
            a = a * distance2(a, &b);
        }
        a
    });
}
*/
