#![feature(test)]

extern crate beagle;
extern crate test;

use self::test::Bencher;

use beagle::vec::Vec3;
use beagle::mat::Mat3x3;
use beagle::vec::{faceforward, distance2};

#[bench]
fn bench_add_mat3s(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Mat3x3<f64> = Mat3x3::from([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a;
        }
        a
    });
}

#[bench]
fn bench_mul_mat3s(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Mat3x3<f64> = Mat3x3::from([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a*a;
        }
        a
    });
}

#[bench]
fn bench_mul_vec_mat3s(b: &mut Bencher) {
    b.iter(|| {
        let a: Mat3x3<f64> = Mat3x3::from([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        let mut b: Vec3<f64> = Vec3::from([1.0, 2.0, 3.0]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            b = a*b;
        }
        b
    });
}

#[bench]
fn bench_add_f64(b: &mut Bencher) {
    b.iter(|| {
        let mut a: f64 = 1.0;

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a;
        }
        a
    });
}

#[bench]
fn bench_faceforward(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f64> = Vec3::from([1.0, 2.0, 3.0]);
        let b: Vec3<f64> = Vec3::from([1.0, 2.0, 3.0]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = faceforward(a, &a, &b);
        }
        a
    });
}

#[bench]
fn bench_distance2(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f64> = Vec3::from([1.0, 2.0, 5.0]);
        let b: Vec3<f64> = Vec3::from([1.0, 2.0, 3.0]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a * distance2(a, &b);
        }
        a
    });
}
