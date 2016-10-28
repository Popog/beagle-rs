#![feature(test)]

extern crate beagle;
extern crate test;

use beagle::mat::Mat3x3;
use test::Bencher;

#[bench]
fn bench_add_mat3s(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Mat3x3<f64> = Mat3x3::new([
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
        let mut a: Mat3x3<f64> = Mat3x3::new([
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
