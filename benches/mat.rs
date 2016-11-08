#![feature(test)]

extern crate beagle;
extern crate test;

use beagle::mat::Mat3x3;
use test::Bencher;

#[bench]
fn bench_add_mat3s(b: &mut Bencher) {
    let mut a: Mat3x3<f64> = test::black_box(Mat3x3::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]));

    b.iter(|| a = a+a );
    test::black_box(a);
}

#[bench]
fn bench_mul_mat3s(b: &mut Bencher) {
    let mut a: Mat3x3<f64> = test::black_box(Mat3x3::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ]));

    b.iter(|| a = a*a );
    test::black_box(a);
}
