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
    b.iter(|| {
        let a: Mat3x3<f64> = Mat3x3::new([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]);
        let mut b: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            b = a*b;
        }
        b
    });
}

#[bench]
fn bench_rsqrt1x32_a(b: &mut Bencher) {
    b.iter(|| {
        let mut a: f32 = 2.0;

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a.sqrt().recip();
        }
        a
    });
}

#[bench]
fn bench_rsqrt1x32_b(b: &mut Bencher) {
    b.iter(|| {

        let mut a: f32 = 2.0;

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a.inverse_sqrt();
        }
        a
    });
}

#[bench]
fn bench_rsqrt3x32_a(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f32> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a.sqrt().recip();
        }
        a
    });
}

#[bench]
fn bench_rsqrt3x32_b(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f32> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a.inverse_sqrt();
        }
        a
    });
}



#[bench]
fn bench_rsqrt1x64_a(b: &mut Bencher) {
    b.iter(|| {
        let mut a: f64 = 2.0;

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a.sqrt().recip();
        }
        a
    });
}

#[bench]
fn bench_rsqrt1x64_b(b: &mut Bencher) {
    b.iter(|| {

        let mut a: f64 = 2.0;

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a.inverse_sqrt();
        }
        a
    });
}


#[bench]
fn bench_rsqrt3x64_a(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a.sqrt().recip();
        }
        a
    });
}

#[bench]
fn bench_rsqrt3x64_b(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a+a.inverse_sqrt();
        }
        a
    });
}

/*
#[bench]
fn bench_faceforward(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);
        let b: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);

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
        let mut a: Vec3<f64> = Vec3::new([1.0, 2.0, 5.0]);
        let b: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(10_000);
        for _ in 0..n {
            a = a * distance2(a, &b);
        }
        a
    });
}
*/
