extern crate beagle;

use beagle::scalar_array::transpose;
use beagle::mat::{Mat3x4,Mat4x3};

#[test]
fn transpose_test() {
    let m = Mat4x3::new([
        [ 1f64,  2f64,  3f64],
        [ 4f64,  5f64,  6f64],
        [ 7f64,  8f64,  9f64],
        [10f64, 11f64, 12f64],
    ]);
    let n = Mat3x4::new([
        [ 1f64,  4f64,  7f64, 10f64],
        [ 2f64,  5f64,  8f64, 11f64],
        [ 3f64,  6f64,  9f64, 12f64],
    ]);
    let result = transpose(m);
    assert_eq!(result, n);
}
