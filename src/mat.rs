use std::borrow::{Borrow, BorrowMut};
use std::cmp::Ordering;
use std::ops::{Index,IndexMut};
use std::slice::{Iter,IterMut};

use traits::{Scalar,Dim, ScalarArray,Cast, ComponentPartialEq,ComponentEq,ComponentPartialOrd,ComponentOrd};
use vec::Vec;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mat<R, C, T: Scalar> (R::Output) where C: Dim<T>, R: Dim<Vec<C, T>>;

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> ScalarArray for Mat<R, C, T> {
    type Scalar = T;
    type Type = Vec<C, T>;
    type Dim = R;

    #[inline(always)]
    fn new(v: R::Output) -> Self { Mat(v) }
    #[inline(always)]
    fn iter(&self) -> Iter<Vec<C, T>> { (self.as_ref() as &[Vec<C, T>]).iter() }
    #[inline(always)]
    fn iter_mut(&mut self) -> IterMut<Vec<C, T>> { (self.as_mut() as &mut [Vec<C, T>]).iter_mut() }
    #[inline(always)]
    fn from_value(v: T) -> Self { Mat(<R as Dim<Vec<C, T>>>::from_value(Vec::from_value(v))) }
}

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Mat<R, C, T> {
    #[inline(always)]
    fn from_2d_array<V: Clone+Into<Vec<C, T>>>(v: &[V]) -> Self {
        Mat(<R as Dim<Vec<C, T>>>::from_iter(v.into_iter().map(|v| (*v).clone().into())))
    }
}

impl <T: Scalar,  U: Scalar, C: Dim<T>+Dim<U>, R: Dim<Vec<C, T>>+Dim<Vec<C, U>>> Cast<U> for Mat<R, C, T> {
    type Output = Mat<R, C, U>;

    #[inline(always)]
    fn fold<F: Fn(U, &<Self as ScalarArray>::Scalar)->U>(&self, default: U, f: F) -> U {
        self.iter().fold(default, |acc, row| Cast::<U>::fold(row, acc, &f))
    }

    #[inline(always)]
    fn unary<F: Fn(&<Self as ScalarArray>::Scalar)->U>(&self, f: F) -> Self::Output {
        Mat::new(<R as Dim<Vec<C, U>>>::from_iter(self.iter().map(|s| Cast::<U>::unary(s, &f))))
    }

    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &<Self as ScalarArray>::Scalar)->U>(&self, rhs: &Self, f: F) -> Self::Output {
        Mat::new(<R as Dim<Vec<C, U>>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| Cast::<U>::binary(l, r, &f))))
    }
}

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Mat<R, C, T>
where R: Dim<T>,
C: Dim<Vec<R,T>> {
    pub fn transpose(&self) -> Mat<C, R, T> {
        Mat(<C as Dim<Vec<R, T>>>::from_iter(self[0].iter().enumerate().map(
            |(c, _)|
            Vec::new(<R as Dim<T>>::from_iter(self.iter().map(|row| row[c])))
        )))
    }
}

impl <T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>+Dim<T>> Mat<R, C, T> {
    #[inline(always)]
    fn mul_vector(&self, rhs: &Vec<C, T>) -> Vec<R, T> {
        Vec::new(<R as Dim<T>>::from_iter(self.iter().map(|lhs_row| lhs_row.dot(rhs))))
    }

    #[inline(always)]
    fn mul_vector_transpose(&self, rhs: &Vec<R, T>) -> Vec<C, T> {
        Vec::new(<C as Dim<T>>::from_iter(self[0].iter().enumerate().map(
            |(c, _)|
            self.iter().zip(rhs.iter()).fold(Default::default(), |sum, (lhs_row, &rhs_value)| sum + lhs_row[c] * rhs_value)
        )))
    }
}

impl <T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>> Mat<R, C, T> {
    #[inline(always)]
    fn mul_matrix<C2: Dim<T>>(&self, rhs: &Mat<C, C2, T>) -> Mat<R, C2, T>
    where R: Dim<Vec<C2,T>>,
    C: Dim<Vec<C2,T>> {
        Mat(<R as Dim<Vec<C2, T>>>::from_iter(self.iter().map(|lhs_row| rhs.mul_vector_transpose(lhs_row))))
    }
}

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Deref for Mat<R, C, T> {
    type Target = R::Output;
    #[inline] fn deref<'a>(&'a self) -> &'a Self::Target { &self.0 }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> DerefMut for Mat<R, C, T> {
    #[inline] fn deref_mut<'a>(&'a mut self) -> &'a mut <Self as Deref>::Target { &mut self.0 }
}

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Borrow   <[Vec<C, T>]> for Mat<R, C, T> {  #[inline(always)] fn borrow    (&    self) -> &    [Vec<C, T>] { self.0.borrow() }  }
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> BorrowMut<[Vec<C, T>]> for Mat<R, C, T> {  #[inline(always)] fn borrow_mut(&mut self) -> &mut [Vec<C, T>] { self.0.borrow_mut() }  }

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> AsRef<[Vec<C, T>]> for Mat<R, C, T> {  #[inline(always)] fn as_ref(&    self) -> &    [Vec<C, T>] { self.0.as_ref() }  }
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> AsMut<[Vec<C, T>]> for Mat<R, C, T> {  #[inline(always)] fn as_mut(&mut self) -> &mut [Vec<C, T>] { self.0.as_mut() }  }

impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> Index<usize> for Mat<R, C, T> {
    type Output = Vec<C, T>;
    #[inline(always)]
    fn index(&self, i: usize) -> &Self::Output { &(self.as_ref() as &[Vec<C, T>])[i] }
}
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IndexMut<usize> for Mat<R, C, T> {
    #[inline(always)]
    fn index_mut(&mut self, i: usize) -> &mut Self::Output { &mut (self.as_mut() as &mut [Vec<C, T>])[i] }
}

impl<'a, T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IntoIterator for &'a Mat<R, C, T> {
    type Item = &'a Vec<C, T>;
    type IntoIter = Iter<'a, Vec<C, T>>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
impl<'a, T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> IntoIterator for &'a mut Mat<R, C, T> {
    type Item = &'a mut Vec<C, T>;
    type IntoIter = IterMut<'a, Vec<C, T>>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

// Do matrix * matrix
impl <T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C2: Dim<T>, C: Dim<T>+Dim<Vec<C2,T>>, R: Dim<Vec<C, T>>+Dim<Vec<C2,T>>> Mul<Mat<C, C2,T>> for Mat<R, C, T> {
    type Output = Mat<R, C2, T>;
    fn mul(self, rhs: Mat<C, C2, T>) -> Self::Output { Mul::mul(&self, &rhs) }
}
impl <'t, T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C2: Dim<T>, C: Dim<T>+Dim<Vec<C2,T>>, R: Dim<Vec<C, T>>+Dim<Vec<C2,T>>> Mul<Mat<C, C2, T>> for &'t Mat<R, C, T> {
    type Output = Mat<R, C2, T>;
    fn mul(self, rhs: Mat<C, C2, T>) -> Self::Output { Mul::mul(self, &rhs) }
}
impl <'r, T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C2: Dim<T>, C: Dim<T>+Dim<Vec<C2,T>>, R: Dim<Vec<C, T>>+Dim<Vec<C2,T>>> Mul<&'r Mat<C, C2, T>> for Mat<R, C, T> {
    type Output = Mat<R, C2, T>;
    fn mul(self, rhs: &'r Mat<C, C2, T>) -> Self::Output { Mul::mul(&self, rhs) }
}
impl <'t, 'r, T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C2: Dim<T>, C: Dim<T>+Dim<Vec<C2,T>>, R: Dim<Vec<C, T>>+Dim<Vec<C2,T>>> Mul<&'r Mat<C, C2, T>> for &'t Mat<R, C, T> {
    type Output = Mat<R, C2, T>;
    fn mul(self, rhs: &'r Mat<C, C2, T>) -> Self::Output { self.mul_matrix(rhs) }
}

// Do matrix * vector
impl <T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>+Dim<T>> Mul<Vec<C, T>> for Mat<R, C, T> {
    type Output = Vec<R, T>;
    fn mul(self, rhs: Vec<C, T>) -> Self::Output { Mul::mul(&self, &rhs) }
}
impl <'t, T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>+Dim<T>> Mul<Vec<C, T>> for &'t Mat<R, C, T> {
    type Output = Vec<R, T>;
    fn mul(self, rhs: Vec<C, T>) -> Self::Output { Mul::mul(self, &rhs) }
}
impl <'r, T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>+Dim<T>> Mul<&'r Vec<C, T>> for Mat<R, C, T> {
    type Output = Vec<R, T>;
    fn mul(self, rhs: &'r Vec<C, T>) -> Self::Output { Mul::mul(&self, rhs) }
}
impl <'t, 'r, T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>+Dim<T>> Mul<&'r Vec<C, T>> for &'t Mat<R, C, T> {
    type Output = Vec<R, T>;
    fn mul(self, rhs: &'r Vec<C, T>) -> Self::Output { self.mul_vector(rhs) }
}

// Do vector * matrix
impl <T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>+Dim<T>> Mul<Mat<R, C, T>> for Vec<R, T> {
    type Output = Vec<C, T>;
    fn mul(self, rhs: Mat<R, C, T>) -> Self::Output { Mul::mul(&self, &rhs) }
}
impl <'t, T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>+Dim<T>> Mul<Mat<R, C, T>> for &'t Vec<R, T> {
    type Output = Vec<C, T>;
    fn mul(self, rhs: Mat<R, C, T>) -> Self::Output { Mul::mul(self, &rhs) }
}
impl <'r, T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>+Dim<T>> Mul<&'r Mat<R, C, T>> for Vec<R, T> {
    type Output = Vec<C, T>;
    fn mul(self, rhs: &'r Mat<R, C, T>) -> Self::Output { Mul::mul(&self, rhs) }
}
impl <'t, 'r, T: Scalar+Default+Add<Output=T>+Mul<Output=T>, C: Dim<T>, R: Dim<Vec<C, T>>+Dim<T>> Mul<&'r Mat<R, C, T>> for &'t Vec<R, T> {
    type Output = Vec<C, T>;
    fn mul(self, rhs: &'r Mat<R, C, T>) -> Self::Output { rhs.mul_vector_transpose(self) }
}

impl <T: Scalar+PartialEq, C: Dim<T>+Dim<bool>, R: Dim<Vec<C, T>>+Dim<Vec<C, bool>>> ComponentPartialEq for Mat<R, C, T> {}
impl <T: Scalar+Eq, C: Dim<T>+Dim<bool>, R: Dim<Vec<C, T>>+Dim<Vec<C, bool>>> ComponentEq for Mat<R, C, T> {}
impl <T: Scalar+PartialOrd, C: Dim<T>+Dim<bool>+Dim<Option<Ordering>>, R: Dim<Vec<C, T>>+Dim<Vec<C, bool>>+Dim<Vec<C, Option<Ordering>>>> ComponentPartialOrd for Mat<R, C, T> {}
impl <T: Scalar+Ord, C: Dim<T>+Dim<bool>+Dim<Option<Ordering>>+Dim<Ordering>, R: Dim<Vec<C, T>>+Dim<Vec<C, bool>>+Dim<Vec<C, Option<Ordering>>>+Dim<Vec<C, Ordering>>> ComponentOrd for Mat<R, C, T> {}


include!(concat!(env!("OUT_DIR"), "/mat.rs"));

#[cfg(test)]
mod tests {
    use super::*;
    use traits::ScalarArray;

    #[test]
    fn test_neg() {
        assert_eq!(-Mat1x1::from([[1f64]]),                   Mat1x1::from([[-1f64]]));
        assert_eq!(-Mat1x2::from([[1f64, 2f64]]),             Mat1x2::from([[-1f64, -2f64]]));
        assert_eq!(-Mat1x3::from([[1f64, 2f64, 3f64]]),       Mat1x3::from([[-1f64, -2f64, -3f64]]));
        assert_eq!(-Mat1x4::from([[1f64, 2f64, 3f64, 4f64]]), Mat1x4::from([[-1f64, -2f64, -3f64, -4f64]]));
    }

    #[test]
    fn test_multiply_zero() {
        assert_eq!(Mat1x1::from_value(0f64)*Mat1x1::from_value(0f64), Mat1x1::from_value(0f64));
        assert_eq!(Mat1x2::from_value(0f64)*Mat2x1::from_value(0f64), Mat1x1::from_value(0f64));
        assert_eq!(Mat1x3::from_value(0f64)*Mat3x1::from_value(0f64), Mat1x1::from_value(0f64));
        assert_eq!(Mat1x4::from_value(0f64)*Mat4x1::from_value(0f64), Mat1x1::from_value(0f64));

        assert_eq!(Mat1x1::from_value(0f64)*Mat1x2::from_value(0f64), Mat1x2::from_value(0f64));
        assert_eq!(Mat1x2::from_value(0f64)*Mat2x2::from_value(0f64), Mat1x2::from_value(0f64));
        assert_eq!(Mat1x3::from_value(0f64)*Mat3x2::from_value(0f64), Mat1x2::from_value(0f64));
        assert_eq!(Mat1x4::from_value(0f64)*Mat4x2::from_value(0f64), Mat1x2::from_value(0f64));

        assert_eq!(Mat1x1::from_value(0f64)*Mat1x3::from_value(0f64), Mat1x3::from_value(0f64));
        assert_eq!(Mat1x2::from_value(0f64)*Mat2x3::from_value(0f64), Mat1x3::from_value(0f64));
        assert_eq!(Mat1x3::from_value(0f64)*Mat3x3::from_value(0f64), Mat1x3::from_value(0f64));
        assert_eq!(Mat1x4::from_value(0f64)*Mat4x3::from_value(0f64), Mat1x3::from_value(0f64));

        assert_eq!(Mat1x1::from_value(0f64)*Mat1x4::from_value(0f64), Mat1x4::from_value(0f64));
        assert_eq!(Mat1x2::from_value(0f64)*Mat2x4::from_value(0f64), Mat1x4::from_value(0f64));
        assert_eq!(Mat1x3::from_value(0f64)*Mat3x4::from_value(0f64), Mat1x4::from_value(0f64));
        assert_eq!(Mat1x4::from_value(0f64)*Mat4x4::from_value(0f64), Mat1x4::from_value(0f64));




        assert_eq!(Mat2x1::from_value(0f64)*Mat1x1::from_value(0f64), Mat2x1::from_value(0f64));
        assert_eq!(Mat2x2::from_value(0f64)*Mat2x1::from_value(0f64), Mat2x1::from_value(0f64));
        assert_eq!(Mat2x3::from_value(0f64)*Mat3x1::from_value(0f64), Mat2x1::from_value(0f64));
        assert_eq!(Mat2x4::from_value(0f64)*Mat4x1::from_value(0f64), Mat2x1::from_value(0f64));

        assert_eq!(Mat2x1::from_value(0f64)*Mat1x2::from_value(0f64), Mat2x2::from_value(0f64));
        assert_eq!(Mat2x2::from_value(0f64)*Mat2x2::from_value(0f64), Mat2x2::from_value(0f64));
        assert_eq!(Mat2x3::from_value(0f64)*Mat3x2::from_value(0f64), Mat2x2::from_value(0f64));
        assert_eq!(Mat2x4::from_value(0f64)*Mat4x2::from_value(0f64), Mat2x2::from_value(0f64));

        assert_eq!(Mat2x1::from_value(0f64)*Mat1x3::from_value(0f64), Mat2x3::from_value(0f64));
        assert_eq!(Mat2x2::from_value(0f64)*Mat2x3::from_value(0f64), Mat2x3::from_value(0f64));
        assert_eq!(Mat2x3::from_value(0f64)*Mat3x3::from_value(0f64), Mat2x3::from_value(0f64));
        assert_eq!(Mat2x4::from_value(0f64)*Mat4x3::from_value(0f64), Mat2x3::from_value(0f64));

        assert_eq!(Mat2x1::from_value(0f64)*Mat1x4::from_value(0f64), Mat2x4::from_value(0f64));
        assert_eq!(Mat2x2::from_value(0f64)*Mat2x4::from_value(0f64), Mat2x4::from_value(0f64));
        assert_eq!(Mat2x3::from_value(0f64)*Mat3x4::from_value(0f64), Mat2x4::from_value(0f64));
        assert_eq!(Mat2x4::from_value(0f64)*Mat4x4::from_value(0f64), Mat2x4::from_value(0f64));




        assert_eq!(Mat3x1::from_value(0f64)*Mat1x1::from_value(0f64), Mat3x1::from_value(0f64));
        assert_eq!(Mat3x2::from_value(0f64)*Mat2x1::from_value(0f64), Mat3x1::from_value(0f64));
        assert_eq!(Mat3x3::from_value(0f64)*Mat3x1::from_value(0f64), Mat3x1::from_value(0f64));
        assert_eq!(Mat3x4::from_value(0f64)*Mat4x1::from_value(0f64), Mat3x1::from_value(0f64));

        assert_eq!(Mat3x1::from_value(0f64)*Mat1x2::from_value(0f64), Mat3x2::from_value(0f64));
        assert_eq!(Mat3x2::from_value(0f64)*Mat2x2::from_value(0f64), Mat3x2::from_value(0f64));
        assert_eq!(Mat3x3::from_value(0f64)*Mat3x2::from_value(0f64), Mat3x2::from_value(0f64));
        assert_eq!(Mat3x4::from_value(0f64)*Mat4x2::from_value(0f64), Mat3x2::from_value(0f64));

        assert_eq!(Mat3x1::from_value(0f64)*Mat1x3::from_value(0f64), Mat3x3::from_value(0f64));
        assert_eq!(Mat3x2::from_value(0f64)*Mat2x3::from_value(0f64), Mat3x3::from_value(0f64));
        assert_eq!(Mat3x3::from_value(0f64)*Mat3x3::from_value(0f64), Mat3x3::from_value(0f64));
        assert_eq!(Mat3x4::from_value(0f64)*Mat4x3::from_value(0f64), Mat3x3::from_value(0f64));

        assert_eq!(Mat3x1::from_value(0f64)*Mat1x4::from_value(0f64), Mat3x4::from_value(0f64));
        assert_eq!(Mat3x2::from_value(0f64)*Mat2x4::from_value(0f64), Mat3x4::from_value(0f64));
        assert_eq!(Mat3x3::from_value(0f64)*Mat3x4::from_value(0f64), Mat3x4::from_value(0f64));
        assert_eq!(Mat3x4::from_value(0f64)*Mat4x4::from_value(0f64), Mat3x4::from_value(0f64));




        assert_eq!(Mat4x1::from_value(0f64)*Mat1x1::from_value(0f64), Mat4x1::from_value(0f64));
        assert_eq!(Mat4x2::from_value(0f64)*Mat2x1::from_value(0f64), Mat4x1::from_value(0f64));
        assert_eq!(Mat4x3::from_value(0f64)*Mat3x1::from_value(0f64), Mat4x1::from_value(0f64));
        assert_eq!(Mat4x4::from_value(0f64)*Mat4x1::from_value(0f64), Mat4x1::from_value(0f64));

        assert_eq!(Mat4x1::from_value(0f64)*Mat1x2::from_value(0f64), Mat4x2::from_value(0f64));
        assert_eq!(Mat4x2::from_value(0f64)*Mat2x2::from_value(0f64), Mat4x2::from_value(0f64));
        assert_eq!(Mat4x3::from_value(0f64)*Mat3x2::from_value(0f64), Mat4x2::from_value(0f64));
        assert_eq!(Mat4x4::from_value(0f64)*Mat4x2::from_value(0f64), Mat4x2::from_value(0f64));

        assert_eq!(Mat4x1::from_value(0f64)*Mat1x3::from_value(0f64), Mat4x3::from_value(0f64));
        assert_eq!(Mat4x2::from_value(0f64)*Mat2x3::from_value(0f64), Mat4x3::from_value(0f64));
        assert_eq!(Mat4x3::from_value(0f64)*Mat3x3::from_value(0f64), Mat4x3::from_value(0f64));
        assert_eq!(Mat4x4::from_value(0f64)*Mat4x3::from_value(0f64), Mat4x3::from_value(0f64));

        assert_eq!(Mat4x1::from_value(0f64)*Mat1x4::from_value(0f64), Mat4x4::from_value(0f64));
        assert_eq!(Mat4x2::from_value(0f64)*Mat2x4::from_value(0f64), Mat4x4::from_value(0f64));
        assert_eq!(Mat4x3::from_value(0f64)*Mat3x4::from_value(0f64), Mat4x4::from_value(0f64));
        assert_eq!(Mat4x4::from_value(0f64)*Mat4x4::from_value(0f64), Mat4x4::from_value(0f64));
    }

    #[test]
    fn test_multiply() {
        let a = Mat3x3::from([[   2f64,    3f64,    5f64], [   7f64,   11f64,   13f64], [  17f64,   19f64,   23f64]]);
        let b = Mat3x3::from([[  29f64,   31f64,   37f64], [  41f64,   43f64,   47f64], [  53f64,   59f64,   61f64]]);
        let c = Mat3x3::from([[ 446f64,  486f64,  520f64], [1343f64, 1457f64, 1569f64], [2491f64, 2701f64, 2925f64]]);
        assert_eq!(a*b, c);
    }
}
