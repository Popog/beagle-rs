use std::borrow::{BorrowMut};
use std::cmp::Ordering;
use std::slice::{Iter,IterMut};

use angle;

pub trait Scalar: Copy {}

pub trait Dim<T:Copy>: Copy {
    type Output: Copy+AsMut<[T]>+AsRef<[T]>+BorrowMut<[T]>;

    #[inline(always)]
    fn from_value(v: T) -> Self::Output;

    #[inline(always)]
    fn from_iter<U>(iterator: U) -> Self::Output
    where U: IntoIterator<Item=T>,
    U::IntoIter: ExactSizeIterator;
}

pub trait ScalarArray {
    type Scalar: Scalar;
    type Type: Copy;
    type Dim: Dim<Self::Type>;

    #[inline(always)]
    fn new(v: <Self::Dim as Dim<Self::Type>>::Output) -> Self;

    #[inline(always)]
    fn iter(&self) -> Iter<Self::Type>;
    #[inline(always)]
    fn iter_mut(&mut self) -> IterMut<Self::Type>;
    #[inline(always)]
    fn from_value(v: Self::Scalar) -> Self;
}

pub trait Cast<T: Scalar>: ScalarArray
where <Self as ScalarArray>::Dim: Dim<<Self::Output as ScalarArray>::Type> {
    type Output: ScalarArray<Scalar=T, Dim=<Self as ScalarArray>::Dim>;

    #[inline(always)]
    fn fold<F: Fn(T, &<Self as ScalarArray>::Scalar)->T>(&self, default: T, f: F) -> T;

    #[inline(always)]
    fn unary<F: Fn(&<Self as ScalarArray>::Scalar)->T>(&self, f: F) -> Self::Output;

    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &<Self as ScalarArray>::Scalar)->T>(&self, rhs: &Self, f: F) -> Self::Output;
}

pub trait ComponentPartialEq: ScalarArray + Cast<bool>
where <Self as ScalarArray>::Scalar: PartialEq,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> {
    fn cpt_eq(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialEq::eq)
    }

    fn cpt_ne(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialEq::ne)
    }
}

pub trait ComponentEq : ComponentPartialEq
where <Self as ScalarArray>::Scalar: Eq,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> {}


pub trait ComponentPartialOrd: ComponentPartialEq + Cast<Option<Ordering>>
where <Self as ScalarArray>::Scalar: PartialOrd,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> + Dim<<<Self as Cast<Option<Ordering>>>::Output as ScalarArray>::Type> {
    fn cpt_partial_cmp(&self, rhs: &Self) -> <Self as Cast<Option<Ordering>>>::Output {
        Cast::<Option<Ordering>>::binary(self, rhs, PartialOrd::partial_cmp)
    }

    fn cpt_lt(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::lt)
    }
    fn cpt_le(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::le)
    }
    fn cpt_gt(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::gt)
    }
    fn cpt_ge(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::ge)
    }
}

pub trait ComponentOrd: ComponentEq + ComponentPartialOrd + Cast<Ordering>
where <Self as ScalarArray>::Scalar: Ord,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> + Dim<<<Self as Cast<Option<Ordering>>>::Output as ScalarArray>::Type> + Dim<<<Self as Cast<Ordering>>::Output as ScalarArray>::Type> {
    fn cpt_cmp(&self, rhs: &Self) -> <Self as Cast<Ordering>>::Output {
        Cast::<Ordering>::binary(self, rhs, Ord::cmp)
    }
}

impl Scalar for Option<Ordering> {}
impl Scalar for Ordering {}

include!(concat!(env!("OUT_DIR"), "/traits.rs"));
