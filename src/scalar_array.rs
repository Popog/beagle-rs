
//! Traits and functions that operate on ScalarArrays

use std::borrow::{BorrowMut};
use std::cmp::Ordering;
use std::slice::{Iter,IterMut};

use angle;

/// Types that can be held in a Matrix/Vector scalar value
pub trait Scalar: Copy {}

/// Types that represent a dimension
pub trait Dim<T:Copy>: Copy {
    /// An array of the size equal to the dimension this type represents
    type Output: Copy+AsMut<[T]>+AsRef<[T]>+BorrowMut<[T]>;

    /// Construct an array from a single value, replicating it
    #[inline(always)]
    fn from_value(v: T) -> Self::Output;

    /// Construct an array from an ExactSizeIterator with len() == the dimension this type
    /// represents
    #[inline(always)]
    fn from_iter<U>(iterator: U) -> Self::Output
    where U: IntoIterator<Item=T>,
    U::IntoIter: ExactSizeIterator;
}

/// Types that represent an array of scalars (a matrix or a vector)
pub trait ScalarArray {
    /// The type of the underlying scalar in the array
    type Scalar: Scalar;
    /// The type of a single element of this type (a single row for matrices/a scalar for vectors)
    type Type: Copy;
    /// The dimension of the scalar array
    type Dim: Dim<Self::Type>;

    /// Construct a matrix/vector from a an array
    #[inline(always)]
    fn new(v: <Self::Dim as Dim<Self::Type>>::Output) -> Self;
    /// Get a slice iterator over the elements (the rows for matrices/the scalars for vectors)
    #[inline(always)]
    fn iter(&self) -> Iter<Self::Type>;
    /// Get a mutable slice iterator over the elements (the rows for matrices/the scalars for
    /// vectors)
    #[inline(always)]
    fn iter_mut(&mut self) -> IterMut<Self::Type>;
    /// Construct a matrix/vector from a single scalar value, setting all elements to that value
    #[inline(always)]
    fn from_value(v: Self::Scalar) -> Self;

    /// Fold all the scalar values into a single output given two folding functions,
    /// The first folding function only applies to the first element of the ScalarArray
    #[inline(always)]
    fn fold<T, F0: FnOnce(&Self::Scalar)->T, F: Fn(T, &Self::Scalar)->T>(&self, f0: F0, f: F) -> T;
}

/// Types that can be transformed from into a `ScalarArray` of `<T>`
pub trait Cast<T: Scalar>: ScalarArray
where <Self as ScalarArray>::Dim: Dim<<Self::Output as ScalarArray>::Type> {
    /// The resulting type
    type Output: ScalarArray<Scalar=T, Dim=<Self as ScalarArray>::Dim>;

    // Fold two `ScalarArray`s together using a binary function
    #[inline(always)]
    fn fold_together<O, F0: FnOnce(&<Self as ScalarArray>::Scalar, &T)->O, F: Fn(O, &<Self as ScalarArray>::Scalar, &T)->O>(&self, rhs: &Self::Output, f0: F0, f: F) -> O;

    /// Transform a single `ScalarArray` using a unary function
    #[inline(always)]
    fn unary<F: Fn(&<Self as ScalarArray>::Scalar)->T>(&self, f: F) -> Self::Output;

    /// Transform two binary `ScalarArray`s using a binary function
    #[inline(always)]
    fn binary<F: Fn(&<Self as ScalarArray>::Scalar, &<Self as ScalarArray>::Scalar)->T>(&self, rhs: &Self, f: F) -> Self::Output;
}

/// Types that can be component-wise compared using PartialEq
pub trait ComponentPartialEq: ScalarArray + Cast<bool>
where <Self as ScalarArray>::Scalar: PartialEq,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> {
    /// This method tests for the components of `self` and `rhs` values to be equal
    fn cpt_eq(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialEq::eq)
    }

    /// This method tests for the components of `self` and `rhs` values to be unequal
    fn cpt_ne(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialEq::ne)
    }
}

/// Types that can be component-wise compared using Eq
pub trait ComponentEq : ComponentPartialEq
where <Self as ScalarArray>::Scalar: Eq,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> {}


/// Types that can be component-wise using PartialOrd
pub trait ComponentPartialOrd: ComponentPartialEq + Cast<Option<Ordering>>
where <Self as ScalarArray>::Scalar: PartialOrd,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> + Dim<<<Self as Cast<Option<Ordering>>>::Output as ScalarArray>::Type> {
    /// This method returns an ordering between the components of `self` and `rhs` values if one
    /// exists
    fn cpt_partial_cmp(&self, rhs: &Self) -> <Self as Cast<Option<Ordering>>>::Output {
        Cast::<Option<Ordering>>::binary(self, rhs, PartialOrd::partial_cmp)
    }

    /// This method tests less than between the components of `self` and `rhs` values
    fn cpt_lt(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::lt)
    }
    /// This method tests less than or equal to between the components of `self` and `rhs` values
    fn cpt_le(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::le)
    }
    /// This method tests greater than between the components of `self` and `rhs` values
    fn cpt_gt(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::gt)
    }
    /// This method tests greater than or equal to between the components of `self` and `rhs`
    /// values
    fn cpt_ge(&self, rhs: &Self) -> <Self as Cast<bool>>::Output {
        Cast::<bool>::binary(self, rhs, PartialOrd::ge)
    }
}

/// Types that can be component-wise using Ord
pub trait ComponentOrd: ComponentEq + ComponentPartialOrd + Cast<Ordering>
where <Self as ScalarArray>::Scalar: Ord,
<Self as ScalarArray>::Dim: Dim<<<Self as Cast<bool>>::Output as ScalarArray>::Type> + Dim<<<Self as Cast<Option<Ordering>>>::Output as ScalarArray>::Type> + Dim<<<Self as Cast<Ordering>>::Output as ScalarArray>::Type> {
    /// This method returns an ordering between the components of `self` and `rhs` values
    fn cpt_cmp(&self, rhs: &Self) -> <Self as Cast<Ordering>>::Output {
        Cast::<Ordering>::binary(self, rhs, Ord::cmp)
    }
}

impl Scalar for Option<Ordering> {}
impl Scalar for Ordering {}

include!(concat!(env!("OUT_DIR"), "/scalar_array.rs"));
