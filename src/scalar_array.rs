//! Traits and functions that operate on ScalarArrays.

// TODO: more commments

use std::cmp::Ordering;
use std::ops::{
    Add,Mul,MulAssign,
};

use num::{Abs,Clamp,Exp,Hyperbolic,Pow,Recip,Round,Sqrt};
use consts::{
    One,
    Array,
    Dim,HasSmaller,DimRef,DimMut,
    TwoDim,TwoDimRef,TwoDimMut,
};

/// Types that represent a 2d array of scalars (a matrix or a vector).
pub trait ScalarArray
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where <<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<Self::Scalar>,
<<Self as ScalarArray>::Dim as HasSmaller>::Smaller: Array<<Self::Row as Array<Self::Scalar>>::Type>,
{
    /// The type of the underlying scalar in the array.
    type Scalar;
    /// The type of a single element of this type (a single row for matrices/a scalar for vectors).
    type Row: Dim<Self::Scalar>;
    /// The dimension of the scalar array.
    type Dim: TwoDim<Self::Scalar, Self::Row>;
}

/// Matrix/Vector types that have an associated constructable type
pub trait HasConcreteScalarArray<S, R = <Self as ScalarArray>::Row, D = <Self as ScalarArray>::Dim>: ScalarArray
where R: Dim<S>,
D: TwoDim<S, R>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<<Self as ScalarArray>::Scalar>,
<<Self as ScalarArray>::Dim as HasSmaller>::Smaller: Array<<<Self as ScalarArray>::Row as Array<<Self as ScalarArray>::Scalar>>::Type>,
<R as HasSmaller>::Smaller: Array<S>,
<D as HasSmaller>::Smaller: Array<<R as Array<S>>::Type>,
{
    /// The type of a concrete ScalarArray of the specified type
    type Concrete: ConcreteScalarArray<Scalar=S, Row=R, Dim=D>;
}

/// Vector types that have an associated constructable type
pub trait HasConcreteVecArray<S, R = <Self as ScalarArray>::Row>: ScalarArray<Dim=One> + HasConcreteScalarArray<S, R, One>
where R: Dim<S>,
Self::Concrete: ConcreteVecArray<Scalar=S, Row=R>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<<Self as ScalarArray>::Scalar>,
<R as HasSmaller>::Smaller: Array<S>,
{}


/// Matrix/Vector types that can be constructed
pub trait ConcreteScalarArray: Sized + ScalarArrayVal + HasConcreteScalarArray<<Self as ScalarArray>::Scalar, Concrete=Self>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where <<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<<Self as ScalarArray>::Scalar>,
<<Self as ScalarArray>::Dim as HasSmaller>::Smaller: Array<<<Self as ScalarArray>::Row as Array<<Self as ScalarArray>::Scalar>>::Type>,
{
    /// Create from an array
    #[inline(always)]
    fn from_val(v: <Self::Dim as Array<<Self::Row as Array<Self::Scalar>>::Type>>::Type) -> Self;
}

/// Vector types that can be constructed
pub trait ConcreteVecArray: Sized + VecArrayVal + ConcreteScalarArray + HasConcreteVecArray<<Self as ScalarArray>::Scalar, Concrete=Self>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where <<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<<Self as ScalarArray>::Scalar>,
{
    /// Create from an array
    #[inline(always)]
    fn from_vec_val(v: <Self::Row as Array<Self::Scalar>>::Type) -> Self;
}

/// Matrix/Vector types that can be deconstructed
pub trait ScalarArrayVal: ScalarArray
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where <<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<<Self as ScalarArray>::Scalar>,
<<Self as ScalarArray>::Dim as HasSmaller>::Smaller: Array<<<Self as ScalarArray>::Row as Array<<Self as ScalarArray>::Scalar>>::Type>,
{
    /// Extract the inner array
    #[inline(always)]
    fn get_val(self) -> <Self::Dim as Array<<Self::Row as Array<Self::Scalar>>::Type>>::Type;
}

/// Vector types that can be deconstructed
pub trait VecArrayVal: ScalarArrayVal<Dim=One>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where <<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<<Self as ScalarArray>::Scalar>,
{
    /// Extract the inner array
    #[inline(always)]
    fn get_vec_val(self) -> <Self::Row as Array<Self::Scalar>>::Type;
}

/// Matrix/Vector types that can be references
pub trait ScalarArrayRef: ScalarArray
where Self::Row: DimRef<Self::Scalar>,
Self::Dim: TwoDimRef<Self::Scalar, Self::Row>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<<Self as ScalarArray>::Scalar>,
<<Self as ScalarArray>::Dim as HasSmaller>::Smaller: Array<<<Self as ScalarArray>::Row as Array<<Self as ScalarArray>::Scalar>>::Type>
{
    /// Extract the inner array references
    #[inline(always)]
    fn get_ref(&self) -> <Self::Dim as Array<<Self::Row as Array<&Self::Scalar>>::Type>>::Type;
}

/// Vector types that can be references
pub trait VecArrayRef: ScalarArrayRef<Dim=One>
where Self::Row: DimRef<Self::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
for<'a> <<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<<Self as ScalarArray>::Scalar>+Array<&'a <Self as ScalarArray>::Scalar>,
{
    /// Extract the inner array references
    #[inline(always)]
    fn get_vec_ref(&self) -> <Self::Row as Array<&Self::Scalar>>::Type;
}

/// Matrix/Vector types that can be mutated
pub trait ScalarArrayMut: ScalarArrayRef
where Self::Row: DimMut<Self::Scalar>,
Self::Dim: TwoDimMut<Self::Scalar, Self::Row>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<<Self as ScalarArray>::Row as HasSmaller>::Smaller: Array<<Self as ScalarArray>::Scalar>,
<<Self as ScalarArray>::Dim as HasSmaller>::Smaller: Array<<<Self as ScalarArray>::Row as Array<<Self as ScalarArray>::Scalar>>::Type>
{
    /// Extract the inner array mutable references
    #[inline(always)]
    fn get_mut(&mut self) -> <Self::Dim as Array<<Self::Row as Array<&mut Self::Scalar>>::Type>>::Type;
}



/// Helpers for 1d arrays
pub mod vec_array {
    use consts::{Array,Dim,CustomArrayThree};
    use std::ops::{Add, Mul, Sub};

    /// Fold all the elements in a 1d array. The first element is mapped with `f0`,
    /// then folding continues with `f` for other elements.
    #[inline(always)]
    pub fn fold<S, R, F0, F, O>(s: R::Type, f0: F0, mut f: F) -> O
    where R: Dim<S>,
    F0: FnOnce(S) -> O,
    F: FnMut(O, S) -> O,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    R::Smaller: Array<S>,
    {
        let (s0, s) = R::split(s);
        let init = f0(s0);
        R::Smaller::fold(s, init, &mut f)
    }

    /// Fold all the elements of two 1d arrays. The first elements are mapped with `f0`,
    /// then folding continues with `f` for other elements.
    #[inline(always)]
    pub fn fold_zip<S, R, T, F0, F, O>(s: <R as Array<S>>::Type, t: <R as Array<T>>::Type, f0: F0, mut f: F) -> O
    where R: Dim<S> + Dim<T>,
    F0: FnOnce(S, T) -> O,
    F: FnMut(O, S, T) -> O,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    R::Smaller: Array<S> + Array<T>,
    {
        let (s0, s) = R::split(s);
        let (t0, t) = R::split(t);
        let init = f0(s0, t0);
        <R::Smaller as Array<S>>::fold_zip(s, t, init, &mut f)
    }

    /// Multiply two Vectors component-wise, summing the results. Known as a dot product.
    #[inline(always)]
    pub fn dot<S, R, T>(s: <R as Array<S>>::Type, t: <R as Array<T>>::Type) -> S::Output
    where R: Dim<S> + Dim<T>,
    S: Mul<T>,
    S::Output: Add<Output=S::Output>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    R::Smaller: Array<S> + Array<T>,
    {
        fold_zip::<S, R, T, _, _, _>(s, t, |s, t| s*t, |init, s, t| init + (s*t))
    }

    #[inline]
    pub fn mul_vector_transpose<S, R, D, T>(v: <D as Array<S>>::Type, m: <D as Array<<R as Array<T>>::Type>>::Type) -> <R as Array<S::Output>>::Type
    where R: Array<S> + Array<T> + Array<S::Output>,
    D: Array<S> + Dim<<R as Array<S>>::Type> + Dim<<R as Array<T>>::Type>,
    S: Mul<T> + Clone,
    S::Output: Add<Output=S::Output>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    D::Smaller: Array<<R as Array<S>>::Type> + Array<<R as Array<T>>::Type>,
    {
        /* We're trying to accomplish the following multiplication
        a   d e f g     a*d + b*h + c*l
        b   h i j k     a*e + b*i + c*m
        c   l m n o     a*f + b*j + c*n
                        a*g + b*k + c*o
        */

        /* Start duplcating out the vec to get the following
        a a a a   d e f g
        b b b b   h i j k
        c c c c   l m n o
        */
        let v = D::map(v, |v| <R as Array<S>>::from_value(v));

        // Now we're going to multiply the items together and then sum the rows down
        // We can do this with a single fold_zip if we're clever, which we are
        fold_zip::<<R as Array<S>>::Type, D, <R as Array<T>>::Type, _, _, _>(
            v, m,
            // multiply the first items (a*x)
            |v, m| <R as Array<S>>::map_zip(v, m, |v, m| v * m),
            |init, v, m| {
                // Multiply the second/etc items (b*x or c*x)
                let r: <R as Array<S::Output>>::Type = <R as Array<S>>::map_zip::<T, S::Output, _>(v, m, |v, m| v * m);
                // Add that result to the running total
                <R as Array<S::Output>>::map_zip(init, r, |init, r| init + r)
            },
        )
    }

    /// Fold all the elements in a 1d array. The first element is mapped with `f0`,
    /// then folding continues with `f` for other elements.
    #[inline(always)]
    pub fn cross<S, T>(s: CustomArrayThree<S>, t: CustomArrayThree<T>) -> CustomArrayThree<S::Output>
    where S: Clone+Mul<T>,
    T: Clone,
    S::Output: Sub<Output=S::Output>, {
        use consts::{Three,Two,One};
        let ((s0, s), (t0, t)) = (Three::split(s), Three::split(t));
        let ((s1, s), (t1, t)) = (Two::split(s), Two::split(t));
        let ((s2, _), (t2, _)) = (One::split(s), One::split(t));

        let (sb0, sb1, sb2) = (s0.clone(), s1.clone(), s2.clone());
        let (tb0, tb1, tb2) = (t0.clone(), t1.clone(), t2.clone());
        CustomArrayThree([
            s1*t2 - sb2*tb1,
            s2*t0 - sb0*tb2,
            s0*t1 - sb1*tb0,
        ])
    }

}

/// Helpers for 2d arrays
pub mod array {
    use consts::{Array,Dim,TwoDim,};
    use super::vec_array;

    /// Construct a 2d array from a single value
    #[inline(always)]
    pub fn from_scalar<S, R, D>(v: S) -> D::Type
    where R: Array<S>, D: Array<R::Type>,
    S: Clone, R::Type: Clone, {
        D::from_value(R::from_value(v))
    }

    /// Apply a function to all elements of a 2d array
    #[inline(always)]
    pub fn apply<S, R, D, F>(s: D::Type, mut f: F)
    where R: Array<S>, D: Array<R::Type>,
    F: FnMut(S), {
        D::apply(s, |s| R::apply(s, &mut f))
    }

    /// Apply a function to all elements of both 2d arrays
    #[inline(always)]
    pub fn apply_zip<S, R, D, T, F>(s: <D as Array<<R as Array<S>>::Type>>::Type, t: <D as Array<<R as Array<T>>::Type>>::Type, mut f: F)
    where R: Array<S> + Array<T>,
    D: Array<<R as Array<S>>::Type> + Array<<R as Array<T>>::Type>,
    F: FnMut(S, T) {
        D::apply_zip(s, t, |s, t| R::apply_zip(s, t, &mut f))
    }

    /// Fold all the elements in a 2d array. The first element is mapped with `f0`,
    /// then folding continues with `f` for other elements.
    #[inline(always)]
    pub fn fold<S, R, D, F0, F, O>(s: D::Type, f0: F0, mut f: F) -> O
    where R: Dim<S>, D: TwoDim<S, R>,
    F0: FnOnce(S) -> O,
    F: FnMut(O, S) -> O,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    R::Smaller: Array<S>,
    D::Smaller: Array<<R as Array<S>>::Type>,
    {
        let (s0, s) = D::split(s);
        let init = vec_array::fold::<S, R, F0, _, O>(s0, f0, &mut f);
        D::Smaller::fold(s, init,
            |init, s| R::fold(s, init, &mut f)
        )
    }

    /// Fold all the elements of two 2d arrays. The first elements are mapped with `f0`,
    /// then folding continues with `f` for other elements.
    #[inline(always)]
    pub fn fold_zip<S, R, D, T, F0, F, O>(s: <D as Array<<R as Array<S>>::Type>>::Type, t: <D as Array<<R as Array<T>>::Type>>::Type, f0: F0, mut f: F) -> O
    where R: Dim<S>+Dim<T>,
    D: TwoDim<S, R>+TwoDim<T, R>,
    F0: FnOnce(S, T) -> O,
    F: FnMut(O, S, T) -> O,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    R::Smaller: Array<S> + Array<T>,
    D::Smaller: Array<<R as Array<S>>::Type> + Array<<R as Array<T>>::Type>,
    {
        let (s0, s) = D::split(s);
        let (t0, t) = D::split(t);
        let init = vec_array::fold_zip::<S, R, T, F0, _, O>(s0, t0, f0, &mut f);
        <D::Smaller as Array<<R as Array<S>>::Type>>::fold_zip(s, t, init,
            |init, s, t| R::fold_zip(s, t, init, &mut f)
        )
    }

    /// Construct a 2d array from another 2d array and a mapping function
    #[inline(always)]
    pub fn map<S, R, D, T, F>(s: <D as Array<<R as Array<S>>::Type>>::Type, mut f: F) -> <D as Array<<R as Array<T>>::Type>>::Type
    where R: Array<S> + Array<T>,
    D: Array<<R as Array<S>>::Type> + Array<<R as Array<T>>::Type>,
    F: FnMut(S) -> T {
        D::map(s, |s| R::map(s, &mut f))
    }

    /// Construct a 2d array from two other 2d arrays and a mapping function
    #[inline(always)]
    pub fn map_zip<S, R, D, T, U, F>(s: <D as Array<<R as Array<S>>::Type>>::Type, t: <D as Array<<R as Array<T>>::Type>>::Type, mut f: F) -> <D as Array<<R as Array<U>>::Type>>::Type
    where R: Array<S> + Array<T> + Array<U>,
    D: Array<<R as Array<S>>::Type> + Array<<R as Array<T>>::Type> + Array<<R as Array<U>>::Type>,
    F: FnMut(S, T) -> U, {
        D::map_zip(s, t, |s, t| R::map_zip(s, t, &mut f))
    }

    /// Transpose a 2d array
    #[inline(always)]
    pub fn transpose<S, R, D>(s: <D as Array<<R as Array<S>>::Type>>::Type) -> <R as Array<<D as Array<S>>::Type>>::Type
    where R: Dim<S> + TwoDim<S, D>,
    D: Dim<S> + TwoDim<S, R>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    R::Smaller: Array<S> + Array<<D as Array<S>>::Type>,
    D::Smaller: Array<S> + Array<<R as Array<S>>::Type>,
    {
        <R as Array<S>>::transpose_helper::<D>(s)
    }
}


/// Construct a ScalarArray from a single value
#[inline(always)]
pub fn from_scalar<S>(v: S::Scalar) -> S
where S: ScalarArray,
(<S::Dim as Array<<S::Row as Array<S::Scalar>>::Type>>::Type,): Into<S>,
S::Scalar: Clone,
<S::Row as Array<S::Scalar>>::Type: Clone,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type>,
{
    (array::from_scalar::<_, S::Row, S::Dim>(v),).into()
}

/// Apply a function to all elements of a ScalarArray
#[inline(always)]
pub fn apply<S, F>(s: S, f: F)
where S: ScalarArrayVal,
F: FnMut(S::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type>,
{
    array::apply::<_, S::Row, S::Dim, F>(s.get_val(), f)
}

/// Apply a function to all elements of a ScalarArray
#[inline(always)]
pub fn apply_ref<'a, S, F>(s: &'a S, f: F)
where S: ScalarArrayRef,
F: FnMut(&'a S::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type>,
{
    array::apply::<_, S::Row, S::Dim, F>(s.get_ref(), f)
}

/// Apply a function to all elements of a ScalarArray
#[inline(always)]
pub fn apply_mut<'a, S, F>(s: &'a mut S, f: F)
where S: ScalarArrayMut,
F: FnMut(&'a mut S::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type>,
{
    array::apply::<_, S::Row, S::Dim, F>(s.get_mut(), f)
}


/// Apply a function to all elements of two ScalarArrays
#[inline(always)]
pub fn apply_zip<S, T, F>(s: S, t: T, f: F)
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar, T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply_zip::<S::Scalar, S::Row, S::Dim, _, F>(s.get_val(), t.get_val(), f)
}

/// Apply a function to all elements of two ScalarArrays
#[inline(always)]
pub fn apply_zip_val_ref<'b, S, T, F>(s: S, t: &'b T, f: F)
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar, &'b T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<T::Scalar>,
S::Dim: TwoDimRef<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply_zip::<S::Scalar, S::Row, S::Dim, _, F>(s.get_val(), t.get_ref(), f)
}

/// Apply a function to all elements of two ScalarArrays
#[inline(always)]
pub fn apply_zip_val_mut<'b, S, T, F>(s: S, t: &'b mut T, f: F)
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayMut<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar, &'b mut T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<T::Scalar>,
S::Dim: TwoDimMut<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply_zip::<S::Scalar, S::Row, S::Dim, _, F>(s.get_val(), t.get_mut(), f)
}

/// Apply a function to all elements of two ScalarArrays
#[inline(always)]
pub fn apply_zip_ref_val<'a, S, T, F>(s: &'a S, t: T, f: F)
where S: ScalarArrayRef,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a S::Scalar, T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply_zip::<_, S::Row, S::Dim, _, F>(s.get_ref(), t.get_val(), f)
}

/// Apply a function to all elements of two ScalarArrays
#[inline(always)]
pub fn apply_zip_ref<'a, 'b, S, T, F>(s: &'a S, t: &'b T, f: F)
where S: ScalarArrayRef,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a S::Scalar, &'b T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply_zip::<_, S::Row, S::Dim, _, F>(s.get_ref(), t.get_ref(), f)
}

/// Apply a function to all elements of two ScalarArrays
#[inline(always)]
pub fn apply_zip_ref_mut<'a, 'b, S, T, F>(s: &'a S, t: &'b mut T, f: F)
where S: ScalarArrayRef,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayMut<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a S::Scalar, &'b mut T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimMut<T::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimMut<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply_zip::<_, S::Row, S::Dim, _, F>(s.get_ref(), t.get_mut(), f)
}

/// Apply a function to all elements of two ScalarArrays
#[inline(always)]
pub fn apply_zip_mut_val<'a, S, T, F>(s: &'a mut S, t: T, f: F)
where S: ScalarArrayMut,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a mut S::Scalar, T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply_zip::<_, S::Row, S::Dim, _, F>(s.get_mut(), t.get_val(), f)
}

/// Apply a function to all elements of two ScalarArrays
#[inline(always)]
pub fn apply_zip_mut_ref<'a, 'b, S, T, F>(s: &'a mut S, t: &'b T, f: F)
where S: ScalarArrayMut,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a mut S::Scalar, &'b T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar> + DimRef<T::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply_zip::<_, S::Row, S::Dim, _, F>(s.get_mut(), t.get_ref(), f)
}

/// Apply a function to all elements of two ScalarArrays
#[inline(always)]
pub fn apply_zip_mut<'a, 'b, S, T, F>(s: &'a mut S, t: &'b mut T, f: F)
where S: ScalarArrayMut,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayMut<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a mut S::Scalar, &'b mut T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar> + DimMut<T::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row> + TwoDimMut<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply_zip::<_, S::Row, S::Dim, _, F>(s.get_mut(), t.get_mut(), f)
}


/// Fold all the scalars in a ScalarArray. The first element is mapped with `f0`,
/// then folding continues with `f` for other elements.
#[inline(always)]
pub fn fold<S, F0, F, O>(s: S, f0: F0, f: F) -> O
where S: ScalarArrayVal,
F0: FnOnce(S::Scalar) -> O,
F: FnMut(O, S::Scalar) -> O,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type>,
{
    array::fold::<_, _, S::Dim, F0, F, O>(s.get_val(), f0, f)
}

/// Fold all the scalars in a ScalarArray by reference. The first element is mapped with `f0`,
/// then folding continues with `f` for other elements.
#[inline(always)]
pub fn fold_ref<'a, S, F0, F, O>(s: &'a S, f0: F0, f: F) -> O
where S: ScalarArrayRef,
F0: FnOnce(&'a S::Scalar) -> O,
F: FnMut(O, &'a S::Scalar) -> O,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<&'a S::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<&'a S::Scalar>>::Type>,
{
    array::fold::<_, _, S::Dim, F0, F, O>(s.get_ref(), f0, f)
}

/// Fold all the scalars in a ScalarArray by mutable reference. The first element is mapped with
/// `f0`, then folding continues with `f` for other elements.
#[inline(always)]
pub fn fold_mut<'a, S, F0, F, O>(s: &'a mut S, f0: F0, f: F) -> O
where S: ScalarArrayMut,
F0: FnOnce(&'a mut S::Scalar) -> O,
F: FnMut(O, &'a mut S::Scalar) -> O,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<&'a mut S::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<&'a mut S::Scalar>>::Type>,
{
    array::fold::<_, _, S::Dim, F0, F, O>(s.get_mut(), f0, f)
}


/// Fold all the scalars in two ScalarArrays. The first elements are mapped with `f0`,
/// then folding continues with `f` for other elements.
#[inline(always)]
pub fn fold_zip<S, T, F0, F, O>(s: S, t: T, f0: F0, f: F) -> O
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
F0: FnOnce(S::Scalar, T::Scalar) -> O,
F: FnMut(O, S::Scalar, T::Scalar) -> O,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type>,
{
    array::fold_zip::<S::Scalar, S::Row, S::Dim, T::Scalar, F0, F, O>(s.get_val(), t.get_val(), f0, f)
}

/// Fold all the scalars in two ScalarArrays. The first elements are mapped with `f0`,
/// then folding continues with `f` for other elements.
#[inline(always)]
pub fn fold_zip_ref<'a, S, T, F0, F, O>(s: &'a S, t: &'a T, f0: F0, f: F) -> O
where S: ScalarArrayRef,
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
F0: FnOnce(&'a S::Scalar, &'a T::Scalar) -> O,
F: FnMut(O, &'a S::Scalar, &'a T::Scalar) -> O,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<&'a S::Scalar> + Array<&'a T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<&'a S::Scalar>>::Type> + Array<<S::Row as Array<&'a T::Scalar>>::Type>,
{
    array::fold_zip::<&'a S::Scalar, S::Row, S::Dim, &'a T::Scalar, F0, F, O>(s.get_ref(), t.get_ref(), f0, f)
}


/// Construct a ScalarArray from another ScalarArray and a mapping function
#[inline(always)]
pub fn map<S, T, F>(s: S, f: F) -> T
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar) -> T::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type>,
{
    T::from_val(array::map::<S::Scalar, S::Row, S::Dim, _, F>(s.get_val(), f))
}

/// Construct a ScalarArray from another ScalarArray by reference and a mapping function
#[inline(always)]
pub fn map_ref<'a, S, T, F>(s: &'a S, f: F) -> T
where S: ScalarArrayRef,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a S::Scalar) -> T::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type>,
{
    T::from_val(array::map::<_, S::Row, S::Dim, _, F>(s.get_ref(), f))
}

/// Construct a ScalarArray from another ScalarArray by reference and a mapping function
#[inline(always)]
pub fn map_mut<'a, S, T, F>(s: &'a mut S, f: F) -> T
where S: ScalarArrayMut,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a mut S::Scalar) -> T::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type>,
{
    T::from_val(array::map::<_, S::Row, S::Dim, _, F>(s.get_mut(), f))
}

/// Construct a ScalarArray from two other ScalarArray and a mapping function
#[inline(always)]
pub fn map_zip<S, T, U, F>(s: S, t: T, f: F) -> U
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar, T::Scalar) -> U::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    U::from_val(array::map_zip::<S::Scalar, S::Row, S::Dim, _, _, F>(s.get_val(), t.get_val(), f))
}

/// Construct a ScalarArray from two other ScalarArray and a mapping function
#[inline(always)]
pub fn map_zip_val_ref<'b, S, T, U, F>(s: S, t: &'b T, f: F) -> U
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar, &'b T::Scalar) -> U::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<T::Scalar>,
S::Dim: TwoDimRef<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    U::from_val(array::map_zip::<S::Scalar, S::Row, S::Dim, _, _, F>(s.get_val(), t.get_ref(), f))
}

/// Construct a ScalarArray from two other ScalarArray and a mapping function
#[inline(always)]
pub fn map_zip_val_mut<'b, S, T, U, F>(s: S, t: &'b mut T, f: F) -> U
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ScalarArrayMut<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar, &'b mut T::Scalar) -> U::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<T::Scalar>,
S::Dim: TwoDimMut<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    U::from_val(array::map_zip::<S::Scalar, S::Row, S::Dim, _, _, F>(s.get_val(), t.get_mut(), f))
}

/// Construct a ScalarArray from two other ScalarArray and a mapping function
#[inline(always)]
pub fn map_zip_ref_val<'a, S, T, U, F>(s: &'a S, t: T, f: F) -> U
where S: ScalarArrayRef,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a S::Scalar, T::Scalar) -> U::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    U::from_val(array::map_zip::<_, S::Row, S::Dim, _, _, F>(s.get_ref(), t.get_val(), f))
}

/// Construct a ScalarArray from two other ScalarArray and a mapping function
#[inline(always)]
pub fn map_zip_ref<'a, 'b, S, T, U, F>(s: &'a S, t: &'b T, f: F) -> U
where S: ScalarArrayRef,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a S::Scalar, &'b T::Scalar) -> U::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    U::from_val(array::map_zip::<_, S::Row, S::Dim, _, _, F>(s.get_ref(), t.get_ref(), f))
}

/// Construct a ScalarArray from two other ScalarArray and a mapping function
#[inline(always)]
pub fn map_zip_ref_mut<'a, 'b, S, T, U, F>(s: &'a S, t: &'b mut T, f: F) -> U
where S: ScalarArrayRef,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ScalarArrayMut<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a S::Scalar, &'b mut T::Scalar) -> U::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimMut<T::Scalar>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimMut<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    U::from_val(array::map_zip::<_, S::Row, S::Dim, _, _, F>(s.get_ref(), t.get_mut(), f))
}

/// Construct a ScalarArray from two other ScalarArray and a mapping function
#[inline(always)]
pub fn map_zip_mut_val<'a, S, T, U, F>(s: &'a mut S, t: T, f: F) -> U
where S: ScalarArrayMut,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a mut S::Scalar, T::Scalar) -> U::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    U::from_val(array::map_zip::<_, S::Row, S::Dim, _, _, F>(s.get_mut(), t.get_val(), f))
}

/// Construct a ScalarArray from two other ScalarArray and a mapping function
#[inline(always)]
pub fn map_zip_mut_ref<'a, 'b, S, T, U, F>(s: &'a mut S, t: &'b T, f: F) -> U
where S: ScalarArrayMut,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a mut S::Scalar, &'b T::Scalar) -> U::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar> + DimRef<T::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    U::from_val(array::map_zip::<_, S::Row, S::Dim, _, _, F>(s.get_mut(), t.get_ref(), f))
}

/// Construct a ScalarArray from two other ScalarArray and a mapping function
#[inline(always)]
pub fn map_zip_mut<'a, S, T, U, F>(s: &'a mut S, t: &'a mut T, f: F) -> U
where S: ScalarArrayMut,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ScalarArrayMut<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(&'a mut S::Scalar, &'a mut T::Scalar) -> U::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar> + DimMut<T::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row> + TwoDimMut<T::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    U::from_val(array::map_zip::<_, S::Row, S::Dim, _, _, F>(s.get_mut(), t.get_mut(), f))
}

/// Construct a ScalarArray based on the equality of the components of two ScalarArray.
#[inline(always)]
pub fn cpt_eq<'a, S, T>(s: &'a S, t: &'a T) -> S::Concrete
where S: ScalarArrayRef + HasConcreteScalarArray<bool>,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
S::Scalar: PartialEq<T::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar> + Dim<bool>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row> + TwoDim<bool, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<bool>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<bool>>::Type>,
{
    map_zip_ref(s, t, |s, t| s == t)
}

/// Construct a ScalarArray based on the inequality of the components of two ScalarArray.
#[inline(always)]
pub fn cpt_ne<'a, S, T>(s: &'a S, t: &'a T) -> S::Concrete
where S: ScalarArrayRef + HasConcreteScalarArray<bool>,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
S::Scalar: PartialEq<T::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar> + Dim<bool>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row> + TwoDim<bool, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<bool>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<bool>>::Type>,
{
    map_zip_ref(s, t, |s, t| s != t)
}

/// Construct a ScalarArray based on the partial ordering of the components of two ScalarArray.
#[inline(always)]
pub fn cpt_partial_cmp<'a, S, T>(s: &'a S, t: &'a T) -> S::Concrete
where S: ScalarArrayRef + HasConcreteScalarArray<Option<Ordering>>,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
S::Scalar: PartialOrd<T::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar> + Dim<Option<Ordering>>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row> + TwoDim<Option<Ordering>, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<Option<Ordering>>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<Option<Ordering>>>::Type>,
{
    map_zip_ref(s, t, |s, t| s.partial_cmp(t))
}

/// Construct a ScalarArray based on the `<` comparison of the components of two ScalarArray.
#[inline(always)]
pub fn cpt_lt<'a, S, T>(s: &'a S, t: &'a T) -> S::Concrete
where S: ScalarArrayRef + HasConcreteScalarArray<bool>,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
S::Scalar: PartialOrd<T::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar> + Dim<bool>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row> + TwoDim<bool, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<bool>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<bool>>::Type>,
{
    map_zip_ref(s, t, |s, t| s < t)
}

/// Construct a ScalarArray based on the `<=` comparison of the components of two ScalarArray.
#[inline(always)]
pub fn cpt_le<'a, S, T>(s: &'a S, t: &'a T) -> S::Concrete
where S: ScalarArrayRef + HasConcreteScalarArray<bool>,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
S::Scalar: PartialOrd<T::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar> + Dim<bool>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row> + TwoDim<bool, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<bool>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<bool>>::Type>,
{
    map_zip_ref(s, t, |s, t| s <= t)
}

/// Construct a ScalarArray based on the `>` comparison of the components of two ScalarArray.
#[inline(always)]
pub fn cpt_gt<'a, S, T>(s: &'a S, t: &'a T) -> S::Concrete
where S: ScalarArrayRef + HasConcreteScalarArray<bool>,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
S::Scalar: PartialOrd<T::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar> + Dim<bool>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row> + TwoDim<bool, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<bool>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<bool>>::Type>,
{
    map_zip_ref(s, t, |s, t| s > t)
}

/// Construct a ScalarArray based on the `>=` comparison of the components of two ScalarArray.
#[inline(always)]
pub fn cpt_ge<'a, S, T>(s: &'a S, t: &'a T) -> S::Concrete
where S: ScalarArrayRef + HasConcreteScalarArray<bool>,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayRef<Row=S::Row, Dim=S::Dim>,
S::Scalar: PartialOrd<T::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<T::Scalar> + Dim<bool>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<T::Scalar, S::Row> + TwoDim<bool, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<bool>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<bool>>::Type>,
{
    map_zip_ref(s, t, |s, t| s >= t)
}

/// Construct a ScalarArray based on the ordering of the components of two ScalarArray.
#[inline(always)]
pub fn cpt_cmp<'a, 'b, S, T>(s1: &'a S, s2: &'b S) -> S::Concrete
where S: ScalarArrayRef + HasConcreteScalarArray<Ordering>,
S::Row: Dim<Ordering>,
S::Dim: TwoDim<Ordering, S::Row>,
S::Scalar: Ord,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimRef<S::Scalar> + DimRef<Ordering>,
S::Dim: TwoDimRef<S::Scalar, S::Row> + TwoDimRef<Ordering, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<Ordering>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<Ordering>>::Type>,
{
    map_zip_ref(s1, s2, |s1, s2| s1.cmp(s2))
}

/// Multiply two ScalarArrays component-wise.
#[inline(always)]
pub fn cpt_mul<S, T>(s: S, t: T) -> S::Concrete
where S: ScalarArrayVal + HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Mul<T::Scalar>>::Output>,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayVal<Row=<S as ScalarArray>::Row, Dim=<S as ScalarArray>::Dim>,
S::Scalar: Mul<T::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as Mul<T::Scalar>>::Output>,
S::Dim: TwoDim<<S::Scalar as Mul<T::Scalar>>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<<<S as ScalarArray>::Scalar as Mul<T::Scalar>>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<<<S as ScalarArray>::Scalar as Mul<T::Scalar>>::Output>>::Type>,
{
    map_zip(s, t, |s, t| s * t)
}

/// Multiply two ScalarArrays component-wise.
#[inline(always)]
pub fn cpt_mul_assign<'a, S, T, U>(s: &'a mut S, t: T)
where S: ScalarArrayMut,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
S::Scalar: MulAssign<T::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: DimMut<S::Scalar>,
S::Dim: TwoDimMut<S::Scalar, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type>,
{
    apply_zip_mut_val(s, t, |s, t| *s *= t);
}

/// Sums all the components of a ScalarArray
#[inline(always)]
pub fn sum<S, O>(s: S) -> O
where S: ScalarArrayVal,
S::Scalar: Into<O>,
O: Add<S::Scalar, Output=O>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type>,
{
    fold(s, Into::into, Add::add)
}

/// Transposes a ScalarArray
#[inline(always)]
pub fn transpose<S>(s: S) -> S::Concrete
where S: ScalarArrayVal + HasConcreteScalarArray<<S as ScalarArray>::Scalar, <S as ScalarArray>::Dim, <S as ScalarArray>::Row>,
S::Row: TwoDim<S::Scalar, S::Dim>,
S::Dim: Dim<S::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S as HasConcreteScalarArray<S::Scalar, S::Dim, S::Row>>::Concrete: HasConcreteScalarArray<S::Scalar, S::Dim, S::Row, Concrete=<S as HasConcreteScalarArray<S::Scalar, S::Dim, S::Row>>::Concrete>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Dim as Array<S::Scalar>>::Type>,
<S::Dim as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Row as Array<S::Scalar>>::Type>,
{
    S::Concrete::from_val(array::transpose::<_, _, S::Dim>(s.get_val()))
}

// Multiply a vector by a matrix
#[inline(always)]
pub fn mul_vector<M, V>(m: M, v: V) -> V::Concrete
where M: ScalarArrayVal,
M::Row: Dim<V::Scalar>,
M::Dim: TwoDim<V::Scalar, V::Row> + Dim<<M::Scalar as Mul<V::Scalar>>::Output>,
V: VecArrayVal<Row=M::Row> + HasConcreteVecArray<<M::Scalar as Mul<<V as ScalarArray>::Scalar>>::Output, M::Dim>,
<V::Row as Array<V::Scalar>>::Type: Clone,
M::Scalar: Mul<V::Scalar>,
<M::Scalar as Mul<V::Scalar>>::Output: Add<Output=<M::Scalar as Mul<V::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<V as HasConcreteScalarArray<<M::Scalar as Mul<V::Scalar>>::Output, M::Dim>>::Concrete: ConcreteVecArray,
<M::Row as HasSmaller>::Smaller: Array<M::Scalar> + Array<V::Scalar>,
<M::Dim as HasSmaller>::Smaller: Array<<M::Row as Array<M::Scalar>>::Type> + Array<<V::Row as Array<V::Scalar>>::Type> + Array<<M::Scalar as Mul<V::Scalar>>::Output>,
{
    /* We're trying to accomplish the following multiplication
    a b c   m     a*m + b*n + c*o
    d e f   n     d*m + e*n + f*o
    g h i   o     g*m + h*n + i*o
    j k l         j*m + k*n + l*o
    */

    /* Start by getting the following
    a b c   m n o
    d e f   m n o
    g h i   m n o
    j k l   m n o
    */
    let m = m.get_val();
    let v = <M::Dim as Array<<V::Row as Array<V::Scalar>>::Type>>::from_value(v.get_vec_val());

    // Now just do a bunch of dot products
    V::Concrete::from_vec_val(<M::Dim as Array<<M::Row as Array<M::Scalar>>::Type>>::map_zip::
    <<V::Row as Array<V::Scalar>>::Type, <M::Scalar as Mul<V::Scalar>>::Output, _>(m, v, |m, v| {
        vec_array::dot::<M::Scalar, M::Row, _>(m, v)
    }))
}

// Multiply a vector by a the tranpose of a matrix
#[inline(always)]
pub fn mul_vector_transpose<V, M>(v: V, m: M) -> V::Concrete
where M: ScalarArrayVal,
M::Row: Dim<V::Scalar> + Dim<<V::Scalar as Mul<M::Scalar>>::Output>,
M::Dim: Dim<V::Scalar> + TwoDim<V::Scalar, M::Row>,
V: VecArrayVal<Row=M::Dim> + HasConcreteVecArray<<<V as ScalarArray>::Scalar as Mul<M::Scalar>>::Output, M::Row>,
V::Scalar: Mul<M::Scalar> + Clone,
<V::Scalar as Mul<M::Scalar>>::Output: Add<Output=<V::Scalar as Mul<M::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<V as HasConcreteScalarArray<<V::Scalar as Mul<M::Scalar>>::Output, M::Row>>::Concrete: ConcreteVecArray,
<M::Row as HasSmaller>::Smaller: Array<M::Scalar> + Array<V::Scalar> + Array<<V::Scalar as Mul<M::Scalar>>::Output>,
<M::Dim as HasSmaller>::Smaller: Array<V::Scalar> + Array<<M::Row as Array<M::Scalar>>::Type> + Array<<M::Row as Array<V::Scalar>>::Type>,
{
    let v = v.get_vec_val();
    let m = m.get_val();
    V::Concrete::from_vec_val(
        vec_array::mul_vector_transpose::<V::Scalar, M::Row, M::Dim, _>(v, m)
    )
}

//// Multiply a two matrices together
#[inline(always)]
pub fn mul_matrix<M, N>(m: M, n: N) -> M::Concrete
where M: ScalarArrayVal + HasConcreteScalarArray<<<M as ScalarArray>::Scalar as Mul<N::Scalar>>::Output, N::Row, <M as ScalarArray>::Dim>,
M::Row: TwoDim<N::Scalar, N::Row> + Dim<<N::Row as Array<M::Scalar>>::Type>,
M::Dim: TwoDim<<M::Scalar as Mul<N::Scalar>>::Output, N::Row> + Array<<N::Dim as Array<<N::Row as Array<N::Scalar>>::Type>>::Type>,
N: ScalarArrayVal<Dim=<M as ScalarArray>::Row>,
N::Row: Array<M::Scalar> + Dim<<M::Scalar as Mul<N::Scalar>>::Output>,
M::Scalar: Mul<N::Scalar> + Clone,
<N::Dim as Array<<N::Row as Array<N::Scalar>>::Type>>::Type: Clone,
<M::Scalar as Mul<N::Scalar>>::Output: Add<Output=<M::Scalar as Mul<N::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<M as HasConcreteScalarArray<<<M as ScalarArray>::Scalar as Mul<N::Scalar>>::Output, N::Row, <M as ScalarArray>::Dim>>::Concrete: ConcreteScalarArray,
<M::Row as HasSmaller>::Smaller: Array<M::Scalar> + Array<<N::Row as Array<N::Scalar>>::Type> + Array<<N::Row as Array<M::Scalar>>::Type> ,
<M::Dim as HasSmaller>::Smaller: Array<<M::Row as Array<M::Scalar>>::Type> + Array<<N::Row as Array<<M::Scalar as Mul<N::Scalar>>::Output>>::Type>,
<N::Row as HasSmaller>::Smaller: Array<N::Scalar> + Array<<M::Scalar as Mul<N::Scalar>>::Output>,
{
    /* We're trying to accomplish the following multiplication
    a b c   m n     a*m + b*o + q*p     a*n + b*p + c*r
    d e f   o p     d*m + e*o + f*p     d*n + e*p + f*r
    g h i   q r     g*m + h*o + i*p     g*n + h*p + i*r
    j k l           j*m + k*o + l*p     j*n + k*p + l*r
    */
    let m = m.get_val();
    let n = n.get_val();

    // Duplicate the right hand side so we have one copy for each row
    let n = M::Dim::from_value(n);

    // Now just go throufh and mul_vector_transpose each row to get the output
    M::Concrete::from_val(
        <M::Dim as Array<<M::Row as Array<M::Scalar>>::Type>>::map_zip::<<N::Dim as Array<<N::Row as Array<N::Scalar>>::Type>>::Type, <N::Row as Array<<M::Scalar as Mul<N::Scalar>>::Output>>::Type, _>(
            m, n, |m, n| {
                vec_array::mul_vector_transpose::<M::Scalar, N::Row, N::Dim, N::Scalar>(m, n)
            }
        )
    )
}

/// Types that can be square-rooted.
impl <S: ScalarArrayVal> Sqrt for S
where S::Scalar: Sqrt,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Sqrt>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as Sqrt>::Output>,
S::Dim: TwoDim<<S::Scalar as Sqrt>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Sqrt>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Sqrt>::Output>>::Type>,
{
    /// The resulting type.
    type Output = S::Concrete;

    /// Takes the square root of a number.
    ///
    /// Returns NaN if self is a negative number.
    fn sqrt(self) -> S::Concrete { map(self, Sqrt::sqrt) }

    /// Returns the inverse of the square root of a number. i.e. the value `1/√x`.
    ///
    /// Returns NaN if `self` is a negative number.
    fn inverse_sqrt(self) -> S::Concrete { map(self, Sqrt::inverse_sqrt) }
}

/// Types that can have the reciprocal taken.
impl <S: ScalarArrayVal> Recip for S
where S::Scalar: Recip,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Recip>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as Recip>::Output>,
S::Dim: TwoDim<<S::Scalar as Recip>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Recip>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Recip>::Output>>::Type>,
{
    /// The output type
    type Output = S::Concrete;
    /// Takes the reciprocal (inverse) of a number, `1/x`.
    fn recip(self) -> S::Concrete { map(self, Recip::recip) }
}

// Types that implment hyperbolic angle functions.
impl <S: ScalarArrayVal> Hyperbolic for S
where S::Scalar: Hyperbolic,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Hyperbolic>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as Hyperbolic>::Output>,
S::Dim: TwoDim<<S::Scalar as Hyperbolic>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Hyperbolic>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Hyperbolic>::Output>>::Type>,
{
    /// The output type
    type Output = S::Concrete;
    // Hyperbolic sine function.
    fn sinh(self) -> S::Concrete { map(self, Hyperbolic::sinh) }
    // Hyperbolic cosine function.
    fn cosh(self) -> S::Concrete { map(self, Hyperbolic::cosh) }
    // Hyperbolic tangent function.
    fn tanh(self) -> S::Concrete { map(self, Hyperbolic::tanh) }
    // Hyperbolic sine function.
    fn asinh(self) -> S::Concrete { map(self, Hyperbolic::asinh) }
    // Hyperbolic cosine function.
    fn acosh(self) -> S::Concrete { map(self, Hyperbolic::acosh) }
    // Hyperbolic tangent function.
    fn atanh(self) -> S::Concrete { map(self, Hyperbolic::atanh) }
}

impl <S: ScalarArrayVal> Abs for S
where S::Scalar: Abs,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Abs>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as Abs>::Output>,
S::Dim: TwoDim<<S::Scalar as Abs>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Abs>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Abs>::Output>>::Type>,
{
    /// The resulting type.
    type Output = S::Concrete;

    /// Computes the absolute value of self. Returns NaN if the number is NaN.
    fn abs(self) -> S::Concrete { map(self, Abs::abs) }

    /// Returns |self-rhs| without modulo overflow.
    fn abs_diff(self, rhs: Self) -> S::Concrete {
        map_zip::<Self, Self, _, _>(self, rhs, Abs::abs_diff)
    }
}

/// Types that implement the Pow function
impl <Lhs: ScalarArrayVal, Rhs: ScalarArrayVal<Row=Lhs::Row,Dim=Lhs::Dim>> Pow<Rhs> for Lhs
where Lhs::Row: Dim<Rhs::Scalar>,
Lhs::Dim: TwoDim<Rhs::Scalar, Lhs::Row>,
Lhs::Scalar: Pow<Rhs::Scalar>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Pow<<Rhs as ScalarArray>::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as Pow<<Rhs as ScalarArray>::Scalar>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as Pow<<Rhs as ScalarArray>::Scalar>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<Rhs::Scalar> + Array<<Lhs::Scalar as Pow<Rhs::Scalar>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<Rhs::Row as Array<Rhs::Scalar>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Pow<Rhs::Scalar>>::Output>>::Type>,
{
    /// The output type
    type Output = Lhs::Concrete;
    /// Returns `self` raised to the power `rhs`
    fn pow(self, rhs: Rhs) -> Lhs::Concrete {
        map_zip(self, rhs, Pow::pow)
    }
}

/// Types that implement exponential functions
impl <S: ScalarArrayVal> Exp for S
where S::Scalar: Exp,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Exp>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<<S as ScalarArray>::Scalar as Exp>::Output>,
S::Dim: TwoDim<<<S as ScalarArray>::Scalar as Exp>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Exp>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Exp>::Output>>::Type>,
{
    /// The output type
    type Output = S::Concrete;
    /// Returns `e^(self)`, (the exponential function).
    fn exp(self) -> S::Concrete { map(self, Exp::exp) }
    /// Returns `2^(self)`.
    fn exp2(self) -> S::Concrete { map(self, Exp::exp2) }
    /// Returns the natural logarithm of the number.
    fn ln(self) -> S::Concrete { map(self, Exp::ln) }
    /// Returns the base 2 logarithm of the number.
    fn log2(self) -> S::Concrete { map(self, Exp::log2) }
    /// Returns the base 10 logarithm of the number.
    fn log10(self) -> S::Concrete { map(self, Exp::log10) }
}

/// Types that can be rounded.
impl <S: ScalarArrayVal> Round for S
where S::Scalar: Round,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Round>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<<S as ScalarArray>::Scalar as Round>::Output>,
S::Dim: TwoDim<<<S as ScalarArray>::Scalar as Round>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Round>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Round>::Output>>::Type>,
{
    /// The output type
    type Output = S::Concrete;
    /// Returns the largest integer less than or equal to a number.
    fn floor(self) -> S::Concrete { map(self, Round::floor) }
    /// Returns the smallest integer greater than or equal to a number.
    fn ceil(self) -> S::Concrete { map(self, Round::ceil) }
    /// Returns the integer part of a number.
    fn trunc(self) -> S::Concrete { map(self, Round::trunc) }
    /// Returns the fractional part of a number.
    fn fract(self) -> S::Concrete { map(self, Round::fract) }
    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    fn round(self) -> S::Concrete { map(self, Round::round) }
}

/// Types that can be clamped
impl <Lhs: ScalarArrayVal, Rhs: ScalarArrayVal<Row=Lhs::Row,Dim=Lhs::Dim>> Clamp<Rhs> for Lhs
where Lhs::Row: Dim<Rhs::Scalar>,
Lhs::Dim: TwoDim<Rhs::Scalar, Lhs::Row>,
Lhs::Scalar: Clamp<Rhs::Scalar>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Clamp<<Rhs as ScalarArray>::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as Clamp<<Rhs as ScalarArray>::Scalar>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as Clamp<<Rhs as ScalarArray>::Scalar>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<Rhs::Scalar> + Array<<Lhs::Scalar as Clamp<Rhs::Scalar>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<Rhs::Row as Array<Rhs::Scalar>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Clamp<Rhs::Scalar>>::Output>>::Type>,
{
    /// The output type
    type Output = Lhs::Concrete;
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn min(self, rhs: Rhs) -> Lhs::Concrete {
        map_zip(self, rhs, Clamp::min)
    }
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn max(self, rhs: Rhs) -> Lhs::Concrete {
        map_zip(self, rhs, Clamp::max)
    }
}

/*

// TODO: common functions

// Returns linear blend of x and y:
//  Tfd mix(Tfd x, Tfd y, Tfd a)
//  Tf  mix(Tf x, Tf y, float a)
//  Td  mix(Td x, Td y, double a)
//  Ti  mix(Ti x, Ti y, Ti a)
//  Tu  mix(Tu x, Tu y, Tu a)
// Components returned come from x when a components are true, from y when a components are false:
//  Tfd mix(Tfd x, Tfd y, Tb a)
//  Tb  mix(Tb x, Tb y, Tb a)
//  Tiu mix(Tiu x, Tiu y, Tb a)
// Returns 0.0 if x < edge, else 1.0:
//  Tfd step(Tfd edge, Tfd x)
//  Tf  step(float edge, Tf x)
//  Td  step(double edge, Td x)
// Clamps and smoothes:
//  Tfd smoothstep(Tfd edge0, Tfd edge1, Tfd x)
//  Tf  smoothstep(float edge0, float edge1, Tf x)
//  Td  smoothstep(double edge0, double edge1, Td x)
// Returns true if x is NaN:
//  Tb  isnan(Tfd x)
// Returns true if x is positive or negative infinity:
//  Tb  isinf(Tfd x)
// Returns signed int or uint value of the encoding of a float :
//  Ti  floatBitsToint (Tf value)
//  Tu  floatBitsToUint (Tf value)
// Returns float value of a signed int  or uint encoding of a float :
//  Tf  intBitsTofloat (Ti value)
//  Tf uintBitsTofloat (Tu value)
// Computes and returns a*b + c.Treated as a single operation when using precise:
//  Tfd  fma(Tfd a, Tfd b, Tfd c)
// Splits x into a floating-point significand in the range[0.5, 1.0) and an integer exponent of 2:
//  Tfd  frexp(Tfd x, out Ti exp)
// Builds a floating-point number from x and the corresponding integral exponent of 2 in exp:
//  Tfd  ldexp(Tfd x, in Ti exp)
*/

//impl Scalar for Option<Ordering> {}
//impl Scalar for Ordering {}
//
//include!(concat!(env!("OUT_DIR"), "/scalar_array.rs"));
