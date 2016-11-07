//! Traits and functions that operate on `ScalarArrays`.

// TODO: more commments

use std::cmp::Ordering;
use std::ops::{
    Add,Mul,MulAssign,
};

use super::Value;
use num::{
    Abs,Clamp,Exp,FloatCategory,FloatTransmute,
    FractionExponent,Hyperbolic,LoadExponent,
    Mix,MulAdd,Pow,Recip,Round,Sqrt,Step,
};
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
    pub fn fold2<S, R, T, F0, F, O>(s: <R as Array<S>>::Type, t: <R as Array<T>>::Type, f0: F0, mut f: F) -> O
    where R: Dim<S> + Dim<T>,
    F0: FnOnce(S, T) -> O,
    F: FnMut(O, S, T) -> O,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    R::Smaller: Array<S> + Array<T>,
    {
        let (s0, s) = R::split(s);
        let (t0, t) = R::split(t);
        let init = f0(s0, t0);
        <R::Smaller as Array<S>>::fold2(s, t, init, &mut f)
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
        fold2::<S, R, T, _, _, _>(s, t, |s, t| s*t, |init, s, t| init + (s*t))
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
        let v = D::map(v, <R as Array<S>>::from_value);

        // Now we're going to multiply the items together and then sum the rows down
        // We can do this with a single fold2 if we're clever, which we are
        fold2::<<R as Array<S>>::Type, D, <R as Array<T>>::Type, _, _, _>(
            v, m,
            // multiply the first items (a*x)
            |v, m| <R as Array<S>>::map2(v, m, |v, m| v * m),
            |init, v, m| {
                // Multiply the second/etc items (b*x or c*x) and add that to the running total
                <R as Array<S::Output>>::map3::<S, T, _, _>(init, v, m, |init, v, m| init + (v * m))
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
    pub fn apply2<S, R, D, T, F>(s: <D as Array<<R as Array<S>>::Type>>::Type, t: <D as Array<<R as Array<T>>::Type>>::Type, mut f: F)
    where R: Array<S> + Array<T>,
    D: Array<<R as Array<S>>::Type> + Array<<R as Array<T>>::Type>,
    F: FnMut(S, T) {
        D::apply2(s, t, |s, t| R::apply2(s, t, &mut f))
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
    pub fn fold2<S, R, D, T, F0, F, O>(s: <D as Array<<R as Array<S>>::Type>>::Type, t: <D as Array<<R as Array<T>>::Type>>::Type, f0: F0, mut f: F) -> O
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
        let init = vec_array::fold2::<S, R, T, F0, _, O>(s0, t0, f0, &mut f);
        <D::Smaller as Array<<R as Array<S>>::Type>>::fold2(s, t, init,
            |init, s, t| R::fold2(s, t, init, &mut f)
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

    /// Construct a 2d array from another 2d array and a mapping function
    #[inline(always)]
    pub fn map_into_2<S, R, D, T, U, F>(s: <D as Array<<R as Array<S>>::Type>>::Type, mut f: F) -> (<D as Array<<R as Array<T>>::Type>>::Type, <D as Array<<R as Array<U>>::Type>>::Type)
    where R: Array<S> + Array<T> + Array<U>,
    D: Array<<R as Array<S>>::Type> + Array<<R as Array<T>>::Type> + Array<<R as Array<U>>::Type>,
    F: FnMut(S) -> (T, U) {
        D::map_into_2(s, |s| R::map_into_2(s, &mut f))
    }

    /// Construct a 2d array from two other 2d arrays and a mapping function
    #[inline(always)]
    pub fn map2<S, R, D, T, U, F>(s: <D as Array<<R as Array<S>>::Type>>::Type, t: <D as Array<<R as Array<T>>::Type>>::Type, mut f: F) -> <D as Array<<R as Array<U>>::Type>>::Type
    where R: Array<S> + Array<T> + Array<U>,
    D: Array<<R as Array<S>>::Type> + Array<<R as Array<T>>::Type> + Array<<R as Array<U>>::Type>,
    F: FnMut(S, T) -> U, {
        D::map2(s, t, |s, t| R::map2(s, t, &mut f))
    }

    /// Construct a 2d array from three other 2d arrays and a mapping function
    #[inline(always)]
    pub fn map3<S, R, D, T, U, V, F>(s: <D as Array<<R as Array<S>>::Type>>::Type, t: <D as Array<<R as Array<T>>::Type>>::Type, u: <D as Array<<R as Array<U>>::Type>>::Type, mut f: F) -> <D as Array<<R as Array<V>>::Type>>::Type
    where R: Array<S> + Array<T> + Array<U> + Array<V>,
    D: Array<<R as Array<S>>::Type> + Array<<R as Array<T>>::Type> + Array<<R as Array<U>>::Type> + Array<<R as Array<V>>::Type>,
    F: FnMut(S, T, U) -> V, {
        D::map3(s, t, u, |s, t, u| R::map3(s, t, u, &mut f))
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


/// Construct a `ScalarArray` from a single value
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

/// Apply a function to all elements of a `ScalarArray`
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

/// Apply a function to all elements of a `ScalarArray`
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

/// Apply a function to all elements of a `ScalarArray`
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


/// Apply a function to all elements of two `ScalarArray`s
#[inline(always)]
pub fn apply2<S, T, F>(s: S, t: T, f: F)
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar, T::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<T::Row as Array<T::Scalar>>::Type>,
{
    array::apply2::<S::Scalar, S::Row, S::Dim, _, F>(s.get_val(), t.get_val(), f)
}

/// Apply a function to all elements of two `ScalarArray`s
#[inline(always)]
pub fn apply2_val_ref<'b, S, T, F>(s: S, t: &'b T, f: F)
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
    array::apply2::<S::Scalar, S::Row, S::Dim, _, F>(s.get_val(), t.get_ref(), f)
}

/// Apply a function to all elements of two `ScalarArray`s
#[inline(always)]
pub fn apply2_val_mut<'b, S, T, F>(s: S, t: &'b mut T, f: F)
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
    array::apply2::<S::Scalar, S::Row, S::Dim, _, F>(s.get_val(), t.get_mut(), f)
}

/// Apply a function to all elements of two `ScalarArray`s
#[inline(always)]
pub fn apply2_ref_val<'a, S, T, F>(s: &'a S, t: T, f: F)
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
    array::apply2::<_, S::Row, S::Dim, _, F>(s.get_ref(), t.get_val(), f)
}

/// Apply a function to all elements of two `ScalarArray`s
#[inline(always)]
pub fn apply2_ref<'a, 'b, S, T, F>(s: &'a S, t: &'b T, f: F)
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
    array::apply2::<_, S::Row, S::Dim, _, F>(s.get_ref(), t.get_ref(), f)
}

/// Apply a function to all elements of two `ScalarArray`s
#[inline(always)]
pub fn apply2_ref_mut<'a, 'b, S, T, F>(s: &'a S, t: &'b mut T, f: F)
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
    array::apply2::<_, S::Row, S::Dim, _, F>(s.get_ref(), t.get_mut(), f)
}

/// Apply a function to all elements of two `ScalarArray`s
#[inline(always)]
pub fn apply2_mut_val<'a, S, T, F>(s: &'a mut S, t: T, f: F)
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
    array::apply2::<_, S::Row, S::Dim, _, F>(s.get_mut(), t.get_val(), f)
}

/// Apply a function to all elements of two `ScalarArray`s
#[inline(always)]
pub fn apply2_mut_ref<'a, 'b, S, T, F>(s: &'a mut S, t: &'b T, f: F)
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
    array::apply2::<_, S::Row, S::Dim, _, F>(s.get_mut(), t.get_ref(), f)
}

/// Apply a function to all elements of two `ScalarArray`s
#[inline(always)]
pub fn apply2_mut<'a, 'b, S, T, F>(s: &'a mut S, t: &'b mut T, f: F)
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
    array::apply2::<_, S::Row, S::Dim, _, F>(s.get_mut(), t.get_mut(), f)
}


/// Fold all the scalars in a `ScalarArray`. The first element is mapped with `f0`,
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

/// Fold all the scalars in a `ScalarArray` by reference. The first element is mapped with `f0`,
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

/// Fold all the scalars in a `ScalarArray` by mutable reference. The first element is mapped with
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


/// Fold all the scalars in two `ScalarArray`s. The first elements are mapped with `f0`,
/// then folding continues with `f` for other elements.
#[inline(always)]
pub fn fold2<S, T, F0, F, O>(s: S, t: T, f0: F0, f: F) -> O
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
    array::fold2::<S::Scalar, S::Row, S::Dim, T::Scalar, F0, F, O>(s.get_val(), t.get_val(), f0, f)
}

/// Fold all the scalars in two `ScalarArray`s. The first elements are mapped with `f0`,
/// then folding continues with `f` for other elements.
#[inline(always)]
pub fn fold2_ref<'a, S, T, F0, F, O>(s: &'a S, t: &'a T, f0: F0, f: F) -> O
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
    array::fold2::<&'a S::Scalar, S::Row, S::Dim, &'a T::Scalar, F0, F, O>(s.get_ref(), t.get_ref(), f0, f)
}


/// Construct a `ScalarArray` from another `ScalarArray` and a mapping function
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

/// Construct a `ScalarArray` from another `ScalarArray` by reference and a mapping function
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

/// Construct a `ScalarArray` from another `ScalarArray` by reference and a mapping function
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


/// Construct two `ScalarArray`s from a `ScalarArray` and a mapping function
#[inline(always)]
pub fn map_into_2<S, T, U, F>(s: S, f: F) -> (T, U)
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar> + Dim<U::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row>,
T: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
U: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar) -> (T::Scalar, U::Scalar),
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type>,
{
    let r = array::map_into_2::<S::Scalar, S::Row, S::Dim, _, _, F>(s.get_val(), f);
    (T::from_val(r.0), U::from_val(r.1))
}

/// Construct a `ScalarArray` from two other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map2<S, T, U, F>(s: S, t: T, f: F) -> U
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
    U::from_val(array::map2::<S::Scalar, S::Row, S::Dim, _, _, F>(s.get_val(), t.get_val(), f))
}

/// Construct a `ScalarArray` from two other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map2_val_ref<'b, S, T, U, F>(s: S, t: &'b T, f: F) -> U
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
    U::from_val(array::map2::<S::Scalar, S::Row, S::Dim, _, _, F>(s.get_val(), t.get_ref(), f))
}

/// Construct a `ScalarArray` from two other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map2_val_mut<'b, S, T, U, F>(s: S, t: &'b mut T, f: F) -> U
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
    U::from_val(array::map2::<S::Scalar, S::Row, S::Dim, _, _, F>(s.get_val(), t.get_mut(), f))
}

/// Construct a `ScalarArray` from two other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map2_ref_val<'a, S, T, U, F>(s: &'a S, t: T, f: F) -> U
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
    U::from_val(array::map2::<_, S::Row, S::Dim, _, _, F>(s.get_ref(), t.get_val(), f))
}

/// Construct a `ScalarArray` from two other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map2_ref<'a, 'b, S, T, U, F>(s: &'a S, t: &'b T, f: F) -> U
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
    U::from_val(array::map2::<_, S::Row, S::Dim, _, _, F>(s.get_ref(), t.get_ref(), f))
}

/// Construct a `ScalarArray` from two other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map2_ref_mut<'a, 'b, S, T, U, F>(s: &'a S, t: &'b mut T, f: F) -> U
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
    U::from_val(array::map2::<_, S::Row, S::Dim, _, _, F>(s.get_ref(), t.get_mut(), f))
}

/// Construct a `ScalarArray` from two other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map2_mut_val<'a, S, T, U, F>(s: &'a mut S, t: T, f: F) -> U
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
    U::from_val(array::map2::<_, S::Row, S::Dim, _, _, F>(s.get_mut(), t.get_val(), f))
}

/// Construct a `ScalarArray` from two other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map2_mut_ref<'a, 'b, S, T, U, F>(s: &'a mut S, t: &'b T, f: F) -> U
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
    U::from_val(array::map2::<_, S::Row, S::Dim, _, _, F>(s.get_mut(), t.get_ref(), f))
}

/// Construct a `ScalarArray` from two other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map2_mut<'a, S, T, U, F>(s: &'a mut S, t: &'a mut T, f: F) -> U
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
    U::from_val(array::map2::<_, S::Row, S::Dim, _, _, F>(s.get_mut(), t.get_mut(), f))
}

/// Construct a `ScalarArray` from three other `ScalarArray` and a mapping function
#[inline(always)]
pub fn map3<S, T, U, V, F>(s: S, t: T, u: U, f: F) -> V
where S: ScalarArrayVal,
S::Row: Dim<T::Scalar> + Dim<U::Scalar> + Dim<V::Scalar>,
S::Dim: TwoDim<T::Scalar, S::Row> + TwoDim<U::Scalar, S::Row> + TwoDim<V::Scalar, S::Row>,
T: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
U: ScalarArrayVal<Row=S::Row, Dim=S::Dim>,
V: ConcreteScalarArray<Row=S::Row, Dim=S::Dim>,
F: FnMut(S::Scalar, T::Scalar, U::Scalar) -> V::Scalar,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar> + Array<U::Scalar> + Array<V::Scalar>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<T::Scalar>>::Type> + Array<<S::Row as Array<U::Scalar>>::Type> + Array<<S::Row as Array<V::Scalar>>::Type>,
{
    V::from_val(array::map3::<S::Scalar, S::Row, S::Dim, _, _, _, F>(s.get_val(), t.get_val(), u.get_val(), f))
}

/// Construct a `ScalarArray` based on the equality of the components of two `ScalarArray`.
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
    map2_ref(s, t, |s, t| s == t)
}

/// Construct a `ScalarArray` based on the inequality of the components of two `ScalarArray`.
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
    map2_ref(s, t, |s, t| s != t)
}

/// Construct a `ScalarArray` based on the partial ordering of the components of two `ScalarArray`.
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
    map2_ref(s, t, |s, t| s.partial_cmp(t))
}

/// Construct a `ScalarArray` based on the `<` comparison of the components of two `ScalarArray`.
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
    map2_ref(s, t, |s, t| s < t)
}

/// Construct a `ScalarArray` based on the `<=` comparison of the components of two `ScalarArray`.
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
    map2_ref(s, t, |s, t| s <= t)
}

/// Construct a `ScalarArray` based on the `>` comparison of the components of two `ScalarArray`.
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
    map2_ref(s, t, |s, t| s > t)
}

/// Construct a `ScalarArray` based on the `>=` comparison of the components of two `ScalarArray`.
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
    map2_ref(s, t, |s, t| s >= t)
}

/// Construct a `ScalarArray` based on the ordering of the components of two `ScalarArray`.
#[inline(always)]
pub fn cpt_cmp<S, T>(s1: &S, s2: &S) -> S::Concrete
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
    map2_ref(s1, s2, |s1, s2| s1.cmp(s2))
}

/// Multiply two `ScalarArray`s component-wise.
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
    map2(s, t, |s, t| s * t)
}

/// Multiply two `ScalarArray`s component-wise.
#[inline(always)]
pub fn cpt_mul_assign<S, T, U>(s: &mut S, t: T)
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
    apply2_mut_val(s, t, |s, t| *s *= t);
}

/// Sums all the components of a `ScalarArray`
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

/// Transposes a `ScalarArray`
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
    V::Concrete::from_vec_val(<M::Dim as Array<<M::Row as Array<M::Scalar>>::Type>>::map2::
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
        <M::Dim as Array<<M::Row as Array<M::Scalar>>::Type>>::map2::<<N::Dim as Array<<N::Row as Array<N::Scalar>>::Type>>::Type, <N::Row as Array<<M::Scalar as Mul<N::Scalar>>::Output>>::Type, _>(
            m, n, |m, n| {
                vec_array::mul_vector_transpose::<M::Scalar, N::Row, N::Dim, N::Scalar>(m, n)
            }
        )
    )
}

// d888888b d8888b.  .d8b.  d888888b d888888b .d8888.
// `~~88~~' 88  `8D d8' `8b   `88'   `~~88~~' 88'  YP
//    88    88oobY' 88ooo88    88       88    `8bo.
//    88    88`8b   88~~~88    88       88      `Y8b.
//    88    88 `88. 88   88   .88.      88    db   8D
//    YP    88   YD YP   YP Y888888P    YP    `8888Y'

impl<Lhs, Rhs> Abs<Rhs> for Lhs
where Lhs: ScalarArrayVal + HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Abs<Rhs::Scalar>>::Output>,
Rhs: ScalarArrayVal<Row=<Lhs as ScalarArray>::Row, Dim=<Lhs as ScalarArray>::Dim>,
Lhs::Row: Dim<Rhs::Scalar>,
Lhs::Dim: TwoDim<Rhs::Scalar, Lhs::Row>,
Lhs::Scalar: Abs<Rhs::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<Lhs::Scalar as Abs<Rhs::Scalar>>::Output>,
Lhs::Dim: TwoDim<<Lhs::Scalar as Abs<Rhs::Scalar>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<Rhs::Scalar> + Array<<Lhs::Scalar as Abs<Rhs::Scalar>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<Lhs::Row as Array<Rhs::Scalar>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Abs<Rhs::Scalar>>::Output>>::Type>,
{
    type Output = Lhs::Concrete;
    fn abs(self) -> Lhs::Concrete { map(self, Abs::abs) }
    fn abs_diff(self, rhs: Rhs) -> Lhs::Concrete { map2(self, rhs, Abs::abs_diff) }
}

impl<Lhs, Rhs> Clamp<Rhs> for Lhs
where Lhs: ScalarArrayVal,
Rhs: ScalarArrayVal<Row=Lhs::Row, Dim=Lhs::Dim>,
Lhs::Row: Dim<Rhs::Scalar>,
Lhs::Dim: TwoDim<Rhs::Scalar, Lhs::Row>,
Lhs::Scalar: Clamp<Rhs::Scalar>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Clamp<<Rhs as ScalarArray>::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as Clamp<<Rhs as ScalarArray>::Scalar>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as Clamp<<Rhs as ScalarArray>::Scalar>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<Rhs::Scalar> + Array<<Lhs::Scalar as Clamp<Rhs::Scalar>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<Rhs::Row as Array<Rhs::Scalar>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Clamp<Rhs::Scalar>>::Output>>::Type>,
{
    type Output = Lhs::Concrete;
    fn min(self, rhs: Rhs) -> Lhs::Concrete { map2(self, rhs, Clamp::min) }
    fn max(self, rhs: Rhs) -> Lhs::Concrete { map2(self, rhs, Clamp::max) }
    fn clamp(self, min: Rhs, max: Rhs) -> Lhs::Concrete { map3(self, min, max, Clamp::clamp) }
}

impl<Lhs, Rhs> Clamp<Value<Rhs>> for Lhs
where Lhs: ScalarArrayVal, Rhs: Clone,
Lhs::Row: Dim<Rhs>,
<Lhs::Row as Array<Rhs>>::Type: Clone,
Lhs::Dim: TwoDim<Rhs, Lhs::Row>,
Lhs::Scalar: Clamp<Rhs>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Clamp<Rhs>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as Clamp<Rhs>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as Clamp<Rhs>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<Rhs> + Array<<Lhs::Scalar as Clamp<Rhs>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<Lhs::Row as Array<Rhs>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Clamp<Rhs>>::Output>>::Type>,
Lhs::Concrete: HasConcreteScalarArray<<Lhs::Scalar as Clamp<Rhs>>::Output, Lhs::Row, Lhs::Dim, Concrete=Lhs::Concrete>,
{
    type Output = Lhs::Concrete;
    fn min(self, rhs: Value<Rhs>) -> Lhs::Concrete {
        let rhs = <Lhs::Row as Array<Rhs>>::from_value(rhs.0);
        let rhs = <Lhs::Dim as Array<<Lhs::Row as Array<Rhs>>::Type>>::from_value(rhs);
        Lhs::Concrete::from_val(array::map2::<Lhs::Scalar, Lhs::Row, Lhs::Dim, _, _, _>(self.get_val(), rhs, Clamp::min))
    }
    fn max(self, rhs: Value<Rhs>) -> Lhs::Concrete {
        let rhs = <Lhs::Row as Array<Rhs>>::from_value(rhs.0);
        let rhs = <Lhs::Dim as Array<<Lhs::Row as Array<Rhs>>::Type>>::from_value(rhs);
        Lhs::Concrete::from_val(array::map2::<Lhs::Scalar, Lhs::Row, Lhs::Dim, _, _, _>(self.get_val(), rhs, Clamp::max))
    }
    fn clamp(self, min: Value<Rhs>, max: Value<Rhs>) -> Lhs::Concrete {
        let min = <Lhs::Row as Array<Rhs>>::from_value(min.0);
        let max = <Lhs::Row as Array<Rhs>>::from_value(max.0);
        let min = <Lhs::Dim as Array<<Lhs::Row as Array<Rhs>>::Type>>::from_value(min);
        let max = <Lhs::Dim as Array<<Lhs::Row as Array<Rhs>>::Type>>::from_value(max);
        Lhs::Concrete::from_val(array::map3::<Lhs::Scalar, Lhs::Row, Lhs::Dim, _, _, _, _>(self.get_val(), min, max, Clamp::clamp))
    }
}

impl<S> Exp for S
where S: ScalarArrayVal,
S::Scalar: Exp,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Exp>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<<S as ScalarArray>::Scalar as Exp>::Output>,
S::Dim: TwoDim<<<S as ScalarArray>::Scalar as Exp>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Exp>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Exp>::Output>>::Type>,
{
    type Output = S::Concrete;
    fn exp(self) -> S::Concrete { map(self, Exp::exp) }
    fn exp2(self) -> S::Concrete { map(self, Exp::exp2) }
    fn ln(self) -> S::Concrete { map(self, Exp::ln) }
    fn log2(self) -> S::Concrete { map(self, Exp::log2) }
    fn log10(self) -> S::Concrete { map(self, Exp::log10) }
}

impl<S> FloatCategory for S
where S: ScalarArrayVal,
S::Scalar: FloatCategory,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as FloatCategory>::Bool> + HasConcreteScalarArray<<<S as ScalarArray>::Scalar as FloatCategory>::FpCategory>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<<S as ScalarArray>::Scalar as FloatCategory>::Bool> + Dim<<<S as ScalarArray>::Scalar as FloatCategory>::FpCategory>,
S::Dim: TwoDim<<<S as ScalarArray>::Scalar as FloatCategory>::Bool, S::Row> + TwoDim<<<S as ScalarArray>::Scalar as FloatCategory>::FpCategory, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as FloatCategory>::Bool> + Array<<S::Scalar as FloatCategory>::FpCategory>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as FloatCategory>::Bool>>::Type> + Array<<S::Row as Array<<S::Scalar as FloatCategory>::FpCategory>>::Type>,
{
    type Bool = <S as HasConcreteScalarArray<<<S as ScalarArray>::Scalar as FloatCategory>::Bool>>::Concrete;
    type FpCategory = <S as HasConcreteScalarArray<<<S as ScalarArray>::Scalar as FloatCategory>::FpCategory>>::Concrete;
    fn is_nan(self) -> Self::Bool { map(self, FloatCategory::is_nan) }
    fn is_infinite(self) -> Self::Bool { map(self, FloatCategory::is_infinite) }
    fn is_finite(self) -> Self::Bool { map(self, FloatCategory::is_finite) }
    fn is_normal(self) -> Self::Bool { map(self, FloatCategory::is_normal) }
    fn classify(self) -> Self::FpCategory { map(self, FloatCategory::classify) }
}


impl<S> FloatTransmute for S
where S: ScalarArrayVal,
S::Scalar: FloatTransmute,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as FloatTransmute>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<<S as ScalarArray>::Scalar as FloatTransmute>::Output>,
S::Dim: TwoDim<<<S as ScalarArray>::Scalar as FloatTransmute>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as FloatTransmute>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as FloatTransmute>::Output>>::Type>,
{
    type Output = S::Concrete;
    fn float_transmute(self) -> S::Concrete { map(self, FloatTransmute::float_transmute) }
}

impl<S> FractionExponent for S
where S: ScalarArrayVal,
S::Scalar: FractionExponent,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as FractionExponent>::Fraction> + HasConcreteScalarArray<<<S as ScalarArray>::Scalar as FractionExponent>::Exponent>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as FractionExponent>::Fraction> + Dim<<S::Scalar as FractionExponent>::Exponent>,
S::Dim: TwoDim<<S::Scalar as FractionExponent>::Fraction, S::Row> + TwoDim<<S::Scalar as FractionExponent>::Exponent, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as FractionExponent>::Fraction> + Array<<S::Scalar as FractionExponent>::Exponent>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as FractionExponent>::Fraction>>::Type> + Array<<S::Row as Array<<S::Scalar as FractionExponent>::Exponent>>::Type>,
{
    type Fraction = <S as HasConcreteScalarArray<<<S as ScalarArray>::Scalar as FractionExponent>::Fraction>>::Concrete;
    type Exponent = <S as HasConcreteScalarArray<<<S as ScalarArray>::Scalar as FractionExponent>::Exponent>>::Concrete;
    fn frexp(self) -> (Self::Fraction, Self::Exponent) { map_into_2(self, FractionExponent::frexp) }
}

impl<S> Hyperbolic for S
where S: ScalarArrayVal,
S::Scalar: Hyperbolic,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Hyperbolic>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as Hyperbolic>::Output>,
S::Dim: TwoDim<<S::Scalar as Hyperbolic>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Hyperbolic>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Hyperbolic>::Output>>::Type>,
{
    type Output = S::Concrete;
    fn sinh(self) -> S::Concrete { map(self, Hyperbolic::sinh) }
    fn cosh(self) -> S::Concrete { map(self, Hyperbolic::cosh) }
    fn tanh(self) -> S::Concrete { map(self, Hyperbolic::tanh) }
    fn asinh(self) -> S::Concrete { map(self, Hyperbolic::asinh) }
    fn acosh(self) -> S::Concrete { map(self, Hyperbolic::acosh) }
    fn atanh(self) -> S::Concrete { map(self, Hyperbolic::atanh) }
}

impl<S, Exponent> LoadExponent<Exponent> for S
where S: ScalarArrayVal + HasConcreteScalarArray<<<S as ScalarArray>::Scalar as LoadExponent<Exponent::Scalar>>::Output>,
Exponent: ScalarArrayVal<Row=<S as ScalarArray>::Row, Dim=<S as ScalarArray>::Dim>,
S::Row: Dim<Exponent::Scalar>,
S::Dim: TwoDim<Exponent::Scalar, S::Row>,
S::Scalar: LoadExponent<Exponent::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as LoadExponent<Exponent::Scalar>>::Output>,
S::Dim: TwoDim<<S::Scalar as LoadExponent<Exponent::Scalar>>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<Exponent::Scalar> + Array<<S::Scalar as LoadExponent<Exponent::Scalar>>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<Exponent::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as LoadExponent<Exponent::Scalar>>::Output>>::Type>,
{
    type Output = S::Concrete;
    fn ldexp(self, exponent: Exponent) -> S::Concrete { map2(self, exponent, LoadExponent::ldexp) }
}

impl<Lhs, A, Rhs> Mix<A, Rhs> for Lhs
where Lhs: ScalarArrayVal,
A: ScalarArrayVal<Row=<Lhs as ScalarArray>::Row, Dim=<Lhs as ScalarArray>::Dim>,
Rhs: ScalarArrayVal<Row=<Lhs as ScalarArray>::Row, Dim=<Lhs as ScalarArray>::Dim>,
Lhs::Row: Dim<A::Scalar> + Dim<Rhs::Scalar>,
Lhs::Dim: TwoDim<A::Scalar, Lhs::Row> + TwoDim<Rhs::Scalar, Lhs::Row>,
Lhs::Scalar: Mix<A::Scalar, Rhs::Scalar>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Mix<A::Scalar, <Rhs as ScalarArray>::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as Mix<A::Scalar, <Rhs as ScalarArray>::Scalar>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as Mix<A::Scalar, <Rhs as ScalarArray>::Scalar>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<A::Scalar> + Array<Rhs::Scalar> + Array<<Lhs::Scalar as Mix<A::Scalar, Rhs::Scalar>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> +  Array<<Lhs::Row as Array<A::Scalar>>::Type> + Array<<Rhs::Row as Array<Rhs::Scalar>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Mix<A::Scalar, Rhs::Scalar>>::Output>>::Type>,
{
    type Output = Lhs::Concrete;
    fn mix(self, y: Rhs, a: A) -> Lhs::Concrete { map3(self, y, a, Mix::mix) }
}

impl<Lhs, A, Rhs> Mix<Value<A>, Rhs> for Lhs
where Lhs: ScalarArrayVal,
A: Clone,
Rhs: ScalarArrayVal<Row=<Lhs as ScalarArray>::Row, Dim=<Lhs as ScalarArray>::Dim>,
Lhs::Row: Dim<A> + Dim<Rhs::Scalar>,
<Lhs::Row as Array<A>>::Type: Clone,
Lhs::Dim: TwoDim<A, Lhs::Row> + TwoDim<Rhs::Scalar, Lhs::Row>,
Lhs::Scalar: Mix<A, Rhs::Scalar>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Mix<A, Rhs::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as Mix<A, Rhs::Scalar>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as Mix<A, Rhs::Scalar>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<A> + Array<Rhs::Scalar> + Array<<Lhs::Scalar as Mix<A, Rhs::Scalar>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<Lhs::Row as Array<A>>::Type> + Array<<Lhs::Row as Array<Rhs::Scalar>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Mix<A, Rhs::Scalar>>::Output>>::Type>,
Lhs::Concrete: HasConcreteScalarArray<<Lhs::Scalar as Mix<A, Rhs::Scalar>>::Output, Lhs::Row, Lhs::Dim, Concrete=Lhs::Concrete>,
{
    type Output = Lhs::Concrete;
    fn mix(self, y: Rhs, a: Value<A>) -> Lhs::Concrete {
        let a = <Lhs::Row as Array<A>>::from_value(a.0);
        let a = <Lhs::Dim as Array<<Lhs::Row as Array<A>>::Type>>::from_value(a);
        Lhs::Concrete::from_val(array::map3::<Lhs::Scalar, Lhs::Row, Lhs::Dim, _, _, _, _>(self.get_val(), y.get_val(), a, Mix::mix))
    }
}

impl<Lhs, A, B> MulAdd<A, B> for Lhs
where Lhs: ScalarArrayVal,
A: ScalarArrayVal<Row=Lhs::Row, Dim=Lhs::Dim>,
B: ScalarArrayVal<Row=Lhs::Row, Dim=Lhs::Dim>,
Lhs::Row: Dim<A::Scalar> + Dim<B::Scalar>,
Lhs::Dim: TwoDim<A::Scalar, Lhs::Row> + TwoDim<B::Scalar, Lhs::Row>,
Lhs::Scalar: MulAdd<A::Scalar, B::Scalar>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as MulAdd<<A as ScalarArray>::Scalar, <B as ScalarArray>::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as MulAdd<<A as ScalarArray>::Scalar, <B as ScalarArray>::Scalar>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as MulAdd<<A as ScalarArray>::Scalar, <B as ScalarArray>::Scalar>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<A::Scalar> + Array<B::Scalar> + Array<<Lhs::Scalar as MulAdd<A::Scalar, B::Scalar>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<A::Row as Array<A::Scalar>>::Type> + Array<<B::Row as Array<B::Scalar>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as MulAdd<A::Scalar, B::Scalar>>::Output>>::Type>,
{
    type Output = Lhs::Concrete;
    fn mul_add(self, a: A, b: B) -> Lhs::Concrete { map3(self, a, b, MulAdd::mul_add) }
}

impl<Lhs, Rhs> Pow<Rhs> for Lhs
where Lhs: ScalarArrayVal,
Rhs: ScalarArrayVal<Row=Lhs::Row, Dim=Lhs::Dim>,
Lhs::Row: Dim<Rhs::Scalar>,
Lhs::Dim: TwoDim<Rhs::Scalar, Lhs::Row>,
Lhs::Scalar: Pow<Rhs::Scalar>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Pow<<Rhs as ScalarArray>::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as Pow<<Rhs as ScalarArray>::Scalar>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as Pow<<Rhs as ScalarArray>::Scalar>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<Rhs::Scalar> + Array<<Lhs::Scalar as Pow<Rhs::Scalar>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<Rhs::Row as Array<Rhs::Scalar>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Pow<Rhs::Scalar>>::Output>>::Type>,
{
    type Output = Lhs::Concrete;
    fn pow(self, rhs: Rhs) -> Lhs::Concrete {
        map2(self, rhs, Pow::pow)
    }
}

impl<S> Recip for S
where S: ScalarArrayVal,
S::Scalar: Recip,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Recip>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as Recip>::Output>,
S::Dim: TwoDim<<S::Scalar as Recip>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Recip>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Recip>::Output>>::Type>,
{
    type Output = S::Concrete;
    fn recip(self) -> S::Concrete { map(self, Recip::recip) }
}

impl<S> Round for S
where S: ScalarArrayVal,
S::Scalar: Round,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Round>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<<S as ScalarArray>::Scalar as Round>::Output>,
S::Dim: TwoDim<<<S as ScalarArray>::Scalar as Round>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Round>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Round>::Output>>::Type>,
{
    type Output = S::Concrete;
    fn floor(self) -> S::Concrete { map(self, Round::floor) }
    fn ceil(self) -> S::Concrete { map(self, Round::ceil) }
    fn trunc(self) -> S::Concrete { map(self, Round::trunc) }
    fn fract(self) -> S::Concrete { map(self, Round::fract) }
    fn round(self) -> S::Concrete { map(self, Round::round) }
}

impl<S> Sqrt for S
where S: ScalarArrayVal,
S::Scalar: Sqrt,
S: HasConcreteScalarArray<<<S as ScalarArray>::Scalar as Sqrt>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
S::Row: Dim<<S::Scalar as Sqrt>::Output>,
S::Dim: TwoDim<<S::Scalar as Sqrt>::Output, S::Row>,
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<<S::Scalar as Sqrt>::Output>,
<S::Dim as HasSmaller>::Smaller: Array<<S::Row as Array<S::Scalar>>::Type> + Array<<S::Row as Array<<S::Scalar as Sqrt>::Output>>::Type>,
{
    type Output = S::Concrete;
    fn sqrt(self) -> S::Concrete { map(self, Sqrt::sqrt) }
    fn inverse_sqrt(self) -> S::Concrete { map(self, Sqrt::inverse_sqrt) }
}

impl<Lhs, Edge0, Edge1> Step<Edge0, Edge1> for Lhs
where Lhs: ScalarArrayVal,
Edge0: ScalarArrayVal<Row=Lhs::Row, Dim=Lhs::Dim>,
Edge1: ScalarArrayVal<Row=Lhs::Row, Dim=Lhs::Dim>,
Lhs::Row: Dim<Edge0::Scalar> + Dim<Edge1::Scalar>,
Lhs::Dim: TwoDim<Edge0::Scalar, Lhs::Row> + TwoDim<Edge1::Scalar, Lhs::Row>,
Lhs::Scalar: Step<Edge0::Scalar, Edge1::Scalar>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Step<<Edge0 as ScalarArray>::Scalar, <Edge1 as ScalarArray>::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as Step<<Edge0 as ScalarArray>::Scalar, <Edge1 as ScalarArray>::Scalar>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as Step<<Edge0 as ScalarArray>::Scalar, <Edge1 as ScalarArray>::Scalar>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<Edge0::Scalar> + Array<Edge1::Scalar> + Array<<Lhs::Scalar as Step<Edge0::Scalar, Edge1::Scalar>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<Edge0::Row as Array<Edge0::Scalar>>::Type> + Array<<Edge1::Row as Array<Edge1::Scalar>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Step<Edge0::Scalar, Edge1::Scalar>>::Output>>::Type>,
{
    type Output = Lhs::Concrete;
    fn step(self, edge: Edge0) -> Lhs::Concrete { map2(self, edge, Step::step) }
    fn smoothstep(self, edge0: Edge0, edge1: Edge1) -> Lhs::Concrete { map3(self, edge0, edge1, Step::smoothstep) }
}

impl<Lhs, Edge0, Edge1> Step<Value<Edge0>, Value<Edge1>> for Lhs
where Lhs: ScalarArrayVal,
Edge0: Clone,
Edge1: Clone,
Lhs::Row: Dim<Edge0> + Dim<Edge1>,
<Lhs::Row as Array<Edge0>>::Type: Clone,
<Lhs::Row as Array<Edge1>>::Type: Clone,
Lhs::Dim: TwoDim<Edge0, Lhs::Row> + TwoDim<Edge1, Lhs::Row>,
Lhs::Scalar: Step<Edge0, Edge1>,
Lhs: HasConcreteScalarArray<<<Lhs as ScalarArray>::Scalar as Step<Edge0, Edge1>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Lhs::Row: Dim<<<Lhs as ScalarArray>::Scalar as Step<Edge0, Edge1>>::Output>,
Lhs::Dim: TwoDim<<<Lhs as ScalarArray>::Scalar as Step<Edge0, Edge1>>::Output, Lhs::Row>,
<Lhs::Row as HasSmaller>::Smaller: Array<Lhs::Scalar> + Array<Edge0> + Array<Edge1> + Array<<Lhs::Scalar as Step<Edge0, Edge1>>::Output>,
<Lhs::Dim as HasSmaller>::Smaller: Array<<Lhs::Row as Array<Lhs::Scalar>>::Type> + Array<<Lhs::Row as Array<Edge0>>::Type> + Array<<Lhs::Row as Array<Edge1>>::Type> + Array<<Lhs::Row as Array<<Lhs::Scalar as Step<Edge0, Edge1>>::Output>>::Type>,
Lhs::Concrete: HasConcreteScalarArray<<Lhs::Scalar as Step<Edge0, Edge1>>::Output, Lhs::Row, Lhs::Dim, Concrete=Lhs::Concrete>,
{
    type Output = Lhs::Concrete;
    fn step(self, edge: Value<Edge0>) -> Lhs::Concrete {
        let edge = <Lhs::Row as Array<Edge0>>::from_value(edge.0);
        let edge = <Lhs::Dim as Array<<Lhs::Row as Array<Edge0>>::Type>>::from_value(edge);
        Lhs::Concrete::from_val(array::map2::<Lhs::Scalar, Lhs::Row, Lhs::Dim, _, _, _>(self.get_val(), edge, Step::step))
    }
    fn smoothstep(self, edge0: Value<Edge0>, edge1: Value<Edge1>) -> Lhs::Concrete {
        let edge0 = <Lhs::Row as Array<Edge0>>::from_value(edge0.0);
        let edge1 = <Lhs::Row as Array<Edge1>>::from_value(edge1.0);
        let edge0 = <Lhs::Dim as Array<<Lhs::Row as Array<Edge0>>::Type>>::from_value(edge0);
        let edge1 = <Lhs::Dim as Array<<Lhs::Row as Array<Edge1>>::Type>>::from_value(edge1);
        Lhs::Concrete::from_val(array::map3::<Lhs::Scalar, Lhs::Row, Lhs::Dim, _, _, _, _>(self.get_val(), edge0, edge1, Step::smoothstep))
    }
}
