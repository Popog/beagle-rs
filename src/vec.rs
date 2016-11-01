//! An array of Scalars, written `Vec<D, V>` but pronounced 'vector'.
//!
//! Vectors support binary operations between two Vectors or between one Vector and one Scalar.
//! All Arithmetic operators and Bitwise operators are supported, where the two Scalar types
//! involved support the operation. All operations operate on each component separately.
//!
//! Vectors also support Negation and Logical Negation where the underlying Scalar Type supports
//! it.
//!
//! # Examples
//!
//! ```
//! use beagle::vec::{Vec3};
//!
//! let v1 = Vec3::new([2f32, 3f32, 5f32]);
//! let v2 = Vec3::new([7f32, 11f32, 13f32]);
//! assert_eq!(v1+v2, Vec3::new([9f32, 14f32, 18f32]));
//! ```

use std::ops::{
    Neg,Not,
    BitAnd,BitOr,BitXor,
    Shl,Shr,
    Add,Div,Mul,Rem,Sub,
    BitAndAssign,BitOrAssign,BitXorAssign,
    ShlAssign,ShrAssign,
    AddAssign,DivAssign,MulAssign,RemAssign,SubAssign,
    Deref,DerefMut,
    Index,IndexMut,
};

use super::{Value,v};
use consts::{
    One,Two,Three,Four,
    CustomArrayOne,
    Array,
    Dim,HasSmaller,DimRef,DimMut,
    TwoDim,
};
use mat::Mat;
use num::{ApproxZero,Sign,Sqrt};
use scalar_array::{
    ScalarArray,ScalarArrayVal,ScalarArrayRef,ScalarArrayMut,
    VecArrayVal,VecArrayRef,
    ConcreteScalarArray,HasConcreteScalarArray,ConcreteVecArray,HasConcreteVecArray,
};
use scalar_array::{
    apply_zip_mut_val,
    fold,fold_ref,fold_zip,fold_zip_ref,
    map,map_zip,
    mul_vector_transpose
};
use scalar_array::vec_array;
use utils::{ArrayRefCast,RefCast};

/// An array of Scalars, written `Vec<D, V>` but pronounced 'vector'.
///
/// Vectors support binary operations between two Vectors or between one Vector and one Scalar.
/// All Arithmetic operators and Bitwise operators are supported, where the two Scalar types
/// involved support the operation. All operations operate on each component separately.
///
/// Vectors also support Negation and Logical Negation where the underlying Scalar Type supports
/// it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Vec<D, S> (D::Type)
where D: Dim<S>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>;

impl<S, D: Dim<S>> Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>,
{
    /// Construct a vector from an array
    pub fn new(v: D::RawType) -> Self
    where D::RawType: Into<D::Type> {
        Vec::from_vec_val(v.into())
    }

    /// Construct a Vector from a single value
    pub fn from_value(s: S) -> Self
    where S: Clone {
        Vec::from_vec_val(D::from_value(s))
    }
}

impl<S, D: Dim<S>> RefCast<D::Type> for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>,
{
    #[inline(always)]
    fn from_ref(v: &D::Type) -> &Self {
        let ptr = v as *const _ as *const Vec<D, S>;
        unsafe { &*ptr }
    }

    #[inline(always)]
    fn from_mut(v: &mut D::Type) -> &mut Self {
        let ptr = v as *mut _ as *mut Vec<D, S>;
        unsafe { &mut*ptr }
    }

    #[inline(always)]
    fn into_ref(&self) -> &D::Type {
        &self.0
    }

    #[inline(always)]
    fn into_mut(&mut self) -> &mut D::Type {
        &mut self.0
    }
}

impl<S, D: Dim<S>> ArrayRefCast<D::Type> for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>,
{
    #[inline(always)]
    fn from_array_ref<A>(v: &<A as Array<D::Type>>::Type) -> &<A as Array<Self>>::Type
    where A: Array<D::Type> + Array<Self> {
        let ptr = v as *const _ as *const <A as Array<Self>>::Type;
        unsafe { &*ptr }
    }

    #[inline(always)]
    fn from_array_mut<A>(v: &mut <A as Array<D::Type>>::Type) -> &mut <A as Array<Self>>::Type
    where A: Array<D::Type> + Array<Self> {
        let ptr = v as *mut _ as *mut <A as Array<Self>>::Type;
        unsafe { &mut*ptr }
    }

    #[inline(always)]
    fn into_array_ref<A>(v: &<A as Array<Self>>::Type) -> &<A as Array<D::Type>>::Type
    where A: Array<D::Type> + Array<Self> {
        let ptr = v as *const _ as *const <A as Array<D::Type>>::Type;
        unsafe { &*ptr }
    }

    #[inline(always)]
    fn into_array_mut<A>(v: &mut <A as Array<Self>>::Type) -> &mut <A as Array<D::Type>>::Type
    where A: Array<D::Type> + Array<Self> {
        let ptr = v as *mut _ as *mut <A as Array<D::Type>>::Type;
        unsafe { &mut*ptr }
    }
}


impl<S, D: Dim<S>> Deref for Vec<D, S>
where D::Type: Deref<Target=D::RawType>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
    type Target = D::RawType;

    #[inline(always)]
    fn deref(&self) -> &D::RawType { self.0.deref() }
}

impl<S, D: Dim<S>> DerefMut for Vec<D, S>
where D::Type: DerefMut<Target=D::RawType>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut D::RawType { self.0.deref_mut() }
}

impl<S, D: Dim<S>> Index<usize> for Vec<D, S>
where D::Type: AsRef<[S]>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
    type Output = S;

    #[inline(always)]
    fn index(&self, rhs: usize) -> &S { self.0.as_ref().index(rhs) }
}

impl<S, D: Dim<S>> IndexMut<usize> for Vec<D, S>
where D::Type: AsRef<[S]>+AsMut<[S]>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
    #[inline(always)]
    fn index_mut(&mut self, rhs: usize) -> &mut S { self.0.as_mut().index_mut(rhs) }
}

/// An alias for Vec&lt;One, V&gt;
pub type Vec1<S> = Vec<One, S>;
/// An alias for Vec&lt;One, V&gt;
pub type Vec2<S> = Vec<Two, S>;
/// An alias for Vec&lt;One, V&gt;
pub type Vec3<S> = Vec<Three, S>;
/// An alias for Vec&lt;One, V&gt;
pub type Vec4<S> = Vec<Four, S>;

impl<S, D: Dim<S>> ScalarArray for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>,
{
    type Scalar = S;
    type Row = D;
    type Dim = One;
}

impl<S, D: Dim<S>> ScalarArrayVal for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>,
{
    #[inline(always)]
    fn get_val(self) -> CustomArrayOne<D::Type> { CustomArrayOne([self.0]) }
}

impl<S, D: DimRef<S>> ScalarArrayRef for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where for<'a> D::Smaller: Array<S> + Array<&'a S>,
{
    #[inline(always)]
    fn get_ref(&self) -> <One as Array<<D as Array<&S>>::Type>>::Type {
        CustomArrayOne([D::get_ref(&self.0)])
    }
}

impl<S, D: DimMut<S>> ScalarArrayMut for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where for<'a> D::Smaller: Array<S> + Array<&'a S> + Array<&'a mut S>,
{
    #[inline(always)]
    fn get_mut(&mut self) -> <One as Array<<D as Array<&mut S>>::Type>>::Type {
        CustomArrayOne([D::get_mut(&mut self.0)])
    }
}

impl<S, D: Dim<S>> VecArrayVal for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>,
{
    #[inline(always)]
    fn get_vec_val(self) -> D::Type {
        self.0
    }
}

impl<S, D: DimRef<S>> VecArrayRef for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where for<'a> D::Smaller: Array<S> + Array<&'a S>,
{
    #[inline(always)]
    fn get_vec_ref(&self) -> <D as Array<&S>>::Type {
        D::get_ref(&self.0)
    }
}

impl<S, T, D: Dim<S>, D2: Dim<T>> HasConcreteScalarArray<T, D2> for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>,
D2::Smaller: Array<T>
{
    /// The type of a concrete ScalarArray of the specified type
    type Concrete = Vec<D2, T>;
}

impl<S, D: Dim<S>> ConcreteScalarArray for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>
{
    #[inline(always)]
    fn from_val(v: <One as Array<D::Type>>::Type) -> Self {
        let [v] = v.0;
        Vec(v)
    }
}

impl<S, T, D: Dim<S>, D2: Dim<T>> HasConcreteVecArray<T, D2> for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>,
D2::Smaller: Array<T>
{}

impl<S, D: Dim<S>> ConcreteVecArray for Vec<D, S>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where D::Smaller: Array<S>
{
    #[inline(always)]
    fn from_vec_val(v: D::Type) -> Self { Vec(v) }
}


/// Multiply two Vectors component-wise, summing the results. Known as a dot product.
pub fn dot<S, T>(s: S, t: T) -> <S::Scalar as Mul<T::Scalar>>::Output
where S: VecArrayVal,
S::Row: Dim<T::Scalar>,
T: VecArrayVal<Row=S::Row>,
S::Scalar: Mul<T::Scalar>,
<S::Scalar as Mul<T::Scalar>>::Output: Add<Output=<S::Scalar as Mul<T::Scalar>>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
{
    vec_array::dot::<S::Scalar, S::Row, _>(s.get_vec_val(), t.get_vec_val())
}

/// Returns the length squared of `s`.
pub fn length2<S>(s: S) -> <S::Scalar as Mul>::Output
where S: VecArrayVal,
S::Scalar: Mul+Clone,
<S::Scalar as Mul>::Output: Add<Output=<S::Scalar as Mul>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar>,
{
    fold(s, |s| s.clone()*s, |init, s| init + (s.clone() * s))
}

/// Returns the length of `s`.
pub fn length<S>(s: S) -> <<S::Scalar as Mul>::Output as Sqrt>::Output
where S: VecArrayVal,
S::Scalar: Mul+Clone,
<S::Scalar as Mul>::Output: Add<Output=<S::Scalar as Mul>::Output> + Sqrt,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar>,
{
    length2(s).sqrt()
}

/// Returns an approximated normalized version of `s`. This is generally faster than `normalize_exact`.
pub fn normalize<S>(s: S) -> <S as Mul<Value<<<S::Scalar as Mul>::Output as Sqrt>::Output>>>::Output
where S: VecArrayRef + Mul<Value<<<<S as ScalarArray>::Scalar as Mul>::Output as Sqrt>::Output>>,
S::Scalar: Mul+Clone,
<S::Scalar as Mul>::Output: Add<Output=<S::Scalar as Mul>::Output> + Sqrt,
S::Row: DimRef<S::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
for<'a> <S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<&'a S::Scalar>,
{
    let length2 = fold_ref(&s, |s| s.clone()*s.clone(), |init, s| init + (s.clone() * s.clone()));
    s * v(length2.inverse_sqrt())
}

/// Returns an exact normalized version of `s`. Basically `v / v.length()`
pub fn normalize_exact<S>(s: S) -> <S as Div<Value<<<S::Scalar as Mul>::Output as Sqrt>::Output>>>::Output
where S: VecArrayRef + Div<Value<<<<S as ScalarArray>::Scalar as Mul>::Output as Sqrt>::Output>>,
S::Scalar: Mul+Clone,
<S::Scalar as Mul>::Output: Add<Output=<S::Scalar as Mul>::Output> + Sqrt,
S::Row: DimRef<S::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
for<'a> <S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<&'a S::Scalar>,
{
    let length2 = fold_ref(&s, |s| s.clone()*s.clone(), |init, s| init + (s.clone() * s.clone()));
    s / v(length2.sqrt())
}

/// Returns the distance squared between `lhs` and `rhs`.
pub fn distance2<S, T>(s: S, t: T) -> <<S::Scalar as Sub<T::Scalar>>::Output as Mul>::Output
where S: VecArrayVal,
S::Row: Dim<T::Scalar>,
T: VecArrayVal<Row=<S as ScalarArray>::Row>,
S::Scalar: Sub<T::Scalar>,
<S::Scalar as Sub<T::Scalar>>::Output: Mul + Clone,
<<S::Scalar as Sub<T::Scalar>>::Output as Mul>::Output: Add<Output=<<S::Scalar as Sub<T::Scalar>>::Output as Mul>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
{
    fold_zip(s, t,
        |s, t| {
            let v = s - t;
            (v.clone() * v)
        },
        |init, s, t| {
            let v = s - t;
            init + (v.clone() * v)
        }
    )
}

/// Returns the distance between `lhs` and `rhs`.
pub fn distance<S, T>(s: S, t: T) -> <<<S::Scalar as Sub<T::Scalar>>::Output as Mul>::Output as Sqrt>::Output
where S: VecArrayVal,
S::Row: Dim<T::Scalar>,
T: VecArrayVal<Row=<S as ScalarArray>::Row>,
S::Scalar: Sub<T::Scalar>,
<S::Scalar as Sub<T::Scalar>>::Output: Mul + Clone,
<<S::Scalar as Sub<T::Scalar>>::Output as Mul>::Output: Add<Output=<<S::Scalar as Sub<T::Scalar>>::Output as Mul>::Output> + Sqrt,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
{
    distance2(s, t).sqrt()
}

/// Returns the cross product of `s` and `t`.
pub fn cross<S, T>(s: S, t: T) -> S::Concrete
where S: VecArrayVal<Row=Three> + HasConcreteVecArray<<<S as ScalarArray>::Scalar as Mul<T::Scalar>>::Output, Three>,
T: VecArrayVal<Row=Three>,
S::Scalar: Clone+Mul<T::Scalar>,
T::Scalar: Clone,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Scalar as Mul<T::Scalar>>::Output: Sub<Output=<S::Scalar as Mul<T::Scalar>>::Output>,
<S as HasConcreteScalarArray<<S::Scalar as Mul<T::Scalar>>::Output, Three, One>>::Concrete: ConcreteVecArray,
{
    S::Concrete::from_vec_val(vec_array::cross(s.get_vec_val(), t.get_vec_val()))
}

/// Returns `cos²(θ) < ε²`, where `θ` = the angle between `a` and `b`.
pub fn is_perpendicular<S, T>(s: S, t: T, epsilon_squared: &ApproxZero<<<S::Scalar as Mul>::Output as Mul<<T::Scalar as Mul>::Output>>::Output>) -> bool
where S: VecArrayVal,
S::Row: Dim<T::Scalar>,
T: VecArrayVal<Row=<S as ScalarArray>::Row>,
S::Scalar: Mul + Mul<T::Scalar> + Clone,
<S::Scalar as Mul>::Output: Add<Output=<S::Scalar as Mul>::Output> + Mul<<T::Scalar as Mul>::Output>,
<S::Scalar as Mul<T::Scalar>>::Output: Add<Output=<S::Scalar as Mul<T::Scalar>>::Output> + Mul<Output=<<S::Scalar as Mul>::Output as Mul<<T::Scalar as Mul>::Output>>::Output> + Clone,
T::Scalar: Mul + Clone,
<T::Scalar as Mul>::Output: Add<Output=<T::Scalar as Mul>::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<S::Row as HasSmaller>::Smaller: Array<S::Scalar> + Array<T::Scalar>,
{
    let (s_dot_t, s_length2, t_length2) = fold_zip(s, t,
        |s, t| (s.clone()*t.clone(), s.clone()*s, t.clone()*t),
        |(st, ss, tt), s, t|(st + s.clone()*t.clone(), ss + s.clone()*s, tt + t.clone()*t),
    );

    // We're looking to return abs(cos(θ)) <= ε
    // Proof:
    // ∵                 a·b  =  |a|*|b|*cos(θ)
    // ∵            cos²(θ)  <=  ε²
    //     |a|²*|b|²*cos²(θ) <=  |a|²*|b|²*ε²
    //   (|a|*|b|*|cos(θ)|)² <=  |a|²*|b|²*ε²
    //                (a·b)² <=  |a|²*|b|²*ε²
    // ∴  (a·b)² / |a|²*|b|² <=  ε²
    //let a_dot_b = a.dot(b);
    epsilon_squared.approx_zero_ratio(s_dot_t.clone() * s_dot_t, s_length2 * t_length2)
}


/// Returns `n` if `nref.dot(i)` is negative, else `-n`.
pub fn faceforward<N, I, NRef>(n: N, i: I, nref: NRef) -> N
where N: Neg<Output=N>,
NRef: VecArrayVal,
NRef::Row: Dim<I::Scalar>,
I: VecArrayVal<Row=NRef::Row>,
NRef::Scalar: Mul<I::Scalar>,
<NRef::Scalar as Mul<I::Scalar>>::Output: Add<Output=<NRef::Scalar as Mul<I::Scalar>>::Output> + Sign<Output=bool>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<NRef::Row as HasSmaller>::Smaller: Array<NRef::Scalar> + Array<I::Scalar>,
{
    if dot(nref, i).is_negative() { n } else { -n }
}

/// Reflects `i` against `n` as `i - n * dot(i,n) * 2`.
pub fn reflect<N, I>(i: I, n: N) -> I::Output
where I: VecArrayRef + Sub<N::Output>,
I::Row: DimRef<I::Scalar> + DimRef<N::Scalar>,
N: VecArrayRef<Row=<I as ScalarArray>::Row> + Mul<Value<<<I as ScalarArray>::Scalar as Mul<<N as ScalarArray>::Scalar>>::Output>>,
I::Scalar: Mul<N::Scalar> + Clone,
N::Scalar: Clone,
<I::Scalar as Mul<N::Scalar>>::Output: Add<Output=<I::Scalar as Mul<N::Scalar>>::Output> + Clone,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
for<'a> <I::Row as HasSmaller>::Smaller: Array<I::Scalar> + Array<&'a I::Scalar> + Array<N::Scalar> + Array<&'a N::Scalar>
{
    let i_dot_n = fold_zip_ref(&i, &n, |i, n| i.clone()*n.clone(), |init, i, n| init + (i.clone() * n.clone()));
    i - n * v(i_dot_n.clone() + i_dot_n)
}

// TODO: geometric functions
// refraction vector
//  Tfd  refract(Tfd I, Tfd N, float eta)


macro_rules! impl_vec_unop {
    ($($trait_name:ident::$method_name:ident)+) => {$(
impl<S, D: Dim<S>> $trait_name for Vec<D, S>
where D: Dim<S::Output>,
S: $trait_name,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S> + Array<S::Output>,
{
    type Output = Vec<D, S::Output>;
    fn $method_name(self) -> Self::Output { map(self, $trait_name::$method_name) }
}
    )+};
}

macro_rules! impl_vec_binop {
    ($($trait_name:ident::$method_name:ident)+) => {$(
        impl_vec_binop!{$trait_name::$method_name for Vec}
        impl_vec_binop!{$trait_name::$method_name for Value}
    )+};

    ($trait_name:ident::$method_name:ident for Vec) => {
impl<S, D: Dim<S>, Rhs> $trait_name<Rhs> for Vec<D, S>
where Rhs: VecArrayVal<Row=D>+,
D: Dim<Rhs::Scalar> + Dim<S::Output>,
S: $trait_name<Rhs::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S> + Array<Rhs::Scalar> + Array<S::Output>,
{
    type Output = Vec<D, S::Output>;
    fn $method_name(self, rhs: Rhs) -> Self::Output { map_zip(self, rhs, $trait_name::$method_name) }
}
    };

    ($trait_name:ident::$method_name:ident for Value) => {
impl<S, D: Dim<S>, Rhs> $trait_name<Value<Rhs>> for Vec<D, S>
where Rhs: Clone,
<D as Array<Rhs>>::Type: Clone,
D: Dim<Rhs> + Dim<S::Output>,
S: $trait_name<Rhs>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S> + Array<Rhs> + Array<S::Output>,
{
    type Output = Vec<D, S::Output>;
    fn $method_name(self, rhs: Value<Rhs>) -> Self::Output {
        map_zip(self, Vec::from_vec_val(D::from_value(rhs.0)), $trait_name::$method_name)
    }
}
impl<S, D: Dim<S>, Lhs> $trait_name<Vec<D, S>> for Value<Lhs>
where Lhs: Clone,
<D as Array<Lhs>>::Type: Clone,
D: Dim<Lhs> + Dim<Lhs::Output>,
Lhs: $trait_name<S>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S> + Array<Lhs> + Array<Lhs::Output>,
{
    type Output = Vec<D, Lhs::Output>;
    fn $method_name(self, rhs: Vec<D, S>) -> Self::Output {
        map_zip(Vec::<_, Lhs>::from_vec_val(D::from_value(self.0)), rhs, $trait_name::$method_name)
    }
}
    };
}

macro_rules! impl_vec_binop_assign {
    ($($trait_name:ident::$method_name:ident)+) => {$(
        impl_vec_binop_assign!{$trait_name::$method_name for Vec}
        impl_vec_binop_assign!{$trait_name::$method_name for Value}
    )+};

    ($trait_name:ident::$method_name:ident for Vec) => {
impl<S, D: DimMut<S>, Rhs> $trait_name<Rhs> for Vec<D, S>
where Rhs: VecArrayVal<Row=D>,
D: Dim<Rhs::Scalar>,
S: $trait_name<Rhs::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
for<'a> D::Smaller: Array<S> + Array<Rhs::Scalar> + Array<&'a S> + Array<&'a mut S>,
{
    fn $method_name(&mut self, rhs: Rhs) { apply_zip_mut_val(self, rhs, $trait_name::$method_name) }
}
    };

    ($trait_name:ident::$method_name:ident for Value) => {
impl<S, D: DimMut<S>, Rhs> $trait_name<Value<Rhs>> for Vec<D, S>
where Rhs: Clone,
<D as Array<Rhs>>::Type: Clone,
D: Dim<Rhs>,
S: $trait_name<Rhs>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
for<'a> D::Smaller: Array<S> + Array<Rhs> + Array<&'a S> + Array<&'a mut S>,
{
    fn $method_name(&mut self, rhs: Value<Rhs>) {
        apply_zip_mut_val(self, Vec::from_vec_val(D::from_value(rhs.0)), $trait_name::$method_name)
    }
}
    };
}

impl_vec_unop!{Neg::neg Not::not}

impl_vec_binop!{
    BitAnd::bitand BitOr::bitor BitXor::bitxor
    Shl::shl Shr::shr
    Add::add Div::div Mul::mul Rem::rem Sub::sub
}
impl_vec_binop_assign!{
    BitAndAssign::bitand_assign BitOrAssign::bitor_assign BitXorAssign::bitxor_assign
    ShlAssign::shl_assign ShrAssign::shr_assign
    AddAssign::add_assign DivAssign::div_assign MulAssign::mul_assign RemAssign::rem_assign SubAssign::sub_assign
}

impl<S, T, C, D> Mul<Mat<D, C, T>> for Vec<D, S>
where C: Dim<T> + Dim<S> + Dim<S::Output>,
D: Dim<S> + TwoDim<S, C> + TwoDim<T, C>,
S: Clone,
S: Mul<T>,
S::Output: Add<Output=S::Output>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
C::Smaller: Array<S> + Array<T> + Array<S::Output>,
D::Smaller: Array<S> + Array<<C as Array<S>>::Type> + Array<<C as Array<T>>::Type>,
{
    type Output = Vec<C, S::Output>;
    fn mul(self, rhs: Mat<D, C, T>) -> Self::Output {
        mul_vector_transpose(self, rhs)
    }
}


#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use super::*;

    /// Test that Vec sizes are what they should be
    #[test]
    fn test_sizes() {
        assert_eq!(size_of::<[i32;1]>(), size_of::<Vec1<i32>>());
        assert_eq!(size_of::<[i32;2]>(), size_of::<Vec2<i32>>());
        assert_eq!(size_of::<[i32;3]>(), size_of::<Vec3<i32>>());
        assert_eq!(size_of::<[i32;4]>(), size_of::<Vec4<i32>>());

        assert_eq!(size_of::<[usize;1]>(), size_of::<Vec1<usize>>());
        assert_eq!(size_of::<[usize;2]>(), size_of::<Vec2<usize>>());
        assert_eq!(size_of::<[usize;3]>(), size_of::<Vec3<usize>>());
        assert_eq!(size_of::<[usize;4]>(), size_of::<Vec4<usize>>());

        assert_eq!(size_of::<[f64;1]>(), size_of::<Vec1<f64>>());
        assert_eq!(size_of::<[f64;2]>(), size_of::<Vec2<f64>>());
        assert_eq!(size_of::<[f64;3]>(), size_of::<Vec3<f64>>());
        assert_eq!(size_of::<[f64;4]>(), size_of::<Vec4<f64>>());

        assert_eq!(size_of::<[bool;1]>(), size_of::<Vec1<bool>>());
        assert_eq!(size_of::<[bool;2]>(), size_of::<Vec2<bool>>());
        assert_eq!(size_of::<[bool;3]>(), size_of::<Vec3<bool>>());
        assert_eq!(size_of::<[bool;4]>(), size_of::<Vec4<bool>>());
    }

    /// Test that Vec::new should work with or without the size or type specified
    #[test]
    fn test_new() {
        assert_eq!(Vec1::new([1f64]),                   Vec::new([1f64]));
        assert_eq!(Vec2::new([1f64, 2f64]),             Vec::new([1f64, 2f64]));
        assert_eq!(Vec3::new([1f64, 2f64, 3f64]),       Vec::new([1f64, 2f64, 3f64]));
        assert_eq!(Vec4::new([1f64, 2f64, 3f64, 4f64]), Vec::new([1f64, 2f64, 3f64, 4f64]));
        assert_eq!(Vec1::new([1]),                      Vec::new([1]));
    }

    /// Test the unary `-` operator
    #[test]
    fn test_vec_neg() {
        assert_eq!(-Vec1::new([1f64]),                   Vec1::new([-1f64]));
        assert_eq!(-Vec2::new([1f64, 2f64]),             Vec2::new([-1f64, -2f64]));
        assert_eq!(-Vec3::new([1f64, 2f64, 3f64]),       Vec3::new([-1f64, -2f64, -3f64]));
        assert_eq!(-Vec4::new([1f64, 2f64, 3f64, 4f64]), Vec4::new([-1f64, -2f64, -3f64, -4f64]));

        assert_eq!(-Vec1::new([1f32]),                   Vec1::new([-1f32]));
        assert_eq!(-Vec2::new([1f32, 2f32]),             Vec2::new([-1f32, -2f32]));
        assert_eq!(-Vec3::new([1f32, 2f32, 3f32]),       Vec3::new([-1f32, -2f32, -3f32]));
        assert_eq!(-Vec4::new([1f32, 2f32, 3f32, 4f32]), Vec4::new([-1f32, -2f32, -3f32, -4f32]));

        assert_eq!(-Vec1::new([1i64]),                   Vec1::new([-1i64]));
        assert_eq!(-Vec2::new([1i64, 2i64]),             Vec2::new([-1i64, -2i64]));
        assert_eq!(-Vec3::new([1i64, 2i64, 3i64]),       Vec3::new([-1i64, -2i64, -3i64]));
        assert_eq!(-Vec4::new([1i64, 2i64, 3i64, 4i64]), Vec4::new([-1i64, -2i64, -3i64, -4i64]));

        assert_eq!(-Vec1::new([1i32]),                   Vec1::new([-1i32]));
        assert_eq!(-Vec2::new([1i32, 2i32]),             Vec2::new([-1i32, -2i32]));
        assert_eq!(-Vec3::new([1i32, 2i32, 3i32]),       Vec3::new([-1i32, -2i32, -3i32]));
        assert_eq!(-Vec4::new([1i32, 2i32, 3i32, 4i32]), Vec4::new([-1i32, -2i32, -3i32, -4i32]));

        assert_eq!(-Vec1::new([1i16]),                   Vec1::new([-1i16]));
        assert_eq!(-Vec2::new([1i16, 2i16]),             Vec2::new([-1i16, -2i16]));
        assert_eq!(-Vec3::new([1i16, 2i16, 3i16]),       Vec3::new([-1i16, -2i16, -3i16]));
        assert_eq!(-Vec4::new([1i16, 2i16, 3i16, 4i16]), Vec4::new([-1i16, -2i16, -3i16, -4i16]));

        assert_eq!(-Vec1::new([1i8 ]),                   Vec1::new([-1i8 ]));
        assert_eq!(-Vec2::new([1i8 , 2i8 ]),             Vec2::new([-1i8 , -2i8 ]));
        assert_eq!(-Vec3::new([1i8 , 2i8 , 3i8 ]),       Vec3::new([-1i8 , -2i8 , -3i8 ]));
        assert_eq!(-Vec4::new([1i8 , 2i8 , 3i8 , 4i8 ]), Vec4::new([-1i8 , -2i8 , -3i8 , -4i8 ]));
    }

    /// Test the `!` operator
    #[test]
    fn test_vec_not() {
        assert_eq!(!Vec1::new([false]),                    Vec1::new([true]));
        assert_eq!(!Vec2::new([false, true]),              Vec2::new([true, false]));
        assert_eq!(!Vec3::new([false, true, false]),       Vec3::new([true, false, true]));
        assert_eq!(!Vec4::new([false, true, false, true]), Vec4::new([true, false, true, false]));
    }

    /// Test the '+' operator
    #[test]
    fn test_vec_add() {
        assert_eq!(Vec1::new([1f64]) + Vec1::new([2f64]), Vec1::new([3f64]));
    }

    /// Test the `[]` operator
    #[test]
    fn test_vec_index() {
        let v = Vec4::new([1f64, 2f64, 3f64, 4f64]);
        assert_eq!(v[3], 4f64);
    }

    //#[test]
    //fn test_vec_add_scalar() {
    //    assert_eq!(Vec1::new([1f64])+v(2f64, Vec1::new([3f64]));
    //}
}
