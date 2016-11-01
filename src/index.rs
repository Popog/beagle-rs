//! Indexing and Swizzling for Vec
//!
//! # Examples
//!
//! ```
//! use beagle::vec::{Vec3,Vec2};
//! use beagle::index::swizzle::xyzw::*;
//!
//! let mut v = Vec3::new([8f32, 12f32, 10f32]);
//! v[YZ] = v[XY] + Vec2::from_value(1f32);
//! assert_eq!(v, Vec3::new([8f32, 9f32, 13f32]));
//! ```
// TODO: More examples

use std::marker::PhantomData;
use std::ops::{
    Neg,Not,
    BitAnd,BitOr,BitXor,
    Shl,Shr,
    Add,Div,Mul,Rem,Sub,
    BitAndAssign,BitOrAssign,BitXorAssign,
    ShlAssign,ShrAssign,
    AddAssign,DivAssign,MulAssign,RemAssign,SubAssign,
};

use super::Value;
use consts::{
    Array,ArrayMut,Dim,DimMut,
    IsSmallerThan,IsLargerThan,NotSame,DecrementIfLargerThan,ExtractItem,HasSmaller,
    One,Two,Three,Four,
    CustomArrayOne,CustomArrayTwo,CustomArrayThree,CustomArrayFour,
};
use utils::RefCast;
use scalar_array::{
    ScalarArray,ScalarArrayVal,ScalarArrayRef,ScalarArrayMut,
    VecArrayVal,VecArrayRef,
    HasConcreteScalarArray,HasConcreteVecArray,
    ConcreteVecArray,
    apply_zip_mut_val,
};
use vec::{Vec};

//  .d88b.  d8b   db d88888b
// .8P  Y8. 888o  88 88'
// 88    88 88V8o 88 88ooooo
// 88    88 88 V8o88 88~~~~~
// `8b  d8' 88  V888 88.
//  `Y88P'  VP   V8P Y88888P

macro_rules! decl_consts {
    ($($id:ident = $val:expr,)*) => {decl_consts!{$($id = $val),*}};
    ($($id:ident = $val:expr),*) => {$(
        /// Constant for unary indexing
        pub const $id: usize = $val;
    )*};
}

decl_consts!{
    I0 = 0, I1 = 1, I2 = 2, I3 = 3,
     X = 0,  Y = 1,  Z = 2,  W = 3,
     R = 0,  G = 1,  B = 2,  A = 3,
     S = 0,  T = 1,  P = 2,  Q = 3,
}



// .88b  d88.  .d8b.   .o88b. d8888b.  .d88b.  .d8888.
// 88'YbdP`88 d8' `8b d8P  Y8 88  `8D .8P  Y8. 88'  YP
// 88  88  88 88ooo88 8P      88oobY' 88    88 `8bo.
// 88  88  88 88~~~88 8b      88`8b   88    88   `Y8b.
// 88  88  88 88   88 Y8b  d8 88 `88. `8b  d8' db   8D
// YP  YP  YP YP   YP  `Y88P' 88   YD  `Y88P'  `8888Y'

macro_rules! impl_vecref_unop_inner {
    ($D:ident, $S:ident, ($($index:ident),+), $id:ty, $dim:ident, $trait_name:ident::$method_name:ident) => {
impl<'a, $S, $D, $($index),+> $trait_name for &'a $id
where $D: Dim<$S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<$D>,)+
$S: $trait_name + Clone,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
$D::Smaller: Array<$S>,
{
    type Output = Vec<$dim, S::Output>;
    fn $method_name(self) -> Self::Output {
        $trait_name::$method_name(Vec::from_vec_val(self.get_vec_val()))
    }
}
    };
}

macro_rules! impl_vecref_unop {
    ($D:ident, $S:ident, $I0:ident, $I1:ident, $id:ty, $dim:ident, $($trait_name:ident::$method_name:ident)+) => {$(
        impl_vecref_unop_inner!{$D, $S, ($I0, $I1), $id, $dim, $trait_name::$method_name}
    )+};

    ($D:ident, $S:ident, $I0:ident, $I1:ident, $I2:ident, $id:ty, $dim:ident, $($trait_name:ident::$method_name:ident)+) => {$(
        impl_vecref_unop_inner!{$D, $S, ($I0, $I1, $I2), $id, $dim, $trait_name::$method_name}
    )+};

    ($D:ident, $S:ident, $I0:ident, $I1:ident, $I2:ident, $I3:ident, $id:ty, $dim:ident, $($trait_name:ident::$method_name:ident)+) => {$(
        impl_vecref_unop_inner!{$D, $S, ($I0, $I1, $I2, $I3), $id, $dim, $trait_name::$method_name}
    )+};
}

macro_rules! impl_vecref_binop_inner {
    ($D:ident, $S:ident, ($($index:ident),+), $id:ty, $dim:ident, $trait_name:ident::$method_name:ident for VecRef) => {
impl<'a, $S, $D, Rhs, $($index),+> $trait_name<Rhs> for &'a $id
where Rhs: VecArrayVal<Row=$dim>,
$D: Dim<$S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<$D>,)+
$S: $trait_name<Rhs::Scalar> + Clone,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
$D::Smaller: Array<$S>,
{
    type Output = Vec<$dim, S::Output>;
    fn $method_name(self, rhs: Rhs) -> Self::Output {
        $trait_name::$method_name(Vec::from_vec_val(self.get_vec_val()), rhs)
    }
}
    };

    ($D:ident, $S:ident, ($($index:ident),+), $id:ty, $dim:ident, $trait_name:ident::$method_name:ident for Value) => {
impl<'a, $S, $D, Rhs, $($index),+> $trait_name<Value<Rhs>> for &'a $id
where Rhs: Clone,
$D: Dim<$S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<$D>,)+
$S: $trait_name<Rhs> + Clone,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
$D::Smaller: Array<$S>,
{
    type Output = Vec<$dim, $S::Output>;
    fn $method_name(self, rhs: Value<Rhs>) -> Self::Output {
        $trait_name::$method_name(Vec::from_vec_val(self.get_vec_val()), rhs)
    }
}

impl<'a, $S, $D, Lhs, $($index),+> $trait_name<&'a $id> for Value<Lhs>
where Lhs: Clone,
$D: Dim<$S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<$D>,)+
Lhs: $trait_name<$S>,
S: Clone,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
$D::Smaller: Array<$S>,
{
    type Output = Vec<$dim, Lhs::Output>;
    fn $method_name(self, rhs: &'a $id) -> Self::Output {
        $trait_name::$method_name(self, Vec::from_vec_val(rhs.get_vec_val()))
    }
}
    };
}

macro_rules! impl_vecref_binop {
    ($D:ident, $S:ident, $I0:ident, $I1:ident, $id:ty, $dim:ident, $($trait_name:ident::$method_name:ident)+) => {$(
        impl_vecref_binop_inner!{$D, $S, ($I0, $I1), $id, $dim, $trait_name::$method_name for VecRef}
        impl_vecref_binop_inner!{$D, $S, ($I0, $I1), $id, $dim, $trait_name::$method_name for Value}
    )+};

    ($D:ident, $S:ident, $I0:ident, $I1:ident, $I2:ident, $id:ty, $dim:ident, $($trait_name:ident::$method_name:ident)+) => {$(
        impl_vecref_binop_inner!{$D, $S, ($I0, $I1, $I2), $id, $dim, $trait_name::$method_name for VecRef}
        impl_vecref_binop_inner!{$D, $S, ($I0, $I1, $I2), $id, $dim, $trait_name::$method_name for Value}
    )+};

    ($D:ident, $S:ident, $I0:ident, $I1:ident, $I2:ident, $I3:ident, $id:ty, $dim:ident, $($trait_name:ident::$method_name:ident)+) => {$(
        impl_vecref_binop_inner!{$D, $S, ($I0, $I1, $I2, $I3), $id, $dim, $trait_name::$method_name for VecRef}
        impl_vecref_binop_inner!{$D, $S, ($I0, $I1, $I2, $I3), $id, $dim, $trait_name::$method_name for Value}
    )+};
}

macro_rules! impl_vecref_binop_assign_inner {
    ($D:ident, $S:ident, ($($index:ident),+), $id:ty, $dim:ident, $trait_name:ident::$method_name:ident for VecRef) => {
impl<$S, $D, Rhs, $($index),+> $trait_name<Rhs> for $id
where Rhs: VecArrayVal<Row=$dim>,
$D: Dim<$S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<$D>,)+
for<'a> &'a mut $id: ScalarArrayMut<Dim=One, Row=$dim, Scalar=S>,
$S: $trait_name<Rhs::Scalar>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
$D::Smaller: Array<$S>,
{
    fn $method_name(&mut self, rhs: Rhs) {
        let mut v = self;
        apply_zip_mut_val(&mut v, rhs, $trait_name::$method_name)
    }
}
    };

($D:ident, $S:ident, ($($index:ident),+), $id:ty, $dim:ident, $trait_name:ident::$method_name:ident for Value) => {
impl<$S, $D, Rhs, $($index),+> $trait_name<Value<Rhs>> for $id
where Rhs: Clone,
$D: Dim<$S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<$D>,)+
for<'a> &'a mut $id: ScalarArrayMut<Dim=One, Row=$dim, Scalar=S>,
S: $trait_name<Rhs>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
    fn $method_name(&mut self, rhs: Value<Rhs>) {
        let mut v = self;
        apply_zip_mut_val(&mut v, Vec::from_vec_val($dim::from_value(rhs.0)), $trait_name::$method_name)
    }
}
    };
}


macro_rules! impl_vecref_binop_assign {
    ($D:ident, $S:ident, $I0:ident, $I1:ident, $id:ty, $dim:ident, $($trait_name:ident::$method_name:ident)+) => {$(
        impl_vecref_binop_assign_inner!{$D, $S, ($I0, $I1), $id, $dim, $trait_name::$method_name for VecRef}
        impl_vecref_binop_assign_inner!{$D, $S, ($I0, $I1), $id, $dim, $trait_name::$method_name for Value}
    )+};

    ($D:ident, $S:ident, $I0:ident, $I1:ident, $I2:ident, $id:ty, $dim:ident, $($trait_name:ident::$method_name:ident)+) => {$(
        impl_vecref_binop_assign_inner!{$D, $S, ($I0, $I1, $I2), $id, $dim, $trait_name::$method_name for VecRef}
        impl_vecref_binop_assign_inner!{$D, $S, ($I0, $I1, $I2), $id, $dim, $trait_name::$method_name for Value}
    )+};

    ($D:ident, $S:ident, $I0:ident, $I1:ident, $I2:ident, $I3:ident, $id:ty, $dim:ident, $($trait_name:ident::$method_name:ident)+) => {$(
        impl_vecref_binop_assign_inner!{$D, $S, ($I0, $I1, $I2, $I3), $id, $dim, $trait_name::$method_name for VecRef}
        impl_vecref_binop_assign_inner!{$D, $S, ($I0, $I1, $I2, $I3), $id, $dim, $trait_name::$method_name for Value}
    )+};
}


macro_rules! decl_refs {
     ($id:ident, $dim:ident, $array:ident, $($index:ident),+) => {
// TODO: Make safe. Blocked on rust-lang/rfcs#997
/// This is a temporary structure until something is done with rust-lang/rfcs#997
/// Do not rely on this type, just rely on the overloaded operators defined for the cases where
// it is returned right now.
pub struct $id<D, S, $($index),+>(PhantomData<(Vec<D, S>, $($index),+)>)
where D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>;

impl<S, D, $($index),+> RefCast<Vec<D, S>> for $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
 // TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
     fn from_ref(v: &Vec<D, S>) -> &Self {
         let ptr = v as *const _ as *const $id<D, S, $($index),+>;
         unsafe { &*ptr }
     }
     fn from_mut(v: &mut Vec<D, S>) -> &mut Self {
         let ptr = v as *mut _ as *mut $id<D, S, $($index),+>;
         unsafe { &mut*ptr }
     }
     fn into_ref(&self) -> &Vec<D, S> {
         let ptr = self as *const _ as *const Vec<D, S>;
         unsafe { &*ptr }
     }
     fn into_mut(&mut self) -> &mut Vec<D, S> {
         let ptr = self as *mut _ as *mut Vec<D, S>;
         unsafe { &mut*ptr }
     }
}

impl<'a, S, D, $($index),+> ScalarArray for &'a $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
     type Scalar = S;
     type Row = $dim;
     type Dim = One;
}

impl<'a, S, D, $($index),+> ScalarArray for &'a mut $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
     type Scalar = S;
     type Row = $dim;
     type Dim = One;
}

impl<'a, S, T, D, D2, $($index),+> HasConcreteScalarArray<T, D2> for &'a $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
D2: Dim<T>,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
D2::Smaller: Array<T>,
{
     /// The type of a concrete ScalarArray of the specified type
     type Concrete = Vec<D2, T>;
}

impl<'a, S, T, D, D2, $($index),+> HasConcreteScalarArray<T, D2> for &'a mut $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
D2: Dim<T>,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
D2::Smaller: Array<T>,
{
     /// The type of a concrete ScalarArray of the specified type
     type Concrete = Vec<D2, T>;
}

impl<'a, S, T, D, D2, $($index),+> HasConcreteVecArray<T, D2> for &'a $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
D2: Dim<T>,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
D2::Smaller: Array<T>,
{}

impl<'a, S, T, D, D2, $($index),+> HasConcreteVecArray<T, D2> for &'a mut $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
D2: Dim<T>,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
D2::Smaller: Array<T>,
{}

impl<'a, S, D, $($index),+> ScalarArrayVal for &'a $id<D, S, $($index),+>
where S: Clone,
D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
     #[inline(always)]
     fn get_val(self) -> CustomArrayOne<$array<S>> {
         CustomArrayOne([self.get_vec_val()])
     }
}

impl<'a, S, D, $($index),+> ScalarArrayVal for &'a mut $id<D, S, $($index),+>
where S: Clone,
D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
     #[inline(always)]
     fn get_val(self) -> CustomArrayOne<$array<S>> {
         CustomArrayOne([self.get_vec_val()])
     }
}

impl<'a, S, D, $($index),+> VecArrayVal for &'a $id<D, S, $($index),+>
where S: Clone,
D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
     #[inline(always)]
     fn get_vec_val(self) -> $array<S> {
         let v = self.into_ref().into_ref();
         $array([
             $($index::index_array(v).clone(),)+
         ])
     }
}

impl<'a, S, D, $($index),+> VecArrayVal for &'a mut $id<D, S, $($index),+>
where S: Clone,
D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
     #[inline(always)]
     fn get_vec_val(self) -> $array<S> {
         let v = self.into_ref().into_ref();
         $array([
             $($index::index_array(v).clone(),)+
         ])
     }
}

impl<'a, S, D, $($index),+> ScalarArrayRef for &'a $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
    #[inline(always)]
    fn get_ref(&self) -> CustomArrayOne<$array<&S>> {
        CustomArrayOne([self.get_vec_ref()])
    }
}

impl<'a, S, D, $($index),+> ScalarArrayRef for &'a mut $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
    #[inline(always)]
    fn get_ref(&self) -> CustomArrayOne<$array<&S>> {
        CustomArrayOne([self.get_vec_ref()])
    }
}

impl<'a, S, D, $($index),+> VecArrayRef for &'a $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
    #[inline(always)]
    fn get_vec_ref(&self) -> $array<&S> {
        let v = self.into_ref().into_ref();
        $array([
            $($index::index_array(v),)+
        ])
    }
}

impl<'a, S, D, $($index),+> VecArrayRef for &'a mut $id<D, S, $($index),+>
where D: Dim<S>$( + IsLargerThan<$index>)+,
$($index: IsSmallerThan<D>,)+
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
D::Smaller: Array<S>,
{
    #[inline(always)]
    fn get_vec_ref(&self) -> $array<&S> {
        let v = self.into_ref().into_ref();
        $array([
            $($index::index_array(v),)+
        ])
    }
}
     };
}

// d888888b db   d8b   db  .d88b.
// `~~88~~' 88   I8I   88 .8P  Y8.
//    88    88   I8I   88 88    88
//    88    Y8   I8I   88 88    88
//    88    `8b d8'8b d8' `8b  d8'
//    YP     `8b8' `8d8'   `Y88P'

decl_refs!{Vec2Ref, Two, CustomArrayTwo, I0, I1}
impl_vecref_unop!{D, S, I0, I1, Vec2Ref<D, S, I0, I1>, Two,
    Neg::neg Not::not
}
impl_vecref_binop!{D, S, I0, I1, Vec2Ref<D, S, I0, I1>, Two,
    BitAnd::bitand BitOr::bitor BitXor::bitxor
    Shl::shl Shr::shr
    Add::add Div::div Mul::mul Rem::rem Sub::sub
}
impl_vecref_binop_assign!{D, S, I0, I1, Vec2Ref<D, S, I0, I1>, Two,
    BitAndAssign::bitand_assign BitOrAssign::bitor_assign BitXorAssign::bitxor_assign
    ShlAssign::shl_assign ShrAssign::shr_assign
    AddAssign::add_assign DivAssign::div_assign MulAssign::mul_assign RemAssign::rem_assign SubAssign::sub_assign
}


impl<'a, S, D, I0, I1> ScalarArrayMut for &'a mut Vec2Ref<D, S, I0, I1>
where D: DimMut<S> + IsLargerThan<I1> + ExtractItem<I0>,
D::Smaller: DimMut<S> + ExtractItem<I1::Result>,
I0: IsSmallerThan<D> + NotSame<I1>,
I1: IsSmallerThan<D> + NotSame<I0> + DecrementIfLargerThan<I0>,
I1::Result: IsSmallerThan<D::Smaller>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<D::Smaller as HasSmaller>::Smaller: ArrayMut<S>,
{
    #[inline(always)]
    fn get_mut(&mut self) -> CustomArrayOne<CustomArrayTwo<&mut S>> {
        let v = D::get_mut(self.into_mut().into_mut());
        let (v0, v) = D::extract::<&mut S>(v);
        let (v1, _) = D::Smaller::extract::<&mut S>(v);
        CustomArrayOne([CustomArrayTwo([v0,v1])])
    }
}

// d888888b db   db d8888b. d88888b d88888b
// `~~88~~' 88   88 88  `8D 88'     88'
//    88    88ooo88 88oobY' 88ooooo 88ooooo
//    88    88~~~88 88`8b   88~~~~~ 88~~~~~
//    88    88   88 88 `88. 88.     88.
//    YP    YP   YP 88   YD Y88888P Y88888P

decl_refs!{Vec3Ref, Three, CustomArrayThree, I0, I1, I2}
impl_vecref_unop!{D, S, I0, I1, I2, Vec3Ref<D, S, I0, I1, I2>, Three,
    Neg::neg Not::not
}
impl_vecref_binop!{D, S, I0, I1, I2, Vec3Ref<D, S, I0, I1, I2>, Three,
    BitAnd::bitand BitOr::bitor BitXor::bitxor
    Shl::shl Shr::shr
    Add::add Div::div Mul::mul Rem::rem Sub::sub
}
impl_vecref_binop_assign!{D, S, I0, I1, I2, Vec3Ref<D, S, I0, I1, I2>, Three,
    BitAndAssign::bitand_assign BitOrAssign::bitor_assign BitXorAssign::bitxor_assign
    ShlAssign::shl_assign ShrAssign::shr_assign
    AddAssign::add_assign DivAssign::div_assign MulAssign::mul_assign RemAssign::rem_assign SubAssign::sub_assign
}

impl<'a, S, D, I0, I1, I2> ScalarArrayMut for &'a mut Vec3Ref<D, S, I0, I1, I2>
where D: DimMut<S> + IsLargerThan<I1> + IsLargerThan<I2> + ExtractItem<I0>,
D::Smaller: DimMut<S> + ExtractItem<I1::Result>,
<D::Smaller as HasSmaller>::Smaller: DimMut<S> + ExtractItem<<I2::Result as DecrementIfLargerThan<I1::Result>>::Result>,
I0: IsSmallerThan<D> + NotSame<I1> + NotSame<I2>,
I1: IsSmallerThan<D> + NotSame<I0> + NotSame<I2> + DecrementIfLargerThan<I0>,
I2: IsSmallerThan<D> + NotSame<I0> + NotSame<I1> + DecrementIfLargerThan<I0>,
I1::Result: IsSmallerThan<D::Smaller> + NotSame<I2::Result>,
I2::Result: DecrementIfLargerThan<I1::Result> + NotSame<I1::Result>,
<I2::Result as DecrementIfLargerThan<I1::Result>>::Result: IsSmallerThan<<D::Smaller as HasSmaller>::Smaller>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<<D::Smaller as HasSmaller>::Smaller as HasSmaller>::Smaller: ArrayMut<S>,
{
    #[inline(always)]
    fn get_mut(&mut self) -> CustomArrayOne<CustomArrayThree<&mut S>> {
        let v = D::get_mut(self.into_mut().into_mut());
        let (v0, v) = D::extract::<&mut S>(v);
        let (v1, v) = D::Smaller::extract::<&mut S>(v);
        let (v2, _) = <D::Smaller as HasSmaller>::Smaller::extract::<&mut S>(v);
        CustomArrayOne([CustomArrayThree([v0,v1,v2])])
    }
}

// d88888b  .d88b.  db    db d8888b.
// 88'     .8P  Y8. 88    88 88  `8D
// 88ooo   88    88 88    88 88oobY'
// 88~~~   88    88 88    88 88`8b
// 88      `8b  d8' 88b  d88 88 `88.
// YP       `Y88P'  ~Y8888P' 88   YD

decl_refs!{Vec4Ref, Four, CustomArrayFour, I0, I1, I2, I3}
impl_vecref_unop!{D, S, I0, I1, I2, I3, Vec4Ref<D, S, I0, I1, I2, I3>, Four,
    Neg::neg Not::not
}
impl_vecref_binop!{D, S, I0, I1, I2, I3, Vec4Ref<D, S, I0, I1, I2, I3>, Four,
    BitAnd::bitand BitOr::bitor BitXor::bitxor
    Shl::shl Shr::shr
    Add::add Div::div Mul::mul Rem::rem Sub::sub
}
impl_vecref_binop_assign!{D, S, I0, I1, I2, I3, Vec4Ref<D, S, I0, I1, I2, I3>, Four,
    BitAndAssign::bitand_assign BitOrAssign::bitor_assign BitXorAssign::bitxor_assign
    ShlAssign::shl_assign ShrAssign::shr_assign
    AddAssign::add_assign DivAssign::div_assign MulAssign::mul_assign RemAssign::rem_assign SubAssign::sub_assign
}

impl<'a, S, D, I0, I1, I2, I3> ScalarArrayMut for &'a mut Vec4Ref<D, S, I0, I1, I2, I3>
where D: DimMut<S> + IsLargerThan<I1> + IsLargerThan<I2> + IsLargerThan<I3> + ExtractItem<I0>,
D::Smaller: DimMut<S> + ExtractItem<I1::Result>,
<D::Smaller as HasSmaller>::Smaller: DimMut<S> + ExtractItem<<I2::Result as DecrementIfLargerThan<I1::Result>>::Result>,
<<D::Smaller as HasSmaller>::Smaller as HasSmaller>::Smaller: DimMut<S> + ExtractItem<<<I3::Result as DecrementIfLargerThan<I1::Result>>::Result as DecrementIfLargerThan<<I2::Result as DecrementIfLargerThan<I1::Result>>::Result>>::Result>,
I0: IsSmallerThan<D> + NotSame<I1> + NotSame<I2> + NotSame<I3>,
I1: IsSmallerThan<D> + NotSame<I0> + NotSame<I2> + NotSame<I3> + DecrementIfLargerThan<I0>,
I2: IsSmallerThan<D> + NotSame<I0> + NotSame<I1> + NotSame<I3> + DecrementIfLargerThan<I0>,
I3: IsSmallerThan<D> + NotSame<I0> + NotSame<I1> + NotSame<I2> + DecrementIfLargerThan<I0>,
I1::Result: IsSmallerThan<D::Smaller> + NotSame<I2::Result> + NotSame<I3::Result>,
I2::Result: DecrementIfLargerThan<I1::Result> + NotSame<I1::Result>,
I3::Result: DecrementIfLargerThan<I1::Result> + NotSame<I1::Result>,
<I2::Result as DecrementIfLargerThan<I1::Result>>::Result: IsSmallerThan<<D::Smaller as HasSmaller>::Smaller> + NotSame<<I3::Result as DecrementIfLargerThan<I1::Result>>::Result>,
<I3::Result as DecrementIfLargerThan<I1::Result>>::Result: DecrementIfLargerThan<<I2::Result as DecrementIfLargerThan<I1::Result>>::Result> + NotSame<<I2::Result as DecrementIfLargerThan<I1::Result>>::Result>,
<<I3::Result as DecrementIfLargerThan<I1::Result>>::Result as DecrementIfLargerThan<<I2::Result as DecrementIfLargerThan<I1::Result>>::Result>>::Result: IsSmallerThan<<<D::Smaller as HasSmaller>::Smaller as HasSmaller>::Smaller>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
<<<D::Smaller as HasSmaller>::Smaller as HasSmaller>::Smaller as HasSmaller>::Smaller: ArrayMut<S>,
{
    #[inline(always)]
    fn get_mut(&mut self) -> CustomArrayOne<CustomArrayFour<&mut S>> {
        let v = D::get_mut(self.into_mut().into_mut());
        let (v0, v) = D::extract::<&mut S>(v);
        let (v1, v) = D::Smaller::extract::<&mut S>(v);
        let (v2, v) = <D::Smaller as HasSmaller>::Smaller::extract::<&mut S>(v);
        let (v3, _) = <<D::Smaller as HasSmaller>::Smaller as HasSmaller>::Smaller::extract::<&mut S>(v);
        CustomArrayOne([CustomArrayFour([v0,v1,v2,v3])])
    }
}

pub mod swizzle {
    use std::marker::PhantomData;
    use std::ops::{Index,IndexMut};

    use super::{Vec2Ref,Vec3Ref,Vec4Ref};
    //use vec::Vec;
    use consts::{
        Array,Dim,
        ExtractArray,Constant,IsLargerThan,IsSmallerThan,NotSame,HasSmaller,
        Zero,One,Two,Three,
    };
    use utils::RefCast;
    use vec::Vec;

    macro_rules! decl_swizzle {
         ($id:ident, $vec_ref:ident, $($index:ident),+) => {
    /// A struct for assisting in swizzling
    pub struct $id<$($index),+>(PhantomData<($($index),+)>)
    where $($index: Constant),+;

    impl<D, S, $($index),+> Index<$id<$($index),+>> for Vec<D, S>
    where D: Dim<S>$( + IsLargerThan<$index>)+,
    $($index: IsSmallerThan<D>,)+
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    D::Smaller: Array<S>,
    {
        type Output = $vec_ref<D, S, $($index),+>;
        fn index(&self, _: $id<$($index),+>) -> &Self::Output {
             RefCast::from_ref(self)
        }
    }
        };
    }
    macro_rules! decl_swizzle_extract {
        ($($id:ident=($($value:ident),+)),+) => {
            decl_swizzle_extract!{$($id=($($value),+)),+}
        };
        ($($id:ident=($($value:ident),+),)+) => {$(
    /// A struct for assisting in swizzling
    pub struct $id;

    impl<D, S> Index<$id> for Vec<D, S>
    where D: Dim<S> + ExtractArray<$($value),+>,
    $($value: IsSmallerThan<D>,)+
    D::Extracted: Dim<S>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    D::Smaller: Array<S>,
    <D::Extracted as HasSmaller>::Smaller: Array<S>,
    {
        type Output = Vec<D::Extracted, S>;
        fn index(&self, _: $id) -> &Self::Output {
            RefCast::from_ref(D::extract_array_ref(self.into_ref()))
        }
    }

    impl<D, S> IndexMut<$id> for Vec<D, S>
    where D: Dim<S> + ExtractArray<$($value),+>,
    $($value: IsSmallerThan<D>,)+
    D::Extracted: Dim<S>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    D::Smaller: Array<S>,
    <D::Extracted as HasSmaller>::Smaller: Array<S>,
    {
        fn index_mut(&mut self, _: $id) -> &mut Self::Output {
            RefCast::from_mut(D::extract_array_mut(self.into_mut()))
        }
    }

        )+};
    }

    macro_rules! decl_constants {
        ($swizzle:ident, $($id:ident=($($value:ident),+),)+) => {
            decl_constants!{$swizzle, $($id=($($value),+)),+}
        };
        ($swizzle:ident, $($id:ident=($($value:ident),+)),+) => {$(
            /// A type alias for assisting in swizzling
            pub type $id = $swizzle<$($value),+>;

            /// Constant for swizzling
            pub const $id: $id = $swizzle(PhantomData);
        )+};
    }

    decl_swizzle!{Swizzle2, Vec2Ref, I0, I1}
    decl_swizzle!{Swizzle3, Vec3Ref, I0, I1, I2}
    decl_swizzle!{Swizzle4, Vec4Ref, I0, I1, I2, I3}

    impl<D, S, I0, I1> IndexMut<Swizzle2<I0, I1>> for Vec<D, S>
    where D: Dim<S> + IsLargerThan<I0> + IsLargerThan<I1>,
    I0: IsSmallerThan<D> + NotSame<I1>,
    I1: IsSmallerThan<D> + NotSame<I0>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    D::Smaller: Array<S>,
    {
         fn index_mut(&mut self, _: Swizzle2<I0, I1>) -> &mut Self::Output {
             RefCast::from_mut(self)
         }
    }

    impl<D, S, I0, I1, I2> IndexMut<Swizzle3<I0, I1, I2>> for Vec<D, S>
    where D: Dim<S> + IsLargerThan<I0> + IsLargerThan<I1> + IsLargerThan<I2>,
    I0: IsSmallerThan<D> + NotSame<I1> + NotSame<I2>,
    I1: IsSmallerThan<D> + NotSame<I0> + NotSame<I2>,
    I2: IsSmallerThan<D> + NotSame<I0> + NotSame<I1>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    D::Smaller: Array<S>,
    {
        fn index_mut(&mut self, _: Swizzle3<I0, I1, I2>) -> &mut Self::Output {
            RefCast::from_mut(self)
        }
    }

    impl<D, S, I0, I1, I2, I3> IndexMut<Swizzle4<I0, I1, I2, I3>> for Vec<D, S>
    where D: Dim<S> + IsLargerThan<I0> + IsLargerThan<I1> + IsLargerThan<I2> + IsLargerThan<I3>,
    I0: IsSmallerThan<D> + NotSame<I1> + NotSame<I2> + NotSame<I3>,
    I1: IsSmallerThan<D> + NotSame<I0> + NotSame<I2> + NotSame<I3>,
    I2: IsSmallerThan<D> + NotSame<I0> + NotSame<I1> + NotSame<I3>,
    I3: IsSmallerThan<D> + NotSame<I0> + NotSame<I1> + NotSame<I2>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    D::Smaller: Array<S>,
    {
         fn index_mut(&mut self, _: Swizzle4<I0, I1, I2, I3>) -> &mut Self::Output {
            RefCast::from_mut(self)
         }
    }

    decl_constants!{Swizzle2,
        I00=(Zero,Zero),                    I02=(Zero,Two),   I03=(Zero,Three),
        I10=(One,Zero),   I11=(One,One),                      I13=(One,Three),
        I20=(Two,Zero),   I21=(Two,One),    I22=(Two,Two),
        I30=(Three,Zero), I31=(Three,One),  I32=(Three,Two),  I33=(Three,Three),
    }

    decl_constants!{Swizzle3,
        I000=(Zero,Zero,Zero),   I001=(Zero,Zero,One),    I002=(Zero,Zero,Two),    I003=(Zero,Zero,Three),
        I010=(Zero,One,Zero),    I011=(Zero,One,One),                              I013=(Zero,One,Three),
        I020=(Zero,Two,Zero),    I021=(Zero,Two,One),     I022=(Zero,Two,Two),     I023=(Zero,Two,Three),
        I030=(Zero,Three,Zero),  I031=(Zero,Three,One),   I032=(Zero,Three,Two),   I033=(Zero,Three,Three),

        I100=(One,Zero,Zero),    I101=(One,Zero,One),     I102=(One,Zero,Two),     I103=(One,Zero,Three),
        I110=(One,One,Zero),     I111=(One,One,One),      I112=(One,One,Two),      I113=(One,One,Three),
        I120=(One,Two,Zero),     I121=(One,Two,One),      I122=(One,Two,Two),
        I130=(One,Three,Zero),   I131=(One,Three,One),    I132=(One,Three,Two),    I133=(One,Three,Three),

        I200=(Two,Zero,Zero),    I201=(Two,Zero,One),     I202=(Two,Zero,Two),     I203=(Two,Zero,Three),
        I210=(Two,One,Zero),     I211=(Two,One,One),      I212=(Two,One,Two),      I213=(Two,One,Three),
        I220=(Two,Two,Zero),     I221=(Two,Two,One),      I222=(Two,Two,Two),      I223=(Two,Two,Three),
        I230=(Two,Three,Zero),   I231=(Two,Three,One),    I232=(Two,Three,Two),    I233=(Two,Three,Three),

        I300=(Three,Zero,Zero),  I301=(Three,Zero,One),   I302=(Three,Zero,Two),   I303=(Three,Zero,Three),
        I310=(Three,One,Zero),   I311=(Three,One,One),    I312=(Three,One,Two),    I313=(Three,One,Three),
        I320=(Three,Two,Zero),   I321=(Three,Two,One),    I322=(Three,Two,Two),    I323=(Three,Two,Three),
        I330=(Three,Three,Zero), I331=(Three,Three,One),  I332=(Three,Three,Two),  I333=(Three,Three,Three),
    }

    decl_constants!{Swizzle4,
        I0000=(Zero,Zero,Zero,Zero),    I0001=(Zero,Zero,Zero,One),     I0002=(Zero,Zero,Zero,Two),     I0003=(Zero,Zero,Zero,Three),
        I0010=(Zero,Zero,One,Zero),     I0011=(Zero,Zero,One,One),      I0012=(Zero,Zero,One,Two),      I0013=(Zero,Zero,One,Three),
        I0020=(Zero,Zero,Two,Zero),     I0021=(Zero,Zero,Two,One),      I0022=(Zero,Zero,Two,Two),      I0023=(Zero,Zero,Two,Three),
        I0030=(Zero,Zero,Three,Zero),   I0031=(Zero,Zero,Three,One),    I0032=(Zero,Zero,Three,Two),    I0033=(Zero,Zero,Three,Three),

        I0100=(Zero,One,Zero,Zero),     I0101=(Zero,One,Zero,One),      I0102=(Zero,One,Zero,Two),      I0103=(Zero,One,Zero,Three),
        I0110=(Zero,One,One,Zero),      I0111=(Zero,One,One,One),       I0112=(Zero,One,One,Two),       I0113=(Zero,One,One,Three),
        I0120=(Zero,One,Two,Zero),      I0121=(Zero,One,Two,One),       I0122=(Zero,One,Two,Two),
        I0130=(Zero,One,Three,Zero),    I0131=(Zero,One,Three,One),     I0132=(Zero,One,Three,Two),     I0133=(Zero,One,Three,Three),

        I0200=(Zero,Two,Zero,Zero),     I0201=(Zero,Two,Zero,One),      I0202=(Zero,Two,Zero,Two),      I0203=(Zero,Two,Zero,Three),
        I0210=(Zero,Two,One,Zero),      I0211=(Zero,Two,One,One),       I0212=(Zero,Two,One,Two),       I0213=(Zero,Two,One,Three),
        I0220=(Zero,Two,Two,Zero),      I0221=(Zero,Two,Two,One),       I0222=(Zero,Two,Two,Two),       I0223=(Zero,Two,Two,Three),
        I0230=(Zero,Two,Three,Zero),    I0231=(Zero,Two,Three,One),     I0232=(Zero,Two,Three,Two),     I0233=(Zero,Two,Three,Three),

        I0300=(Zero,Three,Zero,Zero),   I0301=(Zero,Three,Zero,One),    I0302=(Zero,Three,Zero,Two),    I0303=(Zero,Three,Zero,Three),
        I0310=(Zero,Three,One,Zero),    I0311=(Zero,Three,One,One),     I0312=(Zero,Three,One,Two),     I0313=(Zero,Three,One,Three),
        I0320=(Zero,Three,Two,Zero),    I0321=(Zero,Three,Two,One),     I0322=(Zero,Three,Two,Two),     I0323=(Zero,Three,Two,Three),
        I0330=(Zero,Three,Three,Zero),  I0331=(Zero,Three,Three,One),   I0332=(Zero,Three,Three,Two),   I0333=(Zero,Three,Three,Three),


        I1000=(One,Zero,Zero,Zero),     I1001=(One,Zero,Zero,One),      I1002=(One,Zero,Zero,Two),      I1003=(One,Zero,Zero,Three),
        I1010=(One,Zero,One,Zero),      I1011=(One,Zero,One,One),       I1012=(One,Zero,One,Two),       I1013=(One,Zero,One,Three),
        I1020=(One,Zero,Two,Zero),      I1021=(One,Zero,Two,One),       I1022=(One,Zero,Two,Two),       I1023=(One,Zero,Two,Three),
        I1030=(One,Zero,Three,Zero),    I1031=(One,Zero,Three,One),     I1032=(One,Zero,Three,Two),     I1033=(One,Zero,Three,Three),

        I1100=(One,One,Zero,Zero),      I1101=(One,One,Zero,One),       I1102=(One,One,Zero,Two),       I1103=(One,One,Zero,Three),
        I1110=(One,One,One,Zero),       I1111=(One,One,One,One),        I1112=(One,One,One,Two),        I1113=(One,One,One,Three),
        I1120=(One,One,Two,Zero),       I1121=(One,One,Two,One),        I1122=(One,One,Two,Two),        I1123=(One,One,Two,Three),
        I1130=(One,One,Three,Zero),     I1131=(One,One,Three,One),      I1132=(One,One,Three,Two),      I1133=(One,One,Three,Three),

        I1200=(One,Two,Zero,Zero),      I1201=(One,Two,Zero,One),       I1202=(One,Two,Zero,Two),       I1203=(One,Two,Zero,Three),
        I1210=(One,Two,One,Zero),       I1211=(One,Two,One,One),        I1212=(One,Two,One,Two),        I1213=(One,Two,One,Three),
        I1220=(One,Two,Two,Zero),       I1221=(One,Two,Two,One),        I1222=(One,Two,Two,Two),        I1223=(One,Two,Two,Three),
        I1230=(One,Two,Three,Zero),     I1231=(One,Two,Three,One),      I1232=(One,Two,Three,Two),      I1233=(One,Two,Three,Three),

        I1300=(One,Three,Zero,Zero),    I1301=(One,Three,Zero,One),     I1302=(One,Three,Zero,Two),     I1303=(One,Three,Zero,Three),
        I1310=(One,Three,One,Zero),     I1311=(One,Three,One,One),      I1312=(One,Three,One,Two),      I1313=(One,Three,One,Three),
        I1320=(One,Three,Two,Zero),     I1321=(One,Three,Two,One),      I1322=(One,Three,Two,Two),      I1323=(One,Three,Two,Three),
        I1330=(One,Three,Three,Zero),   I1331=(One,Three,Three,One),    I1332=(One,Three,Three,Two),    I1333=(One,Three,Three,Three),


        I2000=(Two,Zero,Zero,Zero),     I2001=(Two,Zero,Zero,One),      I2002=(Two,Zero,Zero,Two),      I2003=(Two,Zero,Zero,Three),
        I2010=(Two,Zero,One,Zero),      I2011=(Two,Zero,One,One),       I2012=(Two,Zero,One,Two),       I2013=(Two,Zero,One,Three),
        I2020=(Two,Zero,Two,Zero),      I2021=(Two,Zero,Two,One),       I2022=(Two,Zero,Two,Two),       I2023=(Two,Zero,Two,Three),
        I2030=(Two,Zero,Three,Zero),    I2031=(Two,Zero,Three,One),     I2032=(Two,Zero,Three,Two),     I2033=(Two,Zero,Three,Three),

        I2100=(Two,One,Zero,Zero),      I2101=(Two,One,Zero,One),       I2102=(Two,One,Zero,Two),       I2103=(Two,One,Zero,Three),
        I2110=(Two,One,One,Zero),       I2111=(Two,One,One,One),        I2112=(Two,One,One,Two),        I2113=(Two,One,One,Three),
        I2120=(Two,One,Two,Zero),       I2121=(Two,One,Two,One),        I2122=(Two,One,Two,Two),        I2123=(Two,One,Two,Three),
        I2130=(Two,One,Three,Zero),     I2131=(Two,One,Three,One),      I2132=(Two,One,Three,Two),      I2133=(Two,One,Three,Three),

        I2200=(Two,Two,Zero,Zero),      I2201=(Two,Two,Zero,One),       I2202=(Two,Two,Zero,Two),       I2203=(Two,Two,Zero,Three),
        I2210=(Two,Two,One,Zero),       I2211=(Two,Two,One,One),        I2212=(Two,Two,One,Two),        I2213=(Two,Two,One,Three),
        I2220=(Two,Two,Two,Zero),       I2221=(Two,Two,Two,One),        I2222=(Two,Two,Two,Two),        I2223=(Two,Two,Two,Three),
        I2230=(Two,Two,Three,Zero),     I2231=(Two,Two,Three,One),      I2232=(Two,Two,Three,Two),      I2233=(Two,Two,Three,Three),

        I2300=(Two,Three,Zero,Zero),    I2301=(Two,Three,Zero,One),     I2302=(Two,Three,Zero,Two),     I2303=(Two,Three,Zero,Three),
        I2310=(Two,Three,One,Zero),     I2311=(Two,Three,One,One),      I2312=(Two,Three,One,Two),      I2313=(Two,Three,One,Three),
        I2320=(Two,Three,Two,Zero),     I2321=(Two,Three,Two,One),      I2322=(Two,Three,Two,Two),      I2323=(Two,Three,Two,Three),
        I2330=(Two,Three,Three,Zero),   I2331=(Two,Three,Three,One),    I2332=(Two,Three,Three,Two),    I2333=(Two,Three,Three,Three),


        I3000=(Three,Zero,Zero,Zero),   I3001=(Three,Zero,Zero,One),    I3002=(Three,Zero,Zero,Two),    I3003=(Three,Zero,Zero,Three),
        I3010=(Three,Zero,One,Zero),    I3011=(Three,Zero,One,One),     I3012=(Three,Zero,One,Two),     I3013=(Three,Zero,One,Three),
        I3020=(Three,Zero,Two,Zero),    I3021=(Three,Zero,Two,One),     I3022=(Three,Zero,Two,Two),     I3023=(Three,Zero,Two,Three),
        I3030=(Three,Zero,Three,Zero),  I3031=(Three,Zero,Three,One),   I3032=(Three,Zero,Three,Two),   I3033=(Three,Zero,Three,Three),

        I3100=(Three,One,Zero,Zero),    I3101=(Three,One,Zero,One),     I3102=(Three,One,Zero,Two),     I3103=(Three,One,Zero,Three),
        I3110=(Three,One,One,Zero),     I3111=(Three,One,One,One),      I3112=(Three,One,One,Two),      I3113=(Three,One,One,Three),
        I3120=(Three,One,Two,Zero),     I3121=(Three,One,Two,One),      I3122=(Three,One,Two,Two),      I3123=(Three,One,Two,Three),
        I3130=(Three,One,Three,Zero),   I3131=(Three,One,Three,One),    I3132=(Three,One,Three,Two),    I3133=(Three,One,Three,Three),

        I3200=(Three,Two,Zero,Zero),    I3201=(Three,Two,Zero,One),     I3202=(Three,Two,Zero,Two),     I3203=(Three,Two,Zero,Three),
        I3210=(Three,Two,One,Zero),     I3211=(Three,Two,One,One),      I3212=(Three,Two,One,Two),      I3213=(Three,Two,One,Three),
        I3220=(Three,Two,Two,Zero),     I3221=(Three,Two,Two,One),      I3222=(Three,Two,Two,Two),      I3223=(Three,Two,Two,Three),
        I3230=(Three,Two,Three,Zero),   I3231=(Three,Two,Three,One),    I3232=(Three,Two,Three,Two),    I3233=(Three,Two,Three,Three),

        I3300=(Three,Three,Zero,Zero),  I3301=(Three,Three,Zero,One),   I3302=(Three,Three,Zero,Two),   I3303=(Three,Three,Zero,Three),
        I3310=(Three,Three,One,Zero),   I3311=(Three,Three,One,One),    I3312=(Three,Three,One,Two),    I3313=(Three,Three,One,Three),
        I3320=(Three,Three,Two,Zero),   I3321=(Three,Three,Two,One),    I3322=(Three,Three,Two,Two),    I3323=(Three,Three,Two,Three),
        I3330=(Three,Three,Three,Zero), I3331=(Three,Three,Three,One),  I3332=(Three,Three,Three,Two),  I3333=(Three,Three,Three,Three),
    }

    decl_swizzle_extract!{
        I01=(Zero,One), I12=(One,Two), I23=(Two,Three),
        I012=(Zero,Two), I123=(One,Three),
        I0123=(Zero,Three),
    }

    pub mod xyzw {
        pub use super::{
    		// Pair
    		I00 as XX, I01 as XY, I02 as XZ, I03 as XW,
    		I10 as YX, I11 as YY, I12 as YZ, I13 as YW,
    		I20 as ZX, I21 as ZY, I22 as ZZ, I23 as ZW,
    		I30 as WX, I31 as WY, I32 as WZ, I33 as WW,


    		// Triple
    		I000 as XXX, I001 as XXY, I002 as XXZ, I003 as XXW,
    		I010 as XYX, I011 as XYY, I012 as XYZ, I013 as XYW,
    		I020 as XZX, I021 as XZY, I022 as XZZ, I023 as XZW,
    		I030 as XWX, I031 as XWY, I032 as XWZ, I033 as XWW,
    		I100 as YXX, I101 as YXY, I102 as YXZ, I103 as YXW,
    		I110 as YYX, I111 as YYY, I112 as YYZ, I113 as YYW,
    		I120 as YZX, I121 as YZY, I122 as YZZ, I123 as YZW,
    		I130 as YWX, I131 as YWY, I132 as YWZ, I133 as YWW,
    		I200 as ZXX, I201 as ZXY, I202 as ZXZ, I203 as ZXW,
    		I210 as ZYX, I211 as ZYY, I212 as ZYZ, I213 as ZYW,
    		I220 as ZZX, I221 as ZZY, I222 as ZZZ, I223 as ZZW,
    		I230 as ZWX, I231 as ZWY, I232 as ZWZ, I233 as ZWW,
    		I300 as WXX, I301 as WXY, I302 as WXZ, I303 as WXW,
    		I310 as WYX, I311 as WYY, I312 as WYZ, I313 as WYW,
    		I320 as WZX, I321 as WZY, I322 as WZZ, I323 as WZW,
    		I330 as WWX, I331 as WWY, I332 as WWZ, I333 as WWW,


    		// Quadruple
    		I0000 as XXXX, I0001 as XXXY, I0002 as XXXZ, I0003 as XXXW,
    		I0010 as XXYX, I0011 as XXYY, I0012 as XXYZ, I0013 as XXYW,
    		I0020 as XXZX, I0021 as XXZY, I0022 as XXZZ, I0023 as XXZW,
    		I0030 as XXWX, I0031 as XXWY, I0032 as XXWZ, I0033 as XXWW,
    		I0100 as XYXX, I0101 as XYXY, I0102 as XYXZ, I0103 as XYXW,
    		I0110 as XYYX, I0111 as XYYY, I0112 as XYYZ, I0113 as XYYW,
    		I0120 as XYZX, I0121 as XYZY, I0122 as XYZZ, I0123 as XYZW,
    		I0130 as XYWX, I0131 as XYWY, I0132 as XYWZ, I0133 as XYWW,
    		I0200 as XZXX, I0201 as XZXY, I0202 as XZXZ, I0203 as XZXW,
    		I0210 as XZYX, I0211 as XZYY, I0212 as XZYZ, I0213 as XZYW,
    		I0220 as XZZX, I0221 as XZZY, I0222 as XZZZ, I0223 as XZZW,
    		I0230 as XZWX, I0231 as XZWY, I0232 as XZWZ, I0233 as XZWW,
    		I0300 as XWXX, I0301 as XWXY, I0302 as XWXZ, I0303 as XWXW,
    		I0310 as XWYX, I0311 as XWYY, I0312 as XWYZ, I0313 as XWYW,
    		I0320 as XWZX, I0321 as XWZY, I0322 as XWZZ, I0323 as XWZW,
    		I0330 as XWWX, I0331 as XWWY, I0332 as XWWZ, I0333 as XWWW,

    		I1000 as YXXX, I1001 as YXXY, I1002 as YXXZ, I1003 as YXXW,
    		I1010 as YXYX, I1011 as YXYY, I1012 as YXYZ, I1013 as YXYW,
    		I1020 as YXZX, I1021 as YXZY, I1022 as YXZZ, I1023 as YXZW,
    		I1030 as YXWX, I1031 as YXWY, I1032 as YXWZ, I1033 as YXWW,
    		I1100 as YYXX, I1101 as YYXY, I1102 as YYXZ, I1103 as YYXW,
    		I1110 as YYYX, I1111 as YYYY, I1112 as YYYZ, I1113 as YYYW,
    		I1120 as YYZX, I1121 as YYZY, I1122 as YYZZ, I1123 as YYZW,
    		I1130 as YYWX, I1131 as YYWY, I1132 as YYWZ, I1133 as YYWW,
    		I1200 as YZXX, I1201 as YZXY, I1202 as YZXZ, I1203 as YZXW,
    		I1210 as YZYX, I1211 as YZYY, I1212 as YZYZ, I1213 as YZYW,
    		I1220 as YZZX, I1221 as YZZY, I1222 as YZZZ, I1223 as YZZW,
    		I1230 as YZWX, I1231 as YZWY, I1232 as YZWZ, I1233 as YZWW,
    		I1300 as YWXX, I1301 as YWXY, I1302 as YWXZ, I1303 as YWXW,
    		I1310 as YWYX, I1311 as YWYY, I1312 as YWYZ, I1313 as YWYW,
    		I1320 as YWZX, I1321 as YWZY, I1322 as YWZZ, I1323 as YWZW,
    		I1330 as YWWX, I1331 as YWWY, I1332 as YWWZ, I1333 as YWWW,

    		I2000 as ZXXX, I2001 as ZXXY, I2002 as ZXXZ, I2003 as ZXXW,
    		I2010 as ZXYX, I2011 as ZXYY, I2012 as ZXYZ, I2013 as ZXYW,
    		I2020 as ZXZX, I2021 as ZXZY, I2022 as ZXZZ, I2023 as ZXZW,
    		I2030 as ZXWX, I2031 as ZXWY, I2032 as ZXWZ, I2033 as ZXWW,
    		I2100 as ZYXX, I2101 as ZYXY, I2102 as ZYXZ, I2103 as ZYXW,
    		I2110 as ZYYX, I2111 as ZYYY, I2112 as ZYYZ, I2113 as ZYYW,
    		I2120 as ZYZX, I2121 as ZYZY, I2122 as ZYZZ, I2123 as ZYZW,
    		I2130 as ZYWX, I2131 as ZYWY, I2132 as ZYWZ, I2133 as ZYWW,
    		I2200 as ZZXX, I2201 as ZZXY, I2202 as ZZXZ, I2203 as ZZXW,
    		I2210 as ZZYX, I2211 as ZZYY, I2212 as ZZYZ, I2213 as ZZYW,
    		I2220 as ZZZX, I2221 as ZZZY, I2222 as ZZZZ, I2223 as ZZZW,
    		I2230 as ZZWX, I2231 as ZZWY, I2232 as ZZWZ, I2233 as ZZWW,
    		I2300 as ZWXX, I2301 as ZWXY, I2302 as ZWXZ, I2303 as ZWXW,
    		I2310 as ZWYX, I2311 as ZWYY, I2312 as ZWYZ, I2313 as ZWYW,
    		I2320 as ZWZX, I2321 as ZWZY, I2322 as ZWZZ, I2323 as ZWZW,
    		I2330 as ZWWX, I2331 as ZWWY, I2332 as ZWWZ, I2333 as ZWWW,

    		I3000 as WXXX, I3001 as WXXY, I3002 as WXXZ, I3003 as WXXW,
    		I3010 as WXYX, I3011 as WXYY, I3012 as WXYZ, I3013 as WXYW,
    		I3020 as WXZX, I3021 as WXZY, I3022 as WXZZ, I3023 as WXZW,
    		I3030 as WXWX, I3031 as WXWY, I3032 as WXWZ, I3033 as WXWW,
    		I3100 as WYXX, I3101 as WYXY, I3102 as WYXZ, I3103 as WYXW,
    		I3110 as WYYX, I3111 as WYYY, I3112 as WYYZ, I3113 as WYYW,
    		I3120 as WYZX, I3121 as WYZY, I3122 as WYZZ, I3123 as WYZW,
    		I3130 as WYWX, I3131 as WYWY, I3132 as WYWZ, I3133 as WYWW,
    		I3200 as WZXX, I3201 as WZXY, I3202 as WZXZ, I3203 as WZXW,
    		I3210 as WZYX, I3211 as WZYY, I3212 as WZYZ, I3213 as WZYW,
    		I3220 as WZZX, I3221 as WZZY, I3222 as WZZZ, I3223 as WZZW,
    		I3230 as WZWX, I3231 as WZWY, I3232 as WZWZ, I3233 as WZWW,
    		I3300 as WWXX, I3301 as WWXY, I3302 as WWXZ, I3303 as WWXW,
    		I3310 as WWYX, I3311 as WWYY, I3312 as WWYZ, I3313 as WWYW,
    		I3320 as WWZX, I3321 as WWZY, I3322 as WWZZ, I3323 as WWZW,
    		I3330 as WWWX, I3331 as WWWY, I3332 as WWWZ, I3333 as WWWW,
    	};
    }
    pub mod stpq {
        pub use super::{
    		// Pair
    		I00 as SS, I01 as ST, I02 as SP, I03 as SQ,
    		I10 as TS, I11 as TT, I12 as TP, I13 as TQ,
    		I20 as PS, I21 as PT, I22 as PP, I23 as PQ,
    		I30 as QS, I31 as QT, I32 as QP, I33 as QQ,


    		// Triple
    		I000 as SSS, I001 as SST, I002 as SSP, I003 as SSQ,
    		I010 as STS, I011 as STT, I012 as STP, I013 as STQ,
    		I020 as SPS, I021 as SPT, I022 as SPP, I023 as SPQ,
    		I030 as SQS, I031 as SQT, I032 as SQP, I033 as SQQ,
    		I100 as TSS, I101 as TST, I102 as TSP, I103 as TSQ,
    		I110 as TTS, I111 as TTT, I112 as TTP, I113 as TTQ,
    		I120 as TPS, I121 as TPT, I122 as TPP, I123 as TPQ,
    		I130 as TQS, I131 as TQT, I132 as TQP, I133 as TQQ,
    		I200 as PSS, I201 as PST, I202 as PSP, I203 as PSQ,
    		I210 as PTS, I211 as PTT, I212 as PTP, I213 as PTQ,
    		I220 as PPS, I221 as PPT, I222 as PPP, I223 as PPQ,
    		I230 as PQS, I231 as PQT, I232 as PQP, I233 as PQQ,
    		I300 as QSS, I301 as QST, I302 as QSP, I303 as QSQ,
    		I310 as QTS, I311 as QTT, I312 as QTP, I313 as QTQ,
    		I320 as QPS, I321 as QPT, I322 as QPP, I323 as QPQ,
    		I330 as QQS, I331 as QQT, I332 as QQP, I333 as QQQ,


    		// Quadruple
    		I0000 as SSSS, I0001 as SSST, I0002 as SSSP, I0003 as SSSQ,
    		I0010 as SSTS, I0011 as SSTT, I0012 as SSTP, I0013 as SSTQ,
    		I0020 as SSPS, I0021 as SSPT, I0022 as SSPP, I0023 as SSPQ,
    		I0030 as SSQS, I0031 as SSQT, I0032 as SSQP, I0033 as SSQQ,
    		I0100 as STSS, I0101 as STST, I0102 as STSP, I0103 as STSQ,
    		I0110 as STTS, I0111 as STTT, I0112 as STTP, I0113 as STTQ,
    		I0120 as STPS, I0121 as STPT, I0122 as STPP, I0123 as STPQ,
    		I0130 as STQS, I0131 as STQT, I0132 as STQP, I0133 as STQQ,
    		I0200 as SPSS, I0201 as SPST, I0202 as SPSP, I0203 as SPSQ,
    		I0210 as SPTS, I0211 as SPTT, I0212 as SPTP, I0213 as SPTQ,
    		I0220 as SPPS, I0221 as SPPT, I0222 as SPPP, I0223 as SPPQ,
    		I0230 as SPQS, I0231 as SPQT, I0232 as SPQP, I0233 as SPQQ,
    		I0300 as SQSS, I0301 as SQST, I0302 as SQSP, I0303 as SQSQ,
    		I0310 as SQTS, I0311 as SQTT, I0312 as SQTP, I0313 as SQTQ,
    		I0320 as SQPS, I0321 as SQPT, I0322 as SQPP, I0323 as SQPQ,
    		I0330 as SQQS, I0331 as SQQT, I0332 as SQQP, I0333 as SQQQ,

    		I1000 as TSSS, I1001 as TSST, I1002 as TSSP, I1003 as TSSQ,
    		I1010 as TSTS, I1011 as TSTT, I1012 as TSTP, I1013 as TSTQ,
    		I1020 as TSPS, I1021 as TSPT, I1022 as TSPP, I1023 as TSPQ,
    		I1030 as TSQS, I1031 as TSQT, I1032 as TSQP, I1033 as TSQQ,
    		I1100 as TTSS, I1101 as TTST, I1102 as TTSP, I1103 as TTSQ,
    		I1110 as TTTS, I1111 as TTTT, I1112 as TTTP, I1113 as TTTQ,
    		I1120 as TTPS, I1121 as TTPT, I1122 as TTPP, I1123 as TTPQ,
    		I1130 as TTQS, I1131 as TTQT, I1132 as TTQP, I1133 as TTQQ,
    		I1200 as TPSS, I1201 as TPST, I1202 as TPSP, I1203 as TPSQ,
    		I1210 as TPTS, I1211 as TPTT, I1212 as TPTP, I1213 as TPTQ,
    		I1220 as TPPS, I1221 as TPPT, I1222 as TPPP, I1223 as TPPQ,
    		I1230 as TPQS, I1231 as TPQT, I1232 as TPQP, I1233 as TPQQ,
    		I1300 as TQSS, I1301 as TQST, I1302 as TQSP, I1303 as TQSQ,
    		I1310 as TQTS, I1311 as TQTT, I1312 as TQTP, I1313 as TQTQ,
    		I1320 as TQPS, I1321 as TQPT, I1322 as TQPP, I1323 as TQPQ,
    		I1330 as TQQS, I1331 as TQQT, I1332 as TQQP, I1333 as TQQQ,

    		I2000 as PSSS, I2001 as PSST, I2002 as PSSP, I2003 as PSSQ,
    		I2010 as PSTS, I2011 as PSTT, I2012 as PSTP, I2013 as PSTQ,
    		I2020 as PSPS, I2021 as PSPT, I2022 as PSPP, I2023 as PSPQ,
    		I2030 as PSQS, I2031 as PSQT, I2032 as PSQP, I2033 as PSQQ,
    		I2100 as PTSS, I2101 as PTST, I2102 as PTSP, I2103 as PTSQ,
    		I2110 as PTTS, I2111 as PTTT, I2112 as PTTP, I2113 as PTTQ,
    		I2120 as PTPS, I2121 as PTPT, I2122 as PTPP, I2123 as PTPQ,
    		I2130 as PTQS, I2131 as PTQT, I2132 as PTQP, I2133 as PTQQ,
    		I2200 as PPSS, I2201 as PPST, I2202 as PPSP, I2203 as PPSQ,
    		I2210 as PPTS, I2211 as PPTT, I2212 as PPTP, I2213 as PPTQ,
    		I2220 as PPPS, I2221 as PPPT, I2222 as PPPP, I2223 as PPPQ,
    		I2230 as PPQS, I2231 as PPQT, I2232 as PPQP, I2233 as PPQQ,
    		I2300 as PQSS, I2301 as PQST, I2302 as PQSP, I2303 as PQSQ,
    		I2310 as PQTS, I2311 as PQTT, I2312 as PQTP, I2313 as PQTQ,
    		I2320 as PQPS, I2321 as PQPT, I2322 as PQPP, I2323 as PQPQ,
    		I2330 as PQQS, I2331 as PQQT, I2332 as PQQP, I2333 as PQQQ,

    		I3000 as QSSS, I3001 as QSST, I3002 as QSSP, I3003 as QSSQ,
    		I3010 as QSTS, I3011 as QSTT, I3012 as QSTP, I3013 as QSTQ,
    		I3020 as QSPS, I3021 as QSPT, I3022 as QSPP, I3023 as QSPQ,
    		I3030 as QSQS, I3031 as QSQT, I3032 as QSQP, I3033 as QSQQ,
    		I3100 as QTSS, I3101 as QTST, I3102 as QTSP, I3103 as QTSQ,
    		I3110 as QTTS, I3111 as QTTT, I3112 as QTTP, I3113 as QTTQ,
    		I3120 as QTPS, I3121 as QTPT, I3122 as QTPP, I3123 as QTPQ,
    		I3130 as QTQS, I3131 as QTQT, I3132 as QTQP, I3133 as QTQQ,
    		I3200 as QPSS, I3201 as QPST, I3202 as QPSP, I3203 as QPSQ,
    		I3210 as QPTS, I3211 as QPTT, I3212 as QPTP, I3213 as QPTQ,
    		I3220 as QPPS, I3221 as QPPT, I3222 as QPPP, I3223 as QPPQ,
    		I3230 as QPQS, I3231 as QPQT, I3232 as QPQP, I3233 as QPQQ,
    		I3300 as QQSS, I3301 as QQST, I3302 as QQSP, I3303 as QQSQ,
    		I3310 as QQTS, I3311 as QQTT, I3312 as QQTP, I3313 as QQTQ,
    		I3320 as QQPS, I3321 as QQPT, I3322 as QQPP, I3323 as QQPQ,
    		I3330 as QQQS, I3331 as QQQT, I3332 as QQQP, I3333 as QQQQ,
    	};
    }
    pub mod rgba {
        pub use super::{
    		// Pair
    		I00 as RR, I01 as RG, I02 as RB, I03 as RA,
    		I10 as GR, I11 as GG, I12 as GB, I13 as GA,
    		I20 as BR, I21 as BG, I22 as BB, I23 as BA,
    		I30 as AR, I31 as AG, I32 as AB, I33 as AA,


    		// Triple
    		I000 as RRR, I001 as RRG, I002 as RRB, I003 as RRA,
    		I010 as RGR, I011 as RGG, I012 as RGB, I013 as RGA,
    		I020 as RBR, I021 as RBG, I022 as RBB, I023 as RBA,
    		I030 as RAR, I031 as RAG, I032 as RAB, I033 as RAA,
    		I100 as GRR, I101 as GRG, I102 as GRB, I103 as GRA,
    		I110 as GGR, I111 as GGG, I112 as GGB, I113 as GGA,
    		I120 as GBR, I121 as GBG, I122 as GBB, I123 as GBA,
    		I130 as GAR, I131 as GAG, I132 as GAB, I133 as GAA,
    		I200 as BRR, I201 as BRG, I202 as BRB, I203 as BRA,
    		I210 as BGR, I211 as BGG, I212 as BGB, I213 as BGA,
    		I220 as BBR, I221 as BBG, I222 as BBB, I223 as BBA,
    		I230 as BAR, I231 as BAG, I232 as BAB, I233 as BAA,
    		I300 as ARR, I301 as ARG, I302 as ARB, I303 as ARA,
    		I310 as AGR, I311 as AGG, I312 as AGB, I313 as AGA,
    		I320 as ABR, I321 as ABG, I322 as ABB, I323 as ABA,
    		I330 as AAR, I331 as AAG, I332 as AAB, I333 as AAA,


    		// Quadruple
    		I0000 as RRRR, I0001 as RRRG, I0002 as RRRB, I0003 as RRRA,
    		I0010 as RRGR, I0011 as RRGG, I0012 as RRGB, I0013 as RRGA,
    		I0020 as RRBR, I0021 as RRBG, I0022 as RRBB, I0023 as RRBA,
    		I0030 as RRAR, I0031 as RRAG, I0032 as RRAB, I0033 as RRAA,
    		I0100 as RGRR, I0101 as RGRG, I0102 as RGRB, I0103 as RGRA,
    		I0110 as RGGR, I0111 as RGGG, I0112 as RGGB, I0113 as RGGA,
    		I0120 as RGBR, I0121 as RGBG, I0122 as RGBB, I0123 as RGBA,
    		I0130 as RGAR, I0131 as RGAG, I0132 as RGAB, I0133 as RGAA,
    		I0200 as RBRR, I0201 as RBRG, I0202 as RBRB, I0203 as RBRA,
    		I0210 as RBGR, I0211 as RBGG, I0212 as RBGB, I0213 as RBGA,
    		I0220 as RBBR, I0221 as RBBG, I0222 as RBBB, I0223 as RBBA,
    		I0230 as RBAR, I0231 as RBAG, I0232 as RBAB, I0233 as RBAA,
    		I0300 as RARR, I0301 as RARG, I0302 as RARB, I0303 as RARA,
    		I0310 as RAGR, I0311 as RAGG, I0312 as RAGB, I0313 as RAGA,
    		I0320 as RABR, I0321 as RABG, I0322 as RABB, I0323 as RABA,
    		I0330 as RAAR, I0331 as RAAG, I0332 as RAAB, I0333 as RAAA,

    		I1000 as GRRR, I1001 as GRRG, I1002 as GRRB, I1003 as GRRA,
    		I1010 as GRGR, I1011 as GRGG, I1012 as GRGB, I1013 as GRGA,
    		I1020 as GRBR, I1021 as GRBG, I1022 as GRBB, I1023 as GRBA,
    		I1030 as GRAR, I1031 as GRAG, I1032 as GRAB, I1033 as GRAA,
    		I1100 as GGRR, I1101 as GGRG, I1102 as GGRB, I1103 as GGRA,
    		I1110 as GGGR, I1111 as GGGG, I1112 as GGGB, I1113 as GGGA,
    		I1120 as GGBR, I1121 as GGBG, I1122 as GGBB, I1123 as GGBA,
    		I1130 as GGAR, I1131 as GGAG, I1132 as GGAB, I1133 as GGAA,
    		I1200 as GBRR, I1201 as GBRG, I1202 as GBRB, I1203 as GBRA,
    		I1210 as GBGR, I1211 as GBGG, I1212 as GBGB, I1213 as GBGA,
    		I1220 as GBBR, I1221 as GBBG, I1222 as GBBB, I1223 as GBBA,
    		I1230 as GBAR, I1231 as GBAG, I1232 as GBAB, I1233 as GBAA,
    		I1300 as GARR, I1301 as GARG, I1302 as GARB, I1303 as GARA,
    		I1310 as GAGR, I1311 as GAGG, I1312 as GAGB, I1313 as GAGA,
    		I1320 as GABR, I1321 as GABG, I1322 as GABB, I1323 as GABA,
    		I1330 as GAAR, I1331 as GAAG, I1332 as GAAB, I1333 as GAAA,

    		I2000 as BRRR, I2001 as BRRG, I2002 as BRRB, I2003 as BRRA,
    		I2010 as BRGR, I2011 as BRGG, I2012 as BRGB, I2013 as BRGA,
    		I2020 as BRBR, I2021 as BRBG, I2022 as BRBB, I2023 as BRBA,
    		I2030 as BRAR, I2031 as BRAG, I2032 as BRAB, I2033 as BRAA,
    		I2100 as BGRR, I2101 as BGRG, I2102 as BGRB, I2103 as BGRA,
    		I2110 as BGGR, I2111 as BGGG, I2112 as BGGB, I2113 as BGGA,
    		I2120 as BGBR, I2121 as BGBG, I2122 as BGBB, I2123 as BGBA,
    		I2130 as BGAR, I2131 as BGAG, I2132 as BGAB, I2133 as BGAA,
    		I2200 as BBRR, I2201 as BBRG, I2202 as BBRB, I2203 as BBRA,
    		I2210 as BBGR, I2211 as BBGG, I2212 as BBGB, I2213 as BBGA,
    		I2220 as BBBR, I2221 as BBBG, I2222 as BBBB, I2223 as BBBA,
    		I2230 as BBAR, I2231 as BBAG, I2232 as BBAB, I2233 as BBAA,
    		I2300 as BARR, I2301 as BARG, I2302 as BARB, I2303 as BARA,
    		I2310 as BAGR, I2311 as BAGG, I2312 as BAGB, I2313 as BAGA,
    		I2320 as BABR, I2321 as BABG, I2322 as BABB, I2323 as BABA,
    		I2330 as BAAR, I2331 as BAAG, I2332 as BAAB, I2333 as BAAA,

    		I3000 as ARRR, I3001 as ARRG, I3002 as ARRB, I3003 as ARRA,
    		I3010 as ARGR, I3011 as ARGG, I3012 as ARGB, I3013 as ARGA,
    		I3020 as ARBR, I3021 as ARBG, I3022 as ARBB, I3023 as ARBA,
    		I3030 as ARAR, I3031 as ARAG, I3032 as ARAB, I3033 as ARAA,
    		I3100 as AGRR, I3101 as AGRG, I3102 as AGRB, I3103 as AGRA,
    		I3110 as AGGR, I3111 as AGGG, I3112 as AGGB, I3113 as AGGA,
    		I3120 as AGBR, I3121 as AGBG, I3122 as AGBB, I3123 as AGBA,
    		I3130 as AGAR, I3131 as AGAG, I3132 as AGAB, I3133 as AGAA,
    		I3200 as ABRR, I3201 as ABRG, I3202 as ABRB, I3203 as ABRA,
    		I3210 as ABGR, I3211 as ABGG, I3212 as ABGB, I3213 as ABGA,
    		I3220 as ABBR, I3221 as ABBG, I3222 as ABBB, I3223 as ABBA,
    		I3230 as ABAR, I3231 as ABAG, I3232 as ABAB, I3233 as ABAA,
    		I3300 as AARR, I3301 as AARG, I3302 as AARB, I3303 as AARA,
    		I3310 as AAGR, I3311 as AAGG, I3312 as AAGB, I3313 as AAGA,
    		I3320 as AABR, I3321 as AABG, I3322 as AABB, I3323 as AABA,
    		I3330 as AAAR, I3331 as AAAG, I3332 as AAAB, I3333 as AAAA,
    	};
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::swizzle::xyzw::*;
    use vec::Vec4;

    #[test]
    fn test_vec_single() {
        let v = Vec4::new([1i8 , 2i8 , 3i8 , 4i8 ]);
        assert_eq!(v[X], 1);
    }

    #[test]
    fn test_vec_swizzle_add() {
        let v = Vec4::new([1i8 , 2i8 , 3i8 , 4i8 ]);
        assert_eq!(&v[ZWYX] + Vec4::from_value(1i8), Vec4::new([4i8 , 5i8 , 3i8 , 2i8 ]));
    }

    #[test]
    fn test_vec_swizzle_add_assign() {
        let mut v = Vec4::new([1i8 , 2i8 , 3i8 , 4i8 ]);
        v[ZWYX] += Vec4::from_value(1i8);
        assert_eq!(v, Vec4::new([2i8 , 3i8 , 4i8 , 5i8 ]));
    }

}
