//! A basic linear algebra library for computer graphics.
//!
//! It uses generics to provide extensible square and rectangular Matrix types of sizes 1x1 through
//! 4x4 as well as Vector types of size 1 through 4.
#![feature(associated_consts)]
#![feature(slice_patterns)]
#![feature(advanced_slice_patterns)]
#![cfg_attr(test, feature(float_extras))]
#![cfg_attr(feature="serde_all", feature(proc_macro))]

#![allow(unknown_lints)]
#![allow(inline_always)]

#[cfg(feature="rustc-serialize")]
extern crate rustc_serialize;
#[cfg(feature="rand")]
extern crate rand;
#[cfg(feature="serde_all")]
extern crate serde;
#[cfg(feature="serde_all")]
#[macro_use]
extern crate serde_derive;

#[cfg(feature="rand")]
use rand::{Rand,Rng};
use std::ops::{
    Neg,Not,
    BitAnd,BitOr,BitXor,
    Shl,Shr,
    Add,Div,Mul,Rem,Sub,
    BitAndAssign,BitOrAssign,BitXorAssign,
    ShlAssign,ShrAssign,
    AddAssign,DivAssign,MulAssign,RemAssign,SubAssign,
    Deref,DerefMut
};

// TODO: More comments

pub mod angle;
pub mod consts;
pub mod index;
pub mod mat;
pub mod scalar_array;
pub mod vec;
pub mod num;

mod utils;

/// A simple wrapper to allow generic scalar binary ops
///
/// There's generally no reason to store this type, just create it with `v` for use in expressions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Value<S>(pub S);

#[cfg(feature="rand")]
impl <S: Rand> Rand for Value<S> {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        v(Rand::rand(rng))
    }
}

/// An function to produce Value which is short to type
pub fn v<S>(s: S) -> Value<S> { Value(s) }

macro_rules! impl_val_unop {
    ($($trait_name:ident::$method_name:ident)+) => {$(
impl<S: $trait_name> $trait_name for Value<S> {
    type Output = Value<S::Output>;
    fn $method_name(self) -> Self::Output { v($trait_name::$method_name(self.0)) }
}
    )+};
}

macro_rules! impl_val_binop {
    ($($trait_name:ident::$method_name:ident)+) => {$(
impl<Rhs, S: $trait_name<Rhs>> $trait_name<Value<Rhs>> for Value<S> {
    type Output = Value<S::Output>;
    fn $method_name(self, rhs: Value<Rhs>) -> Self::Output { v($trait_name::$method_name(self.0, rhs.0)) }
}
    )+};
}

macro_rules! impl_val_binop_assign {
    ($($trait_name:ident::$method_name:ident)+) => {$(
impl<Rhs, S: $trait_name<Rhs>> $trait_name<Value<Rhs>> for Value<S> {
    fn $method_name(&mut self, rhs: Value<Rhs>) { $trait_name::$method_name(&mut self.0, rhs.0) }
}
    )+};
}

impl_val_unop!{Neg::neg Not::not}
impl_val_binop!{
    BitAnd::bitand BitOr::bitor BitXor::bitxor
    Shl::shl Shr::shr
    Add::add Div::div Rem::rem Sub::sub Mul::mul
}
impl_val_binop_assign!{
    BitAndAssign::bitand_assign BitOrAssign::bitor_assign BitXorAssign::bitxor_assign
    ShlAssign::shl_assign ShrAssign::shr_assign
    AddAssign::add_assign DivAssign::div_assign RemAssign::rem_assign SubAssign::sub_assign MulAssign::mul_assign
}

impl<S> Deref for Value<S> {
    type Target = S;

    #[inline(always)]
    fn deref(&self) -> &S { &self.0 }
}

impl<S> DerefMut for Value<S> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut S { &mut self.0 }
}
