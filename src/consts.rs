//! Traits and structs that emulate integer generics.

// TODO: more commments

use std::borrow::{Borrow,BorrowMut};
use std::fmt;
use std::ops::{Deref,DerefMut};
use std::mem::{forget,replace,uninitialized};
#[cfg(feature="rand")]
use rand::{Rand,Rng};
#[cfg(feature="rustc-serialize")]
use rustc_serialize::{Encodable,Encoder,Decodable,Decoder};
#[cfg(feature="serde_all")]
use serde::{Deserialize,Deserializer,Serialize,Serializer};

use utils::RefCast;

/// Types that represent a constant
pub trait Constant: 'static + Sized {
    /// The value of a constant
    const VALUE: usize;
}

/// A trait to indicate that constants are not the same value
pub trait NotSame<C: NotSame<Self>>: Constant {}

/// A trait to indicate that a constant has a lower value than `C`
pub trait IsSmallerThan<C: IsLargerThan<Self> + NotSame<Self>>: NotSame<C> {
    /// Index an array of size `C` safely.
    #[inline(always)]
    fn index_array<T>(a0: &C::Type) -> &T
    where C: Array<T>;
}

/// A trait to indicate that a constant has a higher value than `C`
pub trait IsLargerThan<C: IsSmallerThan<Self> + NotSame<Self>>: NotSame<C> {}

/// Types that have constants smaller than themselves
pub trait HasSmaller: NotSame<<Self as HasSmaller>::Smaller> {
    type Smaller: HasLarger<Larger=Self> + NotSame<Self>;
}

/// Types that have constants larger than themselves
pub trait HasLarger: NotSame<<Self as HasLarger>::Larger> {
    type Larger: HasSmaller<Smaller=Self> + NotSame<Self>;
}

/// Types that can have an array position extracted.
pub trait ExtractItem<Rhs: IsSmallerThan<Self>>: IsLargerThan<Rhs> {
    #[inline(always)]
    fn extract<T>(a0: Self::Type) -> (T, <Self::Smaller as Array<T>>::Type)
    where Self: Dim<T>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    Self::Smaller: Array<T>;
}

/// Types that can have an array position extracted.
pub trait ExtractArray<Lower, Upper>: IsLargerThan<Lower> + IsLargerThan<Upper>
where Lower: IsSmallerThan<Self> + IsSmallerThan<Upper>,
Upper: IsSmallerThan<Self> + IsLargerThan<Lower> {
    /// The extracted reference array size
    type Extracted: Constant;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &Self::Type) -> &<Self::Extracted as Array<T>>::Type
    where Self: Dim<T>,
    Self::Extracted: Dim<T>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    <Self::Extracted as HasSmaller>::Smaller: Array<T>,
    Self::Smaller: Array<T>;

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut Self::Type) -> &mut <Self::Extracted as Array<T>>::Type
    where Self: Dim<T>,
    Self::Extracted: Dim<T>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    <Self::Extracted as HasSmaller>::Smaller: Array<T>,
    Self::Smaller: Array<T>;

}

/// Types that can be conditionally decremented if `Rhs` is smaller
pub trait DecrementIfLargerThan<Rhs: NotSame<Self>>: NotSame<Rhs> {
    type Result: Constant;
}

/// Types that can be conditionally selected
pub trait SelectTwo<I0: Constant, I1: Constant>: IsSmallerThan<Two>
where Two: IsLargerThan<Self> {
    type Selected: Constant;
}

/// Types that can be conditionally selected
pub trait SelectThree<I0: Constant, I1: Constant, I2: Constant>: IsSmallerThan<Three>
where Three: IsLargerThan<Self> {
    type Selected: Constant;
}

/// Types that can be conditionally selected
pub trait SelectFour<I0: Constant, I1: Constant, I2: Constant, I3: Constant>: IsSmallerThan<Four>
where Four: IsLargerThan<Self> {
    type Selected: Constant;
}

/// Types that represent an array
pub trait Array<T>: Constant {
    /// A raw array of the size equal to the dimension this type represents.
    type RawType;

    /// A custom array of the size equal to the dimension this type represents.
    type Type;

    /// Construct an array from a single value `v`, replicating it to all positions in the array.
    #[inline(always)]
    fn from_value(v: T) -> Self::Type
    where T: Clone;

    /// Apply `f` to all the elements of the array
    #[inline(always)]
    fn apply<F: FnMut(T)>(a0: Self::Type, f: F);

    /// Apply `f` to all elements of two arrays.
    #[inline(always)]
    fn apply2<U, F>(a0: <Self as Array<T>>::Type, a1: <Self as Array<U>>::Type, f: F)
    where Self: Array<U>, F: FnMut(T, U);

    /// Fold all the elements of the array with function `f`
    #[inline(always)]
    fn fold<O, F>(a0: Self::Type, init: O, f: F) -> O
    where F: FnMut(O, T)-> O;

    /// Fold all the elements of two arrays with function `f`
    #[inline(always)]
    fn fold2<U, O, F>(a0: <Self as Array<T>>::Type, a1: <Self as Array<U>>::Type, init: O, f: F) -> O
    where Self: Array<U>, F: FnMut(O, T, U)-> O;

    /// Map all the elements of the array with function `f`
    #[inline(always)]
    fn map<O, F>(a0: <Self as Array<T>>::Type, f: F) -> <Self as Array<O>>::Type
    where Self: Array<O>, F: FnMut(T)-> O;

    /// Map all the elements into two arrays with function `f`
    #[inline(always)]
    fn map_into_2<O1, O2, F>(a0: <Self as Array<T>>::Type, f: F) -> (<Self as Array<O1>>::Type, <Self as Array<O2>>::Type)
    where Self: Array<O1>+Array<O2>, F: FnMut(T)-> (O1, O2);

    /// Map all the elements of two arrays with function `f`
    #[inline(always)]
    fn map2<U, O, F>(a0: <Self as Array<T>>::Type, a1: <Self as Array<U>>::Type, f: F) -> <Self as Array<O>>::Type
    where Self: Array<U>+Array<O>, F: FnMut(T, U)-> O;

    /// Map all the elements of three arrays with function `f`
    #[inline(always)]
    fn map3<T2, T3, O, F>(a0: <Self as Array<T>>::Type, a1: <Self as Array<T2>>::Type, a2: <Self as Array<T3>>::Type, f: F) -> <Self as Array<O>>::Type
    where Self: Array<T2> + Array<T3> + Array<O>, F: FnMut(T, T2, T3)-> O;

    /// Transpose the elements of a 2d array
    #[inline(always)]
    fn transpose<U,S>(a0: <Self as Array<T>>::Type) -> <U as Array<<Self as Array<S>>::Type>>::Type
    where Self: Array<S>,
    U: Dim<S, Type=T> + Dim<<Self as Array<S>>::Type>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    U::Smaller: Array<S>+Array<<Self as Array<S>>::Type>;

    /// A helper to transpose the elements of a 2d array (recursion)
    #[inline(always)]
    fn transpose_helper<U>(a0: <U as Array<<Self as Array<T>>::Type>>::Type) -> <Self as Array<<U as Array<T>>::Type>>::Type
    where U: Array<<Self as Array<T>>::Type> + Array<T>,
    Self: Array<<U as Array<T>>::Type>;
}

pub trait ArrayRef<T>: Array<T>
where for<'a> Self: Array<&'a T> {
    #[inline(always)]
    fn get_ref(a0: &<Self as Array<T>>::Type) -> <Self as Array<&T>>::Type;
}

pub trait ArrayMut<T>: ArrayRef<T>
where for<'a> Self: Array<&'a mut T> {
    #[inline(always)]
    fn get_mut(a0: &mut <Self as Array<T>>::Type) -> <Self as Array<&mut T>>::Type;
}

/// Types that represent a dimension.
pub trait Dim<T>: Array<T> + HasSmaller
where Self::Smaller: Array<T> {
    /// Split the array into an element and a smaller array.
    #[inline(always)]
    fn split(a0: Self::Type) -> (T, <Self::Smaller as Array<T>>::Type);

    /// Split the array into a reference to an element and a reference to a smaller array.
    #[inline(always)]
    fn split_ref(a0: &Self::Type) -> (&T, &<Self::Smaller as Array<T>>::Type);

    /// Split the array into a reference to a smaller array and a reference to an element.
    #[inline(always)]
    fn split_ref_end(a0: &Self::Type) -> (&<Self::Smaller as Array<T>>::Type, &T);

    /// Split the array into a mutable reference to an element and a mutable reference to a smaller array.
    #[inline(always)]
    fn split_mut(a0: &mut Self::Type) -> (&mut T, &mut <Self::Smaller as Array<T>>::Type);

    /// Split the array into a mutable reference to a smaller array and a mutable reference to an element.
    #[inline(always)]
    fn split_mut_end(a0: &mut Self::Type) -> (&mut <Self::Smaller as Array<T>>::Type, &mut T);

    /// The opposite of split.
    #[inline(always)]
    fn chain(v: T, end: <Self::Smaller as Array<T>>::Type) -> Self::Type;
}

pub trait DimRef<T>: Dim<T>+ArrayRef<T>
where for<'a> Self: Dim<&'a T>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Self::Smaller: Array<T>,
{}

pub trait DimMut<T>: DimRef<T>+ArrayMut<T>
where for<'a> Self: Dim<&'a mut T>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Self::Smaller: Array<T>,
{}

pub trait TwoDim<T, D: Dim<T>>: Dim<D::Type>
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
where Self::Smaller: Array<D::Type>,
D::Smaller: Array<T>,
{}

pub trait TwoDimRef<T, D: DimRef<T>>: TwoDim<T, D>+DimRef<<D as Array<T>>::Type>
where for<'a> Self: TwoDim<&'a T, D>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Self::Smaller: Array<<D as Array<T>>::Type>,
D::Smaller: Array<T>,
{}

pub trait TwoDimMut<T, D: DimMut<T>>: TwoDimRef<T, D>+DimMut<<D as Array<T>>::Type>
where for<'a> Self: TwoDim<&'a mut T, D>,
// TODO: remove elaborted bounds. Blocked on rust/issues#20671
Self::Smaller: Array<<D as Array<T>>::Type>,
D::Smaller: Array<T>,
{}

macro_rules! impl_custom_array_inner {
    ($id:ident, $size:expr$(, $index:expr)*) => {
/// An custom array of size $size which has as few restrictions as possible (e.g. doesn't restrict
/// `Clone` impls to `T: Copy`).
#[repr(C)]
#[derive(Copy, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct $id<T>(pub [T; $size]);

impl <T: fmt::Debug> fmt::Debug for $id<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

#[cfg(feature="rand")]
impl <T: Rand> Rand for $id<T> {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        $id(Rand::rand(rng))
    }
}

#[cfg(feature="rustc-serialize")]
impl <T: Encodable> Encodable for $id<T> {
    fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
        Encodable::encode(&self.0, e)
    }
}

#[cfg(feature="rustc-serialize")]
impl <T: Decodable> Decodable for $id<T> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        Ok($id(Decodable::decode(d)?))
    }
}

#[cfg(feature="serde_all")]
impl <T: Serialize> Serialize for $id<T> {
    fn serialize<S: Serializer>(&self, serializer: &mut S) -> Result<(), S::Error> {
        Serialize::serialize(&self.0, serializer)
    }
}

#[cfg(feature="serde_all")]
impl <T: Deserialize> Deserialize for $id<T> {
    fn deserialize<D: Deserializer>(deserializer: &mut D) -> Result<Self, D::Error> {
        Ok($id(Deserialize::deserialize(deserializer)?))
    }
}

impl<T> RefCast<[T; $size]> for $id<T> {
    #[inline(always)]
    fn from_ref(v: &[T; $size]) -> &Self {
        let ptr = v as *const _ as *const $id<T>;
        unsafe { &*ptr }
    }

    #[inline(always)]
    fn from_mut(v: &mut [T; $size]) -> &mut Self {
        let ptr = v as *mut _ as *mut $id<T>;
        unsafe { &mut *ptr }
    }

    #[inline(always)]
    fn into_ref(&self) -> &[T; $size] { &self.0 }

    #[inline(always)]
    fn into_mut(&mut self) -> &mut [T; $size] { &mut self.0 }
}

impl<T> From<[T; $size]> for $id<T> {
    fn from(v: [T; $size]) -> Self { $id(v) }
}
impl<T> Into<[T; $size]> for $id<T> {
    fn into(self) -> [T; $size] { self.0 }
}
impl<T> AsRef<[T; $size]> for $id<T> {
    fn as_ref(&self) -> &[T; $size] { &self.0 }
}
impl<T> AsMut<[T; $size]> for $id<T> {
    fn as_mut(&mut self) -> &mut [T; $size] { &mut self.0 }
}
impl<T> Borrow<[T; $size]> for $id<T> {
    fn borrow(&self) -> &[T; $size] { &self.0 }
}
impl<T> BorrowMut<[T; $size]> for $id<T> {
    fn borrow_mut(&mut self) -> &mut [T; $size] { &mut self.0 }
}
impl<T> Deref for $id<T> {
    type Target = [T; $size];
    fn deref(&self) -> &[T; $size] { &self.0}
}
impl<T> DerefMut for $id<T> {
    fn deref_mut(&mut self) -> &mut [T; $size] { &mut self.0 }
}

impl<T> AsRef<[T]> for $id<T> {
    fn as_ref(&self) -> &[T] { self.0.as_ref() }
}
impl<T> AsMut<[T]> for $id<T> {
    fn as_mut(&mut self) -> &mut [T] { self.0.as_mut() }
}
impl<T> Borrow<[T]> for $id<T> {
    fn borrow(&self) -> &[T] { self.0.borrow() }
}
impl<T> BorrowMut<[T]> for $id<T> {
    fn borrow_mut(&mut self) -> &mut [T] { self.0.borrow_mut() }
}

impl<'a, T> IntoIterator for &'a $id<T> {
    type Item = &'a T;
    type IntoIter = <&'a [T;1] as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

impl<'a, T> IntoIterator for &'a mut $id<T> {
    type Item = &'a mut T;
    type IntoIter = <&'a mut [T;1] as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter { (&mut self.0).into_iter() }
}

    }
}

macro_rules! impl_custom_array {
    ($id:ident) => {
        impl_custom_array_inner!{$id, 0}

        impl<T> Clone for $id<T> {
             fn clone(&self) -> Self {
                $id([])
             }
        }

        impl<T> Default for $id<T> {
             fn default() -> Self {
                $id([])
             }
        }
    };
    ($id:ident, $size:expr$(, $index:expr)*) => {
        impl_custom_array_inner!{$id, $size$(, $index)*}

        impl<T: Clone> Clone for $id<T> {
             fn clone(&self) -> Self {
                $id([
                    $(self.0[$index].clone()),*
                ])
             }
        }

        impl<T: Default> Default for $id<T> {
             fn default() -> Self {
                $id([
                    $({$index; Default::default()}),*
                ])
             }
        }
    }
}

macro_rules! impl_array_inner {
    ($id:ident, $array:ident, $size:expr,
        from($from_value_v:tt)=>$from_value:expr,
        apply($apply_a0:tt, $apply_f:tt)=>$apply:expr,
        apply2($apply2_a0:tt, $apply2_a1:tt, $apply2_f:tt)=>$apply2:expr,
        fold($fold_a0:tt, $fold_init:tt, $fold_f:tt)=>$fold:expr,
        fold2($fold2_a0:tt, $fold2_a1:tt, $fold2_init:tt, $fold2_f:tt)=>$fold2:expr,
        map($map_a0:tt, $map_f:tt)=>$map:expr,
        map_into_2($map_into_2_a0:tt, $map_into_2_f:tt)=>$map_into_2:expr,
        map2($map2_a0:tt, $map2_a1:tt, $map2_f:tt)=>$map2:expr,
        map3($map3_a0:tt, $map3_a1:tt, $map3_a2:tt, $map3_f:tt)=>$map3:expr,
        transpose($transpose_a0:tt)=>$transpose:expr,
        transpose_helper($transpose_helper_a0:tt)=>$transpose_helper:expr,
        array_ref($array_ref_a0:tt)=>$array_ref:expr,
        array_mut($array_mut_a0:tt)=>$array_mut:expr,
        $($index:expr),*) => (
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct $id;

impl Constant for $id {
    const VALUE: usize = $size;
}

impl<T> Array<T> for $id {
    type RawType = [T; $size];
    type Type = $array<T>;

    #[inline(always)]
    fn from_value($from_value_v: T) -> Self::Type
    where T: Clone { $from_value }

    #[inline(always)]
    fn apply<F>($apply_a0: Self::Type, $apply_f: F)
    where F: FnMut(T) { $apply }

    #[inline(always)]
    fn apply2<U, F>($apply2_a0: <Self as Array<T>>::Type, $apply2_a1: <Self as Array<U>>::Type, $apply2_f: F)
    where F: FnMut(T, U) { $apply2 }

    #[inline(always)]
    fn fold<O, F>($fold_a0: Self::Type, $fold_init: O, $fold_f: F) -> O
    where F: FnMut(O, T)-> O { $fold }

    #[inline(always)]
    fn fold2<U, O, F>($fold2_a0: <Self as Array<T>>::Type, $fold2_a1: <Self as Array<U>>::Type, $fold2_init: O, $fold2_f: F) -> O
    where F: FnMut(O, T, U)-> O { $fold2 }

    #[inline(always)]
    fn map<O, F>($map_a0: Self::Type, $map_f: F) -> <Self as Array<O>>::Type
    where F: FnMut(T)-> O { $map }

    #[inline(always)]
    fn map_into_2<O1, O2, F>($map_into_2_a0: <Self as Array<T>>::Type, $map_into_2_f: F) -> (<Self as Array<O1>>::Type, <Self as Array<O2>>::Type)
    where F: FnMut(T)-> (O1, O2) { $map_into_2 }

    #[inline(always)]
    fn map2<U, O, F>($map2_a0: Self::Type, $map2_a1: <Self as Array<U>>::Type, $map2_f: F) -> <Self as Array<O>>::Type
    where F: FnMut(T, U)-> O { $map2 }

    #[inline(always)]
    fn map3<T2, T3, O, F>($map3_a0: <Self as Array<T>>::Type, $map3_a1: <Self as Array<T2>>::Type, $map3_a2: <Self as Array<T3>>::Type, $map3_f: F) -> <Self as Array<O>>::Type
    where F: FnMut(T, T2, T3)-> O { $map3 }

    #[inline(always)]
    fn transpose<U,S>($transpose_a0: <Self as Array<T>>::Type) -> <U as Array<<Self as Array<S>>::Type>>::Type
    where U: Dim<S, Type=T> + Dim<<Self as Array<S>>::Type>,
    U::Smaller: Array<S>+Array<<Self as Array<S>>::Type> { $transpose }

    #[inline(always)]
    fn transpose_helper<U>($transpose_helper_a0: <U as Array<<Self as Array<T>>::Type>>::Type) -> <Self as Array<<U as Array<T>>::Type>>::Type
    where U: Array<<Self as Array<T>>::Type> + Array<T> { $transpose_helper }
}

impl<T> ArrayRef<T> for $id {
    #[inline(always)]
    fn get_ref($array_ref_a0: &<Self as Array<T>>::Type) -> <Self as Array<&T>>::Type {
        $array_ref
    }
}

impl<T> ArrayMut<T> for $id {
    #[inline(always)]
    fn get_mut($array_mut_a0: &mut <Self as Array<T>>::Type) -> <Self as Array<&mut T>>::Type {
        $array_mut
    }
}
    );
}

macro_rules! impl_array {
    ($id:ident, $array:ident) => (
        impl_custom_array!{$array}
        impl_array_inner!{$id, $array, 0,
            from(_)=>[].into(),
            apply(_, _)=>(),
            apply2(_, _, _)=>(),
            fold(_, init, _)=> init,
            fold2(_, _, init, _)=> init,
            map(_, _)=>[].into(),
            map_into_2(_, _)=>([].into(), [].into()),
            map2(_, _, _)=>[].into(),
            map3(_, _, _, _)=>[].into(),
            transpose(_) => <U as Array<<Self as Array<S>>::Type>>::from_value([].into()),
            transpose_helper(_) => [].into(),
            array_ref(_) => [].into(),
            array_mut(_) => [].into(),
        }
    );
    ($id:ident, $array:ident, $size:expr$(, $index:expr;$a0v1:ident;$a1v1:ident;$a2v1:ident)*) => (
        impl_custom_array!{$array, $size, 0$(, $index)*}
        impl_array_inner!{$id, $array, $size,
            from(v)=>{
                $(let $a0v1 = v.clone();)*
                [v$(, $a0v1)*].into()
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            apply(a0, f)=>unsafe {
                let (mut a0, mut f) = (a0, f);
                let a0v0 = replace(&mut a0[0], uninitialized());
        		$(let $a0v1 = replace(&mut a0[$index], uninitialized());)*
        		forget(a0);
        		f(a0v0);
                $(f($a0v1);)*
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            apply2(a0, a1, f)=>unsafe {
                let (mut a0, mut a1, mut f) = (a0, a1, f);
                let a0v0 = replace(&mut a0[0], uninitialized());
                let a1v0 = replace(&mut a1[0], uninitialized());
        		$(let $a0v1 = replace(&mut a0[$index], uninitialized());)*
        		$(let $a1v1 = replace(&mut a1[$index], uninitialized());)*
                forget(a0);
        		forget(a1);
        		f(a0v0, a1v0);
                $(f($a0v1, $a1v1);)*
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            fold(a0, init, f)=>unsafe {
                let (mut a0, mut f) = (a0, f);
                let a0v0 = replace(&mut a0[0], uninitialized());
        		$(let $a0v1 = replace(&mut a0[$index], uninitialized());)*
        		forget(a0);
                let init = f(init, a0v0);
                $(let init = f(init, $a0v1);)*
        		(init)
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            fold2(a0, a1, init, f)=>unsafe {
                let (mut a0, mut a1, mut f) = (a0, a1, f);
                let a0v0 = replace(&mut a0[0], uninitialized());
                let a1v0 = replace(&mut a1[0], uninitialized());
        		$(let $a0v1 = replace(&mut a0[$index], uninitialized());)*
        		$(let $a1v1 = replace(&mut a1[$index], uninitialized());)*
                forget(a0);
        		forget(a1);
                let init = f(init, a0v0, a1v0);
                $(let init = f(init, $a0v1, $a1v1);)*
        		(init)
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            map(a0, f)=>unsafe {
                let (mut a0, mut f) = (a0, f);
                let a0v0 = replace(&mut a0[0], uninitialized());
        		$(let $a0v1 = replace(&mut a0[$index], uninitialized());)*
        		forget(a0);
        		[f(a0v0)$(, f($a0v1))*].into()
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            map_into_2(a0, f)=>unsafe {
                let (mut a0, mut f) = (a0, f);
                let a0v0 = replace(&mut a0[0], uninitialized());
        		$(let $a0v1 = replace(&mut a0[$index], uninitialized());)*
        		forget(a0);
                let (a0v0, a1v0) = f(a0v0);
                $(let ($a0v1, $a1v1) = f($a0v1);)*
        		([a0v0$(, $a0v1)*].into(), [a1v0$(, $a1v1)*].into())
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            map2(a0, a1, f)=>unsafe {
                let (mut a0, mut a1, mut f) = (a0, a1, f);
                let a0v0 = replace(&mut a0[0], uninitialized());
                let a1v0 = replace(&mut a1[0], uninitialized());
        		$(let $a0v1 = replace(&mut a0[$index], uninitialized());)*
        		$(let $a1v1 = replace(&mut a1[$index], uninitialized());)*
                forget(a0);
        		forget(a1);
        		[f(a0v0, a1v0)$(, f($a0v1, $a1v1))*].into()
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            map3(a0, a1, a2, f)=>unsafe {
                let (mut a0, mut a1, mut a2, mut f) = (a0, a1, a2, f);
                let a0v0 = replace(&mut a0[0], uninitialized());
                let a1v0 = replace(&mut a1[0], uninitialized());
                let a2v0 = replace(&mut a2[0], uninitialized());
        		$(let $a0v1 = replace(&mut a0[$index], uninitialized());)*
        		$(let $a1v1 = replace(&mut a1[$index], uninitialized());)*
        		$(let $a2v1 = replace(&mut a2[$index], uninitialized());)*
                forget(a0);
        		forget(a1);
        		forget(a2);
        		[f(a0v0, a1v0, a2v0)$(, f($a0v1, $a1v1, $a2v1))*].into()
            },
            transpose(a0) => unsafe {
                let mut a0 = a0;
                let a0v0 = replace(&mut a0[0], uninitialized());
        		$(let $a0v1 = replace(&mut a0[$index], uninitialized());)*
                forget(a0);

                let a0v0 = <U as Dim<S>>::split(a0v0);
        		$(let $a0v1 = U::split($a0v1);)*
                let r1 = [a0v0.0$(, $a0v1.0)*].into();
                let r2 = [a0v0.1$(, $a0v1.1)*].into();
                let r2 = <U::Smaller as Array<S>>::transpose_helper::<Self>(r2);
                <U as Dim<<Self as Array<S>>::Type>>::chain(r1, r2)
            },
            transpose_helper(a0) => U::transpose::<Self, T>(a0),
            array_ref(a0) => {
                let &[ref a0v0$(, ref $a0v1)*] = &a0.0;
                [a0v0$(, $a0v1)*].into()
            },
            array_mut(a0) => {
                let &mut[ref mut a0v0$(, ref mut $a0v1)*] = &mut a0.0;
                [a0v0$(, $a0v1)*].into()
            },
            $($index),*
        }

        impl<T> Dim<T> for $id {
            #[inline(always)]
            fn split(a0: Self::Type) -> (T, <Self::Smaller as Array<T>>::Type) {
                let mut a0 = a0;
                // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
                let result = unsafe {(
                    replace(&mut a0[0], uninitialized()),
                    [$(replace(&mut a0[$index], uninitialized())),*].into()
                )};
                forget(a0);
                result
            }

            #[inline(always)]
            fn split_ref(a0: &Self::Type) -> (&T, &<Self::Smaller as Array<T>>::Type) {
                let &[ref a0v0, ref a0..] = &a0.0;
                (a0v0, RefCast::from_ref(a0))
            }

            #[inline(always)]
            fn split_ref_end(a0: &Self::Type) -> (&<Self::Smaller as Array<T>>::Type, &T) {
                let &[ref a0.., ref a0v0] = &a0.0;
                (RefCast::from_ref(a0), a0v0)
            }

            #[inline(always)]
            fn split_mut(a0: &mut Self::Type) -> (&mut T, &mut <Self::Smaller as Array<T>>::Type) {
                let &mut[ref mut a0v0, ref mut a0..] = &mut a0.0;
                (a0v0, RefCast::from_mut(a0))
            }

            #[inline(always)]
            fn split_mut_end(a0: &mut Self::Type) -> (&mut <Self::Smaller as Array<T>>::Type, &mut T) {
                let &mut[ref mut a0.., ref mut a0v0] = &mut a0.0;
                (RefCast::from_mut(a0), a0v0)
            }

            #[inline(always)]
            fn chain(a0v0: T, a0: <Self::Smaller as Array<T>>::Type) -> Self::Type {
                // TODO: remove unsafe. Blocked on rust-lang/rust#37302
                let mut a0 = a0; { let _unused = &mut a0; }
                #[allow(eq_op)]
                $(let $a0v1 = replace(&mut a0[$index-1], unsafe { uninitialized() });)*
                forget(a0);
                [a0v0$(, $a0v1)*].into()
            }
        }
        impl ExtractItem<Zero> for $id {
            #[inline(always)]
            fn extract<T>(a0: <Self as Array<T>>::Type) -> (T, <<$id as HasSmaller>::Smaller as Array<T>>::Type) {
                Self::split(a0)
            }
        }
        impl<T> DimRef<T> for $id {}
        impl<T> DimMut<T> for $id {}
        impl<T, D: Dim<T>> TwoDim<T, D> for $id
        // TODO: remove elaborted bounds. Blocked on rust/issues#20671
        where D::Smaller: Array<T>,
        {}
        impl<T, D: DimRef<T>> TwoDimRef<T, D> for $id
        // TODO: remove elaborted bounds. Blocked on rust/issues#20671
        where for<'a> D::Smaller: Array<T>+Array<&'a T>,
        {}
        impl<T, D: DimMut<T>> TwoDimMut<T, D> for $id
        // TODO: remove elaborted bounds. Blocked on rust/issues#20671
        where for<'a> D::Smaller: Array<T>+Array<&'a T>+Array<&'a mut T>,
        {}
    );
}

macro_rules! impl_has_smaller_larger {
    ($smaller:ident, $larger:ident$(, $other:ident)*) => {
        impl HasLarger for $smaller { type Larger = $larger; }
        impl HasSmaller for $larger { type Smaller = $smaller; }
    };
}

macro_rules! impl_is_smaller_larger {
    ($smaller:ident) => {};
    ($smaller:ident, $($larger:ident),+) => {
        $(
impl IsSmallerThan<$larger> for $smaller {
    /// Index an array of size `C` safely.
    #[inline(always)]
    fn index_array<T>(a0: &<$larger as Array<T>>::Type) -> &T {
        fn _dummy(v: <$larger as Array<()>>::Type) { v[$smaller::VALUE] }
        &a0.0[$smaller::VALUE]
    }
}
impl IsLargerThan<$smaller> for $larger {}
impl NotSame<$larger> for $smaller {}
impl NotSame<$smaller> for $larger {}
impl DecrementIfLargerThan<$larger> for $smaller {
    type Result = $smaller;
}
impl DecrementIfLargerThan<$smaller> for $larger {
    type Result = <$larger as HasSmaller>::Smaller;
}
        )+
        impl_has_smaller_larger!{$smaller, $($larger),*}
        impl_is_smaller_larger!{$($larger),*}
    };
}

impl_is_smaller_larger!{Zero, One, Two, Three, Four}

impl_array!{Zero,  CustomArrayZero}
impl_array!{One,   CustomArrayOne,   1}
impl_array!{Two,   CustomArrayTwo,   2, 1;a0v1;a1v1;a2v1}
impl_array!{Three, CustomArrayThree, 3, 1;a0v1;a1v1;a2v1, 2;a0v2;a1v2;a2v2}
impl_array!{Four,  CustomArrayFour,  4, 1;a0v1;a1v1;a2v1, 2;a0v2;a1v2;a2v2, 3;a0v3;a1v3;a2v3}


impl ExtractItem<One> for Two {
    #[inline(always)]
    fn extract<T>(a0: <Self as Array<T>>::Type) -> (T, <One as Array<T>>::Type) {
        let mut a0 = a0;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut a0[1], uninitialized()),
            [
                replace(&mut a0[0], uninitialized()),
            ].into()
        )};
        forget(a0);
        result
    }
}

impl ExtractItem<One> for Three {
    #[inline(always)]
    fn extract<T>(a0: <Self as Array<T>>::Type) -> (T, <Two as Array<T>>::Type) {
        let mut a0 = a0;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut a0[1], uninitialized()),
            [
                replace(&mut a0[0], uninitialized()),
                replace(&mut a0[2], uninitialized()),
            ].into()
        )};
        forget(a0);
        result
    }
}

impl ExtractItem<Two> for Three {
    #[inline(always)]
    fn extract<T>(a0: <Self as Array<T>>::Type) -> (T, <Two as Array<T>>::Type) {
        let mut a0 = a0;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut a0[2], uninitialized()),
            [
                replace(&mut a0[0], uninitialized()),
                replace(&mut a0[1], uninitialized()),
            ].into()
        )};
        forget(a0);
        result
    }
}

impl ExtractItem<One> for Four {
    #[inline(always)]
    fn extract<T>(a0: <Self as Array<T>>::Type) -> (T, <Three as Array<T>>::Type) {
        let mut a0 = a0;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut a0[1], uninitialized()),
            [
                replace(&mut a0[0], uninitialized()),
                replace(&mut a0[2], uninitialized()),
                replace(&mut a0[3], uninitialized()),
            ].into()
        )};
        forget(a0);
        result
    }
}

impl ExtractItem<Two> for Four {
    #[inline(always)]
    fn extract<T>(a0: <Self as Array<T>>::Type) -> (T, <Three as Array<T>>::Type) {
        let mut a0 = a0;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut a0[2], uninitialized()),
            [
                replace(&mut a0[0], uninitialized()),
                replace(&mut a0[1], uninitialized()),
                replace(&mut a0[3], uninitialized()),
            ].into()
        )};
        forget(a0);
        result
    }
}

impl ExtractItem<Three> for Four {
    #[inline(always)]
    fn extract<T>(a0: <Self as Array<T>>::Type) -> (T, <Three as Array<T>>::Type) {
        let mut a0 = a0;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut a0[3], uninitialized()),
            [
                replace(&mut a0[0], uninitialized()),
                replace(&mut a0[1], uninitialized()),
                replace(&mut a0[2], uninitialized()),
            ].into()
        )};
        forget(a0);
        result
    }
}


impl ExtractArray<Zero, One> for Two {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        a0
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        a0
    }
}


impl ExtractArray<Zero, One> for Three {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[ref a0.., _] = &a0.0;
        (RefCast::from_ref(a0))
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [ref mut a0.., _] = &mut a0.0;
        (RefCast::from_mut(a0))
    }
}

impl ExtractArray<Zero, Two> for Three {
    type Extracted = Three;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        a0
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        a0
    }
}

impl ExtractArray<One, Two> for Three {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[_, ref a0..] = &a0.0;
        (RefCast::from_ref(a0))
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [_, ref mut a0..] = &mut a0.0;
        (RefCast::from_mut(a0))
    }
}


impl ExtractArray<Zero, One> for Four {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[ref a0.., _, _] = &a0.0;
        (RefCast::from_ref(a0))
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [ref mut a0.., _, _] = &mut a0.0;
        (RefCast::from_mut(a0))
    }
}

impl ExtractArray<Zero, Two> for Four {
    type Extracted = Three;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[ref a0.., _] = &a0.0;
        (RefCast::from_ref(a0))
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [ref mut a0.., _] = &mut a0.0;
        (RefCast::from_mut(a0))
    }
}

impl ExtractArray<Zero, Three> for Four {
    type Extracted = Four;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        a0
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        a0
    }
}

impl ExtractArray<One, Two> for Four {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[_, ref a0.., _] = &a0.0;
        (RefCast::from_ref(a0))
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [_, ref mut a0.., _] = &mut a0.0;
        (RefCast::from_mut(a0))
    }
}

impl ExtractArray<One, Three> for Four {
    type Extracted = Three;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[_, ref a0..] = &a0.0;
        (RefCast::from_ref(a0))
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [_, ref mut a0..] = &mut a0.0;
        (RefCast::from_mut(a0))
    }
}

impl ExtractArray<Two, Three> for Four {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(a0: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[_, _, ref a0..] = &a0.0;
        (RefCast::from_ref(a0))
    }

    #[inline(always)]
    fn extract_array_mut<T>(a0: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [_, _, ref mut a0..] = &mut a0.0;
        (RefCast::from_mut(a0))
    }
}


impl<I0: Constant, I1: Constant> SelectTwo<I0, I1> for Zero {
    type Selected = I0;
}
impl<I0: Constant, I1: Constant, I2: Constant> SelectThree<I0, I1, I2> for Zero {
    type Selected = I0;
}
impl<I0: Constant, I1: Constant, I2: Constant, I3: Constant> SelectFour<I0, I1, I2, I3> for Zero {
    type Selected = I0;
}

impl<I0: Constant, I1: Constant> SelectTwo<I0, I1> for One {
    type Selected = I1;
}
impl<I0: Constant, I1: Constant, I2: Constant> SelectThree<I0, I1, I2> for One {
    type Selected = I1;
}
impl<I0: Constant, I1: Constant, I2: Constant, I3: Constant> SelectFour<I0, I1, I2, I3> for One {
    type Selected = I1;
}

impl<I0: Constant, I1: Constant, I2: Constant> SelectThree<I0, I1, I2> for Two {
    type Selected = I2;
}
impl<I0: Constant, I1: Constant, I2: Constant, I3: Constant> SelectFour<I0, I1, I2, I3> for Two {
    type Selected = I2;
}

impl<I0: Constant, I1: Constant, I2: Constant, I3: Constant> SelectFour<I0, I1, I2, I3> for Three {
    type Selected = I3;
}
