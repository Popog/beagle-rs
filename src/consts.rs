//! Traits and functions that emulate integer generics.

// TODO: more commments

use std::borrow::{Borrow,BorrowMut};
use std::ops::{Deref,DerefMut};
use std::mem::{forget,replace,uninitialized};

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
    fn index_array<T>(lhs: &C::Type) -> &T
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
    fn extract<T>(lhs: Self::Type) -> (T, <Self::Smaller as Array<T>>::Type)
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
    fn extract_array_ref<T>(lhs: &Self::Type) -> &<Self::Extracted as Array<T>>::Type
    where Self: Dim<T>,
    Self::Extracted: Dim<T>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    <Self::Extracted as HasSmaller>::Smaller: Array<T>,
    Self::Smaller: Array<T>;

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut Self::Type) -> &mut <Self::Extracted as Array<T>>::Type
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
    fn apply<F: FnMut(T)>(lhs: Self::Type, f: F);

    /// Apply `f` to all elements of two arrays.
    #[inline(always)]
    fn apply_zip<U, F>(lhs: <Self as Array<T>>::Type, rhs: <Self as Array<U>>::Type, f: F)
    where Self: Array<U>, F: FnMut(T, U);

    /// Fold all the elements of the array with function `f`
    #[inline(always)]
    fn fold<O, F>(lhs: Self::Type, init: O, f: F) -> O
    where F: FnMut(O, T)-> O;

    /// Fold all the elements of two arrays with function `f`
    #[inline(always)]
    fn fold_zip<U, O, F>(lhs: <Self as Array<T>>::Type, rhs: <Self as Array<U>>::Type, init: O, f: F) -> O
    where Self: Array<U>, F: FnMut(O, T, U)-> O;

    /// Map all the elements of the array with function `f`
    #[inline(always)]
    fn map<O, F>(lhs: <Self as Array<T>>::Type, f: F) -> <Self as Array<O>>::Type
    where Self: Array<O>, F: FnMut(T)-> O;

    /// Map all the elements of two arrays with function `f`
    #[inline(always)]
    fn map_zip<U, O, F>(lhs: <Self as Array<T>>::Type, rhs: <Self as Array<U>>::Type, f: F) -> <Self as Array<O>>::Type
    where Self: Array<U>+Array<O>, F: FnMut(T, U)-> O;

    /// Transpose the elements of a 2d array
    #[inline(always)]
    fn transpose<U,S>(lhs: <Self as Array<T>>::Type) -> <U as Array<<Self as Array<S>>::Type>>::Type
    where Self: Array<S>,
    U: Dim<S, Type=T> + Dim<<Self as Array<S>>::Type>,
    // TODO: remove elaborted bounds. Blocked on rust/issues#20671
    U::Smaller: Array<S>+Array<<Self as Array<S>>::Type>;

    /// A helper to transpose the elements of a 2d array (recursion)
    #[inline(always)]
    fn transpose_helper<U>(lhs: <U as Array<<Self as Array<T>>::Type>>::Type) -> <Self as Array<<U as Array<T>>::Type>>::Type
    where U: Array<<Self as Array<T>>::Type> + Array<T>,
    Self: Array<<U as Array<T>>::Type>;
}

pub trait ArrayRef<T>: Array<T>
where for<'a> Self: Array<&'a T> {
    #[inline(always)]
    fn get_ref(lhs: &<Self as Array<T>>::Type) -> <Self as Array<&T>>::Type;
}

pub trait ArrayMut<T>: ArrayRef<T>
where for<'a> Self: Array<&'a mut T> {
    #[inline(always)]
    fn get_mut(lhs: &mut <Self as Array<T>>::Type) -> <Self as Array<&mut T>>::Type;
}

/// Types that represent a dimension.
pub trait Dim<T>: Array<T> + HasSmaller
where Self::Smaller: Array<T> {
    /// Split the array into an element and a smaller array.
    #[inline(always)]
    fn split(lhs: Self::Type) -> (T, <Self::Smaller as Array<T>>::Type);

    /// Split the array into a reference to an element and a reference to a smaller array.
    #[inline(always)]
    fn split_ref(lhs: &Self::Type) -> (&T, &<Self::Smaller as Array<T>>::Type);

    /// Split the array into a reference to a smaller array and a reference to an element.
    #[inline(always)]
    fn split_ref_end(lhs: &Self::Type) -> (&<Self::Smaller as Array<T>>::Type, &T);

    /// Split the array into a mutable reference to an element and a mutable reference to a smaller array.
    #[inline(always)]
    fn split_mut(lhs: &mut Self::Type) -> (&mut T, &mut <Self::Smaller as Array<T>>::Type);

    /// Split the array into a mutable reference to a smaller array and a mutable reference to an element.
    #[inline(always)]
    fn split_mut_end(lhs: &mut Self::Type) -> (&mut <Self::Smaller as Array<T>>::Type, &mut T);

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
#[derive(Copy, Debug, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct $id<T>(pub [T; $size]);

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
        apply($apply_lhs:tt, $apply_f:tt)=>$apply:expr,
        apply_zip($apply_zip_lhs:tt, $apply_zip_rhs:tt, $apply_zip_f:tt)=>$apply_zip:expr,
        fold($fold_lhs:tt, $fold_init:tt, $fold_f:tt)=>$fold:expr,
        fold_zip($fold_zip_lhs:tt, $fold_zip_rhs:tt, $fold_zip_init:tt, $fold_zip_f:tt)=>$fold_zip:expr,
        map($map_lhs:tt, $map_f:tt)=>$map:expr,
        map_zip($map_zip_lhs:tt, $map_zip_rhs:tt, $map_zip_f:tt)=>$map_zip:expr,
        transpose($transpose_lhs:tt)=>$transpose:expr,
        transpose_helper($transpose_helper_lhs:tt)=>$transpose_helper:expr,
        array_ref($array_ref_lhs:tt)=>$array_ref:expr,
        array_mut($array_mut_lhs:tt)=>$array_mut:expr,
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
    fn apply<F>($apply_lhs: Self::Type, $apply_f: F)
    where F: FnMut(T) { $apply }

    #[inline(always)]
    fn apply_zip<U, F>($apply_zip_lhs: <Self as Array<T>>::Type, $apply_zip_rhs: <Self as Array<U>>::Type, $apply_zip_f: F)
    where F: FnMut(T, U) { $apply_zip }

    #[inline(always)]
    fn fold<O, F>($fold_lhs: Self::Type, $fold_init: O, $fold_f: F) -> O
    where F: FnMut(O, T)-> O { $fold }

    #[inline(always)]
    fn fold_zip<U, O, F>($fold_zip_lhs: <Self as Array<T>>::Type, $fold_zip_rhs: <Self as Array<U>>::Type, $fold_zip_init: O, $fold_zip_f: F) -> O
    where F: FnMut(O, T, U)-> O { $fold_zip }

    #[inline(always)]
    fn map<O, F>($map_lhs: Self::Type, $map_f: F) -> <Self as Array<O>>::Type
    where F: FnMut(T)-> O { $map }

    #[inline(always)]
    fn map_zip<U, O, F>($map_zip_lhs: Self::Type, $map_zip_rhs: <Self as Array<U>>::Type, $map_zip_f: F) -> <Self as Array<O>>::Type
    where F: FnMut(T, U)-> O { $map_zip }

    #[inline(always)]
    fn transpose<U,S>($transpose_lhs: <Self as Array<T>>::Type) -> <U as Array<<Self as Array<S>>::Type>>::Type
    where U: Dim<S, Type=T> + Dim<<Self as Array<S>>::Type>,
    U::Smaller: Array<S>+Array<<Self as Array<S>>::Type> { $transpose }

    #[inline(always)]
    fn transpose_helper<U>($transpose_helper_lhs: <U as Array<<Self as Array<T>>::Type>>::Type) -> <Self as Array<<U as Array<T>>::Type>>::Type
    where U: Array<<Self as Array<T>>::Type> + Array<T> { $transpose_helper }
}

impl<T> ArrayRef<T> for $id {
    #[inline(always)]
    fn get_ref($array_ref_lhs: &<Self as Array<T>>::Type) -> <Self as Array<&T>>::Type {
        $array_ref
    }
}

impl<T> ArrayMut<T> for $id {
    #[inline(always)]
    fn get_mut($array_mut_lhs: &mut <Self as Array<T>>::Type) -> <Self as Array<&mut T>>::Type {
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
            apply_zip(_, _, _)=>(),
            fold(_, init, _)=> init,
            fold_zip(_, _, init, _)=> init,
            map(_, _)=>[].into(),
            map_zip(_, _, _)=>[].into(),
            transpose(_) => <U as Array<<Self as Array<S>>::Type>>::from_value([].into()),
            transpose_helper(_) => [].into(),
            array_ref(_) => [].into(),
            array_mut(_) => [].into(),
        }
    );
    ($id:ident, $array:ident, $size:expr$(, $index:expr;$lh:ident;$rh:ident)*) => (
        impl_custom_array!{$array, $size, 0$(, $index)*}
        impl_array_inner!{$id, $array, $size,
            from(v)=>{
                $(let $lh = v.clone();)*
                [v$(, $lh)*].into()
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            apply(lhs, f)=>unsafe {
                let (mut lhs, mut f) = (lhs, f);
                let lh0 = replace(&mut lhs[0], uninitialized());
        		$(let $lh = replace(&mut lhs[$index], uninitialized());)*
        		forget(lhs);
        		f(lh0);
                $(f($lh);)*
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            apply_zip(lhs, rhs, f)=>unsafe {
                let (mut lhs, mut rhs, mut f) = (lhs, rhs, f);
                let lh0 = replace(&mut lhs[0], uninitialized());
                let rh0 = replace(&mut rhs[0], uninitialized());
        		$(let $lh = replace(&mut lhs[$index], uninitialized());)*
        		$(let $rh = replace(&mut rhs[$index], uninitialized());)*
                forget(lhs);
        		forget(rhs);
        		f(lh0, rh0);
                $(f($lh, $rh);)*
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            fold(lhs, init, f)=>unsafe {
                let (mut lhs, mut f) = (lhs, f);
                let lh0 = replace(&mut lhs[0], uninitialized());
        		$(let $lh = replace(&mut lhs[$index], uninitialized());)*
        		forget(lhs);
                let init = f(init, lh0);
                $(let init = f(init, $lh);)*
        		(init)
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            fold_zip(lhs, rhs, init, f)=>unsafe {
                let (mut lhs, mut rhs, mut f) = (lhs, rhs, f);
                let lh0 = replace(&mut lhs[0], uninitialized());
                let rh0 = replace(&mut rhs[0], uninitialized());
        		$(let $lh = replace(&mut lhs[$index], uninitialized());)*
        		$(let $rh = replace(&mut rhs[$index], uninitialized());)*
                forget(lhs);
        		forget(rhs);
                let init = f(init, lh0, rh0);
                $(let init = f(init, $lh, $rh);)*
        		(init)
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            map(lhs, f)=>unsafe {
                let (mut lhs, mut f) = (lhs, f);
                let lh0 = replace(&mut lhs[0], uninitialized());
        		$(let $lh = replace(&mut lhs[$index], uninitialized());)*
        		forget(lhs);
        		[f(lh0)$(, f($lh))*].into()
            },
            // TODO: remove unsafe. Blocked on rust-lang/rust#37302
            map_zip(lhs, rhs, f)=>unsafe {
                let (mut lhs, mut rhs, mut f) = (lhs, rhs, f);
                let lh0 = replace(&mut lhs[0], uninitialized());
                let rh0 = replace(&mut rhs[0], uninitialized());
        		$(let $lh = replace(&mut lhs[$index], uninitialized());)*
        		$(let $rh = replace(&mut rhs[$index], uninitialized());)*
                forget(lhs);
        		forget(rhs);
        		[f(lh0, rh0)$(, f($lh, $rh))*].into()
            },
            transpose(lhs) => unsafe {
                let mut lhs = lhs;
                let lh0 = replace(&mut lhs[0], uninitialized());
        		$(let $lh = replace(&mut lhs[$index], uninitialized());)*
                forget(lhs);

                let lh0 = <U as Dim<S>>::split(lh0);
        		$(let $lh = U::split($lh);)*
                let r1 = [lh0.0$(, $lh.0)*].into();
                let r2 = [lh0.1$(, $lh.1)*].into();
                let r2 = <U::Smaller as Array<S>>::transpose_helper::<Self>(r2);
                <U as Dim<<Self as Array<S>>::Type>>::chain(r1, r2)
            },
            transpose_helper(lhs) => U::transpose::<Self, T>(lhs),
            array_ref(lhs) => {
                let &[ref lh0$(, ref $lh)*] = &lhs.0;
                [lh0$(, $lh)*].into()
            },
            array_mut(lhs) => {
                let &mut[ref mut lh0$(, ref mut $lh)*] = &mut lhs.0;
                [lh0$(, $lh)*].into()
            },
            $($index),*
        }

        impl<T> Dim<T> for $id {
            #[inline(always)]
            fn split(lhs: Self::Type) -> (T, <Self::Smaller as Array<T>>::Type) {
                let mut lhs = lhs;
                // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
                let result = unsafe {(
                    replace(&mut lhs[0], uninitialized()),
                    [$(replace(&mut lhs[$index], uninitialized())),*].into()
                )};
                forget(lhs);
                result
            }

            #[inline(always)]
            fn split_ref(lhs: &Self::Type) -> (&T, &<Self::Smaller as Array<T>>::Type) {
                let &[ref lh0, ref lhs..] = &lhs.0;
                (lh0, RefCast::from_ref(lhs))
            }

            #[inline(always)]
            fn split_ref_end(lhs: &Self::Type) -> (&<Self::Smaller as Array<T>>::Type, &T) {
                let &[ref lhs.., ref lh0] = &lhs.0;
                (RefCast::from_ref(lhs), lh0)
            }

            #[inline(always)]
            fn split_mut(lhs: &mut Self::Type) -> (&mut T, &mut <Self::Smaller as Array<T>>::Type) {
                let &mut[ref mut lh0, ref mut lhs..] = &mut lhs.0;
                (lh0, RefCast::from_mut(lhs))
            }

            #[inline(always)]
            fn split_mut_end(lhs: &mut Self::Type) -> (&mut <Self::Smaller as Array<T>>::Type, &mut T) {
                let &mut[ref mut lhs.., ref mut lh0] = &mut lhs.0;
                (RefCast::from_mut(lhs), lh0)
            }

            #[inline(always)]
            fn chain(lh0: T, lhs: <Self::Smaller as Array<T>>::Type) -> Self::Type {
                // TODO: remove unsafe. Blocked on rust-lang/rust#37302
                let mut lhs = lhs; { let _unused = &mut lhs; }
                #[allow(eq_op)]
                $(let $lh = replace(&mut lhs[$index-1], unsafe { uninitialized() });)*
                forget(lhs);
                [lh0$(, $lh)*].into()
            }
        }
        impl ExtractItem<Zero> for $id {
            #[inline(always)]
            fn extract<T>(lhs: <Self as Array<T>>::Type) -> (T, <<$id as HasSmaller>::Smaller as Array<T>>::Type) {
                Self::split(lhs)
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
    fn index_array<T>(lhs: &<$larger as Array<T>>::Type) -> &T {
        fn _dummy(v: <$larger as Array<()>>::Type) { v[$smaller::VALUE] }
        &lhs.0[$smaller::VALUE]
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
impl_array!{Two,   CustomArrayTwo,   2, 1;lh1;rh1}
impl_array!{Three, CustomArrayThree, 3, 1;lh1;rh1, 2;lh2;rh2}
impl_array!{Four,  CustomArrayFour,  4, 1;lh1;rh1, 2;lh2;rh2, 3;lh3;rh3}


impl ExtractItem<One> for Two {
    #[inline(always)]
    fn extract<T>(lhs: <Self as Array<T>>::Type) -> (T, <One as Array<T>>::Type) {
        let mut lhs = lhs;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut lhs[1], uninitialized()),
            [
                replace(&mut lhs[0], uninitialized()),
            ].into()
        )};
        forget(lhs);
        result
    }
}

impl ExtractItem<One> for Three {
    #[inline(always)]
    fn extract<T>(lhs: <Self as Array<T>>::Type) -> (T, <Two as Array<T>>::Type) {
        let mut lhs = lhs;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut lhs[1], uninitialized()),
            [
                replace(&mut lhs[0], uninitialized()),
                replace(&mut lhs[2], uninitialized()),
            ].into()
        )};
        forget(lhs);
        result
    }
}

impl ExtractItem<Two> for Three {
    #[inline(always)]
    fn extract<T>(lhs: <Self as Array<T>>::Type) -> (T, <Two as Array<T>>::Type) {
        let mut lhs = lhs;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut lhs[2], uninitialized()),
            [
                replace(&mut lhs[0], uninitialized()),
                replace(&mut lhs[1], uninitialized()),
            ].into()
        )};
        forget(lhs);
        result
    }
}

impl ExtractItem<One> for Four {
    #[inline(always)]
    fn extract<T>(lhs: <Self as Array<T>>::Type) -> (T, <Three as Array<T>>::Type) {
        let mut lhs = lhs;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut lhs[1], uninitialized()),
            [
                replace(&mut lhs[0], uninitialized()),
                replace(&mut lhs[2], uninitialized()),
                replace(&mut lhs[3], uninitialized()),
            ].into()
        )};
        forget(lhs);
        result
    }
}

impl ExtractItem<Two> for Four {
    #[inline(always)]
    fn extract<T>(lhs: <Self as Array<T>>::Type) -> (T, <Three as Array<T>>::Type) {
        let mut lhs = lhs;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut lhs[2], uninitialized()),
            [
                replace(&mut lhs[0], uninitialized()),
                replace(&mut lhs[1], uninitialized()),
                replace(&mut lhs[3], uninitialized()),
            ].into()
        )};
        forget(lhs);
        result
    }
}

impl ExtractItem<Three> for Four {
    #[inline(always)]
    fn extract<T>(lhs: <Self as Array<T>>::Type) -> (T, <Three as Array<T>>::Type) {
        let mut lhs = lhs;
        // TODO: fix to remove unsafe. Blocked on rust-lang/rust#37302
        let result = unsafe {(
            replace(&mut lhs[3], uninitialized()),
            [
                replace(&mut lhs[0], uninitialized()),
                replace(&mut lhs[1], uninitialized()),
                replace(&mut lhs[2], uninitialized()),
            ].into()
        )};
        forget(lhs);
        result
    }
}


impl ExtractArray<Zero, One> for Two {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        lhs
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        lhs
    }
}


impl ExtractArray<Zero, One> for Three {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[ref lhs.., _] = &lhs.0;
        (RefCast::from_ref(lhs))
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [ref mut lhs.., _] = &mut lhs.0;
        (RefCast::from_mut(lhs))
    }
}

impl ExtractArray<Zero, Two> for Three {
    type Extracted = Three;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        lhs
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        lhs
    }
}

impl ExtractArray<One, Two> for Three {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[_, ref lhs..] = &lhs.0;
        (RefCast::from_ref(lhs))
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [_, ref mut lhs..] = &mut lhs.0;
        (RefCast::from_mut(lhs))
    }
}


impl ExtractArray<Zero, One> for Four {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[ref lhs.., _, _] = &lhs.0;
        (RefCast::from_ref(lhs))
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [ref mut lhs.., _, _] = &mut lhs.0;
        (RefCast::from_mut(lhs))
    }
}

impl ExtractArray<Zero, Two> for Four {
    type Extracted = Three;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[ref lhs.., _] = &lhs.0;
        (RefCast::from_ref(lhs))
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [ref mut lhs.., _] = &mut lhs.0;
        (RefCast::from_mut(lhs))
    }
}

impl ExtractArray<Zero, Three> for Four {
    type Extracted = Four;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        lhs
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        lhs
    }
}

impl ExtractArray<One, Two> for Four {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[_, ref lhs.., _] = &lhs.0;
        (RefCast::from_ref(lhs))
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [_, ref mut lhs.., _] = &mut lhs.0;
        (RefCast::from_mut(lhs))
    }
}

impl ExtractArray<One, Three> for Four {
    type Extracted = Three;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[_, ref lhs..] = &lhs.0;
        (RefCast::from_ref(lhs))
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [_, ref mut lhs..] = &mut lhs.0;
        (RefCast::from_mut(lhs))
    }
}

impl ExtractArray<Two, Three> for Four {
    type Extracted = Two;

    #[inline(always)]
    fn extract_array_ref<T>(lhs: &<Self as Array<T>>::Type) -> &<Self::Extracted as Array<T>>::Type {
        let &[_, _, ref lhs..] = &lhs.0;
        (RefCast::from_ref(lhs))
    }

    #[inline(always)]
    fn extract_array_mut<T>(lhs: &mut <Self as Array<T>>::Type) -> &mut <Self::Extracted as Array<T>>::Type {
        let &mut [_, _, ref mut lhs..] = &mut lhs.0;
        (RefCast::from_mut(lhs))
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
