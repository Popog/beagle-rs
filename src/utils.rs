use consts::Array;

/// A trait to cast references back and forth between types
pub trait RefCast<T> {
    #[inline(always)]
    fn from_ref(v: &T) -> &Self;

    #[inline(always)]
    fn from_mut(v: &mut T) -> &mut Self;

    #[inline(always)]
    fn into_ref(&self) -> &T;

    #[inline(always)]
    fn into_mut(&mut self) -> &mut T;
}

/// A trait to cast array references back and forth between types
pub trait ArrayRefCast<T>: Sized {
    #[inline(always)]
    fn from_array_ref<A>(v: &<A as Array<T>>::Type) -> &<A as Array<Self>>::Type
    where A: Array<T> + Array<Self>;

    #[inline(always)]
    fn from_array_mut<A>(v: &mut <A as Array<T>>::Type) -> &mut <A as Array<Self>>::Type
    where A: Array<T> + Array<Self>;

    #[inline(always)]
    fn into_array_ref<A>(v: &<A as Array<Self>>::Type) -> &<A as Array<T>>::Type
    where A: Array<T> + Array<Self>;

    #[inline(always)]
    fn into_array_mut<A>(v: &mut <A as Array<Self>>::Type) -> &mut <A as Array<T>>::Type
    where A: Array<T> + Array<Self>;
}
