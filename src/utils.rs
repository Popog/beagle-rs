
pub trait RefCast<T> {
    fn from_ref(v: &T) -> &Self;
    fn from_mut(v: &mut T) -> &mut Self;
    fn into_ref(&self) -> &T;
    fn into_mut(&mut self) -> &T;
}
