use std::fs::File;
use std::io::{Write,Result};
use std::path::Path;


pub fn types() -> &'static [&'static str] {
    static TYPES: [&'static str; 11] = ["i64","i32","i16","i8","u64","u32","u16","u8","f64","f32","bool"];
    return &TYPES;
}
pub fn angle_types()  -> &'static [&'static str] {
    static TYPES: [&'static str; 4] = ["Rad32","Rad64", "Deg32","Deg64"];
    return &TYPES;
}

fn impl_dim(f: &mut File) -> Result<()> {
    for (i,d) in ["One","Two","Three","Four"].iter().enumerate() {
        let i = i+1;
        try!(write!(f,"#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct {d};
impl <T: Copy> Dim<T> for {d} {{
    type Output = [T;{i}];
    #[inline(always)]
    fn from_value(v: T) -> Self::Output {{ [v; {i}] }}

    #[inline(always)]
    fn from_iter<U>(iterator: U) -> Self::Output
    where U: IntoIterator<Item=T>,
    <U as IntoIterator>::IntoIter: ExactSizeIterator {{
        let mut iterator = iterator.into_iter();
        assert!(iterator.len() == {i});
        [", d=d, i=i));
        for _ in 0..i {
            try!(write!(f,"iterator.next().unwrap(), "));
        }
        try!(write!(f,"]
    }}
}}\n"));
    }
    write!(f, "\n")
}

fn impl_scalar_type(f: &mut File) -> Result<()> {
    for t in types().iter() {
        try!(write!(f,"impl Scalar for {t} {{}}\n", t=t));
    }
    for t in angle_types().iter() {
        try!(write!(f,"impl Scalar for angle::{t} {{}}\n", t=t));
    }
    write!(f, "\n")
}

pub fn main(out_dir: &String) {
    let dest_path = Path::new(&out_dir).join("scalar_array.rs");
    let mut f = File::create(&dest_path).unwrap();

    impl_dim(&mut f).unwrap();
    impl_scalar_type(&mut f).unwrap();
}
