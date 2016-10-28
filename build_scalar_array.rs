use std::fs::File;
use std::cmp::max;
use std::io::{Write,Result};
use std::ops::Range;
use std::path::Path;


pub fn types() -> &'static [&'static str] {
    static TYPES: [&'static str; 11] = ["i64","i32","i16","i8","u64","u32","u16","u8","f64","f32","bool"];
    &TYPES
}
pub fn angle_types() -> &'static [&'static str] {
    static TYPES: [&'static str; 4] = ["Rad32","Rad64", "Deg32","Deg64"];
    &TYPES
}

pub fn dims() -> &'static [&'static str] {
    static DIMS: [&'static str; 4] = ["One","Two","Three","Four"];
    &DIMS
}

pub enum IndicesClass {
    InOrder,
    Unique,
    Duplicates,
}

pub trait Indices {
    fn classify(self) -> IndicesClass;
    fn max(self) -> usize;
}

impl Indices for (usize,) {
    fn classify(self) -> IndicesClass { IndicesClass::InOrder }

    fn max(self) -> usize {
        self.0
    }
}
impl Indices for (usize,usize) {
    fn classify(self) -> IndicesClass {
        let (a, b) = self;
        if a+1 == b { IndicesClass::InOrder }
        else if a != b { IndicesClass::Unique }
        else { IndicesClass::Duplicates }
    }
    fn max(self) -> usize {
        let (a, b) = self;
        max(a, b)
    }
}
impl Indices for (usize,usize,usize) {
    fn classify(self) -> IndicesClass {
        let (a, b, c) = self;
        if a+1 == b && b+1 == c { IndicesClass::InOrder }
        else if a != b && a != c && b != c { IndicesClass::Unique }
        else { IndicesClass::Duplicates }
    }
    fn max(self) -> usize {
        let (a, b, c) = self;
        max(max(a, b), c)
    }
}
impl Indices for (usize,usize,usize,usize) {
    fn classify(self) -> IndicesClass {
        let (a, b, c, d) = self;
        if a+1 == b && b+1 == c && c+1 == d { IndicesClass::InOrder }
        else if a != b && a != c && a != d && b != c && b != d && c != d { IndicesClass::Unique }
        else { IndicesClass::Duplicates }
    }
    fn max(self) -> usize {
        let (a, b, c, d) = self;
        max(max(max(a, b), c), d)
    }
}

pub struct TupleRange<T> {
    msb: Range<usize>,
    start: T,
    range: Range<T>,
}

impl From<Range<(usize, usize)>> for TupleRange<usize> {
    fn from(val: Range<(usize, usize)>) -> Self {
        TupleRange{
            msb: val.start.0..val.end.0,
            start: val.start.1,
            range: val.start.1..val.end.1,
        }
    }
}

impl From<Range<(usize, usize, usize)>> for TupleRange<(usize, usize)> {
    fn from(val: Range<(usize, usize, usize)>) -> Self {
        TupleRange{
            msb: val.start.0..val.end.0,
            start: (val.start.1, val.start.2),
            range: (val.start.1, val.start.2)..(val.end.1, val.end.2),
        }
    }
}

impl From<Range<(usize, usize, usize, usize)>> for TupleRange<(usize, usize, usize)> {
    fn from(val: Range<(usize, usize, usize, usize)>) -> Self {
        TupleRange{
            msb: val.start.0..val.end.0,
            start: (val.start.1, val.start.2, val.start.3),
            range: (val.start.1, val.start.2, val.start.3)..(val.end.1, val.end.2, val.end.3),
        }
    }
}

impl Iterator for TupleRange<usize> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.msb.start >= self.msb.end {
            return None
        } else if self.range.start >= self.range.end {
            self.msb.start += 1;
            self.range.start = self.start;
            self.next()
        } else {
            let a = self.msb.start;
            let b = self.range.start;
            self.range.start += 1;
            Some((a, b))
        }
    }
}

impl Iterator for TupleRange<(usize, usize)> {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.msb.start >= self.msb.end {
            return None
        } else if self.range.start.0 >= self.range.end.0 {
            self.msb.start += 1;
            self.range.start.0 = self.start.0;
            self.range.start.1 = self.start.1;
            self.next()
        } else if self.range.start.1 >= self.range.end.1 {
            self.range.start.0 += 1;
            self.range.start.1 = self.start.1;
            self.next()
        } else {
            let a = self.msb.start;
            let (b, c) = self.range.start;
            self.range.start.1 += 1;
            Some((a, b, c))
        }
    }
}

impl Iterator for TupleRange<(usize, usize, usize)> {
    type Item = (usize, usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.msb.start >= self.msb.end {
            return None
        } else if self.range.start.0 >= self.range.end.0 {
            self.msb.start += 1;
            self.range.start.0 = self.start.0;
            self.range.start.1 = self.start.1;
            self.range.start.2 = self.start.2;
            self.next()
        } else if self.range.start.1 >= self.range.end.1 {
            self.range.start.0 += 1;
            self.range.start.1 = self.start.1;
            self.range.start.2 = self.start.2;
            self.next()
        } else if self.range.start.2 >= self.range.end.2 {
            self.range.start.1 += 1;
            self.range.start.2 = self.start.2;
            self.next()
        } else {
            let a = self.msb.start;
            let (b, c, d) = self.range.start;
            self.range.start.2 += 1;
            Some((a, b, c, d))
        }
    }
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


    //impl_dim(&mut f).unwrap();
    impl_scalar_type(&mut f).unwrap();
}
