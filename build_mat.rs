use std::fs::File;
use std::io::{Write,Result};
use std::path::Path;

use build_traits::{types,angle_types};

fn declare_mat(f: &mut File) -> Result<()> {
    let un_ops = ["Neg", "Not"];
    let bin_ops = ["BitAnd", "BitOr", "BitXor", "Add", "Div", "Rem", "Sub", "Shl", "Shr"];

    try!(write!(f,"use std::ops::{{"));
    for trait_name in un_ops.iter().chain(bin_ops.iter()) {
        try!(write!(f,"{},", trait_name));
    }
    try!(write!(f,"Mul,Deref,DerefMut}};\n\n"));

    try!(write!(f,"use angle::{{"));
    for t in angle_types().iter() {
        try!(write!(f,"{},",t));
    }
    try!(write!(f,"}};\n\n"));

    for t in types().iter().chain(angle_types().iter()).filter(|t| **t != "bool") {
        for u in types().iter().chain(angle_types().iter()).filter(|t| **t != "bool") {
            if t == u { continue; }
            try!(write!(f,"impl <C: Dim<{t}>+Dim<{u}>, R: Dim<Vec<C, {t}>>+Dim<Vec<C, {u}>>> From<Mat<R, C, {u}>> for Mat<R, C, {t}> {{  fn from(v: Mat<R, C, {u}>) -> Self {{ Mat(<R as Dim<Vec<C, {t}>>>::from_iter(v.iter().map(|v| (*v).clone().into()))) }}  }}\n", t=t, u=u));
        }
    }

    for trait_name in un_ops.iter() {
        try!(impl_unop(f, trait_name));
    }

    for trait_name in bin_ops.iter() {
        try!(impl_binop(f, trait_name));
        try!(impl_binop_scalar(f, trait_name));
    }
    try!(impl_binop_scalar(f, "Mul"));

    for (r, dr) in ["One","Two","Three","Four"].iter().enumerate() {
        let r = r+1;
        try!(write!(f,"use traits::{};\n", dr));
        try!(write!(f,"use vec::Vec{};\n", r));
        for (c, dc) in ["One","Two","Three","Four"].iter().enumerate() {
            let c = c+1;
            try!(write!(f,"
pub type Mat{r}x{c}<T> = Mat<{dr}, {dc}, T>;

impl <T: Scalar> From<[Vec{c}<T>; {r}]> for Mat{r}x{c}<T> {{  fn from(v: [Vec{c}<T>; {r}]) -> Self {{ Mat(v) }}  }}
impl <T: Scalar> Into<[Vec{c}<T>; {r}]> for Mat{r}x{c}<T> {{  fn into(self) -> [Vec{c}<T>; {r}] {{ self.0 }}  }}

impl <T: Scalar> From<[[T; {c}]; {r}]> for Mat{r}x{c}<T> {{ fn from(v: [[T; {c}]; {r}]) -> Self {{ Mat::from_2d_array(v.borrow()) }} }}

impl <T: Scalar> Borrow<[Vec{c}<T>; {r}]> for Mat{r}x{c}<T> {{  #[inline] fn borrow(&self) -> &[Vec{c}<T>; {r}] {{ &self.0 }}  }}
impl <T: Scalar> BorrowMut<[Vec{c}<T>; {r}]> for Mat{r}x{c}<T> {{  #[inline] fn borrow_mut(&mut self) -> &mut [Vec{c}<T>; {r}] {{ &mut self.0 }}  }}

impl <T: Scalar> AsRef<[Vec{c}<T>; {r}]> for Mat{r}x{c}<T> {{  #[inline] fn as_ref(&self) -> &[Vec{c}<T>; {r}] {{ &self.0 }}  }}
impl <T: Scalar> AsMut<[Vec{c}<T>; {r}]> for Mat{r}x{c}<T> {{  #[inline] fn as_mut(&mut self) -> &mut [Vec{c}<T>; {r}] {{ &mut self.0 }}  }}\n", r=r, c=c, dr=dr, dc=dc));
        }
    }

    write!(f, "\n")
}

fn impl_unop(f: &mut File, trait_name: &str) -> Result<()> {
    write!(f, "
impl <T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> {trait_name} for Mat<R, C, T>
where T: {trait_name}<Output=T> {{
    type Output = Mat<R, C, T>;
    fn {method_name}(mut self) -> Self::Output {{
        for a in self.iter_mut() {{ *a = {trait_name}::{method_name}(*a) }}
        self
    }}
}}
impl <'a, T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> {trait_name} for &'a Mat<R, C, T>
where &'a T: {trait_name}<Output=T> {{
    type Output = Mat<R, C, T>;
    fn {method_name}(self) -> Self::Output {{
        Mat(<R as Dim<Vec<C, T>>>::from_iter(self.iter().map({trait_name}::{method_name})))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}

fn impl_binop(f: &mut File, trait_name: &str) -> Result<()> {
    write!(f, "
impl <T: Scalar, U: Scalar, C: Dim<T>+Dim<U>, R: Dim<Vec<C, T>>+Dim<Vec<C, U>>> {trait_name}<Mat<R, C, U>> for Mat<R, C, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Mat<R, C, T>;
    fn {method_name}(mut self, rhs: Mat<R, C, U>) -> Self::Output {{
        for (l, r) in self.iter_mut().zip(rhs.iter()) {{ *l = {trait_name}::{method_name}(*l, *r); }}
        self
    }}
}}
impl <'t, T: Scalar, U: Scalar, C: Dim<T>+Dim<U>, R: Dim<Vec<C, T>>+Dim<Vec<C, U>>> {trait_name}<Mat<R, C, U>> for &'t Mat<R, C, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Mat<R, C, T>;
    fn {method_name}(self, rhs: Mat<R, C, U>) -> Self::Output {{
        Mat(<R as Dim<Vec<C, T>>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| {trait_name}::{method_name}(*l, *r))))
    }}
}}
impl <'r, T: Scalar, U: Scalar, C: Dim<T>+Dim<U>, R: Dim<Vec<C, T>>+Dim<Vec<C, U>>> {trait_name}<&'r Mat<R, C, U>> for Mat<R, C, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Mat<R, C, T>;
    fn {method_name}(mut self, rhs: &'r Mat<R, C, U>) -> Self::Output {{
        for (l, r) in self.iter_mut().zip(rhs.iter()) {{ *l = {trait_name}::{method_name}(*l, *r); }}
        self
    }}
}}
impl <'t, 'r, T: Scalar, U: Scalar, C: Dim<T>+Dim<U>, R: Dim<Vec<C, T>>+Dim<Vec<C, U>>> {trait_name}<&'r Mat<R, C, U>> for &'t Mat<R, C, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Mat<R, C, T>;
    fn {method_name}(self, rhs: &'r Mat<R, C, U>) -> Self::Output {{
        Mat(<R as Dim<Vec<C, T>>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| {trait_name}::{method_name}(*l, *r))))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}

fn impl_binop_scalar(f: &mut File, trait_name: &str) -> Result<()> {
    write!(f, "
impl <T: Scalar, U: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> {trait_name}<U> for Mat<R, C, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Mat<R, C, T>;
    fn {method_name}(mut self, rhs: U) -> Self::Output {{
        for l in self.iter_mut() {{ *l = {trait_name}::{method_name}(*l, rhs); }}
        self
    }}
}}
impl <'t, T: Scalar, U: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> {trait_name}<U> for &'t Mat<R, C, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Mat<R, C, T>;
    fn {method_name}(self, rhs: U) -> Self::Output {{
        Mat(<R as Dim<Vec<C, T>>>::from_iter(self.iter().map(|l| {trait_name}::{method_name}(*l, rhs))))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}


pub fn main(out_dir: &String) {
    let dest_path = Path::new(out_dir).join("mat.rs");
    let mut f = File::create(&dest_path).unwrap();
    declare_mat(&mut f).unwrap();
}