use std::fs::File;
use std::io::{Write,Result};
use std::path::Path;

use build_scalar_array::{types,angle_types,dims};

fn declare_mat(f: &mut File) -> Result<()> {
    let un_ops = ["Neg", "Not"];
    let bin_ops = ["BitAnd", "BitOr", "BitXor", "Add", "Div", "Rem", "Sub", "Shl", "Shr"];

    try!(write!(f,"use std::ops::{{"));
    for trait_name in un_ops.iter().chain(bin_ops.iter()) {
        try!(write!(f,"{},", trait_name));
    }
    try!(write!(f,"Mul}};\n\n"));

    try!(write!(f,"use angle::{{"));
    for t in angle_types().iter() {
        try!(write!(f,"{},",t));
    }
    try!(write!(f,"}};\n"));

    try!(write!(f,"use scalar_array::{{"));
    for d in dims().iter() {
        try!(write!(f,"{},", d));
    }
    try!(write!(f,"}};\n"));

    try!(write!(f,"use vec::{{"));
    for (d, _) in dims().iter().enumerate() {
        try!(write!(f,"Vec{},", d+1));
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

    for (r, dr) in dims().iter().enumerate() {
        let r = r+1;
        try!(write!(f,"
/// An alias for Mat&lt;{dr}, {dr}, T&gt;
pub type Mat{r}<T> = Mat<{dr}, {dr}, T>;\n", r=r, dr=dr));
        for (c, dc) in dims().iter().enumerate() {
            let c = c+1;
            try!(write!(f,"
/// An alias for Mat&lt;{dr}, {dc}, T&gt;
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
where T: {trait_name},
<T as {trait_name}>::Output: Scalar,
C: Dim<<T as {trait_name}>::Output>,
R: Dim<Vec<C, <T as {trait_name}>::Output>> {{
    type Output = Mat<R, C, <T as {trait_name}>::Output>;
    fn {method_name}(self) -> Self::Output {{ {trait_name}::{method_name}(&self) }}
}}
impl <'a, T: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> {trait_name} for &'a Mat<R, C, T>
where T: {trait_name},
<T as {trait_name}>::Output: Scalar,
C: Dim<<T as {trait_name}>::Output>,
R: Dim<Vec<C, <T as {trait_name}>::Output>> {{
    type Output = Mat<R, C, <T as {trait_name}>::Output>;
    fn {method_name}(self) -> Self::Output {{
        Mat(<R as Dim<Vec<C, <T as {trait_name}>::Output>>>::from_iter(self.iter().map(|v| {trait_name}::{method_name}(*v))))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}

fn impl_binop(f: &mut File, trait_name: &str) -> Result<()> {
    write!(f, "
impl <T: Scalar, U: Scalar, C: Dim<T>+Dim<U>, R: Dim<Vec<C, T>>+Dim<Vec<C, U>>> {trait_name}<Mat<R, C, U>> for Mat<R, C, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
C: Dim<<T as {trait_name}<U>>::Output>,
R: Dim<Vec<C, <T as {trait_name}<U>>::Output>> {{
    type Output = Mat<R, C, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: Mat<R, C, U>) -> Self::Output {{ {trait_name}::{method_name}(self, &rhs) }}
}}
impl <'t, T: Scalar, U: Scalar, C: Dim<T>+Dim<U>, R: Dim<Vec<C, T>>+Dim<Vec<C, U>>> {trait_name}<Mat<R, C, U>> for &'t Mat<R, C, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
C: Dim<<T as {trait_name}<U>>::Output>,
R: Dim<Vec<C, <T as {trait_name}<U>>::Output>> {{
    type Output = Mat<R, C, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: Mat<R, C, U>) -> Self::Output {{ {trait_name}::{method_name}(self, &rhs) }}
}}
impl <'r, T: Scalar, U: Scalar, C: Dim<T>+Dim<U>, R: Dim<Vec<C, T>>+Dim<Vec<C, U>>> {trait_name}<&'r Mat<R, C, U>> for Mat<R, C, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
C: Dim<<T as {trait_name}<U>>::Output>,
R: Dim<Vec<C, <T as {trait_name}<U>>::Output>> {{
    type Output = Mat<R, C, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: &'r Mat<R, C, U>) -> Self::Output {{ {trait_name}::{method_name}(&self, rhs) }}
}}
impl <'t, 'r, T: Scalar, U: Scalar, C: Dim<T>+Dim<U>, R: Dim<Vec<C, T>>+Dim<Vec<C, U>>> {trait_name}<&'r Mat<R, C, U>> for &'t Mat<R, C, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
C: Dim<<T as {trait_name}<U>>::Output>,
R: Dim<Vec<C, <T as {trait_name}<U>>::Output>> {{
    type Output = Mat<R, C, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: &'r Mat<R, C, U>) -> Self::Output {{
        Mat(<R as Dim<Vec<C, <T as {trait_name}<U>>::Output>>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| {trait_name}::{method_name}(*l, *r))))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}

fn impl_binop_scalar(f: &mut File, trait_name: &str) -> Result<()> {
    write!(f, "
impl <T: Scalar, U: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> {trait_name}<U> for Mat<R, C, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
C: Dim<<T as {trait_name}<U>>::Output>,
R: Dim<Vec<C, <T as {trait_name}<U>>::Output>> {{
    type Output = Mat<R, C, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: U) -> Self::Output {{ {trait_name}::{method_name}(&self, rhs) }}
}}
impl <'t, T: Scalar, U: Scalar, C: Dim<T>, R: Dim<Vec<C, T>>> {trait_name}<U> for &'t Mat<R, C, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
C: Dim<<T as {trait_name}<U>>::Output>,
R: Dim<Vec<C, <T as {trait_name}<U>>::Output>> {{
    type Output = Mat<R, C, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: U) -> Self::Output {{
        Mat(<R as Dim<Vec<C, <T as {trait_name}<U>>::Output>>>::from_iter(self.iter().map(|l| {trait_name}::{method_name}(*l, rhs))))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}


pub fn main(out_dir: &String) {
    let dest_path = Path::new(out_dir).join("mat.rs");
    let mut f = File::create(&dest_path).unwrap();
    declare_mat(&mut f).unwrap();
}
