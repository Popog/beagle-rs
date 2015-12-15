use std::fs::File;
use std::io::{Write,Result};
use std::path::Path;

use build_traits::{types, angle_types};

fn declare_vec(f: &mut File) -> Result<()> {
    let un_ops = ["Neg", "Not"];
    let bin_ops = ["BitAnd", "BitOr", "BitXor", "Add", "Div", "Mul", "Rem", "Sub", "Shl", "Shr"];

    try!(write!(f,"use std::ops::{{"));
    for trait_name in un_ops.iter().chain(bin_ops.iter()) {
        try!(write!(f,"{},", trait_name));
    }
    try!(write!(f,"Deref,DerefMut}};\n"));

    try!(write!(f,"use angle::{{"));
    for t in angle_types().iter() {
        try!(write!(f,"{},",t));
    }
    try!(write!(f,"}};\n\n"));


    for trait_name in un_ops.iter() {
        try!(impl_unop(f, trait_name));
    }

    for trait_name in bin_ops.iter() {
        try!(impl_binop(f, trait_name));
    }
    try!(impl_from(f));

    for (i, d) in ["One","Two","Three","Four"].iter().enumerate() {
        let i = i+1;
        try!(write!(f,"
use traits::{d};
pub type Vec{i}<T> = Vec<{d}, T>;

impl <T: Scalar> From<[T; {i}]> for Vec{i}<T> {{  fn from(v: [T; {i}]) -> Self {{ Vec(v) }}  }}
impl <T: Scalar> Into<[T; {i}]> for Vec{i}<T> {{  fn into(self) -> [T; {i}] {{ self.0 }}  }}

impl <T: Scalar> Borrow   <[T; {i}]> for Vec{i}<T> {{  #[inline] fn borrow    (&    self) -> &    [T; {i}] {{ &    self.0 }}  }}
impl <T: Scalar> BorrowMut<[T; {i}]> for Vec{i}<T> {{  #[inline] fn borrow_mut(&mut self) -> &mut [T; {i}] {{ &mut self.0 }}  }}

impl <T: Scalar> AsRef<[T; {i}]> for Vec{i}<T> {{  #[inline] fn as_ref(&    self) -> &    [T; {i}] {{ &    self.0 }}  }}
impl <T: Scalar> AsMut<[T; {i}]> for Vec{i}<T> {{  #[inline] fn as_mut(&mut self) -> &mut [T; {i}] {{ &mut self.0 }}  }}\n", i=i, d=d));
    }

    // TODO: re-enable when it stop breaking rust
    /*for scalar_type in types().iter() {
        for trait_name in bin_ops.iter() {
            try!(write!(f, "
impl <T: Scalar, D: Dim<T>> {trait_name}<Vec<D, T>> for {scalar_type}
where {scalar_type}: {trait_name}<T>,
<{scalar_type} as {trait_name}<T>>::Output: Scalar,
D: Dim<<{scalar_type} as {trait_name}<T>>::Output> {{
    type Output = Vec<D, <{scalar_type} as {trait_name}<T>>::Output>;
    fn {method_name}(self, rhs: Vec<D, T>) -> Self::Output {{
        Vec(<D as Dim<<{scalar_type} as {trait_name}<T>>::Output>>::from_iter(rhs.iter().map(|r| {trait_name}::<T>::{method_name}(self, *r))))
    }}
}}
impl <'t, T: Scalar, D: Dim<T>> {trait_name}<Vec<D, T>> for &'t {scalar_type}
where {scalar_type}: {trait_name}<T>,
<{scalar_type} as {trait_name}<T>>::Output: Scalar,
D: Dim<<{scalar_type} as {trait_name}<T>>::Output> {{
    type Output = Vec<D, <{scalar_type} as {trait_name}<T>>::Output>;
    fn {method_name}(self, rhs: Vec<D, T>) -> Self::Output {{
        Vec(<D as Dim<<{scalar_type} as {trait_name}<T>>::Output>>::from_iter(rhs.iter().map(|r| {trait_name}::<T>::{method_name}(*self, *r))))
    }}
}}
impl <'r, T: Scalar, D: Dim<T>> {trait_name}<&'r Vec<D, T>> for {scalar_type}
where {scalar_type}: {trait_name}<T>,
<{scalar_type} as {trait_name}<T>>::Output: Scalar,
D: Dim<<{scalar_type} as {trait_name}<T>>::Output> {{
    type Output = Vec<D, <{scalar_type} as {trait_name}<T>>::Output>;
    fn {method_name}(self, rhs: &'r Vec<D, T>) -> Self::Output {{
        Vec(<D as Dim<<{scalar_type} as {trait_name}<T>>::Output>>::from_iter(rhs.iter().map(|r| {trait_name}::<T>::{method_name}(self, *r))))
    }}
}}
impl <'t, 'r, T: Scalar, D: Dim<T>> {trait_name}<&'r Vec<D, T>> for &'t {scalar_type}
where {scalar_type}: {trait_name}<T>,
<{scalar_type} as {trait_name}<T>>::Output: Scalar,
D: Dim<<{scalar_type} as {trait_name}<T>>::Output> {{
    type Output = Vec<D, <{scalar_type} as {trait_name}<T>>::Output>;
    fn {method_name}(self, rhs: &'r Vec<D, T>) -> Self::Output {{
        Vec(<D as Dim<<{scalar_type} as {trait_name}<T>>::Output>>::from_iter(rhs.iter().map(|r| {trait_name}::<T>::{method_name}(*self, *r))))
    }}
}}", trait_name=trait_name, method_name=trait_name.to_lowercase(), scalar_type=scalar_type));
        }
    }*/

    write!(f, "\n")
}

fn impl_unop(f: &mut File, trait_name: &str) -> Result<()> {
    write!(f, "
impl <T: Scalar, D: Dim<T>> {trait_name} for Vec<D, T>
where T: {trait_name}<Output=T> {{
    type Output = Vec<D, T>;
    fn {method_name}(mut self) -> Self::Output {{
        for a in self.iter_mut() {{ *a = {trait_name}::{method_name}(*a) }}
        self
    }}
}}
impl <'a, T: Scalar, D: Dim<T>> {trait_name} for &'a Vec<D, T>
where &'a T: {trait_name}<Output=T> {{
    type Output = Vec<D, T>;
    fn {method_name}(self) -> Self::Output {{
        Vec(<D as Dim<T>>::from_iter(self.iter().map({trait_name}::{method_name})))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}

fn impl_binop(f: &mut File, trait_name: &str) -> Result<()> {
    write!(f, "
impl <T: Scalar, U: Scalar, D: Dim<T> + Dim<U>> {trait_name}<Vec<D, U>> for Vec<D, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Vec<D, T>;
    fn {method_name}(mut self, rhs: Vec<D, U>) -> Self::Output {{
        for (l, r) in self.iter_mut().zip(rhs.iter()) {{ *l = {trait_name}::{method_name}(*l, *r); }}
        self
    }}
}}
impl <'t, T: Scalar, U: Scalar, D: Dim<T> + Dim<U>> {trait_name}<Vec<D, U>> for &'t Vec<D, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Vec<D, T>;
    fn {method_name}(self, rhs: Vec<D, U>) -> Self::Output {{
        Vec(<D as Dim<T>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| {trait_name}::{method_name}(*l, *r))))
    }}
}}
impl <'r, T: Scalar, U: Scalar, D: Dim<T> + Dim<U>> {trait_name}<&'r Vec<D, U>> for Vec<D, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Vec<D, T>;
    fn {method_name}(mut self, rhs: &'r Vec<D, U>) -> Self::Output {{
        for (l, r) in self.iter_mut().zip(rhs.iter()) {{ *l = {trait_name}::{method_name}(*l, *r); }}
        self
    }}
}}
impl <'t, 'r, T: Scalar, U: Scalar, D: Dim<T> + Dim<U>> {trait_name}<&'r Vec<D, U>> for &'t Vec<D, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Vec<D, T>;
    fn {method_name}(self, rhs: &'r Vec<D, U>) -> Self::Output {{
        Vec(<D as Dim<T>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| {trait_name}::{method_name}(*l, *r))))
    }}
}}

impl <T: Scalar, U: Scalar, D: Dim<T>> {trait_name}<U> for Vec<D, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Vec<D, T>;
    fn {method_name}(mut self, rhs: U) -> Self::Output {{
        for l in self.iter_mut() {{ *l = {trait_name}::{method_name}(*l, rhs); }}
        self
    }}
}}
impl <'t, T: Scalar, U: Scalar, D: Dim<T>> {trait_name}<U> for &'t Vec<D, T>
where T: {trait_name}<U, Output=T> {{
    type Output = Vec<D, T>;
    fn {method_name}(self, rhs: U) -> Self::Output {{
        Vec(<D as Dim<T>>::from_iter(self.iter().map(|l| {trait_name}::{method_name}(*l, rhs))))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}

fn impl_from(f: &mut File) -> Result<()> {
    for t in types().iter().filter(|&t| *t != "bool") {
        for u in types().iter().filter(|&t| *t != "bool") {
            if t == u { continue; }
            try!(write!(f,"impl <D: Dim<{t}>+Dim<{u}>> From<Vec<D, {u}>> for Vec<D, {t}> {{  fn from(v: Vec<D, {u}>) -> Self {{ Vec(<D as Dim<{t}>>::from_iter(v.iter().map(|&v| v as {t}))) }}  }}\n", t=t, u=u));
        }
        for u in angle_types().iter() {
            if t == u { continue; }
            try!(write!(f,"impl <D: Dim<{t}>+Dim<{u}>> From<Vec<D, {u}>> for Vec<D, {t}> {{  fn from(v: Vec<D, {u}>) -> Self {{ Vec(<D as Dim<{t}>>::from_iter(v.iter().map(|&v| v.into()))) }}  }}\n", t=t, u=u));
        }
    }
    for t in angle_types().iter() {
        for u in types().iter().chain(angle_types().iter()).filter(|t| **t != "bool") {
            if t == u { continue; }
            try!(write!(f,"impl <D: Dim<{t}>+Dim<{u}>> From<Vec<D, {u}>> for Vec<D, {t}> {{  fn from(v: Vec<D, {u}>) -> Self {{ Vec(<D as Dim<{t}>>::from_iter(v.iter().map(|&v| v.into()))) }}  }}\n", t=t, u=u));
        }
    }
    write!(f, "\n")
}


pub fn main(out_dir: &String) {
    let dest_path = Path::new(out_dir).join("vec.rs");
    let mut f = File::create(&dest_path).unwrap();
    declare_vec(&mut f).unwrap();
}