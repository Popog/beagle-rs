use std::fs::File;
use std::io::{Write,Result};
use std::path::Path;

use build_scalar_array::{types, angle_types, dims};

fn declare_vec(f: &mut File) -> Result<()> {
    let un_ops = ["Neg", "Not"];
    let bin_ops = ["BitAnd", "BitOr", "BitXor", "Add", "Div", "Mul", "Rem", "Sub", "Shl", "Shr"];

    try!(write!(f,"use std::ops::{{"));
    for trait_name in un_ops.iter().chain(bin_ops.iter()) {
        try!(write!(f,"{},", trait_name));
    }
    try!(write!(f,"}};\n\n"));

    try!(write!(f,"use angle::{{"));
    for t in angle_types().iter() {
        try!(write!(f,"{},",t));
    }
    try!(write!(f,"}};\n"));

    try!(write!(f,"use scalar_array::{{"));
    for d in dims().iter() {
        try!(write!(f,"{},", d));
    }
    try!(write!(f,"}};\n\n"));


    for trait_name in un_ops.iter() {
        try!(impl_unop(f, trait_name));
    }

    for trait_name in bin_ops.iter() {
        try!(impl_binop(f, trait_name));
    }
    try!(impl_from(f));

    for (i, d) in dims().iter().enumerate() {
        let i = i+1;
        let tuple = (0..i).fold(String::with_capacity(i*2), |acc, _| acc+"T,");
        let tpat = (0..i).fold(String::with_capacity(i*3), |acc, i| acc+"v"+i.to_string().as_ref()+",");
        let selfpat = (0..i).fold(String::with_capacity(i*8), |acc, i| acc+"self["+i.to_string().as_ref()+"],");
        try!(write!(f,"
/// An alias for Vec&lt;{d}, T&gt;
pub type Vec{i}<T> = Vec<{d}, T>;

impl <T: Scalar> From<[T; {i}]> for Vec{i}<T> {{  fn from(v: [T; {i}]) -> Self {{ Vec(v) }}  }}
impl <T: Scalar> Into<[T; {i}]> for Vec{i}<T> {{  fn into(self) -> [T; {i}] {{ self.0 }}  }}

impl <T: Scalar> From<({tuple})> for Vec{i}<T> {{  fn from(({tpat}): ({tuple})) -> Self {{ Vec([{tpat}]) }}  }}
impl <T: Scalar> Into<({tuple})> for Vec{i}<T> {{  fn into(self) -> ({tuple}) {{ ({selfpat}) }}  }}

impl <T: Scalar> Borrow   <[T; {i}]> for Vec{i}<T> {{  #[inline] fn borrow    (&    self) -> &    [T; {i}] {{ &    self.0 }}  }}
impl <T: Scalar> BorrowMut<[T; {i}]> for Vec{i}<T> {{  #[inline] fn borrow_mut(&mut self) -> &mut [T; {i}] {{ &mut self.0 }}  }}

impl <T: Scalar> AsRef<[T; {i}]> for Vec{i}<T> {{  #[inline] fn as_ref(&    self) -> &    [T; {i}] {{ &    self.0 }}  }}
impl <T: Scalar> AsMut<[T; {i}]> for Vec{i}<T> {{  #[inline] fn as_mut(&mut self) -> &mut [T; {i}] {{ &mut self.0 }}  }}\n", i=i, d=d, tuple=tuple, tpat=tpat, selfpat=selfpat));
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
impl <'u, T: Scalar, D: Dim<T>> {trait_name}<&'u Vec<D, T>> for {scalar_type}
where {scalar_type}: {trait_name}<T>,
<{scalar_type} as {trait_name}<T>>::Output: Scalar,
D: Dim<<{scalar_type} as {trait_name}<T>>::Output> {{
    type Output = Vec<D, <{scalar_type} as {trait_name}<T>>::Output>;
    fn {method_name}(self, rhs: &'u Vec<D, T>) -> Self::Output {{
        Vec(<D as Dim<<{scalar_type} as {trait_name}<T>>::Output>>::from_iter(rhs.iter().map(|r| {trait_name}::<T>::{method_name}(self, *r))))
    }}
}}
impl <'t, 'u, T: Scalar, D: Dim<T>> {trait_name}<&'u Vec<D, T>> for &'t {scalar_type}
where {scalar_type}: {trait_name}<T>,
<{scalar_type} as {trait_name}<T>>::Output: Scalar,
D: Dim<<{scalar_type} as {trait_name}<T>>::Output> {{
    type Output = Vec<D, <{scalar_type} as {trait_name}<T>>::Output>;
    fn {method_name}(self, rhs: &'u Vec<D, T>) -> Self::Output {{
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
where T: {trait_name},
<T as {trait_name}>::Output: Scalar,
D: Dim<<T as {trait_name}>::Output> {{
    type Output = Vec<D, <T as {trait_name}>::Output>;
    fn {method_name}(self) -> Self::Output {{ {trait_name}::{method_name}(&self) }}
}}
impl <'t, T: Scalar, D: Dim<T>> {trait_name} for &'t Vec<D, T>
where T: {trait_name},
<T as {trait_name}>::Output: Scalar,
D: Dim<<T as {trait_name}>::Output> {{
    type Output = Vec<D, <T as {trait_name}>::Output>;
    fn {method_name}(self) -> Self::Output {{
        Vec(<D as Dim<<T as {trait_name}>::Output>>::from_iter(self.iter().map(|v| {trait_name}::{method_name}(*v))))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}

fn impl_binop(f: &mut File, trait_name: &str) -> Result<()> {
    write!(f, "
impl <T: Scalar, U: Scalar, D: Dim<T> + Dim<U>> {trait_name}<Vec<D, U>> for Vec<D, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
D: Dim<<T as {trait_name}<U>>::Output> {{
    type Output = Vec<D, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: Vec<D, U>) -> Self::Output {{ {trait_name}::{method_name}(&self, &rhs) }}
}}
impl <'t, T: Scalar, U: Scalar, D: Dim<T> + Dim<U>> {trait_name}<Vec<D, U>> for &'t Vec<D, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
D: Dim<<T as {trait_name}<U>>::Output> {{
    type Output = Vec<D, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: Vec<D, U>) -> Self::Output {{ {trait_name}::{method_name}(self, &rhs) }}
}}
impl <'u, T: Scalar, U: Scalar, D: Dim<T> + Dim<U>> {trait_name}<&'u Vec<D, U>> for Vec<D, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
D: Dim<<T as {trait_name}<U>>::Output> {{
    type Output = Vec<D, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: &'u Vec<D, U>) -> Self::Output {{ {trait_name}::{method_name}(&self, rhs) }}
}}
impl <'t, 'u, T: Scalar, U: Scalar, D: Dim<T> + Dim<U>> {trait_name}<&'u Vec<D, U>> for &'t Vec<D, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
D: Dim<<T as {trait_name}<U>>::Output> {{
    type Output = Vec<D, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: &'u Vec<D, U>) -> Self::Output {{
        Vec(<D as Dim<<T as {trait_name}<U>>::Output>>::from_iter(self.iter().zip(rhs.iter()).map(|(l, r)| {trait_name}::{method_name}(*l, *r))))
    }}
}}

impl <T: Scalar, U: Scalar, D: Dim<T>> {trait_name}<U> for Vec<D, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
D: Dim<<T as {trait_name}<U>>::Output> {{
    type Output = Vec<D, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: U) -> Self::Output {{ {trait_name}::{method_name}(&self, rhs) }}
}}
impl <'t, T: Scalar, U: Scalar, D: Dim<T>> {trait_name}<U> for &'t Vec<D, T>
where T: {trait_name}<U>,
<T as {trait_name}<U>>::Output: Scalar,
D: Dim<<T as {trait_name}<U>>::Output> {{
    type Output = Vec<D, <T as {trait_name}<U>>::Output>;
    fn {method_name}(self, rhs: U) -> Self::Output {{
        Vec(<D as Dim<<T as {trait_name}<U>>::Output>>::from_iter(self.iter().map(|l| {trait_name}::{method_name}(*l, rhs))))
    }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase())
}

fn impl_from(f: &mut File) -> Result<()> {
    for t in types().iter().filter(|&t| *t != "bool") {
        for u in types().iter().filter(|&t| *t != "bool") {
            if t == u { continue; }
            try!(write!(f,"impl <D: Dim<{t:5}>+Dim<{u:5}>> From<Vec<D, {u:5}>> for Vec<D, {t:5}> {{  fn from(v: Vec<D, {u:5}>) -> Self {{ Vec(<D as Dim<{t:5}>>::from_iter(v.iter().map(|&v| v as {t}))) }}  }}\n", t=t, u=u));
        }
        for u in angle_types().iter() {
            if t == u { continue; }
            try!(write!(f,"impl <D: Dim<{t:5}>+Dim<{u:5}>> From<Vec<D, {u:5}>> for Vec<D, {t:5}> {{  fn from(v: Vec<D, {u:5}>) -> Self {{ Vec(<D as Dim<{t:5}>>::from_iter(v.iter().map(|&v| v.into()))) }}  }}\n", t=t, u=u));
        }
    }
    for t in angle_types().iter() {
        for u in types().iter().chain(angle_types().iter()).filter(|t| **t != "bool") {
            if t == u { continue; }
            try!(write!(f,"impl <D: Dim<{t:5}>+Dim<{u:5}>> From<Vec<D, {u:5}>> for Vec<D, {t:5}> {{  fn from(v: Vec<D, {u:5}>) -> Self {{ Vec(<D as Dim<{t:5}>>::from_iter(v.iter().map(|&v| v.into()))) }}  }}\n", t=t, u=u));
        }
    }
    write!(f, "\n")
}


pub fn main(out_dir: &String) {
    let dest_path = Path::new(out_dir).join("vec.rs");
    let mut f = File::create(&dest_path).unwrap();
    declare_vec(&mut f).unwrap();
}
