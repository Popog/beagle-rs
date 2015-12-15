use std::fs::File;
use std::io::{Write,Result};
use std::path::Path;

use build_traits::{types};

fn declare_angle(f: &mut File) -> Result<()> {
    let un_ops = ["Neg"];
    let bin_ops = ["Add", "Div", "Rem", "Sub", "Mul"];

    try!(write!(f,"use std::ops::{{"));
    for trait_name in un_ops.iter().chain(bin_ops.iter()) {
        try!(write!(f,"{},", trait_name));
    }

    try!(write!(f,"}};\n"));
    for name in ["Rad", "Deg"].iter() {
        for size in [32, 64].iter() {
            try!(write!(f,"
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct {name}{size}(pub f{size});

impl Angle for {name}{size} {{
    type Type = f{size};

    #[inline] fn sin(self) -> Self::Type {{ Rad{size}::from(self).0.sin() }}
    #[inline] fn cos(self) -> Self::Type {{ Rad{size}::from(self).0.cos() }}
    #[inline] fn tan(self) -> Self::Type {{ Rad{size}::from(self).0.tan() }}
    #[inline] fn sin_cos(self) -> (Self::Type, Self::Type) {{ Rad{size}::from(self).0.sin_cos() }}

    #[inline] fn asin(s: Self::Type) -> Self {{ s.asin().into() }}
    #[inline] fn acos(s: Self::Type) -> Self {{ s.acos().into() }}
    #[inline] fn atan(s: Self::Type) -> Self {{ s.atan().into() }}
    #[inline] fn atan2(a: Self::Type, b: Self::Type) -> Self {{ a.atan2(b).into() }}
}}

impl AsRef<f{size}> for {name}{size} {{ fn as_ref(&self) -> &f{size} {{ &self.0 }} }}
impl AsMut<f{size}> for {name}{size} {{ fn as_mut(&mut self) -> &mut f{size} {{ &mut self.0 }} }}
\n", name=name, size=size));
            for t in types().iter().filter(|t| **t != "bool") {
                try!(write!(f,"
impl From<{t:5}> for {name}{size} {{  fn from(v: {t:5}) -> Self {{ {name}{size}(v as f{size}) }}  }}
impl From<{name}{size}> for {t:5} {{  fn from(v: {name}{size}) -> Self {{ v.0 as {t} }}  }}\n", name=name, size=size, t=t));
            }

            try!(impl_unop(f, "Neg", name, *size));
            try!(impl_binop(f, "Add", name, name, *size));
            try!(impl_binop(f, "Sub", name, name, *size));
            try!(impl_binop(f, "Div", name, "f", *size));
            try!(impl_binop(f, "Rem", name, name, *size));
            try!(impl_binop_scalar(f, "Mul", name, name, *size));
        }
    }

    write!(f, "\n")
}

fn impl_unop(f: &mut File, trait_name: &str, name: &str, size: usize) -> Result<()> {
    write!(f, "
impl {trait_name} for {name}{size} {{
    type Output = {name}{size};
    fn {method_name}(self) -> Self::Output {{ {name}{size}({trait_name}::{method_name}(self.0)) }}
}}
impl <'t> {trait_name} for &'t {name}{size} {{
    type Output = {name}{size};
    fn {method_name}(self) -> Self::Output {{ {name}{size}({trait_name}::{method_name}(&self.0)) }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase(), name=name, size=size)
}

fn impl_binop(f: &mut File, trait_name: &str, name: &str, output: &str, size: usize) -> Result<()> {
    write!(f, "
impl {trait_name} for {name}{size} {{
    type Output = {output}{size};
    fn {method_name}(self, rhs: {name}{size}) -> Self::Output {{ {trait_name}::<f{size}>::{method_name}(self.0, rhs.0).into() }}
}}
impl <'t> {trait_name}<{name}{size}> for &'t {name}{size} {{
    type Output = {output}{size};
    fn {method_name}(self, rhs: {name}{size}) -> Self::Output {{ {trait_name}::<f{size}>::{method_name}(&self.0, rhs.0).into() }}
}}
impl <'r> {trait_name}<&'r {name}{size}> for {name}{size} {{
    type Output = {output}{size};
    fn {method_name}(self, rhs: &'r {name}{size}) -> Self::Output {{ {trait_name}::<&'r f{size}>::{method_name}(self.0, &rhs.0).into() }}
}}
impl <'t, 'r> {trait_name}<&'r {name}{size}> for &'t {name}{size} {{
    type Output = {output}{size};
    fn {method_name}(self, rhs: &'r {name}{size}) -> Self::Output {{ {trait_name}::<&'r f{size}>::{method_name}(&self.0, &rhs.0).into() }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase(), name=name, output=output, size=size)
}

fn impl_binop_scalar(f: &mut File, trait_name: &str, name: &str, output: &str, size: usize) -> Result<()> {
    write!(f, "
impl {trait_name}<f{size}> for {name}{size} {{
    type Output = {output}{size};
    fn {method_name}(self, rhs: f{size}) -> Self::Output {{ {trait_name}::<f{size}>::{method_name}(self.0, rhs).into() }}
}}
impl <'t> {trait_name}<f{size}> for &'t {name}{size} {{
    type Output = {output}{size};
    fn {method_name}(self, rhs: f{size}) -> Self::Output {{ {trait_name}::<f{size}>::{method_name}(&self.0, rhs).into() }}
}}
impl <'r> {trait_name}<&'r f{size}> for {name}{size} {{
    type Output = {output}{size};
    fn {method_name}(self, rhs: &'r f{size}) -> Self::Output {{ {trait_name}::<&'r f{size}>::{method_name}(self.0, rhs).into() }}
}}
impl <'t, 'r> {trait_name}<&'r f{size}> for &'t {name}{size} {{
    type Output = {output}{size};
    fn {method_name}(self, rhs: &'r f{size}) -> Self::Output {{ {trait_name}::<&'r f{size}>::{method_name}(&self.0, rhs).into() }}
}}\n", trait_name=trait_name, method_name=trait_name.to_lowercase(), name=name, output=output, size=size)
}

pub fn main(out_dir: &String) {
    let dest_path = Path::new(out_dir).join("angle.rs");
    let mut f = File::create(&dest_path).unwrap();
    declare_angle(&mut f).unwrap();
}
