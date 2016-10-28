use std::ascii::AsciiExt;
use std::fs::File;
use std::io::{Write,Result};
use std::path::Path;

use build_scalar_array::{TupleRange,Indices, dims};
use build_scalar_array::IndicesClass::{InOrder,Unique,Duplicates};

fn declare_mod(f: &mut File, prefix: &str, (x,y,z,w): (char,char,char,char)) -> Result<()> {
    try!(write!(f,"pub mod {prefix}{x}{y}{z}{w} {{
    use std::iter::{{Chain,Once, once}};
    use std::ops::{{Index,IndexMut}};
    use std::vec::Vec;

    use super::{{Apply,VecRef}};
    use scalar_array::{{Construct, Scalar, Two, Three, Four}};
    use vec::{{", prefix=prefix.to_lowercase(), x=x.to_ascii_lowercase(), y=y.to_ascii_lowercase(), z=z.to_ascii_lowercase(), w=w.to_ascii_lowercase()));
    for i in 2..4+1 {
        try!(write!(f,"Vec{i},", i=i));
    }
    try!(write!(f,"}};\n"));


    let prefix = prefix.to_uppercase();
    let xyzw = [x.to_ascii_uppercase(), y.to_ascii_uppercase(), z.to_ascii_uppercase(), w.to_ascii_uppercase()];

    try!(write!(f,"\n\t// Unary\n\t"));
    for i in 0..4 {
        try!(write!(f,"pub const {}{}: usize = {};  ", prefix, xyzw[i], {i}));
    }

    try!(write!(f,"\n\n\n\t// Pair\n\t"));
    for (i,j) in TupleRange::from((0,0)..(4,4)) {
        try!(write!(f,"pub struct {}{}{};  ", prefix, xyzw[i], xyzw[j]));
        if j == 3 && i != 3 { try!(write!(f,"\n\t")) }
    }

    try!(write!(f,"\n\n\n\t// Triple\n\t"));
    for (i,j,k) in TupleRange::from((0,0,0)..(4,4,4)) {
        try!(write!(f,"pub struct {}{}{}{};  ", prefix, xyzw[i], xyzw[j], xyzw[k]));
        if k == 3 && (j != 3 || i != 3) { try!(write!(f,"\n\t")) }
    }

    try!(write!(f,"\n\n\n\t// Quadruple\n\t"));
    for (i,j,k,l) in TupleRange::from((0,0,0,0)..(4,4,4,4)) {
        try!(write!(f,"pub struct {}{}{}{}{};  ", prefix, xyzw[i], xyzw[j], xyzw[k], xyzw[l]));
        if l == 3 && k == 3 && j == 3 && i != 3 { try!(write!(f,"\n")) }
        if l == 3 && (k != 3 || j != 3 || i != 3) { try!(write!(f,"\n\t")) }
    }

    try!(write!(f,"\n\n\t// Pair Index"));
    for (i,j) in TupleRange::from((0,0)..(4,4)) {
        for v in (i,j).max()+1..4+1 {
            let vlong = dims()[v-1];
            match (i,j).classify() {
                InOrder => {
                    try!(write!(f,"
    impl <V: Scalar> Index<{prefix}{x}{y}> for Vec{v}<V> {{
        type Output = Vec2<V>;
        #[inline]
        fn index(&self, _: {prefix}{x}{y}) -> &Self::Output {{\n", prefix=prefix, x=xyzw[i], y=xyzw[j], v=v));

                    try!(write!(f,"\t\t\tlet ["));
                    for _ in 0..i { try!(write!(f,"_, ")); }
                    try!(write!(f,"ref v.."));
                    for _ in j+1..v { try!(write!(f,", _")); }

                    try!(write!(f,"] = **self;
            Construct::from_ref(v)
        }}
    }}
    impl <V: Scalar> IndexMut<{prefix}{x}{y}> for Vec{v}<V> {{
        #[inline]
        fn index_mut(&mut self, _: {prefix}{x}{y}) -> &mut Self::Output {{\n", prefix=prefix, x=xyzw[i], y=xyzw[j], v=v));

                    try!(write!(f,"\t\t\tlet ["));
                    for _ in 0..i { try!(write!(f,"_, ")); }
                    try!(write!(f,"ref mut v.."));
                    for _ in j+1..v { try!(write!(f,", _")); }

                    try!(write!(f,"] = **self;
            Construct::from_mut(v)
        }}
    }}"))
                },

                // TODO IndexSet
                Unique => try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <V: Scalar> IndexSet<{prefix}{x}{y}> for Vec{v}<V> {{}}
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <V: Scalar> IndexGet<{prefix}{x}{y}> for Vec{v}<V> {{}}
    // TODO: Make safe. Blocked on rust-lang/rfcs#997
    impl <V: Scalar> Index<{prefix}{x}{y}> for Vec{v}<V> {{
        type Output = VecRef<{vlong}, V, {prefix}{x}{y}>;
        #[inline]
        fn index(&self, _: {prefix}{x}{y}) -> &Self::Output {{ VecRef::from_ref(self) }}
    }}
    // TODO: Make safe. Blocked on rust-lang/rfcs#997
    impl <V: Scalar> IndexMut<{prefix}{x}{y}> for Vec{v}<V> {{
        #[inline]
        fn index_mut(&mut self, _: {prefix}{x}{y}) -> &mut Self::Output {{ VecRef::from_mut(self) }}
    }}

    impl<'a, V: Scalar> IntoIterator for &'a VecRef<{vlong}, V, {prefix}{x}{y}> {{
        type Item = &'a V;
        type IntoIter = Chain<Once<&'a V>,Once<&'a V>>;
        fn into_iter(self) -> Self::IntoIter {{
            let v: &Vec{v}<V> = self.to_ref();
            let x = once(&v[{prefix}{x}]);
            let y = once(&v[{prefix}{y}]);
            x.chain(y)
        }}
    }}
    impl<'a, V: Scalar> IntoIterator for &'a mut VecRef<{vlong}, V, {prefix}{x}{y}> {{
        type Item = &'a mut V;
        type IntoIter = Chain<Once<&'a mut V>,Once<&'a mut V>>;
        fn into_iter(self) -> Self::IntoIter {{
            let v: &mut Vec{v}<V> = self.to_mut();
            let mut v: Vec<&mut V> = v.into_iter().collect();

            v.swap({prefix}{x}, {v}-1);
            let x = once(v.remove({v}-1));
            v.swap({prefix}{y}, {v}-2);
            let y = once(v.remove({v}-2));
            x.chain(y)
        }}
    }}

    impl <V: Scalar> Apply<Two, V> for VecRef<{vlong}, V, {prefix}{x}{y}> {{
        #[inline]
        // TODO: Make safe. Blocked on rust-lang/rfcs#997
        fn apply_rhs<U: Scalar, F:FnMut(&mut V, &U)>(&mut self, mut f: F, rhs: &Vec2<U>) {{
            let v: &mut Vec{v}<V> = self.to_mut();
            f(&mut v[{prefix}{x}], &rhs[0]);
            f(&mut v[{prefix}{y}], &rhs[1]);
        }}
        #[inline]
        // TODO: Make safe. Blocked on rust-lang/rfcs#997
        fn apply_lhs<U: Scalar, F:FnMut(&V, &mut U)>(&self, mut f: F, lhs: &mut Vec2<U>) {{
            let v: &Vec{v}<V> = self.to_ref();
            f(&v[{prefix}{x}], &mut lhs[0]);
            f(&v[{prefix}{y}], &mut lhs[1]);
        }}
    }}", prefix=prefix, x=xyzw[i], y=xyzw[j], v=v, vlong=vlong)),

                // TODO IndexGet
                Duplicates => try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <V: Scalar> IndexGet<{prefix}{x}{y}> for Vec{v}<V> {{}}", prefix=prefix, x=xyzw[i], y=xyzw[j], v=v)),
            }
        }
    }

    try!(write!(f,"\n\n    // Triple Index"));
    for (i,j,k) in TupleRange::from((0,0,0)..(4,4,4)) {
        for v in (i,j,k).max()+1..4+1 {
            let vlong = dims()[v-1];
            match (i,j,k).classify() {
                InOrder => {
                    try!(write!(f,"
    impl <V: Scalar> Index<{prefix}{x}{y}{z}> for Vec{v}<V> {{
        type Output = Vec3<V>;
        #[inline]
        fn index(&self, _: {prefix}{x}{y}{z}) -> &Self::Output {{\n", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], v=v));

                    try!(write!(f,"\t\t\tlet ["));
                    for _ in 0..i { try!(write!(f,"_, ")); }
                    try!(write!(f,"ref v.."));
                    for _ in k+1..v { try!(write!(f,", _")); }

                    try!(write!(f,"] = **self;
            Construct::from_ref(v)
        }}
    }}
    impl <V: Scalar> IndexMut<{prefix}{x}{y}{z}> for Vec{v}<V> {{
        #[inline]
        fn index_mut(&mut self, _: {prefix}{x}{y}{z}) -> &mut Self::Output {{\n", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], v=v));

                    try!(write!(f,"\t\t\tlet ["));
                    for _ in 0..i { try!(write!(f,"_, ")); }
                    try!(write!(f,"ref mut v.."));
                    for _ in k+1..v { try!(write!(f,", _")); }

                    try!(write!(f,"] = **self;
            Construct::from_mut(v)
        }}
    }}"))
                },

                // TODO IndexSet
                Unique => try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <V: Scalar> IndexSet<{prefix}{x}{y}{z}> for Vec{v}<V> {{}}
    // TODO: Make safe. Blocked on rust-lang/rfcs#997
    impl <V: Scalar> Index<{prefix}{x}{y}{z}> for Vec{v}<V> {{
        type Output = VecRef<{vlong}, V, {prefix}{x}{y}{z}>;
        #[inline]
        fn index(&self, _: {prefix}{x}{y}{z}) -> &Self::Output {{ VecRef::from_ref(self) }}
    }}
    // TODO: Make safe. Blocked on rust-lang/rfcs#997
    impl <V: Scalar> IndexMut<{prefix}{x}{y}{z}> for Vec{v}<V> {{
        #[inline]
        fn index_mut(&mut self, _: {prefix}{x}{y}{z}) -> &mut Self::Output {{ VecRef::from_mut(self) }}
    }}
    impl <V: Scalar> Apply<Three, V> for VecRef<{vlong}, V, {prefix}{x}{y}{z}> {{
        #[inline]
        fn apply_rhs<U: Scalar, F:FnMut(&mut V, &U)>(&mut self, mut f: F, rhs: &Vec3<U>) {{
            let v: &mut Vec{v}<V> = self.to_mut();
            f(&mut v[{prefix}{x}], &rhs[0]);
            f(&mut v[{prefix}{y}], &rhs[1]);
            f(&mut v[{prefix}{z}], &rhs[2]);
        }}
        #[inline]
        fn apply_lhs<U: Scalar, F:FnMut(&V, &mut U)>(&self, mut f: F, lhs: &mut Vec3<U>) {{
            let v: &Vec{v}<V> = self.to_ref();
            f(&v[{prefix}{x}], &mut lhs[0]);
            f(&v[{prefix}{y}], &mut lhs[1]);
            f(&v[{prefix}{z}], &mut lhs[2]);
        }}
    }}", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], v=v, vlong=vlong)),

                // TODO IndexGet
                Duplicates => try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <V: Scalar> IndexGet<{prefix}{x}{y}{z}> for Vec{v}<V> {{}}", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], v=v)),
            }
        }
    }

    try!(write!(f,"\n\n    // Quadruple Index"));
    for (i,j,k,l) in TupleRange::from((0,0,0,0)..(4,4,4,4)) {
        for v in (i,j,k,l).max()+1..4+1 {
            let vlong = dims()[v-1];
            match (i,j,k,l).classify() {
                InOrder => {
                    try!(write!(f,"
    impl <V: Scalar> Index<{prefix}{x}{y}{z}{w}> for Vec{v}<V> {{
        type Output = Vec4<V>;
        #[inline]
        fn index(&self, _: {prefix}{x}{y}{z}{w}) -> &Self::Output {{\n", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], w=xyzw[l], v=v));
                    try!(write!(f,"\t\t\tlet ["));
                    for _ in 0..i { try!(write!(f,"_, ")); }
                    try!(write!(f,"ref v.."));
                    for _ in l+1..v { try!(write!(f,", _")); }

                    try!(write!(f,"] = **self;
            Construct::from_ref(v)
        }}
    }}
    impl <V: Scalar> IndexMut<{prefix}{x}{y}{z}{w}> for Vec{v}<V> {{
        #[inline]
        fn index_mut(&mut self, _: {prefix}{x}{y}{z}{w}) -> &mut Self::Output {{\n", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], w=xyzw[l], v=v));
                    try!(write!(f,"\t\t\tlet ["));
                    for _ in 0..i { try!(write!(f,"_, ")); }
                    try!(write!(f,"ref mut v.."));
                    for _ in l+1..v { try!(write!(f,", _")); }

                    try!(write!(f,"] = **self;
            Construct::from_mut(v)
        }}
    }}"))
                },

                // TODO IndexSet
                Unique => try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <V: Scalar> IndexSet<{prefix}{x}{y}{z}{w}> for Vec{v}<V> {{}}
    // TODO: Make safe. Blocked on rust-lang/rfcs#997
    impl <V: Scalar> Index<{prefix}{x}{y}{z}{w}> for Vec{v}<V> {{
        type Output = VecRef<{vlong}, V, {prefix}{x}{y}{z}{w}>;
        #[inline]
        fn index(&self, _: {prefix}{x}{y}{z}{w}) -> &Self::Output {{ VecRef::from_ref(self) }}
    }}
    // TODO: Make safe. Blocked on rust-lang/rfcs#997
    impl <V: Scalar> IndexMut<{prefix}{x}{y}{z}{w}> for Vec{v}<V> {{
        #[inline]
        fn index_mut(&mut self, _: {prefix}{x}{y}{z}{w}) -> &mut Self::Output {{ VecRef::from_mut(self) }}
    }}
    impl <V: Scalar> Apply<Four, V> for VecRef<{vlong}, V, {prefix}{x}{y}{z}{w}> {{
        #[inline]
        fn apply_rhs<U: Scalar, F:FnMut(&mut V, &U)>(&mut self, mut f: F, rhs: &Vec4<U>) {{
            let v: &mut Vec{v}<V> = self.to_mut();
            f(&mut v[{prefix}{x}], &rhs[0]);
            f(&mut v[{prefix}{y}], &rhs[1]);
            f(&mut v[{prefix}{z}], &rhs[2]);
            f(&mut v[{prefix}{w}], &rhs[3]);
        }}
        #[inline]
        fn apply_lhs<U: Scalar, F:FnMut(&V, &mut U)>(&self, mut f: F, rhs: &mut Vec4<U>) {{
            let v: &Vec{v}<V> = self.to_ref();
            f(&v[{prefix}{x}], &mut rhs[0]);
            f(&v[{prefix}{y}], &mut rhs[1]);
            f(&v[{prefix}{z}], &mut rhs[2]);
            f(&v[{prefix}{w}], &mut rhs[3]);
        }}
    }}",
                prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], w=xyzw[l], v=v, vlong=vlong)),

                // TODO IndexGet
                Duplicates => try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <V: Scalar> IndexGet<{prefix}{x}{y}{z}{w}> for Vec{v}<V> {{}}",
                prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], w=xyzw[l], v=v)),
            }
        }
    }

    write!(f, "\n}}\n")
}

fn declare_alias_mod(f: &mut File, prefix: &str, (x,y,z,w): (char,char,char,char)) -> Result<()> {
    try!(write!(f,"pub mod {prefix}{x}{y}{z}{w} {{
    pub use super::s0123::{{", prefix=prefix.to_lowercase(), x=x.to_ascii_lowercase(), y=y.to_ascii_lowercase(), z=z.to_ascii_lowercase(), w=w.to_ascii_lowercase()));

    let prefix = prefix.to_uppercase();
    let xyzw = [x.to_ascii_uppercase(), y.to_ascii_uppercase(), z.to_ascii_uppercase(), w.to_ascii_uppercase()];

    let s0123 = ['0','1','2','3'];

    try!(write!(f,"\n\t\t// Unary\n\t\t"));
    for i in 0..4 {
        try!(write!(f,"S{} as {}{},", s0123[i], prefix, xyzw[i]));
    }

    try!(write!(f,"\n\n\n\t\t// Pair\n\t\t"));
    for (i,j) in TupleRange::from((0,0)..(4,4)) {
        try!(write!(f,"S{}{} as {}{}{}, ", s0123[i], s0123[j], prefix, xyzw[i], xyzw[j]));
        if j == 3 && i != 3{ try!(write!(f,"\n\t\t")) }
    }

    try!(write!(f,"\n\n\n\t\t// Triple\n\t\t"));
    for (i,j,k) in TupleRange::from((0,0,0)..(4,4,4)) {
        try!(write!(f,"S{}{}{} as {}{}{}{}, ", s0123[i], s0123[j], s0123[k], prefix, xyzw[i], xyzw[j], xyzw[k]));
        if k == 3 && (j != 3 || i != 3) { try!(write!(f,"\n\t\t")) }
    }

    try!(write!(f,"\n\n\n\t\t// Quadruple\n\t\t"));
    for (i,j,k,l) in TupleRange::from((0,0,0,0)..(4,4,4,4)) {
        try!(write!(f,"S{}{}{}{} as {}{}{}{}{}, ", s0123[i], s0123[j], s0123[k], s0123[l], prefix, xyzw[i], xyzw[j], xyzw[k], xyzw[l]));
        if l == 3 && k == 3 && j == 3 && i != 3 { try!(write!(f,"\n")) }
        if l == 3 && (k != 3 || j != 3 || i != 3) { try!(write!(f,"\n\t\t")) }
    }

    write!(f, "\n\t}};\n}}\n")
}

pub fn main(out_dir: &String) {
    let dest_path = Path::new(&out_dir).join("index.rs");
    let mut f = File::create(&dest_path).unwrap();

    declare_mod(&mut f, "s", ('0', '1', '2', '3')).unwrap();
    declare_alias_mod(&mut f, "", ('x', 'y', 'z', 'w')).unwrap();
    declare_alias_mod(&mut f, "", ('s', 't', 'p', 'q')).unwrap();
    declare_alias_mod(&mut f, "", ('r', 'g', 'b', 'a')).unwrap();

}
