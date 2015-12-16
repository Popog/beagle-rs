use std::ascii::AsciiExt;
use std::cmp::max;
use std::fs::File;
use std::io::{Write,Result};
use std::path::Path;

fn declare_mod(out_dir: &String, prefix: &str, (x,y,z,w): (char,char,char,char)) -> Result<()> {
    let dest_path = Path::new(out_dir).join(format!("{prefix}{x}{y}{z}{w}.rs", prefix=prefix.to_lowercase(), x=x.to_ascii_lowercase(), y=y.to_ascii_lowercase(), z=z.to_ascii_lowercase(), w=w.to_ascii_lowercase()));
    let mut f = try!(File::create(&dest_path));

    try!(write!(f,"pub mod {prefix}{x}{y}{z}{w} {{
    // TODO: Remove. Blocked by rust-lang/rust#23121
    use std::mem;
    use std::ops::{{Index,IndexMut}};

    use traits::Scalar;
    use vec::{{", prefix=prefix.to_lowercase(), x=x.to_ascii_lowercase(), y=y.to_ascii_lowercase(), z=z.to_ascii_lowercase(), w=w.to_ascii_lowercase()));
    for i in 1..4+1 {
        try!(write!(f,"Vec{},", i));
    }
    try!(write!(f,"}};\n"));


    let prefix = prefix.to_uppercase();
    let xyzw = [x.to_ascii_uppercase(), y.to_ascii_uppercase(), z.to_ascii_uppercase(), w.to_ascii_uppercase()];

    try!(write!(f,"\n    // Unary\n  "));
    for i in 0..4 {
        try!(write!(f,"  pub struct {}{};", prefix, xyzw[i]));
    }

    try!(write!(f,"\n\n\n    // Pair"));
    for i in 0..4 {
        try!(write!(f,"\n  "));
        for j in 0..4 {
            try!(write!(f,"  pub struct {}{}{};", prefix, xyzw[i], xyzw[j]));
        }
    }

    try!(write!(f,"\n\n\n    // Triple"));
    for i in 0..4 {
        for j in 0..4 {
            try!(write!(f,"\n  "));
            for k in 0..4 {
                try!(write!(f,"  pub struct {}{}{}{};", prefix, xyzw[i], xyzw[j], xyzw[k]));
            }
        }
        try!(write!(f,"\n"));
    }

    try!(write!(f,"\n\n    // Quadruple"));
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                try!(write!(f,"\n  "));
                for l in 0..4 {
                    try!(write!(f,"  pub struct {}{}{}{}{};", prefix, xyzw[i], xyzw[j], xyzw[k], xyzw[l]));
                }
            }
            try!(write!(f,"\n"));
        }
        try!(write!(f,"\n"));
    }

    try!(write!(f,"\n    // Unary Index"));
    for i in 0..4 {
        for v in i+1..4+1 {
            try!(write!(f,"
    impl <T: Scalar> Index<{prefix}{x}> for Vec{v}<T> {{
        type Output = Vec1<T>;
        #[inline]
        // TODO: Convert to slice pattern. Blocked by rust-lang/rust#23121
        fn index(&self, _: {prefix}{x}) -> &Self::Output {{ unsafe {{ mem::transmute(&self[{i}usize]) }} }}
    }}
    impl <T: Scalar> IndexMut<{prefix}{x}> for Vec{v}<T> {{
        #[inline]
        // TODO: Convert to slice pattern. Blocked by rust-lang/rust#23121
        fn index_mut(&mut self, _: {prefix}{x}) -> &mut Self::Output {{ unsafe {{ mem::transmute(&mut self[{i}usize]) }} }}
    }}", prefix=prefix, x=xyzw[i], v=v, i=i));
        }
    }

    try!(write!(f,"\n\n    // Pair Index"));
    for i in 0..4 {
        for j in 0..4 {
            for v in max(i,j)+1..4+1 {
                if i+1 == j {
                    try!(write!(f,"
    impl <T: Scalar> Index<{prefix}{x}{y}> for Vec{v}<T> {{
        type Output = Vec2<T>;
        #[inline]
        // TODO: Convert to slice pattern. Blocked by rust-lang/rust#23121
        fn index(&self, _: {prefix}{x}{y}) -> &Self::Output {{ unsafe {{ mem::transmute(&self[{i}usize]) }} }}
    }}
    impl <T: Scalar> IndexMut<{prefix}{x}{y}> for Vec{v}<T> {{
        #[inline]
        // TODO: Convert to slice pattern. Blocked by rust-lang/rust#23121
        fn index_mut(&mut self, _: {prefix}{x}{y}) -> &mut Self::Output {{ unsafe {{ mem::transmute(&mut self[{i}usize]) }} }}
    }}", prefix=prefix, x=xyzw[i], y=xyzw[j], v=v, i=i));
                } else {
                    if i != j {
                        // TODO IndexSet
                        try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <T: Scalar> IndexSet<{prefix}{x}{y}> for Vec{v}<T> {{}}", prefix=prefix, x=xyzw[i], y=xyzw[j], v=v));
                    }
                    // TODO IndexGet
                    try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <T: Scalar> IndexGet<{prefix}{x}{y}> for Vec{v}<T> {{}}", prefix=prefix, x=xyzw[i], y=xyzw[j], v=v));
                }
            }
        }
    }

    try!(write!(f,"\n\n    // Triple Index"));
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                for v in max(max(i,j),k)+1..4+1 {
                    if i+1 == j && j+1 == k {
                        try!(write!(f,"
    impl <T: Scalar> Index<{prefix}{x}{y}{z}> for Vec{v}<T> {{
        type Output = Vec3<T>;
        #[inline]
        // TODO: Convert to slice pattern. Blocked by rust-lang/rust#23121
        fn index(&self, _: {prefix}{x}{y}{z}) -> &Self::Output {{ unsafe {{ mem::transmute(&self[{i}usize]) }} }}
    }}
    impl <T: Scalar> IndexMut<{prefix}{x}{y}{z}> for Vec{v}<T> {{
        #[inline]
        // TODO: Convert to slice pattern. Blocked by rust-lang/rust#23121
        fn index_mut(&mut self, _: {prefix}{x}{y}{z}) -> &mut Self::Output {{ unsafe {{ mem::transmute(&mut self[{i}usize]) }} }}
    }}", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], v=v, i=i));
                    } else {
                        if i != j && i != k && j != k {
                            // TODO IndexSet
                            try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <T: Scalar> IndexSet<{prefix}{x}{y}{z}> for Vec{v}<T> {{}}", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], v=v));
                        }
                        // TODO IndexGet
                        try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <T: Scalar> IndexGet<{prefix}{x}{y}{z}> for Vec{v}<T> {{}}", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], v=v));
                    }
                }
            }
        }
    }

    try!(write!(f,"\n\n    // Quadruple Index"));
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                for l in 0..4 {
                    for v in max(max(max(i,j),k),l)+1..4+1 {
                        if i+1 == j && j+1 == k && k+1 == l {
                            try!(write!(f,"
    impl <T: Scalar> Index<{prefix}{x}{y}{z}{w}> for Vec{v}<T> {{
        type Output = Vec4<T>;
        #[inline]
        // TODO: Convert to slice pattern. Blocked by rust-lang/rust#23121
        fn index(&self, _: {prefix}{x}{y}{z}{w}) -> &Self::Output {{ unsafe {{ mem::transmute(&self[{i}usize]) }} }}
    }}
    impl <T: Scalar> IndexMut<{prefix}{x}{y}{z}{w}> for Vec{v}<T> {{
        #[inline]
        // TODO: Convert to slice pattern. Blocked by rust-lang/rust#23121
        fn index_mut(&mut self, _: {prefix}{x}{y}{z}{w}) -> &mut Self::Output {{ unsafe {{ mem::transmute(&mut self[{i}usize]) }} }}
    }}", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], w=xyzw[l], v=v, i=i));
                        } else {
                            if i != j && i != k && i != l && j != k && j != l && k != l {
                                // TODO IndexSet
                                try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <T: Scalar> IndexSet<{prefix}{x}{y}{z}{w}> for Vec{v}<T> {{}}", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], w=xyzw[l], v=v));
                            }
                            // TODO IndexGet
                            try!(write!(f,"
    // TODO: Implement. Blocked on rust-lang/rfcs#997
    //impl <T: Scalar> IndexGet<{prefix}{x}{y}{z}{w}> for Vec{v}<T> {{}}", prefix=prefix, x=xyzw[i], y=xyzw[j], z=xyzw[k], w=xyzw[l], v=v));
                        }
                    }
                }
            }
        }
    }

    write!(f, "\n}}\n")
}

pub fn main(out_dir: &String) {
    declare_mod(out_dir, "s", ('0', '1', '2', '3')).unwrap();
    declare_mod(out_dir, "", ('x', 'y', 'z', 'w')).unwrap();
    declare_mod(out_dir, "", ('s', 't', 'p', 'q')).unwrap();
    declare_mod(out_dir, "", ('r', 'g', 'b', 'a')).unwrap();

}
