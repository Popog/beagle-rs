use std::fs::File;
use std::io::{Write,Result};
use std::path::Path;

fn declare_mat(f: &mut File) -> Result<()> {
    for r in 1..4 {
        let r = r+1;
        try!(write!(f,"
impl <T: Copy+Mul<Output=T>+Neg<Output=T>+Add<Output=T>> Mat{r}<T> {{
    /// The determinant of the matrix
    pub fn determinant(&self) -> T {{\n", r=r));

        let mut lehmer_sequence = vec![0i32; r];
        loop {
            try!(write!(f,"        "));
            let mut even = true;
            let mut taken = vec![false; r];
            let mut indices = vec![];
            for (row, &lehmer) in lehmer_sequence.iter().enumerate() {
                let index = taken.iter().enumerate()
                    .scan(lehmer+1, |lehmer, (i, &b)| {
                        //write!(f,"//Scanning {}, {}, {}\n", row, lehmer, i).unwrap();
                        if !b { *lehmer -= 1; }
                        if *lehmer > 0 { Some(i)
                        } else if *lehmer == 0 {
                            *lehmer -= 1;
                            Some(i)
                        } else { None }
                    }).last().unwrap();
                taken[index] = true;
                indices.push((row, index));
                let mut lehmer = lehmer;
                for &lehmer2 in lehmer_sequence.iter() {
                    if lehmer == 0 { break; }
                    if lehmer2 < lehmer {
                        lehmer -= 1;
                        even=!even;
                    }
                }
            }
            let count = lehmer_sequence.iter_mut().rev().enumerate().map(|(i, l)| {
                //write!(f,"//inc {}, {}\n", i, l).unwrap();
                *l = *l+1;
                if *l > i as i32 {
                    *l = 0;
                    true
                } else {
                    false
                }
            }).take_while(|&v| v).count();

            if !even {
                try!(write!(f,"-"));
            } else {
                try!(write!(f," "));
            }

            try!(write!(f,"("));
            {
                let (row, index) = indices[0];
                try!(write!(f,"self[{row}][{index}]", row=row, index=index));
            }
            for &(row, index) in indices.iter().skip(1) {
                try!(write!(f,"*self[{row}][{index}]", row=row, index=index));
            }
            try!(write!(f,")"));

            if count >= r {
                try!(write!(f,"\n"));
                break;
            } else {
                try!(write!(f," + \n"));
            }
        }
        try!(write!(f,"    }}\n}}\n"));
    }

    write!(f, "\n")
}

pub fn main(out_dir: &String) {
    let dest_path = Path::new(out_dir).join("mat.rs");
    let mut f = File::create(&dest_path).unwrap();
    declare_mat(&mut f).unwrap();
}
