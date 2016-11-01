use std::env;

mod build_mat;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    build_mat::main(&out_dir);
}
