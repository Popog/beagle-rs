use std::env;

mod build_index;
mod build_mat;
mod build_scalar_array;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    build_index::main(&out_dir);
    build_mat::main(&out_dir);
}
