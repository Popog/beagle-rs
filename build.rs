use std::env;

mod build_index;
mod build_mat;
mod build_scalar_array;
mod build_vec;
mod build_angle;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();

    build_vec::main(&out_dir);
    build_scalar_array::main(&out_dir);
    build_index::main(&out_dir);
    build_mat::main(&out_dir);
    build_angle::main(&out_dir);
}
