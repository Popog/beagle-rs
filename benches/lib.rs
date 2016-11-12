#![feature(test)]

extern crate beagle;
#[cfg(feature="rand")]
extern crate rand;
extern crate test;

#[path="modules/mat.rs"]
mod mat;

#[path="modules/vec.rs"]
mod vec;
