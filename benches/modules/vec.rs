#[macro_use] mod macros;

macro_rules! bench_binop_vec {
    ($($mtype:ty, $mod_name:ident, $Vec1:ident,)+) => {$(
#[cfg(feature="rand")]
mod $mod_name {
    use beagle::{Value, v};
    use beagle::vec::{$Vec1};
    use rand;
    use test;
    use test::Bencher;
    bench_binop!{
        vec_add, $mtype, $Vec1<$mtype>, $Vec1<$mtype>, +,
        vec_sub, $mtype, $Vec1<$mtype>, $Vec1<$mtype>, -,
        vec_mul, $mtype, $Vec1<$mtype>, $Vec1<$mtype>, *,
        vec_div, $mtype, $Vec1<$mtype>, $Vec1<$mtype>, /,
        vec_rem, $mtype, $Vec1<$mtype>, $Vec1<$mtype>, %,

        value_add, $mtype, $Vec1<$mtype>, Value<$mtype>, +,
        value_sub, $mtype, $Vec1<$mtype>, Value<$mtype>, -,
        value_div, $mtype, $Vec1<$mtype>, Value<$mtype>, /,
        value_mul, $mtype, $Vec1<$mtype>, Value<$mtype>, *,
        value_rem, $mtype, $Vec1<$mtype>, Value<$mtype>, %,
    }
}
    )+};
}

macro_rules! bench_binop_vec_types {
    ($($f32:ident)+) => {$(
mod $f32 {
    bench_binop_vec!{
        $f32, vec1_bench, Vec1,
        $f32, vec2_bench, Vec2,
        $f32, vec3_bench, Vec3,
        $f32, vec4_bench, Vec4,
    }
}
    )+};
}

bench_binop_vec_types!{
    f32 f64
}

#[cfg(not(feature="quick_bench"))]
bench_binop_vec_types!{
    i8 i16 i32 i64
    u8 u16 u32 u64
}



/*
#[bench]
fn bench_faceforward(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);
        let b: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(ROUNDS);
        for _ in 0..n {
            a = faceforward(a, &a, &b);
        }
        a
    });
}

#[bench]
fn bench_distance2(b: &mut Bencher) {
    b.iter(|| {
        let mut a: Vec3<f64> = Vec3::new([1.0, 2.0, 5.0]);
        let b: Vec3<f64> = Vec3::new([1.0, 2.0, 3.0]);

        let n = test::black_box(ROUNDS);
        for _ in 0..n {
            a = a * distance2(a, &b);
        }
        a
    });
}
*/
