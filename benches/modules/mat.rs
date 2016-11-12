#[macro_use] mod macros;

macro_rules! bench_binop_mat {
    ($($mtype:ty, $mod_name:ident, $Mat1x2:ident, $Mat2x1:ident, $Mat2x2:ident, $Mat2x3:ident, $Mat2x4:ident, $Vec1:ident, $Vec2:ident,)+) => {$(
#[cfg(feature="rand")]
mod $mod_name {
    #[cfg(not(feature="quick_bench"))]
    use beagle::Value;
    use beagle::v;
    use beagle::vec::{$Vec1 as LhsVec, $Vec2 as RhsVec};
    #[cfg(not(feature="quick_bench"))]
    use beagle::mat::$Mat2x1;
    use beagle::mat::{$Mat1x2,$Mat2x2,$Mat2x3,$Mat2x4};
    use rand;
    use test;
    use test::Bencher;

    bench_binop!{
        mat_add, $mtype, $Mat1x2<$mtype>, $Mat1x2<$mtype>, +,
        mat_sub, $mtype, $Mat1x2<$mtype>, $Mat1x2<$mtype>, -,
        mat_div, $mtype, $Mat1x2<$mtype>, $Mat1x2<$mtype>, /,
        mat_rem, $mtype, $Mat1x2<$mtype>, $Mat1x2<$mtype>, %,
    }
    #[cfg(not(feature="quick_bench"))]
    bench_binop!{
        value_add, $mtype, $Mat1x2<$mtype>, Value<$mtype>, +,
        value_sub, $mtype, $Mat1x2<$mtype>, Value<$mtype>, -,
        value_div, $mtype, $Mat1x2<$mtype>, Value<$mtype>, /,
        value_mul, $mtype, $Mat1x2<$mtype>, Value<$mtype>, *,
        value_rem, $mtype, $Mat1x2<$mtype>, Value<$mtype>, %,
    }

    #[cfg(not(feature="quick_bench"))]
    bench_binop!{
        mat_mulx1, $mtype, $Mat1x2<$mtype>, $Mat2x1<$mtype>, *,
    }

    bench_binop!{
        mat_mulx2, $mtype, $Mat1x2<$mtype>, $Mat2x2<$mtype>, *,
        mat_mulx3, $mtype, $Mat1x2<$mtype>, $Mat2x3<$mtype>, *,
        mat_mulx4, $mtype, $Mat1x2<$mtype>, $Mat2x4<$mtype>, *,

        vec_mul, $mtype, $Mat1x2<$mtype>, RhsVec<$mtype>, *,
        mul_vec, $mtype, LhsVec<$mtype>, $Mat1x2<$mtype>, *,
    }
}
    )+};
}

macro_rules! bench_binop_mat_types {
    ($($f32:ident)+) => {$(
mod $f32 {
    #[cfg(not(feature="quick_bench"))]
    bench_binop_mat!{
        $f32, mat1x1_bench, Mat1,   Mat1x1, Mat1x2, Mat1x3, Mat1x4, Vec1, Vec1,
        $f32, mat1x2_bench, Mat1x2, Mat2x1, Mat2x2, Mat2x3, Mat2x4, Vec1, Vec2,
        $f32, mat1x3_bench, Mat1x3, Mat3x1, Mat3x2, Mat3x3, Mat3x4, Vec1, Vec3,
        $f32, mat1x4_bench, Mat1x4, Mat4x1, Mat4x2, Mat4x3, Mat4x4, Vec1, Vec4,

        $f32, mat2x1_bench, Mat2x1, Mat1x1, Mat1x2, Mat1x3, Mat1x4, Vec2, Vec1,
    }
    bench_binop_mat!{
        $f32, mat2x2_bench, Mat2,   Mat2x1, Mat2x2, Mat2x3, Mat2x4, Vec2, Vec2,
        $f32, mat2x3_bench, Mat2x3, Mat3x1, Mat3x2, Mat3x3, Mat3x4, Vec2, Vec3,
        $f32, mat2x4_bench, Mat2x4, Mat4x1, Mat4x2, Mat4x3, Mat4x4, Vec2, Vec4,

        $f32, mat3x1_bench, Mat3x1, Mat1x1, Mat1x2, Mat1x3, Mat1x4, Vec3, Vec1,
        $f32, mat3x2_bench, Mat3x2, Mat2x1, Mat2x2, Mat2x3, Mat2x4, Vec3, Vec2,
        $f32, mat3x3_bench, Mat3,   Mat3x1, Mat3x2, Mat3x3, Mat3x4, Vec3, Vec3,
        $f32, mat3x4_bench, Mat3x4, Mat4x1, Mat4x2, Mat4x3, Mat4x4, Vec3, Vec4,

        $f32, mat4x1_bench, Mat4x1, Mat1x1, Mat1x2, Mat1x3, Mat1x4, Vec4, Vec1,
        $f32, mat4x2_bench, Mat4x2, Mat2x1, Mat2x2, Mat2x3, Mat2x4, Vec4, Vec2,
        $f32, mat4x3_bench, Mat4x3, Mat3x1, Mat3x2, Mat3x3, Mat3x4, Vec4, Vec3,
        $f32, mat4x4_bench, Mat4,   Mat4x1, Mat4x2, Mat4x3, Mat4x4, Vec4, Vec4,
    }
}
    )+};
}

bench_binop_mat_types!{
    f32 f64
}
#[cfg(not(feature="quick_bench"))]
bench_binop_mat_types!{
    i8 i16 i32 i64
    u8 u16 u32 u64
}
