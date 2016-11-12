

macro_rules! bench_binop {
    ($($name:ident, $mtype:ty, $t1:ty, $t2:ty, $binop: tt,)+) => {
        bench_binop!{$($name, $mtype, $t1, $t2, $binop),+}
    };
    ($($name:ident, $mtype:ty, $t1:ty, $t2:ty, $binop:tt),+) => {$(
#[cfg(feature="rand")]
#[bench]
fn $name(b: &mut Bencher) {
    const LEN: usize = 10;

    let v1: [$t1; LEN] = test::black_box(rand::random());
    let mut v2: [$t2; LEN] = test::black_box(rand::random());
    for i in 0..LEN { v2[i] += v(1 as $mtype); }
    let v2 = v2;

    b.iter(|| {
        for i in 0..LEN {
            test::black_box(v1[i] $binop v2[i]);
        }
    });
}
    )+};
}

macro_rules! bench_unop {
    ($name: ident, $t1: ty, $unop: ident) => {
#[bench]
fn $name(b: &mut Bencher) {
    let elems1: $t1 = test::black_box(rand::random());

    b.iter(|| test::black_box(v1.$unop()));
}
    };
}
