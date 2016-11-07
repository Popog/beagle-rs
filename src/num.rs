//! A collection of numeric traits and functions.

use std::ops::Mul;
use std::cmp;
use std::cmp::Ordering;
use std::num::{FpCategory,Wrapping};
use std::mem::transmute;

/// Types that can have their absolute value taken
pub trait Abs<Rhs = Self> {
    /// The resulting type.
    type Output;

    /// Computes the absolute value of self. Returns NaN if the number is NaN.
    fn abs(self) -> Self::Output;

    /// Returns |self-rhs| without modulo overflow.
    fn abs_diff(self, rhs: Rhs) -> Self::Output;
}

/// Types that can be used with compared to be approximate equal.
pub trait Approx<T> {
    /// Returns true if `lhs` is approximately `rhs`.
    fn approx(&self, lhs: T, rhs: T) -> bool;
}

/// Types that can be used to compare to zero efficiently.
pub trait ApproxZero<T> {
    /// Returns true if `a` is approximately zero.
    fn approx_zero(&self, a: T) -> bool;

    /// Returns true if `a/b` is approximately zero.
    fn approx_zero_ratio(&self, a: T, b: T) -> bool;
}

/// Types that can be clamped
pub trait Clamp<Rhs = Self>: Sized {
    /// The output type
    type Output;
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn min(self, rhs: Rhs) -> Self::Output;
    /// Returns the minimum of the two numbers.
    /// If one of the arguments is NaN, then the other argument is returned.
    fn max(self, rhs: Rhs) -> Self::Output;
    /// Returns the value of `self` constrained to the range [`min_val`, `max_val`]
    /// The returned value is computed as min(max(self, `min_val`), `mal_val`).
    fn clamp(self, min: Rhs, max: Rhs) -> Self::Output;
}

/// Types that implement exponential functions
pub trait Exp {
    /// The output type
    type Output;
    /// Returns `e^(self)`, (the exponential function).
    fn exp(self) -> Self::Output;
    /// Returns `2^(self)`.
    fn exp2(self) -> Self::Output;
    /// Returns the natural logarithm of the number.
    fn ln(self) -> Self::Output;
    /// Returns the base 2 logarithm of the number.
    fn log2(self) -> Self::Output;
    /// Returns the base 10 logarithm of the number.
    fn log10(self) -> Self::Output;
}

/// Types that can be categorized via `FpCategory`
pub trait FloatCategory {
    /// The `bool` output type
    type Bool;
    /// The `FpCategory` output type
    type FpCategory;
    /// Returns `true` if this value is `NaN` and false otherwise.
    fn is_nan(self) -> Self::Bool;
    /// Returns `true` if this value is positive infinity or negative infinity and
    /// false otherwise.
    fn is_infinite(self) -> Self::Bool;
    /// Returns `true` if this number is neither infinite nor `NaN`.
    fn is_finite(self) -> Self::Bool;
    /// Returns `true` if the number is neither zero, infinite,
    /// [subnormal][subnormal], or `NaN`.
    fn is_normal(self) -> Self::Bool;
    /// Returns the floating point category of the number. If only one property
    /// is going to be tested, it is generally faster to use the specific
    /// predicate instead.
    fn classify(self) -> Self::FpCategory;
}

/// Types that can be transmuted either from a float to an integer  representation of the bits, or
/// vice-versa
pub trait FloatTransmute {
    /// The type after transmuting
    type Output;
    /// Decodes a float into its bit representation
    #[inline(always)]
    fn float_transmute(self) -> Self::Output;
}

/// Types that can be broken into a fractional component and an exponent
pub trait FractionExponent {
    /// The floating point fractional component type
    type Fraction;
    /// The integral exponent component type
    type Exponent;
    /// Extracts a floating point significand in the range [`0.5`, `1.0`) and an integer such that:
    /// `self = Fraction * 2^Exponent`
    fn frexp(self) -> (Self::Fraction, Self::Exponent);
}

/// Types that implment hyperbolic angle functions.
pub trait Hyperbolic {
    /// The output type
    type Output;
    /// Hyperbolic sine function.
    fn sinh(self) -> Self::Output;
    /// Hyperbolic cosine function.
    fn cosh(self) -> Self::Output;
    /// Hyperbolic tangent function.
    fn tanh(self) -> Self::Output;
    /// Hyperbolic sine function.
    fn asinh(self) -> Self::Output;
    /// Hyperbolic cosine function.
    fn acosh(self) -> Self::Output;
    /// Hyperbolic tangent function.
    fn atanh(self) -> Self::Output;
}

/// Types that can be used to load the exponent of a floating point number
pub trait LoadExponent<Exponent> {
    /// The output type
    type Output;
    /// Creates a floating point number equal to `self * 2^exp`
    fn ldexp(self, exponent: Exponent) -> Self::Output;
}

/// Types which can be mixed
pub trait Mix<A = Self, Rhs = Self> {
    /// The output type
    type Output;
    /// Performs a linear interpolation between a number and y using `a` to weight between them.
    /// For non-boolean `a`, the return value is computed as `self*(1-a) + y*a*self*(1-a) + y*a`.
    /// For boolean `a`, the return value is composed of selected components of `x` and `y`.
    /// False components of `a` select the corresponding component of `x`. True componenets
    /// of `a` select the corresponding components of 'y'.
    fn mix(self, y: Rhs, a: A) -> Self::Output;
}

/// Types that implement fused multiply-add
pub trait MulAdd<A = Self, B = A> {
    /// The output type
    type Output;
    /// Fused multiply-add. Computes `(self * a) + b` with only one rounding
    /// error. This produces a more accurate result than a separate multiplication operation
    /// followed by an add. Usually it's a little slower, depending on hardware.
    fn mul_add(self, a: A, b: B) -> Self::Output;
}

/// Types that implement the Pow function
pub trait Pow<Rhs = Self> {
    /// The output type
    type Output;
    /// Returns `self` raised to the power `rhs`
    fn pow(self, rhs: Rhs) -> Self::Output;
}

/// Types that can have the reciprocal taken.
pub trait Recip {
    /// The output type
    type Output;
    /// Takes the reciprocal (inverse) of a number, `1/x`.
    fn recip(self) -> Self::Output;
}

/// Types that can be rounded.
pub trait Round {
    /// The output type
    type Output;
    /// Returns the largest integer less than or equal to a number.
    fn floor(self) -> Self::Output;
    /// Returns the smallest integer greater than or equal to a number.
    fn ceil(self) -> Self::Output;
    /// Returns the integer part of a number.
    fn trunc(self) -> Self::Output;
    /// Returns the fractional part of a number.
    fn fract(self) -> Self::Output;
    /// Returns the nearest integer to a number. Round half-way cases away from 0.0.
    fn round(self) -> Self::Output;
}

/// Types that can be sign functions.
pub trait Sign {
    /// The resulting boolean type.
    type Output;

    /// Returns `true` if `self` is strictly less than zero.
    fn is_negative(&self) -> Self::Output;
    /// Returns `true` if `self` is strictly greater than zero.
    fn is_positive(&self) -> Self::Output;

    /// Returns `true` if `self`'s sign is negative (including `-0`).
    fn is_sign_negative(&self) -> Self::Output;
    /// Returns `true` if `self`'s sign is positive (including `+0`).
    fn is_sign_positive(&self) -> Self::Output;

    /// Returns a number that represents the sign of self.
    /// NaN if `self` is NaN
    /// One if the `self.is_sign_positive()`
    /// Negative one if `self.is_sign_negative()`
    fn signum(self) -> Self;

    /// Returns a number that represents the sign of self.
    /// NaN if `self` is NaN
    /// One if the `self.is_positive()`
    /// Negative one if `self.is_negative()`
    /// else Zero
    fn sign(self) -> Self;
}

/// Types that can be square-rooted.
pub trait Sqrt {
    /// The output type
    type Output;

    /// Takes the square root of a number.
    ///
    /// Returns NaN if `self` is a negative number.
    fn sqrt(self) -> Self::Output;

    /// Returns the inverse of the square root of a number. i.e. the value `1/√x`.
    ///
    /// Returns NaN if `self` is a negative number.
    ///
    /// ```
    /// use beagle::num::Sqrt;
    ///
    /// let positive = 4.0_f32;
    /// let negative = -4.0_f32;
    /// let inf = std::f32::INFINITY;
    /// let nan = std::f32::NAN;
    /// let zero = 0f32;
    /// let negzero = -0f32;
    ///
    /// assert!((positive.inverse_sqrt() - 0.5).abs() < 1e-3);
    /// assert!(negative.inverse_sqrt().is_nan());
    /// assert!(nan.inverse_sqrt().is_nan());
    /// assert_eq!(inf.inverse_sqrt(), 0f32);
    /// assert_eq!(zero.inverse_sqrt(), std::f32::INFINITY);
    /// assert_eq!(negzero.inverse_sqrt(), std::f32::INFINITY);
    /// ```
    fn inverse_sqrt(self) -> Self::Output;
}

/// Types that can generate a step function by comparing two values
pub trait Step<Edge0 = Self, Edge1 = Edge0> {
    /// The output type
    type Output;
    // Returns `0.0` if `self < edge`, else `1.0`
    fn step(self, edge: Edge0) -> Self::Output;
    /// Performs smooth Hermite interpolation between 0 and 1 when `edge0 < self < edge1`.
    /// smoothstep is equivalent to:
    /// ```
    /// t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    /// return t * t * (3.0 - 2.0 * t);
    /// ```
    /// Results are undefined if `edge0 ≥ edge1`.
    fn smoothstep(self, edge0: Edge0, edge1: Edge1) -> Self::Output;
}



macro_rules! float_impl {
    ($($float:ident=($bits:expr,$int:ident, $uint:ident)),+) => {$(
impl Abs for $float {
    type Output = $float;
    fn abs(self) -> $float { $float::abs(self) }
    fn abs_diff(self, rhs: $float) -> $float { $float::abs(self - rhs) }
}

impl Clamp for $float {
    type Output = $float;
    fn min(self, rhs: $float) -> $float { $float::min(self, rhs) }
    fn max(self, rhs: $float) -> $float { $float::max(self, rhs) }
    fn clamp(self, min: $float, max: $float) -> $float { $float::min($float::max(self, max), min) }
}

impl Exp for $float {
    type Output = $float;
    fn exp(self) -> $float { $float::exp(self) }
    fn exp2(self) -> $float { $float::exp2(self) }
    fn ln(self) -> $float { $float::ln(self) }
    fn log2(self) -> $float { $float::log2(self) }
    fn log10(self) -> $float { $float::log10(self) }
}

impl FloatCategory for $float {
    type Bool = bool;
    type FpCategory = FpCategory;
    fn is_nan(self) -> bool { $float::is_nan(self) }
    fn is_infinite(self) -> bool { $float::is_infinite(self) }
    fn is_finite(self) -> bool { $float::is_finite(self) }
    fn is_normal(self) -> bool { $float::is_normal(self) }
    fn classify(self) -> FpCategory { $float::classify(self) }
}

impl FloatTransmute for $float {
    type Output = $uint;
    #[inline(always)]
    fn float_transmute(self) -> $uint { unsafe { transmute(self) } }
}

impl FloatTransmute for $uint {
    type Output = $float;
    #[inline(always)]
    fn float_transmute(self) -> $float { unsafe { transmute(self) } }
}

impl FractionExponent for $float {
    type Fraction = $float;
    type Exponent = $int;
    fn frexp(self) -> ($float, $int) {
        use std::$float;
        if self == 0.0 { return (self, 0); }

        let v = self.float_transmute();
        // The number of bits in the mantissa
        const MANTISSA_BITS: $uint = $float::MANTISSA_DIGITS as $uint - 1;
        // The number of bits outside the mantissa
        const SIGN_EXPONENT_BITS: $uint = $bits - MANTISSA_BITS;
        /// The number of digits in the exponent
        const EXPONENT_BITS: $uint = SIGN_EXPONENT_BITS - 1;

        // Get the exponent
        let mask = (1 << EXPONENT_BITS) - 1;
        let mask = mask << MANTISSA_BITS;
        let exponent = v & mask;

        if exponent == 0 {
            // Offset the exponent properly
            let exponent = $float::MIN_EXP as $int;

            // Save the sign bit and clear it
            let mask = 1 << ($bits - 1);
            let sign = v & mask;
            let v = v & !mask;

            // Determine how many doublings it would take for the mantissa to reach the exponent
            let leading_zeros = v.leading_zeros() as $uint - EXPONENT_BITS;

            // Decrementing the exponent by the number of doublings necessary
            let exponent = exponent - leading_zeros as $int;

            // Normalize v by by doubling
            let v = (v << leading_zeros) | sign;

            // Set the exponent to -1
            let mask = ($float::MAX_EXP as $uint - 1) << MANTISSA_BITS;
            let v = v ^ mask;

            (v.float_transmute(), exponent)
        } else if exponent == mask {
            (v.float_transmute(), 0)
        } else {
            // Shift the exponent down
            let exponent = exponent as $int >> MANTISSA_BITS;

            // Offset the exponent properly
            let exponent = exponent + $float::MIN_EXP as $int - 1;

            // Clear the exponent and set it to -1
            let v = v & !mask;

            let mask = ($float::MAX_EXP as $uint - 2) << MANTISSA_BITS;
            let v = v | mask;

            (v.float_transmute(), exponent)
        }
    }
}

impl Hyperbolic for $float {
    type Output = $float;
    fn sinh(self) -> $float { $float::sinh(self) }
    fn cosh(self) -> $float { $float::cosh(self) }
    fn tanh(self) -> $float { $float::tanh(self) }
    fn asinh(self) -> $float { $float::asinh(self) }
    fn acosh(self) -> $float { $float::acosh(self) }
    fn atanh(self) -> $float { $float::atanh(self) }
}

impl LoadExponent<$int> for $float {
    type Output = $float;
    fn ldexp(self, exponent: $int) -> $float {
        use std::$float;
        if self == 0.0  { return self; }

        let v = self.float_transmute();
        // The number of bits in the mantissa
        const MANTISSA_BITS: $uint = $float::MANTISSA_DIGITS as $uint - 1;
        // The number of bits outside the mantissa
        const SIGN_EXPONENT_BITS: $uint = $bits - MANTISSA_BITS;
        /// The number of digits in the exponent
        const EXPONENT_BITS: $uint = SIGN_EXPONENT_BITS - 1;
        /// The mask for the exponent bits
        const EXPONENT_MASK: $uint = (1 << EXPONENT_BITS) - 1;
        /// The mask for the exponent bits
        const MANTISSA_MASK: $uint = (1 << MANTISSA_BITS) - 1;

        // Get the exponent
        let mask = EXPONENT_MASK << MANTISSA_BITS;
        let v_exponent = v & mask;

        if v_exponent == 0 {
            if exponent > 0 {
                // Exponent is positive
                let exponent = exponent as $uint;

                // Save the sign bit and clear it
                let mask = 1 << ($bits - 1);
                let sign = v & mask;
                let v = v & !mask;

                // Determine how many doublings it would take for the mantissa to reach the exponent
                let leading_zeros = v.leading_zeros() as $uint - EXPONENT_BITS;

                let v = if leading_zeros < exponent {
                    // Normalize v by by doubling
                    let v = v << leading_zeros;

                    // Decrementing the exponent by the number of doublings necessary
                    let exponent = exponent - leading_zeros + 1;

                    // Set the exponent, but clamp to infinity
                    if exponent >= EXPONENT_MASK {
                        $float::INFINITY.float_transmute()
                    } else {
                        let mask = (exponent ^ 1) << MANTISSA_BITS;
                        v ^ mask
                    }
                } else {
                    // Normalize v by by doubling
                    v << exponent
                };

                // Restore the sign bit
                let v = v | sign;

                v.float_transmute()
            } else if exponent < 0 {
                // Set the exponent adjustment to a positive number less than $bits
                let exponent = cmp::min(-exponent as $uint, $bits - 1);

                // Save the sign bit and clear it
                let mask = 1 << (32 - 1);
                let sign = v & mask;
                let v = v & !mask;

                // TODO: enable other rounding modes blocked by rust-lang/rust#10186
                // Compute the rounding adjustment
                let adjustment = // match get_rounding mode {
                    // FE_TONEAREST =>
                    {
                        let adjustment = (1 << exponent) - 1;
                        let adjustment = v & adjustment;
                        if adjustment << 1 > 1 << exponent { 1 }
                        else if adjustment << 1 == 1 << exponent { (v >> exponent) & 1 }
                        else { 0 }
                    };
                    // FE_TOWARDZERO => 0,
                //}

                // Finish shrinking and apply the adjusting
                let v = (v >> exponent) + adjustment;

                // Restore the sign bit
                let v = v | sign;

                v.float_transmute()
            } else {
                v.float_transmute()
            }
        } else if v_exponent == mask {
            v.float_transmute()
        } else {
            // Shift the exponent down
            let v_exponent = v_exponent as $int >> MANTISSA_BITS;

            // Calculate the new exponent
            let exponent = v_exponent + exponent;

            // Determine if the result is subnormal or overflow
            if exponent >= (1 << EXPONENT_BITS) - 1 {
                // Set the exponent to all ones
                let v = v | mask;

                // Clear the mantissa
                let v = v & !MANTISSA_MASK;

                v.float_transmute()
            } else if exponent > 0 {
                // Shift the new exponent to the correct position
                let exponent = (exponent as $uint) << MANTISSA_BITS;

                // Clear the exponent
                let v = v & !mask;

                // Set the exponent to the new value
                let v = v | exponent;

                v.float_transmute()
            } else {
                // Set the exponent adjustment to a positive number less than $bits
                let exponent = cmp::min(-exponent as $uint, $bits-2);

                // Add 1 to shift the 1 out of the exponent bits
                let exponent = exponent + 1;

                // Save the sign bit
                let mask = 1 << ($bits - 1);
                let sign = v & mask;

                // Clear the exponent and sign bit
                let v = v & MANTISSA_MASK;

                // Set the lowest bit of the exponent for use in subnormal
                let v = v | (1 << MANTISSA_BITS);

                // TODO: enable other rounding modes blocked by rust-lang/rust#10186
                // Compute the rounding adjustment
                let adjustment = // match get_rounding mode {
                    // FE_TONEAREST =>
                    {
                        let adjustment = (1 << exponent) - 1;
                        let adjustment = v & adjustment;
                        if adjustment << 1 > 1 << exponent { 1 }
                        else if adjustment << 1 == 1 << exponent { (v >> exponent) & 1 }
                        else { 0 }
                    };
                    // FE_TOWARDZERO => 0,
                //}

                // Finish shrinking and apply adjustment
                let v = (v >> exponent) + adjustment;

                // Restore the sign bit
                let v = v | sign;

                v.float_transmute()
            }
        }
    }
}

impl Mix for $float {
    type Output = $float;
    fn mix(self, y: $float, a: $float) -> $float {
        self*(1.0-a) + y*a*self*(1.0-a) + y*a
    }
}

impl Mix<bool> for $float {
    type Output = $float;
    fn mix(self, y: $float, a: bool) -> $float {
        if a { y } else { self }
    }
}

impl MulAdd for $float {
    type Output = $float;
    fn mul_add(self, a: $float, b: $float) -> $float { $float::mul_add(self, a, b) }
}

impl Pow for $float {
    type Output = $float;
    fn pow(self, rhs: $float) -> $float { $float::powf(self, rhs) }
}

impl Pow<i32> for $float {
    type Output = $float;
    fn pow(self, rhs: i32) -> $float { $float::powi(self, rhs) }
}

impl Recip for $float {
    type Output = $float;
    fn recip(self) -> $float { $float::recip(self) }
}

impl Round for $float {
    type Output = $float;
    fn floor(self) -> $float { $float::floor(self) }
    fn ceil(self) -> $float { $float::ceil(self) }
    fn trunc(self) -> $float { $float::trunc(self) }
    fn fract(self) -> $float { $float::fract(self) }
    fn round(self) -> $float { $float::round(self) }
}

impl Sign for $float {
    type Output = bool;

    fn is_negative(&self) -> bool { self.lt(&0.0) }
    fn is_positive(&self) -> bool { self.gt(&0.0) }

    fn is_sign_negative(&self) -> bool { $float::is_sign_negative(*self) }
    fn is_sign_positive(&self) -> bool { $float::is_sign_positive(*self) }

    fn signum(self) -> $float { $float::signum(self) }

    fn sign(self) -> $float {
        match self.partial_cmp(&0.0) {
            Some(Ordering::Less) => -1.0,
            Some(Ordering::Greater) => 1.0,
            _ => self,
        }
    }
}

impl Sqrt for $float {
    type Output = $float;

    fn sqrt(self) -> Self { $float::sqrt(self) }

    fn inverse_sqrt(self) -> Self {
        use std::$float;

        // Turns out, this already handles NaN input.
        if self < 0.0 { return $float::NAN; }
        if self == 0.0 { return $float::INFINITY; }
        if self == $float::INFINITY { return 0.0; }

        let y = Wrapping(self.float_transmute());   // evil floating point bit level hacking
        let y = Wrapping(0x5f375a86) - ( y >> 1 );  // what the fuck?
        let y = y.0.float_transmute();
        let x2 = self * 0.5;
        // Assign to y and repeat for a more precise result
        y * ( 1.5 - ( x2 * y * y ) )
    }
}

impl Step for $float {
    type Output = $float;
    fn step(self, edge: $float) -> $float {
        if self < edge { 0.0 } else { 1.0 }
    }
    fn smoothstep(self, edge0: $float, edge1: $float) -> $float {
        let t = Clamp::clamp((self - edge0) / (edge1 - edge0), 0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }
}
    )+};
}

macro_rules! int_impl {
    ($($int:ident)+) => {$(
impl Clamp for $int {
    type Output = $int;
    fn min(self, rhs: $int) -> $int { cmp::min(self, rhs) }
    fn max(self, rhs: $int) -> $int { cmp::max(self, rhs) }
    fn clamp(self, min: $int, max: $int) -> $int { cmp::min(cmp::max(self, max), min) }
}

impl Mix<bool> for $int {
    type Output = $int;
    fn mix(self, y: $int, a: bool) -> $int {
        if a { y } else { self }
    }
}
    )+};
}

float_impl!{f32=(32,i32,u32),f64=(64,i64,u64)}
int_impl!{i8 i16 i32 i64 u8 u16 u32 u64}

impl Mix<bool> for bool {
    type Output = bool;
    fn mix(self, y: bool, a: bool) -> bool {
        if a { y } else { self }
    }
}

/// An approximator using simple episilon.
pub struct AbsoluteEpsilon<F>(pub F);

impl <T: Copy+Mul> AbsoluteEpsilon<T> {
    /// Tighten approximation so that `self.approx_zero(x) == self.squared().approx_zero(x*x)`.
    pub fn squared(&self) -> AbsoluteEpsilon<T::Output> { AbsoluteEpsilon(self.0*self.0) }
}

impl <T: Abs> Approx<T> for AbsoluteEpsilon<<T as Abs>::Output>
where <T as Abs>::Output: Copy+PartialOrd {
    fn approx(&self, lhs: T, rhs: T) -> bool {
        Abs::abs_diff(lhs, rhs) <= self.0
    }
}

impl <T: Copy+Abs> ApproxZero<T> for AbsoluteEpsilon<<T as Abs>::Output>
where <T as Abs>::Output: Copy+PartialOrd+Mul<Output=<T as Abs>::Output> {
    fn approx_zero(&self, a: T) -> bool {
        Abs::abs(a) <= self.0
    }
    fn approx_zero_ratio(&self, a: T, b: T) -> bool {
        Abs::abs(a) <= Abs::abs(b) * self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(deprecated)]
    #[test]
    fn test_frexp() {
        use std::f64;
        let f = 45.0;
        assert_eq!(FractionExponent::frexp(f).0, f64::frexp(f).0);
        assert_eq!(FractionExponent::frexp(f).1, f64::frexp(f).1 as i64);
    }

    #[allow(deprecated)]
    #[test]
    fn test_ldexp() {
        let f = 45.0;
        let r = FractionExponent::frexp(45.0);
        let r = LoadExponent::ldexp(r.0, r.1);
        assert_eq!(r, f);
    }

    #[ignore]
    #[allow(deprecated)]
    #[test]
    // Exhaustively testing would take 10k years
    // Test all 32-bit floats, but cast them to f64 to exercise those functions
    fn extensive_frexp64() {
        use std::f64;

        let mut i = 0u32;
        loop {
            i = i.wrapping_add(1);
            if i == 0 { break; }
            let j = i.float_transmute();
            let r1 = FractionExponent::frexp(j as f64);
            let r2 = f64::frexp(j as f64);
            let match_float = (r1.0.is_nan() && r2.0.is_nan()) || r1.0 == r2.0;
            assert!(match_float && r1.1 == r2.1 as i64, "mismatch {:?} != {:?}", r1, r2);
        }
    }

    #[ignore]
    #[allow(deprecated)]
    #[test]
    // Exhaustively test all floating point numbers against the deprecated std::f32
    fn exhaustive_frexp32() {
        use std::f32;

        let mut i = 0u32;
        loop {
            i = i.wrapping_add(1);
            if i == 0 { break; }
            let j = i.float_transmute();
            let r1 = FractionExponent::frexp(j);
            let r2 = f32::frexp(j);
            let match_float = (r1.0.is_nan() && r2.0.is_nan()) || r1.0 == r2.0;
            assert!(match_float && r1.1 == r2.1 as i32, "mismatch {:?} != {:?}", r1, r2);
        }
    }

    #[ignore]
    #[allow(deprecated)]
    #[test]
    // Exhaustively testing would take 10k years
    // Test all that it inverses properly,
    // Test all values for -5 to 5, and some select larger values.
    fn extensive_ldexp32() {
        use std::f32;

        let mut i = 0u32;
        for exp in [-200,-50,50,200].iter().map(|&v| v).chain(-5..5) {
            loop {
                i = i.wrapping_add(1);
                if i == 0 { break; }
                let j = i.float_transmute();
                let r1 = f32::ldexp(j, exp as isize);
                let r2 = LoadExponent::ldexp(j, exp);

                let match_float = (r1.is_nan() && r2.is_nan()) || r1 == r2;
                assert!(match_float, "mismatch {:?} != {:?}\t{}\t{}", r1, r2, i, exp);
            }
        }
    }

    #[ignore]
    #[allow(deprecated)]
    #[test]
    // Exhaustively test all outputs of frexp
    fn exhaustive_ldexp32_inverse() {
        let mut i = 0u32;
        loop {
            i = i.wrapping_add(1);
            if i == 0 { break; }
            let j = i.float_transmute();
            let r1 = FractionExponent::frexp(j);
            let j2 = LoadExponent::ldexp(r1.0, r1.1);

            let match_float = (j.is_nan() && j2.is_nan()) || j == j2;
            assert!(match_float, "mismatch {:?} != {:?}", j, j2);
        }
    }

    #[ignore]
    #[test]
    // Exhaustively test all NaN and negative numbers
    fn exhaustive_inverse_sqrt_nan() {
        use std::f32;

        let mut nanu = f32::INFINITY.float_transmute();
        loop {
             nanu = nanu.wrapping_add(1);
             if nanu == 0 { break; }
             let nan: f32 = nanu.float_transmute();
             if nan == 0.0 { continue; }
             assert!(nan.inverse_sqrt().is_nan(), "inverse_sqrt was not nan for {} ({})", nan, nanu);
        }
    }
}
