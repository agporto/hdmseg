//! Vectorizable elementwise `exp` for non-positive arguments.
//!
//! Derived from `rustcpd/src/fastexp.rs`; see `THIRD_PARTY.md`.
//!
//! The Gaussian kernel construction spends most of its time in `exp`. This
//! module evaluates the Cephes rational approximation (the same one used by
//! many libm implementations; ≤ 2 ulp over the relevant range) in a form
//! LLVM auto-vectorizes. On x86-64 the vector path requires AVX2+FMA and is
//! selected at runtime with the same scalar approximation as fallback, so
//! every supported CPU follows the same kernel and cutoff.
//! always produces identical results run-over-run.
//!
//! Arguments are exponents of Gaussian weights, hence always ≤ 0. Inputs
//! below −706 return exactly 0 instead of a subnormal; those weights are
//! smaller than 1e−300 and contribute nothing to any accumulated statistic.

/// Cephes `exp` coefficients (Moshier, public domain).
const C1: f64 = 6.931_457_519_531_25e-1;
const C2: f64 = 1.428_606_820_309_417_2e-6;
const P0: f64 = 1.261_771_930_748_105_9e-4;
const P1: f64 = 3.029_944_077_074_419_6e-2;
const P2: f64 = 1.0; // Cephes P2 rounds to exactly 1.0 in f64.
const Q0: f64 = 3.001_985_051_386_644_6e-6;
const Q1: f64 = 2.524_483_403_496_841e-3;
const Q2: f64 = 2.272_655_482_081_550_3e-1;
const Q3: f64 = 2.0;

#[inline(always)]
fn exp_body(values: &mut [f64]) {
    for value in values {
        let x = *value;
        let n = f64::mul_add(std::f64::consts::LOG2_E, x, 0.5).floor();
        let r = f64::mul_add(-n, C2, f64::mul_add(-n, C1, x));
        let rr = r * r;
        let p = r * f64::mul_add(f64::mul_add(P0, rr, P1), rr, P2);
        let q = f64::mul_add(f64::mul_add(f64::mul_add(Q0, rr, Q1), rr, Q2), rr, Q3);
        let e = 1.0 + 2.0 * p / (q - p);
        let scale = f64::from_bits(((n as i64 + 1023) as u64) << 52);
        *value = if x < -706.0 { 0.0 } else { e * scale };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn exp_avx2(values: &mut [f64]) {
    exp_body(values);
}

/// Replace every element `x ≤ 0` of `values` with `exp(x)`.
pub(crate) fn exp_non_positive(values: &mut [f64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the required target features were just detected.
            return unsafe { exp_avx2(values) };
        }
        exp_body(values);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // FMA is baseline on aarch64 and most other modern targets.
        exp_body(values);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn matches_libm_to_two_ulp() {
        let mut worst = 0.0_f64;
        let mut x = -750.0_f64;
        while x <= 0.0 {
            let mut buffer = [x];
            super::exp_non_positive(&mut buffer);
            let reference = x.exp();
            if reference > 1e-300 {
                worst = worst.max(((buffer[0] - reference) / reference).abs());
            }
            x += 1.03e-3;
        }
        assert!(worst <= 2.0 * f64::EPSILON, "worst relative error {worst}");
    }
}
