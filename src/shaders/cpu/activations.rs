//! CPU twins for `shaders/include/activations.metal`.

const GELU_TANH_COEF: f32 = 0.797_884_6;

/// Mirror of `gelu_tanh` in `include/activations.metal`.
pub fn gelu_tanh(x: f32) -> f32 {
    let x3 = x * x * x;
    let u = GELU_TANH_COEF * (x + 0.044_715 * x3);
    let t = if u > 8.0 {
        1.0
    } else if u < -8.0 {
        -1.0
    } else {
        u.tanh()
    };
    0.5 * x * (1.0 + t)
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn swiglu_gelu(gate: f32, up: f32) -> f32 {
    gelu_tanh(gate) * up
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_monotonic_near_origin() {
        assert!(gelu_tanh(0.0).abs() < 1e-6);
        assert!(gelu_tanh(1.0) > gelu_tanh(0.0));
        assert!(gelu_tanh(-1.0) < gelu_tanh(0.0));
    }

    #[test]
    fn silu_swiglu_finite() {
        let g = silu(0.5);
        assert!(g.is_finite());
        assert!(swiglu_gelu(0.5, 1.0).is_finite());
    }
}
