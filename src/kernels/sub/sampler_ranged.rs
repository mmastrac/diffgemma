//! Ranged elementwise sampler kernels: copy_f32, logit_softcapping, scale_logits.

use super::gpu_common;
use super::test_util::ElemFormat;
use crate::model::embed::logit_softcapping;
use crate::safetensors::Error;

pub const COPY_ENTRY: &str = "copy_f32";
pub const SOFTCAP_ENTRY: &str = "logit_softcapping";
pub const SCALE_ENTRY: &str = "scale_logits";

const COPY_SHADER: &str = shader_include::include_metal!("kernels/copy_f32.metal");
const SOFTCAP_SHADER: &str = shader_include::include_metal!("kernels/logit_softcapping.metal");
const SCALE_SHADER: &str = shader_include::include_metal!("kernels/scale_logits.metal");

/// Production canvas × vocab flat length (forces multi-chunk ranged dispatch).
pub const CANVAS_LEN: usize = 256;
pub const VOCAB: usize = 262_144;
pub const CANVAS_VOCAB_LEN: usize = CANVAS_LEN * VOCAB;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangedOp {
    Copy,
    Softcap,
    Scale,
}

#[derive(Debug, Clone)]
pub struct Fixture {
    pub src: Vec<f32>,
    pub cap: f32,
    pub inv_t: f32,
    pub op: RangedOp,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.src.len()
    }
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.len()
}

fn fill_logits(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * seed).sin() * 1.7 + ((i as f32) * seed * 0.31).cos() * 0.9)
        .collect()
}

pub fn tiny_softcap_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        src: vec![-40.0, -5.0, 0.0, 3.0, 40.0],
        cap: 30.0,
        inv_t: 1.0,
        op: RangedOp::Softcap,
    }
}

pub fn tiny_scale_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        src: vec![1.0, -2.0, 4.0, 0.0],
        cap: 30.0,
        inv_t: 0.65,
        op: RangedOp::Scale,
    }
}

pub fn tiny_copy_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        src: vec![0.25, -1.5, 2.0, 7.0, -3.0],
        cap: 30.0,
        inv_t: 1.0,
        op: RangedOp::Copy,
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    match f.op {
        RangedOp::Copy => f.src.clone(),
        RangedOp::Softcap => {
            let mut out = f.src.clone();
            logit_softcapping(&mut out, f.cap);
            out
        }
        RangedOp::Scale => f.src.iter().map(|v| v * f.inv_t).collect(),
    }
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

fn shader_and_entry(op: RangedOp) -> (&'static str, &'static str) {
    match op {
        RangedOp::Copy => (COPY_SHADER, COPY_ENTRY),
        RangedOp::Softcap => (SOFTCAP_SHADER, SOFTCAP_ENTRY),
        RangedOp::Scale => (SCALE_SHADER, SCALE_ENTRY),
    }
}

#[cfg(target_os = "macos")]
pub fn gpu(f: &Fixture) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use objc2_metal::MTLComputeCommandEncoder;

    let (shader, entry) = shader_and_entry(f.op);
    let ctx = MetalContext::new()?;
    let pipeline = ctx.compile_kernel(shader, entry)?;
    let mut pool = BufferPool::new();
    let len = f.len();
    let bytes = len * 4;

    let buf = pool
        .allocate(&ctx.device, bytes)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_f32(&buf, &f.src);

    let buf_dst = if f.op == RangedOp::Copy {
        let b = pool
            .allocate(&ctx.device, bytes)
            .ok_or(Error::Format("alloc"))?;
        Some(b)
    } else {
        None
    };

    gpu_common::dispatch_1d_ranged(&ctx.queue, &pipeline.pipeline, len, |enc, base, chunk| {
        let range = [base, chunk];
        match f.op {
            RangedOp::Copy => {
                let dst = buf_dst.as_ref().expect("dst");
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(dst), 0, 1);
                }
                gpu_common::set_bytes(enc, &range, 2);
            }
            RangedOp::Softcap => {
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf), 0, 0);
                }
                gpu_common::set_bytes(enc, &f.cap, 1);
                gpu_common::set_bytes(enc, &range, 2);
            }
            RangedOp::Scale => {
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&buf), 0, 0);
                }
                gpu_common::set_bytes(enc, &f.inv_t, 1);
                gpu_common::set_bytes(enc, &range, 2);
            }
        }
    })?;

    let mut out = vec![0.0f32; len];
    let read_buf = if f.op == RangedOp::Copy {
        buf_dst.as_ref().expect("dst")
    } else {
        &buf
    };
    BufferPool::read_f32(read_buf, &mut out);
    Ok(out)
}

#[cfg(target_os = "macos")]
pub fn gpu_variant(f: &Fixture, _: super::variant::KernelVariant) -> Result<Vec<f32>, Error> {
    gpu(f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;

    kernel_oracle_matrix! {
        mod tiny_softcap,
        cpu = crate::kernels::sub::sampler_ranged::cpu,
        cpu_oracle = crate::kernels::sub::sampler_ranged::cpu_oracle,
        gpu = crate::kernels::sub::sampler_ranged::gpu_variant,
        fixture = crate::kernels::sub::sampler_ranged::tiny_softcap_fixture,
        out_len = crate::kernels::sub::sampler_ranged::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod tiny_scale,
        cpu = crate::kernels::sub::sampler_ranged::cpu,
        cpu_oracle = crate::kernels::sub::sampler_ranged::cpu_oracle,
        gpu = crate::kernels::sub::sampler_ranged::gpu_variant,
        fixture = crate::kernels::sub::sampler_ranged::tiny_scale_fixture,
        out_len = crate::kernels::sub::sampler_ranged::fixture_len,
        formats: [F32],
        max_tol = 1e-5,
        min_cos = 0.9999,
    }

    kernel_oracle_matrix! {
        mod tiny_copy,
        cpu = crate::kernels::sub::sampler_ranged::cpu,
        cpu_oracle = crate::kernels::sub::sampler_ranged::cpu_oracle,
        gpu = crate::kernels::sub::sampler_ranged::gpu_variant,
        fixture = crate::kernels::sub::sampler_ranged::tiny_copy_fixture,
        out_len = crate::kernels::sub::sampler_ranged::fixture_len,
        formats: [F32],
        max_tol = 0.0,
        min_cos = 1.0,
    }

    #[cfg(target_os = "macos")]
    mod canvas_vocab {
        use super::*;
        use crate::kernels::sub::test_util::assert_oracle;

        fn run(op: RangedOp, seed: f32) {
            let fix = Fixture {
                src: fill_logits(CANVAS_VOCAB_LEN, seed),
                cap: 30.0,
                inv_t: 1.0 / 0.8,
                op,
            };
            let cpu = cpu(&fix);
            let gpu = gpu(&fix).expect("gpu");
            assert_oracle(&gpu, &cpu, 1e-4, 0.9999);
        }

        #[test]
        fn gpu_softcap_matches_cpu() {
            run(RangedOp::Softcap, 0.000011);
        }

        #[test]
        fn gpu_scale_matches_cpu() {
            run(RangedOp::Scale, 0.000013);
        }

        #[test]
        fn gpu_copy_matches_cpu() {
            run(RangedOp::Copy, 0.000017);
        }
    }
}
