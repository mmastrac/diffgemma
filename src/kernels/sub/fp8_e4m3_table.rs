//! FP8 E4M3 decode table — GPU vs CPU oracle (tier-1, no model weights).

use super::gpu_common;
use super::test_util::ElemFormat;
use super::variant::KernelVariant;
use crate::dgq::fp4::fp8_e4m3_to_f32;
use crate::safetensors::Error;

pub const ENTRY: &str = "fp8_e4m3_table";

const SHADER: &str = shader_include::include_metal!("kernels/fp8_e4m3_table.metal");

/// Canonical E4M3 codes exercised in tier-1 (oracle-free reference values).
pub const TABLE: &[(u8, f32)] = &[
    (0x00, 0.0),
    (0x01, 1.0 / 512.0),
    (0x08, 1.0 / 64.0),
    (0x28, 0.25),
    (0x38, 1.0),
    (0x3C, 1.5),
    (0x40, 2.0),
    (0x50, 8.0),
    (0x7E, 448.0),
];

#[derive(Debug, Clone)]
pub struct Fixture {
    pub codes: Vec<u8>,
}

pub fn fixture_len(f: &Fixture) -> usize {
    f.codes.len()
}

pub fn table_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        codes: TABLE.iter().map(|(b, _)| *b).collect(),
    }
}

pub fn full_fixture(_: ElemFormat) -> Fixture {
    Fixture {
        codes: (0u8..=0x7E).collect(),
    }
}

pub fn cpu(f: &Fixture) -> Vec<f32> {
    f.codes.iter().map(|&b| fp8_e4m3_to_f32(b)).collect()
}

pub fn cpu_oracle(f: &Fixture) -> Vec<f32> {
    cpu(f)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn pipeline_for(
    ctx: &crate::metal::device::MetalContext,
    variant: KernelVariant,
) -> Result<crate::metal::device::ComputePipeline, Error> {
    ctx.compile_subkernel(SHADER, ENTRY, variant)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn bind_gpu_buffers(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    codes: &ProtocolObject<dyn MTLBuffer>,
    out: &ProtocolObject<dyn MTLBuffer>,
    n: u32,
) {
    unsafe {
        enc.setBuffer_offset_atIndex(Some(codes), 0, 0);
        enc.setBuffer_offset_atIndex(Some(out), 0, 2);
    }
    gpu_common::set_bytes(enc, &n, 1);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
pub fn gpu(f: &Fixture, variant: KernelVariant) -> Result<Vec<f32>, Error> {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;

    let ctx = MetalContext::new()?;
    let pipeline = pipeline_for(&ctx, variant)?;
    let n = f.codes.len() as u32;
    let mut pool = BufferPool::new();
    let buf_codes = pool
        .allocate(&ctx.device, f.codes.len())
        .ok_or(Error::Format("alloc"))?;
    let buf_out = pool
        .allocate(&ctx.device, f.codes.len() * 4)
        .ok_or(Error::Format("alloc"))?;
    BufferPool::write_bytes(&buf_codes, &f.codes);
    gpu_common::dispatch_1d(&ctx.queue, &pipeline.pipeline, f.codes.len(), |enc| {
        bind_gpu_buffers(enc, &buf_codes, &buf_out, n);
    })?;

    let mut out = vec![0.0f32; f.codes.len()];
    BufferPool::read_f32(&buf_out, &mut out);
    Ok(out)
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub fn gpu(_: &Fixture, _: KernelVariant) -> Result<Vec<f32>, Error> {
    Err(Error::Format("Metal unavailable"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_oracle_matrix;
    use crate::kernels::sub::test_util::ElemFormat;

    #[test]
    fn cpu_table_matches_reference() {
        for &(bits, expected) in TABLE {
            let got = fp8_e4m3_to_f32(bits);
            assert!(
                (got - expected).abs() < 1e-6,
                "0x{bits:02X}: got {got} expected {expected}"
            );
        }
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn gpu_shader_table_matches_reference() {
        use crate::kernels::sub::variant::KernelVariant;
        use crate::kernels::sub::test_util::assert_oracle;

        let fix = table_fixture(ElemFormat::F32);
        let gpu_out = gpu(&fix, KernelVariant::PRODUCTION).expect("gpu decode");
        let expected: Vec<f32> = TABLE.iter().map(|(_, v)| *v).collect();
        assert_oracle(&gpu_out, &expected, 1e-6, 1.0);

        let full = full_fixture(ElemFormat::F32);
        let gpu_full = gpu(&full, KernelVariant::PRODUCTION).expect("gpu full decode");
        let cpu_full = cpu(&full);
        assert_oracle(&gpu_full, &cpu_full, 1e-6, 1.0);
    }

    kernel_oracle_matrix! {
        mod table,
        cpu = crate::kernels::sub::fp8_e4m3_table::cpu,
        cpu_oracle = crate::kernels::sub::fp8_e4m3_table::cpu_oracle,
        gpu = crate::kernels::sub::fp8_e4m3_table::gpu,
        fixture = crate::kernels::sub::fp8_e4m3_table::table_fixture,
        out_len = crate::kernels::sub::fp8_e4m3_table::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 1.0,
    }

    kernel_oracle_matrix! {
        mod full,
        cpu = crate::kernels::sub::fp8_e4m3_table::cpu,
        cpu_oracle = crate::kernels::sub::fp8_e4m3_table::cpu_oracle,
        gpu = crate::kernels::sub::fp8_e4m3_table::gpu,
        fixture = crate::kernels::sub::fp8_e4m3_table::full_fixture,
        out_len = crate::kernels::sub::fp8_e4m3_table::fixture_len,
        formats: [F32],
        max_tol = 1e-6,
        min_cos = 1.0,
    }
}
