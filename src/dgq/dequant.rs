//! CPU dequant: quantized blob slices → f32 (oracle / parity path).

use crate::dgq::block::{dequant_matrix_q4, dequant_matrix_q8};
use crate::dgq::layout::{QuantKind, q4_matrix_bytes, q8_matrix_bytes};
use crate::dgq::nvfp4::dequant_matrix_nvfp4_payload;
use crate::safetensors::Error;
use crate::shaders::cpu::bf16_to_f32;

/// Dequant a tensor payload to f32 row-major flat buffer.
pub fn dequant_to_f32(
    kind: QuantKind,
    src: &[u8],
    shape: &[i64],
    dst: &mut [f32],
) -> Result<(), Error> {
    match kind {
        QuantKind::Q4Block => dequant_q4_tensor(src, shape, dst),
        QuantKind::Q6Block => dequant_q6_tensor(src, shape, dst),
        QuantKind::Nvfp4Block => dequant_nvfp4_tensor(src, shape, dst),
        QuantKind::Q8Row => dequant_q8_tensor(src, shape, dst),
        QuantKind::Raw => dequant_raw_tensor(src, shape, dst),
    }
}

fn dequant_q6_tensor(src: &[u8], shape: &[i64], dst: &mut [f32]) -> Result<(), Error> {
    use crate::dgq::block::q6_weight_at;
    use crate::dgq::layout::q6_matrix_bytes;
    let (mats, out, inp) = match shape.len() {
        2 => (1usize, shape[0] as usize, shape[1] as usize),
        3 => (shape[0] as usize, shape[1] as usize, shape[2] as usize),
        _ => return Err(Error::Format("q6 tensor must be 2D or 3D")),
    };
    let mat_bytes = q6_matrix_bytes(out, inp);
    let row_bytes = crate::dgq::layout::q6_row_bytes(inp);
    if src.len() != mats * mat_bytes || dst.len() != mats * out * inp {
        return Err(Error::Format("q6 shape/size mismatch"));
    }
    for m in 0..mats {
        for r in 0..out {
            let row = &src[m * mat_bytes + r * row_bytes..m * mat_bytes + (r + 1) * row_bytes];
            let base = (m * out + r) * inp;
            for c in 0..inp {
                dst[base + c] = q6_weight_at(row, c);
            }
        }
    }
    Ok(())
}

fn dequant_q4_tensor(src: &[u8], shape: &[i64], dst: &mut [f32]) -> Result<(), Error> {
    match shape.len() {
        2 => {
            let out = shape[0] as usize;
            let inp = shape[1] as usize;
            if src.len() != q4_matrix_bytes(out, inp) || dst.len() != out * inp {
                return Err(Error::Format("q4 2d shape/size mismatch"));
            }
            dequant_matrix_q4(src, out, inp, dst);
            Ok(())
        }
        3 => {
            let experts = shape[0] as usize;
            let out = shape[1] as usize;
            let inp = shape[2] as usize;
            let per = q4_matrix_bytes(out, inp);
            if src.len() != experts * per || dst.len() != experts * out * inp {
                return Err(Error::Format("q4 expert shape/size mismatch"));
            }
            for e in 0..experts {
                dequant_matrix_q4(
                    &src[e * per..(e + 1) * per],
                    out,
                    inp,
                    &mut dst[e * out * inp..(e + 1) * out * inp],
                );
            }
            Ok(())
        }
        _ => Err(Error::Format("q4 unsupported rank")),
    }
}

fn dequant_nvfp4_tensor(src: &[u8], shape: &[i64], dst: &mut [f32]) -> Result<(), Error> {
    match shape.len() {
        2 => {
            let out = shape[0] as usize;
            let inp = shape[1] as usize;
            dequant_matrix_nvfp4_payload(src, out, inp, dst)?;
            Ok(())
        }
        3 => {
            let experts = shape[0] as usize;
            let out = shape[1] as usize;
            let inp = shape[2] as usize;
            let per = crate::dgq::layout::nvfp4_matrix_bytes(out, inp);
            if src.len() != experts * per || dst.len() != experts * out * inp {
                return Err(Error::Format("nvfp4 expert shape/size mismatch"));
            }
            for e in 0..experts {
                dequant_matrix_nvfp4_payload(
                    &src[e * per..(e + 1) * per],
                    out,
                    inp,
                    &mut dst[e * out * inp..(e + 1) * out * inp],
                )?;
            }
            Ok(())
        }
        _ => Err(Error::Format("nvfp4 unsupported rank")),
    }
}

fn dequant_q8_tensor(src: &[u8], shape: &[i64], dst: &mut [f32]) -> Result<(), Error> {
    if shape.len() != 2 {
        return Err(Error::Format("q8 expects rank 2"));
    }
    let out = shape[0] as usize;
    let inp = shape[1] as usize;
    if src.len() != q8_matrix_bytes(out, inp) || dst.len() != out * inp {
        return Err(Error::Format("q8 shape/size mismatch"));
    }
    dequant_matrix_q8(src, out, inp, dst);
    Ok(())
}

fn dequant_raw_tensor(src: &[u8], shape: &[i64], dst: &mut [f32]) -> Result<(), Error> {
    let numel: usize = shape.iter().product::<i64>() as usize;
    if dst.len() != numel {
        return Err(Error::Format("raw dst numel mismatch"));
    }
    if src.len() == numel * 4 {
        for (i, chunk) in src.chunks_exact(4).enumerate() {
            dst[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(())
    } else if src.len() == numel * 2 {
        for i in 0..numel {
            let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
            dst[i] = bf16_to_f32(bits);
        }
        Ok(())
    } else if src.len() == numel {
        for (i, &b) in src.iter().enumerate() {
            dst[i] = b as f32;
        }
        Ok(())
    } else {
        Err(Error::Format("raw element size mismatch"))
    }
}

/// Max abs error vs bf16 reference for a 2-D Q4 matrix.
#[cfg(test)]
pub fn q4_max_abs_error_vs_bf16(src_bf16: &[u8], out_dim: usize, in_dim: usize) -> f32 {
    let mut q = vec![0u8; q4_matrix_bytes(out_dim, in_dim)];
    crate::dgq::block::quantize_bf16_matrix_q4(src_bf16, out_dim, in_dim, &mut q);
    let mut dec = vec![0.0f32; out_dim * in_dim];
    dequant_matrix_q4(&q, out_dim, in_dim, &mut dec);
    let mut max_err = 0.0f32;
    for i in 0..out_dim * in_dim {
        let bits = u16::from_le_bytes([src_bf16[i * 2], src_bf16[i * 2 + 1]]);
        let orig = bf16_to_f32(bits);
        max_err = max_err.max((orig - dec[i]).abs());
    }
    max_err
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dgq::block::quantize_bf16_matrix_q4;
    use crate::dgq::layout::QuantKind;

    fn f32_to_bf16_bits(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    #[test]
    fn dequant_q4_shape_2d() {
        let out = 2usize;
        let inp = 32usize;
        let mut bf16 = vec![0u8; out * inp * 2];
        for row in 0..out {
            for col in 0..inp {
                let v = col as f32 * 0.01 + row as f32 * 0.1 - 0.5;
                let bits = f32_to_bf16_bits(v).to_le_bytes();
                let i = (row * inp + col) * 2;
                bf16[i] = bits[0];
                bf16[i + 1] = bits[1];
            }
        }
        let mut q = vec![0u8; q4_matrix_bytes(out, inp)];
        quantize_bf16_matrix_q4(&bf16, out, inp, &mut q);
        let mut f32_out = vec![0.0f32; out * inp];
        dequant_to_f32(
            QuantKind::Q4Block,
            &q,
            &[out as i64, inp as i64],
            &mut f32_out,
        )
        .unwrap();
        let err = q4_max_abs_error_vs_bf16(&bf16, out, inp);
        assert!(err < 0.15, "err={err}");
    }
}
