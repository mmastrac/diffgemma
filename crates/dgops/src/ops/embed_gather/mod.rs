//! Token-embedding gather from a bf16 or q8-row table, scaled.
//!
//! The table lives in the pack blob at a byte offset; each row is either raw
//! bf16 (\`hidden\` \u00d7 2 bytes) or a q8 row (bf16 scale + int8 codes). This is
//! the engine's first pipeline stage, ported so it can run on CUDA against the
//! same CPU oracle.

crate::op_kernel! {
    name = "embed_gather",
    metal = "embed_gather.metal",
    cuda = "embed_gather.cu",
    fixture = Fixture => fix,
    abi = [
        in_u32(buf_blob = fix.blob_words(), as blob_words),
        in_u32(buf_ids = fix.ids),
        out(buf_out = fix.len()),
        u32x2(fix.hidden, fix.ids.len()),
        u32(fix.w_off_lo()),
        u32(fix.w_off_hi()),
        f32(fix.embed_scale),
        u32(fix.vocab),
        u32(fix.raw as u32),
    ],
    launch = 1d(fix.len()),
    result = (buf_out, fix.len()),
    tests = [
        raw_tiny => raw_tiny_fixture => (1e-6, 0.999999),
        raw_gemma_shape => raw_gemma_shape_fixture => (1e-6, 0.999999),
        q8_tiny => q8_tiny_fixture => (1e-6, 0.999999),
    ],
}

/// Bytes of one bf16 embed row.
pub fn raw_row_bytes(hidden: usize) -> usize {
    hidden * 2
}

/// Bytes of one q8 embed row: bf16 scale + int8 codes.
pub fn q8_row_bytes(hidden: usize) -> usize {
    2 + hidden
}

#[derive(Debug, Clone)]
pub struct Fixture {
    /// Raw table payload as stored in the blob (bf16 or q8 rows).
    pub blob: Vec<u8>,
    pub ids: Vec<u32>,
    pub hidden: usize,
    pub vocab: usize,
    pub embed_scale: f32,
    /// Table format: true = raw bf16, false = q8 row.
    pub raw: bool,
    /// Byte offset of the table inside the blob (exercises non-zero offsets).
    pub w_off: u64,
}

impl Fixture {
    pub fn len(&self) -> usize {
        self.ids.len() * self.hidden
    }

    /// The blob as the u32 words the generated ABI uploads. Padded to a
    /// multiple of 4 bytes so the tail read is in bounds.
    pub fn blob_words(&self) -> Vec<u32> {
        let mut bytes = self.blob.clone();
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn w_off_lo(&self) -> u32 {
        self.w_off as u32
    }

    fn w_off_hi(&self) -> u32 {
        (self.w_off >> 32) as u32
    }

    fn row_bytes(&self) -> usize {
        if self.raw {
            raw_row_bytes(self.hidden)
        } else {
            q8_row_bytes(self.hidden)
        }
    }

    /// A row of the table as f32 (the raw, unscaled weight).
    pub fn table_row_f32(&self, row: usize) -> Vec<f32> {
        let off = self.w_off as usize + row * self.row_bytes();
        let src = &self.blob[off..off + self.row_bytes()];
        let mut out = vec![0.0f32; self.hidden];
        if self.raw {
            for (i, o) in out.iter_mut().enumerate() {
                let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
                *o = f32::from_bits((bits as u32) << 16);
            }
        } else {
            let scale = f32::from_bits((u16::from_le_bytes([src[0], src[1]]) as u32) << 16);
            for (i, o) in out.iter_mut().enumerate() {
                *o = scale * (src[2 + i] as i8) as f32;
            }
        }
        out
    }
}

fn bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn raw_table(rows: &[&[f32]], hidden: usize) -> Vec<u8> {
    let mut blob = Vec::with_capacity(rows.len() * raw_row_bytes(hidden));
    for row in rows {
        assert_eq!(row.len(), hidden);
        for &v in row.iter() {
            blob.extend_from_slice(&bf16_bits(v).to_le_bytes());
        }
    }
    blob
}

/// q8 row: bf16 scale = max|x|/127, codes = round(x/scale) clamped to i8.
fn q8_table(rows: &[&[f32]], hidden: usize) -> Vec<u8> {
    let mut blob = Vec::with_capacity(rows.len() * q8_row_bytes(hidden));
    for row in rows {
        let amax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if amax == 0.0 { 1.0 } else { amax / 127.0 };
        blob.extend_from_slice(&bf16_bits(scale).to_le_bytes());
        for &v in row.iter() {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            blob.push(q as u8);
        }
    }
    blob
}

/// Two 8-wide bf16 rows, gathered out of order with a non-zero table offset.
pub fn raw_tiny_fixture() -> Fixture {
    let hidden = 8;
    let r0: Vec<f32> = (0..hidden).map(|i| 0.25 + i as f32 * 0.125).collect();
    let r1: Vec<f32> = (0..hidden).map(|i| -1.5 + i as f32 * 0.25).collect();
    let r2: Vec<f32> = (0..hidden).map(|i| 3.0 - i as f32 * 0.5).collect();
    let mut blob = vec![0xABu8; 16]; // leading bytes the gather must skip
    let w_off = blob.len() as u64;
    blob.extend_from_slice(&raw_table(&[&r0, &r1, &r2], hidden));
    Fixture {
        blob,
        ids: vec![2, 0, 1, 2],
        hidden,
        vocab: 3,
        embed_scale: 53.065_998,
        raw: true,
        w_off,
    }
}

/// The real model shape: vocab 262144, hidden 2816 (a 5-row slice).
pub fn raw_gemma_shape_fixture() -> Fixture {
    let hidden = 2816;
    let rows = 5usize;
    let table: Vec<Vec<f32>> = (0..rows)
        .map(|r| {
            (0..hidden)
                .map(|i| ((i as f32) * 0.0017 + r as f32).sin() * 0.08)
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = table.iter().map(|r| r.as_slice()).collect();
    Fixture {
        blob: raw_table(&refs, hidden),
        ids: vec![4, 0, 3, 1, 4],
        hidden,
        vocab: rows,
        embed_scale: 53.065_998,
        raw: true,
        w_off: 0,
    }
}

pub fn q8_tiny_fixture() -> Fixture {
    let hidden = 8;
    let r0: Vec<f32> = (0..hidden).map(|i| 0.5 - i as f32 * 0.1).collect();
    let r1: Vec<f32> = (0..hidden).map(|i| -0.25 + i as f32 * 0.05).collect();
    let mut blob = vec![0u8; 8];
    let w_off = blob.len() as u64;
    blob.extend_from_slice(&q8_table(&[&r0, &r1], hidden));
    Fixture {
        blob,
        ids: vec![1, 0, 1],
        hidden,
        vocab: 2,
        embed_scale: 2.0,
        raw: false,
        w_off,
    }
}

/// out[t, d] = decode(table[ids[t]], d) * embed_scale.
pub fn cpu(fix: &Fixture) -> Vec<f32> {
    let mut out = vec![0.0f32; fix.len()];
    for (t, &id) in fix.ids.iter().enumerate() {
        let row = fix.table_row_f32(id as usize);
        let dst = &mut out[t * fix.hidden..(t + 1) * fix.hidden];
        for i in 0..fix.hidden {
            dst[i] = row[i] * fix.embed_scale;
        }
    }
    out
}
