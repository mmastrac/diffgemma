//! Arena scratch layout for the monolithic step kernel (host-built, GPU-bindable).

/// Must match `shaders/include/arena_layout.metal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaLayout {
    pub a_hidden: u32,
    pub a_resid: u32,
    pub a_attnq: u32,
    pub a_attnk: u32,
    pub a_attnv: u32,
    pub a_attno: u32,
    pub a_ffg: u32,
    pub a_ffu: u32,
    pub a_moein: u32,
    pub a_dense: u32,
    pub a_moeout: u32,
    pub a_soft: u32,
    pub a_stream: u32,
    pub a_tmp: u32,
    pub a_rs_sc: u32,
    pub a_rs_samp: u32,
    pub arena_bytes: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ArenaLayoutParams {
    pub canvas: usize,
    pub hidden: usize,
    pub dense_ff: usize,
    /// Max Q/O projection width (full-layer: n_q_heads * head_dim).
    pub max_attn_q_cols: usize,
    /// Max K/V projection width per token.
    pub max_attn_kv_cols: usize,
    /// f32 hidden + stream planes (`DGQ_HIDDEN_F32`): the residual accumulator
    /// keeps 23-bit mantissas across the 30 layers instead of re-rounding to
    /// bf16 (7-bit) at every residual add — the dominant per-step self-noise
    /// vs the f32 engine (layer-26 cos 0.975). All other planes stay bf16.
    pub hidden_f32: bool,
}

fn align16(off: u64) -> u64 {
    (off + 15) & !15
}

fn half_plane_bytes(canvas: usize, cols: usize) -> u64 {
    (canvas * cols * 2) as u64
}

fn f32_plane_bytes(canvas: usize, cols: usize) -> u64 {
    (canvas * cols * 4) as u64
}

fn rowstat_plane_bytes(canvas: usize) -> u64 {
    (canvas * 2 * std::mem::size_of::<f32>()) as u64
}

/// Stack arena planes in the order used by the step schedule; 16-byte align between planes.
pub fn build_arena_layout(p: &ArenaLayoutParams) -> ArenaLayout {
    let mut off = 0u64;
    let hidden_plane = |canvas: usize, cols: usize| {
        if p.hidden_f32 {
            f32_plane_bytes(canvas, cols)
        } else {
            half_plane_bytes(canvas, cols)
        }
    };

    let a_hidden = off;
    off = align16(off + hidden_plane(p.canvas, p.hidden));

    let a_resid = off;
    off = align16(off + half_plane_bytes(p.canvas, p.hidden));

    let a_attnq = off;
    off = align16(off + half_plane_bytes(p.canvas, p.max_attn_q_cols));

    let a_attnk = off;
    off = align16(off + half_plane_bytes(p.canvas, p.max_attn_kv_cols));

    let a_attnv = off;
    off = align16(off + half_plane_bytes(p.canvas, p.max_attn_kv_cols));

    let a_attno = off;
    off = align16(off + half_plane_bytes(p.canvas, p.max_attn_q_cols));

    let a_ffg = off;
    off = align16(off + half_plane_bytes(p.canvas, p.dense_ff));

    let a_ffu = off;
    off = align16(off + half_plane_bytes(p.canvas, p.dense_ff));

    let a_moein = off;
    off = align16(off + half_plane_bytes(p.canvas, p.hidden));

    let a_dense = off;
    off = align16(off + half_plane_bytes(p.canvas, p.hidden));

    let a_moeout = off;
    off = align16(off + f32_plane_bytes(p.canvas, p.hidden));

    let a_soft = off;
    off = align16(off + half_plane_bytes(p.canvas, p.hidden));

    let a_stream = off;
    off = align16(off + hidden_plane(p.canvas, p.hidden));

    let a_tmp = off;
    off = align16(off + half_plane_bytes(p.canvas, p.hidden));

    let a_rs_sc = off;
    off = align16(off + rowstat_plane_bytes(p.canvas));

    let a_rs_samp = off;
    off = align16(off + rowstat_plane_bytes(p.canvas));

    let arena_bytes = off as u32;

    ArenaLayout {
        a_hidden: a_hidden as u32,
        a_resid: a_resid as u32,
        a_attnq: a_attnq as u32,
        a_attnk: a_attnk as u32,
        a_attnv: a_attnv as u32,
        a_attno: a_attno as u32,
        a_ffg: a_ffg as u32,
        a_ffu: a_ffu as u32,
        a_moein: a_moein as u32,
        a_dense: a_dense as u32,
        a_moeout: a_moeout as u32,
        a_soft: a_soft as u32,
        a_stream: a_stream as u32,
        a_tmp: a_tmp as u32,
        a_rs_sc: a_rs_sc as u32,
        a_rs_samp: a_rs_samp as u32,
        arena_bytes,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    }
}

impl ArenaLayout {
    #[inline]
    pub fn hidden_off(&self) -> u64 {
        self.a_hidden as u64
    }
    #[inline]
    #[allow(dead_code)]
    pub fn resid_off(&self) -> u64 {
        self.a_resid as u64
    }
    #[inline]
    pub fn attnq_off(&self) -> u64 {
        self.a_attnq as u64
    }
    #[inline]
    pub fn attnk_off(&self) -> u64 {
        self.a_attnk as u64
    }
    #[inline]
    pub fn attnv_off(&self) -> u64 {
        self.a_attnv as u64
    }
    #[inline]
    pub fn attno_off(&self) -> u64 {
        self.a_attno as u64
    }
    #[inline]
    pub fn ffg_off(&self) -> u64 {
        self.a_ffg as u64
    }
    #[inline]
    pub fn ffu_off(&self) -> u64 {
        self.a_ffu as u64
    }
    #[inline]
    pub fn moein_off(&self) -> u64 {
        self.a_moein as u64
    }
    #[inline]
    pub fn dense_off(&self) -> u64 {
        self.a_dense as u64
    }
    #[inline]
    pub fn moeout_off(&self) -> u64 {
        self.a_moeout as u64
    }
    #[inline]
    pub fn soft_off(&self) -> u64 {
        self.a_soft as u64
    }
    #[inline]
    pub fn stream_off(&self) -> u64 {
        self.a_stream as u64
    }
    #[inline]
    pub fn tmp_off(&self) -> u64 {
        self.a_tmp as u64
    }
    #[inline]
    pub fn rs_sc_off(&self) -> u64 {
        self.a_rs_sc as u64
    }
    #[inline]
    pub fn rs_samp_off(&self) -> u64 {
        self.a_rs_samp as u64
    }
    #[inline]
    pub fn bytes(&self) -> u64 {
        self.arena_bytes as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_layout_matches_legacy_offsets() {
        let layout = build_arena_layout(&ArenaLayoutParams {
            canvas: 256,
            hidden: 2816,
            dense_ff: 2112,
            max_attn_q_cols: 8192,
            max_attn_kv_cols: 2048,
            hidden_f32: false,
        });
        assert_eq!(layout.a_hidden, 0);
        assert_eq!(layout.a_resid, 1_441_792);
        assert_eq!(layout.a_attnq, 2_883_584);
        assert_eq!(layout.a_attnk, 7_077_888);
        assert_eq!(layout.a_attnv, 8_126_464);
        assert_eq!(layout.a_attno, 9_175_040);
        assert_eq!(layout.a_ffg, 13_369_344);
        assert_eq!(layout.a_ffu, 14_450_688);
        assert_eq!(layout.a_moein, 15_532_032);
        assert_eq!(layout.a_dense, 16_973_824);
        assert_eq!(layout.a_moeout, 18_415_616);
        assert_eq!(layout.a_soft, 21_299_200);
        assert_eq!(layout.a_stream, 22_740_992);
        assert_eq!(layout.a_tmp, 24_182_784);
        assert_eq!(layout.a_rs_sc, 25_624_576);
        assert_eq!(layout.a_rs_samp, 25_626_624);
        assert_eq!(layout.arena_bytes, 25_628_672);
    }
}
