//! VA-splice probe: can ONE `newBufferWithBytesNoCopy` wrap a contiguous
//! virtual range assembled from MULTIPLE mmap regions?
//!
//! Motivation (pack-format thought exercise): a layered pack's head could be
//! served with ZERO copies — no `model.dgq.head` cache at all — by reserving
//! the canonical address range and `MAP_FIXED`-splicing each HF-safetensors
//! shard's contiguous run at its (page-congruent) canonical offset, then
//! wrapping the whole range in one no-copy `MTLBuffer` so `w_off` addressing
//! survives untouched. Everything about that plan is standard except one
//! UNDOCUMENTED question: does Metal accept a no-copy buffer whose range
//! spans multiple distinct VM mappings (several files + anon fill), and do
//! GPU shader loads read correctly across the mapping seams?
//!
//! This probe answers exactly that with a [file A | anon | file B] splice and
//! a `convert_scale` (f32->f32, scale=1.0) dispatch straddling both seams.
//! PASS = bit-exact readback (idea viable); FAIL/panic = Metal refused or
//! misread (idea dead, head cache stays). Diagnostic only — undocumented
//! behavior must not gate CI, so it is #[ignore]d; run with:
//!
//!     cargo test --release va_splice -- --ignored --nocapture

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use crate::metal::buffer::BufferPool;
    use crate::metal::device::MetalContext;
    use crate::shaders::convert_scale;
    use crate::shaders::variant::KernelVariant;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::os::unix::io::AsRawFd;
    use std::ptr::NonNull;

    const PAGE: usize = 16384;

    /// MAP_FIXED a whole file read-only at `addr` (mapping outlives the fd).
    unsafe fn map_fixed_file(addr: *mut c_void, len: usize, path: &std::path::Path) {
        let f = std::fs::File::open(path).expect("probe: open");
        let p = unsafe {
            libc::mmap(
                addr,
                len,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_FIXED,
                f.as_raw_fd(),
                0,
            )
        };
        assert!(p != libc::MAP_FAILED, "probe: MAP_FIXED file map failed");
        assert_eq!(p, addr, "probe: MAP_FIXED landed elsewhere");
    }

    #[test]
    #[ignore = "diagnostic probe: undocumented Metal behavior (multi-mapping no-copy buffer)"]
    fn va_splice_multi_mapping_nocopy_buffer() {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        assert_eq!(page, PAGE, "probe assumes 16 KiB pages (Apple Silicon)");
        // Tiny (semantics) then 3x512 MiB (production-scale wiring).
        probe(4 * PAGE);
        probe(512 * 1024 * 1024);
    }

    fn probe(seg: usize) {
        // [file A: seg][anon fill: seg][file B: seg]; distinct per-third
        // value bands so a wrong-page read is unmistakable.
        let total = 3 * seg;
        let per_seg = seg / 4; // f32 count per segment
        let n_f32 = total / 4;
        let expect: Vec<f32> = (0..n_f32)
            .map(|i| (i as f32).sin() + (i / per_seg) as f32 * 1000.0)
            .collect();
        let bytes_of = |lo: usize, hi: usize| -> Vec<u8> {
            expect[lo..hi].iter().flat_map(|v| v.to_le_bytes()).collect()
        };

        let dir = std::env::temp_dir().join(format!("dgq-va-splice-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("probe: tmp dir");
        let fa = dir.join("a.bin");
        let fb = dir.join("b.bin");
        std::fs::write(&fa, bytes_of(0, per_seg)).expect("probe: write A");
        std::fs::write(&fb, bytes_of(2 * per_seg, 3 * per_seg)).expect("probe: write B");

        unsafe {
            // Reserve the contiguous range, then stitch the three mappings.
            let base = libc::mmap(
                std::ptr::null_mut(),
                total,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            );
            assert!(base != libc::MAP_FAILED, "probe: reserve failed");
            map_fixed_file(base, seg, &fa);
            let anon = libc::mmap(
                base.add(seg),
                seg,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
                -1,
                0,
            );
            assert!(anon != libc::MAP_FAILED, "probe: anon fill failed");
            std::slice::from_raw_parts_mut(anon as *mut u8, seg)
                .copy_from_slice(&bytes_of(per_seg, 2 * per_seg));
            map_fixed_file(base.add(2 * seg), seg, &fb);

            // CPU sanity across all three regions before involving Metal.
            let want_bytes = bytes_of(0, n_f32);
            let cpu = std::slice::from_raw_parts(base as *const u8, total);
            assert_eq!(cpu, &want_bytes[..], "probe: CPU composite mismatch");

            // Probe question 1: does buffer creation accept the range?
            let ctx = MetalContext::new().expect("metal ctx");
            let ptr = NonNull::new(base).expect("probe: null base");
            let buf = ctx
                .device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    total,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
                .expect(
                    "VERDICT NEGATIVE: newBufferWithBytesNoCopy returned nil \
                     over a multi-mapping range",
                );

            // Probe question 2: do compute-shader loads read correctly across
            // both seams? convert_scale f32->f32 at scale 1.0 = shader copy.
            let mut pool = BufferPool::new();
            let dst = pool.allocate(&ctx.device, total).expect("dst");
            let dump = pool.allocate(&ctx.device, 16).expect("dump");
            let pipe = convert_scale::pipeline_for_fmt(&ctx, KernelVariant::PRODUCTION, true, true)
                .expect("pipe");
            let cmd = ctx.queue.commandBuffer().expect("cmd");
            let enc = cmd.computeCommandEncoder().expect("enc");
            enc.setComputePipelineState(&pipe.pipeline);
            convert_scale::bind_gpu_buffers(&enc, &buf, 0, &dst, 0, 0, 0, n_f32 as u32, 1.0, &dump);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_f32.div_ceil(64),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 64,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
            assert!(
                cmd.error().is_none(),
                "VERDICT NEGATIVE: command buffer error (GPU fault?) reading \
                 across mapping seams: {:?}",
                cmd.error()
            );

            let got = std::slice::from_raw_parts(dst.contents().as_ptr() as *const u8, total);
            assert_eq!(
                got,
                &want_bytes[..],
                "VERDICT NEGATIVE: shader readback differs across mapping seams"
            );

            eprintln!(
                "VERDICT POSITIVE: one no-copy MTLBuffer over 3 mappings \
                 (fileA|anon|fileB, {} KiB) — compute-shader readback bit-exact \
                 across both seams",
                total / 1024
            );
            libc::munmap(base, total);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
