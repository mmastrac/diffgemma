//! UMA memory-pressure watch (`DGQ_MEM_WATCH=1`): detect when timings are
//! being poisoned by unified-memory pressure (swap/compressor activity, GPU
//! working-set saturation) instead of real kernel cost.
//!
//! On Apple Silicon the GPU shares one memory pool with the OS; when the
//! session footprint (weights + KV + arena) approaches the Metal
//! `recommendedMaxWorkingSetSize` or the system starts swapping, kernel
//! wall-clock degrades in ways that look like perf regressions but aren't.
//! This module samples, per labeled section:
//! - `vm.swapusage` (sysctl): swap-used delta during the section,
//! - `kern.memorystatus_vm_pressure_level` (1 normal / 2 warn / 4 critical),
//! - Metal `currentAllocatedSize` vs `recommendedMaxWorkingSetSize`.
//! A nonzero swap delta or elevated pressure prints a LOUD "timings suspect"
//! line so benches/probes can be discarded instead of misread (see the
//! perf-timing methodology note: full-run wall-clock is questionable when
//! the machine isn't idle).

#![cfg(target_os = "macos")]

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

unsafe extern "C" {
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        oldp: *mut std::ffi::c_void,
        oldlenp: *mut usize,
        newp: *mut std::ffi::c_void,
        newlen: usize,
    ) -> i32;
}

/// Mirror of macOS `struct xsw_usage` (`vm.swapusage`).
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct XswUsage {
    total: u64,
    avail: u64,
    used: u64,
    pagesize: u32,
    encrypted: bool,
}

fn sysctl_read<T: Default>(name: &std::ffi::CStr) -> Option<T> {
    let mut val = T::default();
    let mut len = std::mem::size_of::<T>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut val as *mut T as *mut std::ffi::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0).then_some(val)
}

fn swap_used_bytes() -> u64 {
    sysctl_read::<XswUsage>(c"vm.swapusage")
        .map(|s| s.used)
        .unwrap_or(0)
}

/// Total physical RAM (`hw.memsize`), for the fail-fast `--ctx` budget guard
/// before a Metal device exists. 0 if unavailable.
pub fn physical_ram_bytes() -> u64 {
    sysctl_read::<u64>(c"hw.memsize").unwrap_or(0)
}

/// 1 = normal, 2 = warn, 4 = critical (0 if unreadable).
fn pressure_level() -> u32 {
    sysctl_read::<u32>(c"kern.memorystatus_vm_pressure_level").unwrap_or(0)
}

#[derive(Clone, Copy)]
pub struct MemSnapshot {
    swap_used: u64,
    pressure: u32,
}

pub fn snapshot() -> MemSnapshot {
    MemSnapshot {
        swap_used: swap_used_bytes(),
        pressure: pressure_level(),
    }
}

fn gib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Print the section's memory verdict; returns true when the section's
/// timings should be considered SUSPECT (swap grew, or pressure elevated,
/// or the GPU allocation sits above 90% of the recommended working set).
pub fn report_section(
    label: &str,
    before: MemSnapshot,
    device: &ProtocolObject<dyn MTLDevice>,
) -> bool {
    let after = snapshot();
    let swap_delta = after.swap_used as i64 - before.swap_used as i64;
    let alloc = device.currentAllocatedSize() as u64;
    let cap = device.recommendedMaxWorkingSetSize();
    let frac = if cap > 0 {
        alloc as f64 / cap as f64
    } else {
        0.0
    };
    let pressure = after.pressure.max(before.pressure);
    let suspect = swap_delta > 0 || pressure > 1 || frac > 0.90;
    eprintln!(
        "mem-watch [{label}]: gpu_alloc {:.2}/{:.2} GiB ({:.0}%), swap used {:.2} GiB ({}{:.0} MiB during section), pressure {}",
        gib(alloc),
        gib(cap),
        frac * 100.0,
        gib(after.swap_used),
        if swap_delta >= 0 { "+" } else { "" },
        swap_delta as f64 / (1024.0 * 1024.0),
        match pressure {
            1 => "normal",
            2 => "WARN",
            4 => "CRITICAL",
            _ => "unknown",
        },
    );
    if suspect {
        eprintln!(
            "mem-watch [{label}]: TIMINGS SUSPECT — unified-memory pressure \
             (swap delta {swap_delta:+} B, pressure level {pressure}, working set {:.0}%); \
             treat wall-clock from this section as noise",
            frac * 100.0
        );
    }
    suspect
}
