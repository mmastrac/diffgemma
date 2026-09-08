//! Minimal CUDA driver-API bindings, resolved at runtime with dlopen.
//!
//! Linking libcuda at build time would require the CUDA stub library on every
//! host -- and it does not exist on macOS at all. Resolving the ~24 entry points
//! used here from libcuda.so.1 on first use keeps the crate buildable (and the
//! backend type-checkable) on hosts with no CUDA installed; a dispatch on such
//! a host fails with the name of the missing symbol instead of a link error.
//! DGQ_CUDA_DRIVER overrides the library name.

use crate::Error;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
/// CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR.
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

const RTLD_NOW: c_int = 2;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

type CuInit = unsafe extern "C" fn(u32) -> CUresult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type CuCtxCreate = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
type CuCtxDestroy = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSynchronize = unsafe extern "C" fn() -> CUresult;
type CuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type CuModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
type CuModuleGetFunction = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type CuMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type CuMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type CuMemcpyHtoD = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type CuMemcpyDtoH = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type CuMemsetD8 = unsafe extern "C" fn(CUdeviceptr, u8, usize) -> CUresult;
type CuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type CuFuncSetAttribute = unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult;
type CuGetErrorName = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;
type CuGetErrorString = unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult;

/// Resolved CUDA driver entry points. Holds only extern "C" fn pointers, so it
/// is Sync without a wrapper; the dlopen handle is intentionally leaked (a
/// driver library outlives the process in every sane deployment).
pub struct Driver {
    pub cu_init: CuInit,
    pub cu_device_get_count: CuDeviceGetCount,
    pub cu_device_get: CuDeviceGet,
    pub cu_device_get_name: CuDeviceGetName,
    pub cu_device_get_attribute: CuDeviceGetAttribute,
    pub cu_ctx_create: CuCtxCreate,
    pub cu_ctx_destroy: CuCtxDestroy,
    pub cu_ctx_set_current: CuCtxSetCurrent,
    pub cu_ctx_synchronize: CuCtxSynchronize,
    pub cu_module_load_data: CuModuleLoadData,
    pub cu_module_unload: CuModuleUnload,
    pub cu_module_get_function: CuModuleGetFunction,
    pub cu_mem_alloc: CuMemAlloc,
    pub cu_mem_free: CuMemFree,
    pub cu_memcpy_htod: CuMemcpyHtoD,
    pub cu_memcpy_dtoh: CuMemcpyDtoH,
    pub cu_memset_d8: CuMemsetD8,
    pub cu_launch_kernel: CuLaunchKernel,
    pub cu_func_set_attribute: CuFuncSetAttribute,
    cu_get_error_name: CuGetErrorName,
    cu_get_error_string: CuGetErrorString,
}

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let raw = unsafe { dlsym($lib, concat!($name, "\0").as_ptr().cast()) };
        if raw.is_null() {
            return Err(format!("CUDA driver is missing symbol {}", $name));
        }
        unsafe { std::mem::transmute::<*mut c_void, $ty>(raw) }
    }};
}

fn last_dl_error() -> String {
    let p = unsafe { dlerror() };
    if p.is_null() {
        "unknown dlopen error".to_string()
    } else {
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

impl Driver {
    fn open() -> Result<Self, String> {
        let lib_name = std::env::var("DGQ_CUDA_DRIVER").unwrap_or_else(|_| "libcuda.so.1".into());
        let c_name = CString::new(lib_name.clone()).map_err(|e| format!("bad driver name: {e}"))?;
        let lib = unsafe { dlopen(c_name.as_ptr(), RTLD_NOW) };
        if lib.is_null() {
            return Err(format!("cannot load {lib_name}: {}", last_dl_error()));
        }
        Ok(Driver {
            cu_init: sym!(lib, "cuInit", CuInit),
            cu_device_get_count: sym!(lib, "cuDeviceGetCount", CuDeviceGetCount),
            cu_device_get: sym!(lib, "cuDeviceGet", CuDeviceGet),
            cu_device_get_name: sym!(lib, "cuDeviceGetName", CuDeviceGetName),
            cu_device_get_attribute: sym!(lib, "cuDeviceGetAttribute", CuDeviceGetAttribute),
            cu_ctx_create: sym!(lib, "cuCtxCreate_v2", CuCtxCreate),
            cu_ctx_destroy: sym!(lib, "cuCtxDestroy_v2", CuCtxDestroy),
            cu_ctx_set_current: sym!(lib, "cuCtxSetCurrent", CuCtxSetCurrent),
            cu_ctx_synchronize: sym!(lib, "cuCtxSynchronize", CuCtxSynchronize),
            cu_module_load_data: sym!(lib, "cuModuleLoadData", CuModuleLoadData),
            cu_module_unload: sym!(lib, "cuModuleUnload", CuModuleUnload),
            cu_module_get_function: sym!(lib, "cuModuleGetFunction", CuModuleGetFunction),
            cu_mem_alloc: sym!(lib, "cuMemAlloc_v2", CuMemAlloc),
            cu_mem_free: sym!(lib, "cuMemFree_v2", CuMemFree),
            cu_memcpy_htod: sym!(lib, "cuMemcpyHtoD_v2", CuMemcpyHtoD),
            cu_memcpy_dtoh: sym!(lib, "cuMemcpyDtoH_v2", CuMemcpyDtoH),
            cu_memset_d8: sym!(lib, "cuMemsetD8_v2", CuMemsetD8),
            cu_launch_kernel: sym!(lib, "cuLaunchKernel", CuLaunchKernel),
            cu_func_set_attribute: sym!(lib, "cuFuncSetAttribute", CuFuncSetAttribute),
            cu_get_error_name: sym!(lib, "cuGetErrorName", CuGetErrorName),
            cu_get_error_string: sym!(lib, "cuGetErrorString", CuGetErrorString),
        })
    }

    /// Human-readable name for a driver return code (cuGetErrorName).
    pub fn error_name(&self, code: CUresult) -> String {
        let mut p: *const c_char = std::ptr::null();
        unsafe { (self.cu_get_error_name)(code, &mut p) };
        if p.is_null() {
            return format!("CUDA error {code}");
        }
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }

    /// Turn a non-success driver return code into an Error::Cuda, naming the
    /// call that failed.
    pub fn check(&self, code: CUresult, what: &str) -> Result<(), Error> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        let mut p: *const c_char = std::ptr::null();
        unsafe { (self.cu_get_error_string)(code, &mut p) };
        let msg = if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        };
        Err(Error::Cuda(format!(
            "{what} failed: {} ({code}) {msg}",
            self.error_name(code)
        )))
    }
}

/// The process-wide driver, opened on first use.
pub fn driver() -> Result<&'static Driver, Error> {
    static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();
    match DRIVER.get_or_init(Driver::open) {
        Ok(d) => Ok(d),
        Err(msg) => Err(Error::Cuda(format!("CUDA driver unavailable: {msg}"))),
    }
}
