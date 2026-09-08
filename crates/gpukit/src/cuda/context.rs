//! CUDA context, module cache, and kernel handles.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int};
use std::sync::{Arc, Mutex};

use super::dispatch::KernelArgs;
use super::driver::{
    self, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CUcontext, CUdevice, CUfunction, CUmodule,
    Driver,
};
use crate::Error;

/// How to open the CUDA context.
#[derive(Debug, Clone, Default)]
pub struct ContextConfig {
    /// Zero-based device ordinal (CUDA_VISIBLE_DEVICES still applies).
    pub device_ordinal: usize,
}

/// A CUDA context plus a content-addressed cache of loaded modules.
///
/// Cheap to clone (an Arc); every operation makes the context current on the
/// calling thread first, so tests may dispatch from parallel threads.
#[derive(Clone)]
pub struct Context(Arc<Inner>);

struct Inner {
    driver: &'static Driver,
    raw: CUcontext,
    device: CUdevice,
    name: String,
    compute_capability: (i32, i32),
    modules: Mutex<HashMap<u64, Arc<Module>>>,
}

// A CUcontext is a driver-owned opaque handle. The CUDA driver API is
// thread-safe, every operation makes the context current on the calling
// thread first, and the module cache is behind a Mutex -- so sharing one
// Context across threads is sound.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        // Unload modules before destroying the context they were loaded into.
        if let Ok(mut modules) = self.modules.lock() {
            modules.clear();
        }
        unsafe {
            (self.driver.cu_ctx_destroy)(self.raw);
        }
    }
}

fn device_attr(driver: &Driver, device: CUdevice, attr: c_int) -> Result<i32, Error> {
    let mut value: c_int = 0;
    driver.check(
        unsafe { (driver.cu_device_get_attribute)(&mut value, attr, device) },
        "cuDeviceGetAttribute",
    )?;
    Ok(value)
}

/// Stable (no-random-seed) hash of a module image, used as its cache key.
fn image_hash(image: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in image {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl Context {
    /// Open (and initialize) the CUDA driver and a context on the configured
    /// device ordinal.
    pub fn new(config: ContextConfig) -> Result<Self, Error> {
        let driver = driver::driver()?;
        driver.check(unsafe { (driver.cu_init)(0) }, "cuInit")?;

        let mut count: c_int = 0;
        driver.check(
            unsafe { (driver.cu_device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        if config.device_ordinal >= count as usize {
            return Err(Error::Cuda(format!(
                "device ordinal {} out of range ({count} visible CUDA devices)",
                config.device_ordinal
            )));
        }

        let mut device: CUdevice = 0;
        driver.check(
            unsafe { (driver.cu_device_get)(&mut device, config.device_ordinal as c_int) },
            "cuDeviceGet",
        )?;

        let mut raw: CUcontext = std::ptr::null_mut();
        driver.check(
            unsafe { (driver.cu_ctx_create)(&mut raw, 0, device) },
            "cuCtxCreate",
        )?;

        let mut name_buf = [0 as c_char; 256];
        driver.check(
            unsafe { (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 256, device) },
            "cuDeviceGetName",
        )?;
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let major = device_attr(driver, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = device_attr(driver, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;

        Ok(Self(Arc::new(Inner {
            driver,
            raw,
            device,
            name,
            compute_capability: (major, minor),
            modules: Mutex::new(HashMap::new()),
        })))
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// (major, minor), e.g. (12, 1) for sm_121.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.0.compute_capability
    }

    pub fn device(&self) -> CUdevice {
        self.0.device
    }

    pub(crate) fn driver(&self) -> &'static Driver {
        self.0.driver
    }

    /// Make this context current on the calling thread.
    pub(crate) fn set_current(&self) -> Result<(), Error> {
        self.0.driver.check(
            unsafe { (self.0.driver.cu_ctx_set_current)(self.0.raw) },
            "cuCtxSetCurrent",
        )
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        self.set_current()?;
        self.0.driver.check(
            unsafe { (self.0.driver.cu_ctx_synchronize)() },
            "cuCtxSynchronize",
        )
    }

    /// Load a cubin/PTX image, reusing an already-loaded module for identical
    /// bytes. The image must be a valid cuModuleLoadData argument (ELF cubin or
    /// NUL-terminated PTX).
    pub fn load_module(&self, image: &[u8]) -> Result<Arc<Module>, Error> {
        let key = image_hash(image);
        if let Some(existing) = self.0.modules.lock().unwrap().get(&key) {
            return Ok(Arc::clone(existing));
        }
        self.set_current()?;
        let mut handle: CUmodule = std::ptr::null_mut();
        self.0.driver.check(
            unsafe { (self.0.driver.cu_module_load_data)(&mut handle, image.as_ptr().cast()) },
            "cuModuleLoadData",
        )?;
        let module = Arc::new(Module {
            driver: self.0.driver,
            raw: handle,
        });
        self.0
            .modules
            .lock()
            .unwrap()
            .insert(key, Arc::clone(&module));
        Ok(module)
    }

    /// Enqueue a kernel launch; returns once the launch is enqueued, not
    /// completed. Call synchronize (or copy to host) to wait.
    pub fn launch(
        &self,
        kernel: &Kernel,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_bytes: u32,
        args: &mut KernelArgs,
    ) -> Result<(), Error> {
        self.set_current()?;
        let mut params = args.pointers();
        self.0.driver.check(
            unsafe {
                (self.0.driver.cu_launch_kernel)(
                    kernel.raw,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_bytes,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }
}

/// A loaded CUDA module. Unloaded when the last Kernel referencing it drops
/// (or when its Context drops).
pub struct Module {
    driver: &'static Driver,
    raw: CUmodule,
}

impl Module {
    pub fn function(self: &Arc<Self>, name: &str) -> Result<Kernel, Error> {
        let c_name = CString::new(name)
            .map_err(|e| Error::Cuda(format!("kernel name {name:?} has a NUL: {e}")))?;
        let mut raw: CUfunction = std::ptr::null_mut();
        self.driver.check(
            unsafe { (self.driver.cu_module_get_function)(&mut raw, self.raw, c_name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(Kernel {
            module: Arc::clone(self),
            raw,
            name: name.to_string(),
        })
    }
}

// Same reasoning as Inner: a CUmodule is a driver-owned handle and the CUDA
// driver API is thread-safe, so a loaded module may be shared and used from
// any thread (each launch makes its context current first).
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            (self.driver.cu_module_unload)(self.raw);
        }
    }
}

/// A resolved kernel entry point. Cheap to clone; keeps its module alive.
#[derive(Clone)]
pub struct Kernel {
    module: Arc<Module>,
    raw: CUfunction,
    name: String,
}

// The handle is a driver-owned opaque pointer, valid for as long as the module
// it came from lives -- which the Arc<Module> field guarantees. Launches are
// safe from any thread (each makes its context current first).
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Raise the dynamic shared-memory limit for this kernel (needed above
    /// 48 KiB). Returns the driver error when the limit is unsupported.
    pub fn set_max_dynamic_shared(&self, bytes: i32) -> Result<(), Error> {
        const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
        self.module.driver.check(
            unsafe {
                (self.module.driver.cu_func_set_attribute)(
                    self.raw,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes,
                )
            },
            "cuFuncSetAttribute",
        )
    }
}
