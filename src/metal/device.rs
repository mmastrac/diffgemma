use crate::safetensors::Error;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLCommandQueue, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};

pub struct MetalContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl MetalContext {
    pub fn new() -> Result<Self, Error> {
        let device = MTLCreateSystemDefaultDevice().ok_or(Error::Format("no Metal device"))?;
        let queue = device
            .newCommandQueue()
            .ok_or(Error::Format("failed to create Metal command queue"))?;
        Ok(Self { device, queue })
    }

    pub fn compile_kernel(&self, source: &str, entry: &str) -> Result<ComputePipeline, Error> {
        let ns_source = NSString::from_str(source);
        let library = self
            .device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(|e| shader_compile_error(e))?;

        let name = NSString::from_str(entry);
        let function = library
            .newFunctionWithName(&name)
            .ok_or(Error::Format("Metal kernel not found"))?;

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|_| Error::Format("Metal pipeline compile failed"))?;

        Ok(ComputePipeline { pipeline })
    }

    pub fn compile_kernels(
        &self,
        source: &str,
        entries: &[&str],
    ) -> Result<Vec<ComputePipeline>, Error> {
        let ns_source = NSString::from_str(source);
        let library = self
            .device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(|e| shader_compile_error(e))?;

        let mut pipelines = Vec::with_capacity(entries.len());
        for entry in entries {
            let name = NSString::from_str(entry);
            let function = library
                .newFunctionWithName(&name)
                .ok_or(Error::Format("Metal kernel not found"))?;
            let pipeline = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|_| Error::Format("Metal pipeline compile failed"))?;
            pipelines.push(ComputePipeline { pipeline });
        }
        Ok(pipelines)
    }
}

pub struct ComputePipeline {
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

pub type Bf16GemmPipeline = ComputePipeline;

fn shader_compile_error(err: Retained<NSError>) -> Error {
    Error::NotFound(format!(
        "Metal shader compile failed: {}",
        err.localizedDescription()
    ))
}
