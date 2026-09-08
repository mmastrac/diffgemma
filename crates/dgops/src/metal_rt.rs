//! Shared Metal context for the portable ops: no include table, no on-disk
//! pipeline archive (the process-global in-memory archive still dedups).

use gpukit::metal::{CacheConfig, Context, ContextConfig};

pub fn context() -> Result<Context, gpukit::Error> {
    Context::new(ContextConfig {
        includes: &[],
        cache: CacheConfig {
            enabled: false,
            dir: None,
            namespace: "dgops",
            key: 0x6467_6f70,
            verbose: false,
        },
    })
}
