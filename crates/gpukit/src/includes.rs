//! Link-time registry of shader include folders.
//!
//! Each crate that owns shared headers registers its folder once with
//! [`register_includes!`]; [`include_table`] collects every registered folder
//! into one name → contents table, which the Metal context uses to resolve
//! `#include "name.metal"`. Registration is the only step — a header on disk
//! in a registered folder cannot be forgotten.

use crate::Error;

pub use include_dir;
pub use include_dir::Dir;
pub use scattered_collect;
pub use scattered_collect::declarative::{gather, scatter};

/// One registered folder of shader headers.
pub struct IncludeFolder(pub &'static Dir<'static>);

gather! {
    #[gather]
    pub static INCLUDE_DIRS: scattered_collect::ScatteredSlice<IncludeFolder>;
}

/// Embed a folder of shader headers and register it for `#include`
/// resolution in every [`crate::metal::Context`].
///
/// `path` is an `include_dir!` path, e.g.
/// `"$CARGO_MANIFEST_DIR/src/shaders/include"`. Invoke at most once per
/// module. The registering crate's build script must emit
/// `rerun-if-changed` for the folder — cargo does not track embedded files
/// on its own.
#[macro_export]
macro_rules! register_includes {
    ($path:tt) => {
        const _: () = {
            // `include_dir!` emits paths naming its own crate; this alias
            // resolves them without the caller depending on it directly.
            use $crate::includes::include_dir;
            static DIR: include_dir::Dir<'static> = include_dir::include_dir!($path);
            $crate::includes::scatter! {
                #[scatter($crate::includes::INCLUDE_DIRS)]
                static REG: $crate::includes::IncludeFolder =
                    $crate::includes::IncludeFolder(&DIR);
            }
        };
    };
}

/// The union of every registered folder as a name → contents table.
///
/// Folders are read non-recursively. Fails on a header that is not UTF-8 or
/// a name registered by two folders — bare-name `#include` resolution has no
/// way to pick between duplicates.
pub fn include_table() -> Result<Vec<(&'static str, &'static str)>, Error> {
    let mut table: Vec<(&'static str, &'static str)> = Vec::new();
    for folder in INCLUDE_DIRS.iter() {
        for file in folder.0.files() {
            let path = file.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(contents) = file.contents_utf8() else {
                return Err(Error::Compile(format!(
                    "registered include {name:?} is not UTF-8"
                )));
            };
            if table.iter().any(|(n, _)| *n == name) {
                return Err(Error::Compile(format!(
                    "include {name:?} is registered by two folders; bare-name \
                     #include resolution cannot pick one"
                )));
            }
            table.push((name, contents));
        }
    }
    Ok(table)
}
