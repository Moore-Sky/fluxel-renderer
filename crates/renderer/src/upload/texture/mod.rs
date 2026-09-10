//! Separates linear and sRGB texture upload domains so formats cannot be mixed implicitly.

mod linear;
mod srgb;

pub use linear::*;
pub use srgb::*;

#[cfg(test)]
pub(in crate::upload) use linear::{BaseColorTextureSnapshotDebug, rgba8_byte_len};
