//! Internal shader ownership boundary for renderer material implementations.
//!
//! This module deliberately contains no shader compiler or native API calls.
//! A later renderer backend will consume [`ShaderModule`] values and provide
//! compilation and reflection behind this boundary.

#![allow(dead_code)]

/// Source owned by a renderer material before backend compilation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ShaderModule {
    source: &'static str,
    stage: ShaderStage,
}

impl ShaderModule {
    /// Creates a shader module description for a material implementation.
    pub(crate) const fn new(stage: ShaderStage, source: &'static str) -> Self {
        Self { source, stage }
    }

    /// Returns the shader stage this source targets.
    pub(crate) const fn stage(self) -> ShaderStage {
        self.stage
    }

    /// Returns the source that a later backend will compile and reflect.
    pub(crate) const fn source(self) -> &'static str {
        self.source
    }
}

/// A programmable shader stage required by a renderer material.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ShaderStage {
    /// Vertex processing.
    Vertex,
    /// Fragment processing.
    Fragment,
}

/// Returns the source modules selected by the current basic material.
///
/// This is intentionally a module-selection boundary, not shader compilation
/// or reflection. Both steps remain backend work for a later milestone.
pub(crate) const fn basic_material_modules() -> [ShaderModule; 2] {
    [
        ShaderModule::new(ShaderStage::Vertex, BASIC_VERTEX_SOURCE),
        ShaderModule::new(ShaderStage::Fragment, BASIC_FRAGMENT_SOURCE),
    ]
}

const BASIC_VERTEX_SOURCE: &str = "// BasicMaterial vertex shader is not implemented yet.";
const BASIC_FRAGMENT_SOURCE: &str = "// BasicMaterial fragment shader is not implemented yet.";

#[cfg(test)]
mod tests;
