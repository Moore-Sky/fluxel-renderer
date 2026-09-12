//! Experimental APIs that intentionally do not belong to the stable RHI facade.
//!
//! The fixed raster artifacts here support the current renderer validation
//! slice. They are closed recipes, not a general graphics-pipeline API.

/// Closed renderer-shaped raster recipes and their native execution objects.
pub mod fixed_artifacts {
    pub use crate::execution::{RasterBackend, RasterBindings, RasterObjectProvider};
    pub use crate::resource::{
        RasterArtifactIdentity, RasterBindingVisibility, RasterCreateError,
        RasterFixedLightDirection, RasterKernel, RasterLightingModel, RasterLightingSpace,
        RasterNormalBindings, RasterNormalBindingsLease, RasterNormalInterpolation,
        RasterNormalNormalization, RasterPipeline, RasterPipelineLease, RasterSamplerAddressMode,
        RasterSamplerBindingType, RasterSamplerFilter, RasterTextureBindings,
        RasterTextureBindingsLease, RasterTextureSampleType, RasterUniformBindings,
        RasterUniformBindingsLease, RasterUvLinearClampTextureBindings,
        RasterUvLinearClampTextureBindingsLease, RasterUvTextureBindings,
        RasterUvTextureBindingsLease, RasterVertexColorBindings, RasterVertexColorBindingsLease,
        RasterVertexLayout,
    };
}

/// Closed browser WebGL2 execution for the retained Stage 1 unlit scene.
///
/// This module exists only in the browser build.  It deliberately exposes no
/// WebGL objects: the binding crate owns the JavaScript canvas value and RHI
/// owns the context, program, buffers, fences, and generations behind this
/// small session façade.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub mod webgl2;
