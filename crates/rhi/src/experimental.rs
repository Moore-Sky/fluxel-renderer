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
