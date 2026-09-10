//! Fixed compute artifact construction.
use super::super::artifact::{ComputeBindingsShared, ComputePipelineShared};
use super::super::bindings::TexturePackBindingsShared;
use super::super::validation::*;
use super::super::*;
use std::sync::Arc;
impl Device {
    /// Creates one fixed-artifact compute pipeline for this device.
    pub fn create_compute_pipeline(
        &self,
        kernel: ComputeKernel,
    ) -> Result<ComputePipeline, ComputeCreateError> {
        validate_compute_workgroup_limits(
            kernel.workgroup_size(),
            self.capabilities.max_compute_workgroup_size,
            self.capabilities.max_compute_invocations_per_workgroup,
        )?;
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        let native = crate::imp::create_compute_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.entry_point(),
        )
        .map_err(crate::resource::artifact::map_compute_pipeline_create_error)?;
        #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
        let native = crate::imp::create_compute_pipeline(
            &self.inner,
            kernel.wgsl_source(),
            kernel.entry_point(),
        )
        .map_err(ComputeCreateError::NativeObjectCreation)?;
        Ok(ComputePipeline(Arc::new(ComputePipelineShared {
            _native: native,
            kernel,
            device: self.identity,
        })))
    }

    /// Creates the only binding layout accepted by the fixed compute artifacts.
    pub fn create_compute_bindings(
        &self,
        pipeline: &ComputePipeline,
        buffer: &Buffer,
        offset: u64,
        size: u64,
    ) -> Result<ComputeBindings, ComputeCreateError> {
        if pipeline.kernel() == ComputeKernel::TexturePackRgba8 {
            return Err(ComputeCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity || buffer.device_identity() != self.identity
        {
            return Err(ComputeCreateError::ForeignDevice);
        }
        validate_compute_binding_range(
            offset,
            size,
            buffer.descriptor().buffer.size,
            self.capabilities.min_storage_buffer_offset_alignment,
            self.capabilities.max_storage_buffer_binding_size,
        )?;
        if !buffer
            .allowed_usage()
            .contains(BufferUsageKind::StorageRead)
            || !buffer
                .allowed_usage()
                .contains(BufferUsageKind::StorageWrite)
        {
            return Err(ComputeCreateError::StorageUsageRequired);
        }
        let native = crate::imp::create_compute_bindings(
            &self.inner,
            pipeline.native(),
            buffer.native(),
            offset,
            size,
        )
        .map_err(ComputeCreateError::NativeFailure)?;
        Ok(ComputeBindings(Arc::new(ComputeBindingsShared {
            _native: native,
            pipeline: pipeline.clone(),
            _buffer: buffer.lease(),
            offset,
            size,
            device: self.identity,
        })))
    }

    /// Creates the only binding layout accepted by the X01 texture-pack artifact.
    ///
    /// The complete `Rgba8Unorm` texture is sampled with `textureLoad`; each
    /// pixel occupies exactly one little-endian RGBA8-packed `u32` in the
    /// authorized destination range.
    pub fn create_texture_pack_bindings(
        &self,
        pipeline: &ComputePipeline,
        texture: &Texture,
        buffer: &Buffer,
        offset: u64,
        size: u64,
    ) -> Result<TexturePackBindings, ComputeCreateError> {
        if pipeline.kernel() != ComputeKernel::TexturePackRgba8 {
            return Err(ComputeCreateError::BindingRecipeMismatch);
        }
        if pipeline.device_identity() != self.identity
            || texture.device_identity() != self.identity
            || buffer.device_identity() != self.identity
        {
            return Err(ComputeCreateError::ForeignDevice);
        }
        validate_texture_pack_texture(texture)?;
        validate_compute_binding_range(
            offset,
            size,
            buffer.descriptor().buffer.size,
            self.capabilities.min_storage_buffer_offset_alignment,
            self.capabilities.max_storage_buffer_binding_size,
        )?;
        if !buffer
            .allowed_usage()
            .contains(BufferUsageKind::StorageRead)
            || !buffer
                .allowed_usage()
                .contains(BufferUsageKind::StorageWrite)
        {
            return Err(ComputeCreateError::StorageUsageRequired);
        }
        let required_size = texture_pack_required_size(texture.descriptor().texture)?;
        if size < required_size {
            return Err(ComputeCreateError::InvalidBindingRange);
        }
        let native = crate::imp::create_texture_pack_bindings(
            &self.inner,
            pipeline.native(),
            texture.native(),
            buffer.native(),
            offset,
            size,
        )
        .map_err(ComputeCreateError::NativeFailure)?;
        Ok(TexturePackBindings(Arc::new(TexturePackBindingsShared {
            _native: native,
            pipeline: pipeline.clone(),
            _texture: texture.lease(),
            _buffer: buffer.lease(),
            offset,
            size,
            device: self.identity,
        })))
    }
}
