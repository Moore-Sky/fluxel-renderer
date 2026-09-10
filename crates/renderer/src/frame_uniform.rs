//! Fixed camera/material uniform ABI for closed raster recipes.

use crate::{BasicMaterial, Camera};

/// Exact byte length of the closed `mat4x4<f32> + vec4<f32>` ABI.
pub(crate) const FRAME_UNIFORM_BYTES: usize = 80;

/// A validated, serialized fixed frame uniform.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameUniform {
    bytes: [u8; FRAME_UNIFORM_BYTES],
    view_projection: [[f32; 4]; 4],
}

impl FrameUniform {
    /// Validates and serializes `projection * view` followed by base color.
    pub(crate) fn new(
        camera: &Camera,
        material: &BasicMaterial,
    ) -> Result<Self, FrameUniformError> {
        let view = camera.view();
        let projection = camera.projection();
        if !view.iter().flatten().all(|value| value.is_finite())
            || !projection.iter().flatten().all(|value| value.is_finite())
        {
            return Err(FrameUniformError::NonFiniteCamera);
        }
        let color = material.base_color();
        if !color.iter().all(|value| value.is_finite()) {
            return Err(FrameUniformError::NonFiniteColor);
        }
        if !color.iter().all(|value| (0.0..=1.0).contains(value)) {
            return Err(FrameUniformError::ColorOutOfRange);
        }
        let matrix = projection_times_view(*projection, *view)?;
        let mut bytes = [0; FRAME_UNIFORM_BYTES];
        for (column, values) in matrix.iter().enumerate() {
            for (row, value) in values.iter().enumerate() {
                let start = (column * 4 + row) * 4;
                bytes[start..start + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        for (index, value) in color.iter().enumerate() {
            let start = 64 + index * 4;
            bytes[start..start + 4].copy_from_slice(&value.to_bits().to_le_bytes());
        }
        Ok(Self {
            bytes,
            view_projection: matrix,
        })
    }

    /// Returns the exact immutable-upload payload.
    pub(crate) fn bytes(&self) -> &[u8; FRAME_UNIFORM_BYTES] {
        &self.bytes
    }

    /// Returns the validated column-major projection-times-view matrix.
    pub(crate) const fn view_projection(&self) -> &[[f32; 4]; 4] {
        &self.view_projection
    }
}

/// Why a Camera/material pair cannot become a fixed frame uniform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FrameUniformError {
    /// A view or projection matrix element was NaN or infinite.
    NonFiniteCamera,
    /// A material color component was NaN or infinite.
    NonFiniteColor,
    /// A finite material color component was outside the closed unit interval.
    ColorOutOfRange,
    /// A finite matrix input produced a non-finite product or accumulation.
    MatrixProductNonFinite,
}

fn projection_times_view(
    projection: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
) -> Result<[[f32; 4]; 4], FrameUniformError> {
    let mut output = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            let mut value = 0.0;
            for (k, projection_column) in projection.iter().enumerate() {
                let product = projection_column[row] * view[column][k];
                if !product.is_finite() {
                    return Err(FrameUniformError::MatrixProductNonFinite);
                }
                value += product;
                if !value.is_finite() {
                    return Err(FrameUniformError::MatrixProductNonFinite);
                }
            }
            output[column][row] = value;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
