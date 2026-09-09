//! Fixed Camera/material uniform ABI for the 0.2.2 raster slice.

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
mod tests {
    use super::*;

    #[test]
    fn serializes_column_major_projection_times_view_and_color() {
        let view = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let projection = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [7.0, 0.0, 0.0, 11.0],
        ];
        let uniform = FrameUniform::new(
            &Camera::new(view, projection),
            &BasicMaterial::new([0.0, 0.5, 1.0, 0.25]),
        )
        .unwrap();
        let words: Vec<_> = uniform
            .bytes()
            .chunks_exact(4)
            .map(|word| f32::from_bits(u32::from_le_bytes(word.try_into().unwrap())))
            .collect();
        assert_eq!(
            words,
            vec![
                30.0, 6.0, 15.0, 44.0, 66.0, 18.0, 35.0, 88.0, 102.0, 30.0, 55.0, 132.0, 138.0,
                42.0, 75.0, 176.0, 0.0, 0.5, 1.0, 0.25,
            ]
        );
    }

    #[test]
    fn rejects_non_finite_out_of_range_and_overflowing_inputs() {
        assert_eq!(
            FrameUniform::new(
                &Camera::new([[f32::NAN; 4]; 4], [[0.0; 4]; 4]),
                &BasicMaterial::default(),
            ),
            Err(FrameUniformError::NonFiniteCamera)
        );
        assert_eq!(
            FrameUniform::new(
                &Camera::default(),
                &BasicMaterial::new([0.0, 0.0, 0.0, f32::INFINITY])
            ),
            Err(FrameUniformError::NonFiniteColor)
        );
        assert_eq!(
            FrameUniform::new(
                &Camera::default(),
                &BasicMaterial::new([1.1, 0.0, 0.0, 1.0])
            ),
            Err(FrameUniformError::ColorOutOfRange)
        );
        assert_eq!(
            FrameUniform::new(
                &Camera::new([[f32::MAX; 4]; 4], [[f32::MAX; 4]; 4]),
                &BasicMaterial::default(),
            ),
            Err(FrameUniformError::MatrixProductNonFinite)
        );
    }
}
