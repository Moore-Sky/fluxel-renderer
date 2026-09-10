//! CPU admission checks and transactional snapshot-reservation helpers.

use super::*;

pub(super) fn rgba8_unorm_filterable(capabilities: &DeviceCapabilities) -> bool {
    capabilities.texture_formats.iter().any(|format| {
        format.format == TextureFormat::Rgba8Unorm && format.sampled && format.filterable
    })
}
pub(super) fn rgba8_unorm_srgb_filterable(capabilities: &DeviceCapabilities) -> bool {
    capabilities.texture_formats.iter().any(|format| {
        format.format == TextureFormat::Rgba8UnormSrgb && format.sampled && format.filterable
    })
}
pub(super) fn map_snapshot_use(error: SnapshotUseError) -> DrawStartError {
    match error {
        SnapshotUseError::InFlight => DrawStartError::SnapshotInFlight,
        SnapshotUseError::Poisoned => DrawStartError::SnapshotPoisoned,
    }
}
fn map_texture_use(error: SnapshotUseError) -> DrawStartError {
    match error {
        SnapshotUseError::InFlight => DrawStartError::TextureInFlight,
        SnapshotUseError::Poisoned => DrawStartError::TexturePoisoned,
    }
}
pub(in crate::fixed_frame) enum PairReservationError<E> {
    First(E),
    Second(E),
}
pub(super) fn map_pair_error(error: PairReservationError<SnapshotUseError>) -> DrawStartError {
    match error {
        PairReservationError::First(error) => map_snapshot_use(error),
        PairReservationError::Second(error) => map_texture_use(error),
    }
}

/// Acquires two independent gates transactionally: a failed second acquisition
/// rolls the first one back before the error crosses the public boundary.
pub(in crate::fixed_frame) fn reserve_pair<A, B, E>(
    first: Result<A, E>,
    second: impl FnOnce() -> Result<B, E>,
    rollback: impl FnOnce(A),
) -> Result<(A, B), PairReservationError<E>> {
    let first = first.map_err(PairReservationError::First)?;
    match second() {
        Ok(second) => Ok((first, second)),
        Err(error) => {
            rollback(first);
            Err(PairReservationError::Second(error))
        }
    }
}

pub(in crate::fixed_frame) fn validate_clip(
    positions: &[[f32; 3]],
    indices: &[u32],
    matrix: &[[f32; 4]; 4],
) -> Result<(), DrawStartError> {
    // FrameUniform stores a column-major projection*view matrix. This mirrors
    // the portable D3D/WebGPU clip volume rather than one backend's convention.
    for &index in indices {
        let position = positions
            .get(index as usize)
            .ok_or(DrawStartError::InvalidIndexCount)?;
        if !position.iter().all(|value| value.is_finite()) {
            return Err(DrawStartError::NonFinitePosition);
        }
        let input = [position[0], position[1], position[2], 1.0];
        let mut clip = [0.0; 4];
        for (row, component) in clip.iter_mut().enumerate() {
            for column in 0..4 {
                let product = matrix[column][row] * input[column];
                if !product.is_finite() {
                    return Err(DrawStartError::NonFiniteClipPosition);
                }
                *component += product;
                if !component.is_finite() {
                    return Err(DrawStartError::NonFiniteClipPosition);
                }
            }
        }
        let w = clip[3];
        if w <= 0.0 {
            return Err(DrawStartError::ClipWNonPositive);
        }
        if clip[0] < -w
            || clip[0] > w
            || clip[1] < -w
            || clip[1] > w
            || clip[2] < 0.0
            || clip[2] > w
        {
            return Err(DrawStartError::ClipOutOfBounds);
        }
    }
    Ok(())
}

pub(super) fn validate_textured_clip(
    positions: &[[f32; 3]],
    indices: &[u32],
    texture_coordinates: &[[f32; 2]],
    matrix: &[[f32; 4]; 4],
) -> Result<(), DrawStartError> {
    validate_clip(positions, indices, matrix)?;
    for &index in indices {
        let coordinate = texture_coordinates
            .get(index as usize)
            .ok_or(DrawStartError::InvalidIndexCount)?;
        if !coordinate.iter().all(|value| value.is_finite()) {
            return Err(DrawStartError::NonFiniteTextureCoordinate);
        }
    }
    Ok(())
}
