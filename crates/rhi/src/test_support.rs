//! Doc-hidden conformance observation helpers for workspace hardware fixtures.
//!
//! This non-default module exposes neither host mapping, command recording,
//! nor native handles. It is intentionally limited to observing completed
//! fixture work and injecting private conformance faults; production callers
//! must not use it as a general readback or submission API.

use crate::{
    BufferUploadError, BufferUploadStage, Device, TextureUploadError, TextureUploadStage,
    UploadedBuffer, UploadedTexture,
};

/// Clears diagnostics collected after `Device::open` enabled capture.
pub fn clear_validation_diagnostics(device: &Device) {
    #[cfg(windows)]
    crate::imp::clear_validation_diagnostics(&device.inner);
    #[cfg(not(windows))]
    let _ = device;
}

/// Returns validation warnings and errors collected for `device`.
#[must_use]
pub fn validation_diagnostics(device: &Device) -> Vec<String> {
    #[cfg(windows)]
    {
        crate::imp::validation_diagnostics(&device.inner)
    }
    #[cfg(not(windows))]
    {
        let _ = device;
        Vec::new()
    }
}

/// Makes the next native submission fail before queue acceptance.
///
/// This conformance-only hook is one-shot and has no production build
/// surface. It exercises the path which must release reservations because
/// native work was never accepted.
pub fn inject_submit_rejected_once() {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_submit_rejected_once();
}

/// Makes the submission after exactly `successful_submits` accepted
/// submissions fail before queue acceptance.
///
/// Passing zero is identical to [`inject_submit_rejected_once`]. This is
/// conformance-only and fixtures must serialize configuration with their
/// existing guard.
pub fn inject_submit_rejected_after(successful_submits: usize) {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_submit_rejected_after(successful_submits);
    #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
    let _ = successful_submits;
}

/// Makes the next native submission report accepted-but-unknown failure.
///
/// This conformance-only hook is one-shot and exercises quarantine of all
/// submitted leases: callers must not infer a final resource state.
pub fn inject_submit_accepted_unknown_once() {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_submit_accepted_unknown_once();
}

/// Makes the submission after exactly `successful_submits` accepted
/// submissions report accepted-but-unknown failure.
///
/// Passing zero is identical to [`inject_submit_accepted_unknown_once`].
/// This is conformance-only and fixtures must serialize configuration with
/// their existing guard.
pub fn inject_submit_accepted_unknown_after(successful_submits: usize) {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_submit_accepted_unknown_after(successful_submits);
    #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
    let _ = successful_submits;
}

/// Makes the next nonblocking native completion observation report `Pending`.
///
/// This conformance-only hook is one-shot and is consumed by the same
/// completion-status path used by renderer polling. It never alters a
/// later blocking readback wait.
pub fn inject_completion_pending_once() {
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    crate::imp::inject_completion_pending_once();
}

/// Reads a finalized immutable upload using its exact published state.
///
/// The helper is only for CPU-oracle hardware fixtures. It consumes the
/// upload's `CopyDestination` state as the true incoming state before the
/// private readback transition, so it cannot repair a wrong upload state.
pub fn readback_uploaded_buffer(
    device: &Device,
    uploaded: &UploadedBuffer,
) -> Result<Vec<u8>, BufferUploadError> {
    if uploaded.buffer().device_identity() != device.identity() {
        return Err(BufferUploadError::ForeignDevice);
    }
    if !uploaded
        .buffer()
        .allowed_usage()
        .contains(fluxel_rendergraph::BufferUsageKind::CopySource)
    {
        return Err(BufferUploadError::InvalidRequest(
            crate::InvalidBufferUploadReason::CopySourceUsageRequired,
        ));
    }
    #[cfg(windows)]
    {
        crate::imp::readback_buffer_for_test(
            &device.inner,
            uploaded.buffer().native(),
            uploaded.lease().into(),
            uploaded.outgoing_state(),
            uploaded.buffer().descriptor().buffer.size,
        )
        .map_err(|reason| BufferUploadError::Native {
            backend: device.hardware().backend,
            stage: BufferUploadStage::Completion,
            reason,
        })
    }
    #[cfg(not(windows))]
    {
        let _ = uploaded;
        Err(BufferUploadError::Native {
            backend: device.hardware().backend,
            stage: BufferUploadStage::Completion,
            reason: "native readback is only supported on Windows".into(),
        })
    }
}

/// Reads a finalized immutable texture using its exact exported incoming
/// state. The returned tuple is `(tight_rgba8, padded_native_rows,
/// bytes_per_row)` for hardware CPU-oracle fixtures only.
///
/// This helper consumes [`UploadedTexture::outgoing_state`] as its actual
/// incoming state and restores that state afterward; it never assumes or
/// silently repairs an incorrect graph export.
pub fn readback_uploaded_texture(
    device: &Device,
    uploaded: &UploadedTexture,
) -> Result<(Vec<u8>, Vec<u8>, u32), TextureUploadError> {
    if uploaded.texture().device_identity() != device.identity() {
        return Err(TextureUploadError::ForeignDevice);
    }
    if !uploaded
        .texture()
        .allowed_usage()
        .contains(fluxel_rendergraph::TextureUsageKind::CopySource)
    {
        return Err(TextureUploadError::InvalidRequest(
            crate::InvalidTextureUploadReason::UnexpectedUsage,
        ));
    }
    #[cfg(windows)]
    {
        let readback = crate::imp::readback_texture_for_test(
            &device.inner,
            uploaded.texture().native(),
            uploaded.lease().into(),
            uploaded.texture().descriptor().texture,
            uploaded.outgoing_state(),
        )
        .map_err(|reason| TextureUploadError::Native {
            backend: device.hardware().backend,
            stage: TextureUploadStage::Completion,
            reason,
        })?;
        Ok((readback.tight, readback.padded, readback.bytes_per_row))
    }
    #[cfg(not(windows))]
    {
        let _ = uploaded;
        Err(TextureUploadError::Native {
            backend: device.hardware().backend,
            stage: TextureUploadStage::Completion,
            reason: "native readback is only supported on Windows".into(),
        })
    }
}
