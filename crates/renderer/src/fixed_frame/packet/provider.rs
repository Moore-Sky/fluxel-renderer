//! Resolves packet-owned snapshot and uniform imports for the raster graph.
//!
//! This adapter maps only the private binding identities emitted by
//! `packet::graph` to opaque RHI buffers and their actual incoming states.  It
//! owns no graph declaration and does not create native binding objects; those
//! remain respectively in RenderGraph and the existing closed RHI provider.

#[cfg(windows)]
use std::cell::RefCell;
use std::collections::HashMap;

use fluxel_rendergraph::{
    BoundBuffer, BufferBindingId, DeviceIdentity, FrameBindingError, FrameBindingErrorKind,
    FrameResourceProvider, ResourceAccessState, TextureBindingId,
};
#[cfg(windows)]
use fluxel_rendergraph::{BoundSurfaceTexture, SurfaceBindingId, SurfaceBindingResult};
use fluxel_rhi::experimental::fixed_artifacts::RasterBackend;
use fluxel_rhi::{Buffer, ResourceLease, UploadedBuffer};
#[cfg(windows)]
use fluxel_rhi::{Texture, presentation::PresentationToken};

use super::{RenderPacket, graph::PacketGraph};

/// Concrete data behind one packet-private buffer binding identity.
struct PacketBuffer {
    physical: Buffer,
    initial_state: ResourceAccessState,
}

/// Provider inputs assembled after every packet uniform upload is ready.
pub(in crate::fixed_frame) struct PacketResources {
    device: DeviceIdentity,
    buffers: HashMap<BufferBindingId, PacketBuffer>,
    #[cfg(windows)]
    surface: RefCell<Option<BoundSurfaceTexture<Texture, ResourceLease, PresentationToken>>>,
}

impl PacketResources {
    /// Retains exactly the buffers selected by one compiled packet graph.
    pub(in crate::fixed_frame) fn new(
        packet: &RenderPacket,
        graph: &PacketGraph,
        uniforms: &[UploadedBuffer],
        #[cfg(windows)] surface: Option<
            BoundSurfaceTexture<Texture, ResourceLease, PresentationToken>,
        >,
    ) -> Self {
        assert_eq!(
            uniforms.len(),
            graph.draws.len(),
            "packet submission finalizes exactly one uniform per draw"
        );
        let mut buffers = HashMap::with_capacity(graph.snapshots.len() * 2 + graph.draws.len());
        for snapshot in &graph.snapshots {
            let draw = packet
                .draws()
                .get(snapshot.source_draw)
                .expect("packet graph source draw was built from this packet");
            insert_buffer(
                &mut buffers,
                snapshot.position_binding,
                draw.snapshot().positions(),
            );
            insert_buffer(
                &mut buffers,
                snapshot.index_binding,
                draw.snapshot().indices(),
            );
        }
        for (draw_index, (draw, uniform)) in graph.draws.iter().zip(uniforms).enumerate() {
            assert_eq!(
                uniform.buffer().device_identity(),
                packet.device(),
                "finalized uniform for packet draw {draw_index} belongs to the packet device"
            );
            insert_buffer(&mut buffers, draw.uniform_binding, uniform);
        }
        Self {
            device: packet.device(),
            buffers,
            #[cfg(windows)]
            surface: RefCell::new(surface),
        }
    }

    #[cfg(windows)]
    pub(in crate::fixed_frame) fn take_surface(
        &self,
    ) -> Option<BoundSurfaceTexture<Texture, ResourceLease, PresentationToken>> {
        self.surface.borrow_mut().take()
    }
}

impl FrameResourceProvider<RasterBackend> for PacketResources {
    fn texture(
        &self,
        _: TextureBindingId,
    ) -> Result<
        fluxel_rendergraph::BoundTexture<fluxel_rhi::Texture, ResourceLease>,
        FrameBindingError,
    > {
        Err(missing_binding(
            FrameBindingErrorKind::MissingTexture,
            "legacy packet graph has no texture imports",
        ))
    }

    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
        let buffer = self.buffers.get(&id).ok_or_else(|| {
            missing_binding(
                FrameBindingErrorKind::MissingBuffer,
                "unknown packet buffer binding",
            )
        })?;
        Ok(BoundBuffer {
            device: self.device,
            identity: buffer.physical.identity(),
            physical: buffer.physical.clone(),
            descriptor: buffer.physical.descriptor().buffer,
            usage: buffer.physical.allowed_usage(),
            initial_state: buffer.initial_state,
            lease: buffer.physical.lease().into(),
        })
    }

    #[cfg(windows)]
    fn surface(&self, id: SurfaceBindingId) -> SurfaceBindingResult<RasterBackend> {
        if id != crate::fixed_frame::surface_binding() {
            return Err(missing_binding(
                FrameBindingErrorKind::MissingSurface,
                "unknown packet presentation binding",
            ));
        }
        self.surface.borrow_mut().take().ok_or_else(|| {
            missing_binding(
                FrameBindingErrorKind::MissingSurface,
                "packet graph requested an unavailable acquired presentation image",
            )
        })
    }
}

fn insert_buffer(
    buffers: &mut HashMap<BufferBindingId, PacketBuffer>,
    binding: BufferBindingId,
    uploaded: &UploadedBuffer,
) {
    let previous = buffers.insert(
        binding,
        PacketBuffer {
            physical: uploaded.buffer().clone(),
            initial_state: uploaded.outgoing_state(),
        },
    );
    assert!(
        previous.is_none(),
        "packet graph emitted duplicate buffer binding {binding:?}"
    );
}

fn missing_binding(kind: FrameBindingErrorKind, detail: &str) -> FrameBindingError {
    FrameBindingError {
        kind,
        texture_slot: None,
        buffer_slot: None,
        resource: None,
        surface_binding: None,
        detail: detail.into(),
    }
}
