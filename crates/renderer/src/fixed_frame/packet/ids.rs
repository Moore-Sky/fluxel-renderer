//! Issues packet-local identities for RenderGraph frame-resource bindings.
//!
//! These identities select caller-owned buffers through a
//! `FrameResourceProvider`; they are neither resource handles nor binding
//! recipes.  A packet has one issuer so a duplicate snapshot can deliberately
//! reuse its two identities while every separately uploaded uniform receives
//! its own identity.

use core::fmt;

use fluxel_rendergraph::BufferBindingId;

const FIRST_PACKET_BUFFER_ID: u64 = 0x0290_0000_0000_0000;

/// Failure to allocate another private packet binding identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PacketIdError {
    /// The private identity namespace has no remaining value.
    Exhausted,
}

impl fmt::Display for PacketIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted => formatter.write_str("packet buffer binding identities exhausted"),
        }
    }
}

impl std::error::Error for PacketIdError {}

/// Monotonic issuer scoped to exactly one packet graph declaration.
pub(super) struct PacketIdIssuer {
    next_buffer: u64,
}

impl PacketIdIssuer {
    /// Starts at the renderer-private packet namespace.
    pub(super) const fn new() -> Self {
        Self {
            next_buffer: FIRST_PACKET_BUFFER_ID,
        }
    }

    /// Allocates one frame-resource buffer identity.
    pub(super) fn buffer(&mut self) -> Result<BufferBindingId, PacketIdError> {
        let value = self.next_buffer;
        self.next_buffer = self
            .next_buffer
            .checked_add(1)
            .ok_or(PacketIdError::Exhausted)?;
        Ok(BufferBindingId::new(value))
    }
}
