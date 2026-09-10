//! Freezes packet-private frame-resource identity allocation.

use super::super::ids::PacketIdIssuer;
use fluxel_rendergraph::BufferBindingId;

#[test]
fn packet_buffer_identities_are_monotonic_and_packet_private() {
    let mut issuer = PacketIdIssuer::new();
    let first = issuer.buffer().unwrap();
    let second = issuer.buffer().unwrap();

    assert_eq!(first, BufferBindingId::new(0x0290_0000_0000_0000));
    assert_eq!(second, BufferBindingId::new(0x0290_0000_0000_0001));
}
