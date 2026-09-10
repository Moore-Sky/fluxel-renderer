//! Native fixed-frame conformance test suites.

use super::renderer::{PairReservationError, reserve_pair, validate_clip};
use super::*;

#[cfg(windows)]
mod camera_material;
mod clip;
#[cfg(windows)]
mod legacy_fixed_color;
#[cfg(windows)]
mod linear_clamp_sampling;
#[cfg(windows)]
mod normal_lambert;
#[cfg(windows)]
mod srgb_sampling;
#[cfg(windows)]
mod texture_load;
#[cfg(windows)]
mod uv_texture_load;
#[cfg(windows)]
mod vertex_color;
