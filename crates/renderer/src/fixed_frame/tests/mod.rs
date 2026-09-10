//! Native fixed-frame conformance test suites.

use super::renderer::{PairReservationError, reserve_pair, validate_clip};
use super::*;

mod clip;
#[cfg(windows)]
mod legacy_u02;
#[cfg(windows)]
mod u03;
#[cfg(windows)]
mod u04;
#[cfg(windows)]
mod u05;
#[cfg(windows)]
mod u06;
#[cfg(windows)]
mod u07;
#[cfg(windows)]
mod u08;
#[cfg(windows)]
mod u09;
