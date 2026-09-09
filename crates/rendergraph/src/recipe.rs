//! Internal closure adapters used by graph authoring methods.
//!
//! Applications should pass ordinary closures to [`crate::RenderGraph`] rather
//! than naming or implementing the traits in this module.

use crate::{
    error::RecordResult,
    pass::{
        ComputeCommands, ComputePassBuilder, CopyCommands, CopyPassBuilder, PassResourceResolver,
        RasterCommands, RasterPassBuilder,
    },
};

/// Internal closure adapter for static pass data retained by an immutable graph.
///
/// Callers should pass ordinary `Send + Sync + 'static` data to the pass
/// declaration methods rather than implementing this trait directly.
#[doc(hidden)]
pub trait PassData: Send + Sync + 'static {}

impl<T: Send + Sync + 'static> PassData for T {}

/// Internal closure adapter for one-time raster declarations.
///
/// Callers should pass an ordinary closure rather than implementing this trait.
#[doc(hidden)]
pub trait RasterSetup<O, D: PassData>: FnOnce(&mut RasterPassBuilder) -> (O, D) {}

impl<O, D, T> RasterSetup<O, D> for T
where
    D: PassData,
    T: FnOnce(&mut RasterPassBuilder) -> (O, D),
{
}

/// Internal closure adapter for one-time compute declarations.
///
/// Callers should pass an ordinary closure rather than implementing this trait.
#[doc(hidden)]
pub trait ComputeSetup<O, D: PassData>: FnOnce(&mut ComputePassBuilder) -> (O, D) {}

impl<O, D, T> ComputeSetup<O, D> for T
where
    D: PassData,
    T: FnOnce(&mut ComputePassBuilder) -> (O, D),
{
}

/// Internal closure adapter for one-time copy declarations.
///
/// Callers should pass an ordinary closure rather than implementing this trait.
#[doc(hidden)]
pub trait CopySetup<O, D: PassData>: FnOnce(&mut CopyPassBuilder) -> (O, D) {}

impl<O, D, T> CopySetup<O, D> for T
where
    D: PassData,
    T: FnOnce(&mut CopyPassBuilder) -> (O, D),
{
}

/// Internal closure adapter for repeatable raster recording.
///
/// Callers should pass an ordinary closure rather than implementing this trait.
#[doc(hidden)]
pub trait RasterExecute<F, D: PassData>:
    for<'a> Fn(&mut RasterCommands<'a>, &mut PassResourceResolver<'a>, &D, &F) -> RecordResult
    + Send
    + Sync
    + 'static
{
}

impl<F, D, T> RasterExecute<F, D> for T
where
    D: PassData,
    T: for<'a> Fn(&mut RasterCommands<'a>, &mut PassResourceResolver<'a>, &D, &F) -> RecordResult
        + Send
        + Sync
        + 'static,
{
}

/// Internal closure adapter for repeatable compute recording.
///
/// Callers should pass an ordinary closure rather than implementing this trait.
#[doc(hidden)]
pub trait ComputeExecute<F, D: PassData>:
    for<'a> Fn(&mut ComputeCommands<'a>, &mut PassResourceResolver<'a>, &D, &F) -> RecordResult
    + Send
    + Sync
    + 'static
{
}

impl<F, D, T> ComputeExecute<F, D> for T
where
    D: PassData,
    T: for<'a> Fn(&mut ComputeCommands<'a>, &mut PassResourceResolver<'a>, &D, &F) -> RecordResult
        + Send
        + Sync
        + 'static,
{
}

/// Internal closure adapter for repeatable copy recording.
///
/// Callers should pass an ordinary closure rather than implementing this trait.
#[doc(hidden)]
pub trait CopyExecute<F, D: PassData>:
    for<'a> Fn(&mut CopyCommands<'a>, &mut PassResourceResolver<'a>, &D, &F) -> RecordResult
    + Send
    + Sync
    + 'static
{
}

impl<F, D, T> CopyExecute<F, D> for T
where
    D: PassData,
    T: for<'a> Fn(&mut CopyCommands<'a>, &mut PassResourceResolver<'a>, &D, &F) -> RecordResult
        + Send
        + Sync
        + 'static,
{
}
