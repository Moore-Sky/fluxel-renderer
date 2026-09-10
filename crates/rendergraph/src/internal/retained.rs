//! Type-erased retained pass callback adapters.

use std::sync::Arc;

use crate::{
    pass::{ComputeCommands, CopyCommands, PassResourceResolver, RasterCommands},
    recipe::{ComputeExecute, CopyExecute, PassData, RasterExecute},
};
pub(crate) trait RasterRecipe<F>: Send + Sync {
    fn call<'a>(
        &self,
        commands: &mut RasterCommands<'a>,
        resolver: &mut PassResourceResolver<'a>,
        frame: &F,
    ) -> crate::RecordResult;
}
pub(crate) trait ComputeRecipe<F>: Send + Sync {
    fn call<'a>(
        &self,
        commands: &mut ComputeCommands<'a>,
        resolver: &mut PassResourceResolver<'a>,
        frame: &F,
    ) -> crate::RecordResult;
}
pub(crate) trait CopyRecipe<F>: Send + Sync {
    fn call<'a>(
        &self,
        commands: &mut CopyCommands<'a>,
        resolver: &mut PassResourceResolver<'a>,
        frame: &F,
    ) -> crate::RecordResult;
}

struct Retained<D, E> {
    data: D,
    execute: E,
}
impl<F, D, E> RasterRecipe<F> for Retained<D, E>
where
    D: PassData,
    E: RasterExecute<F, D>,
{
    fn call<'a>(
        &self,
        c: &mut RasterCommands<'a>,
        r: &mut PassResourceResolver<'a>,
        f: &F,
    ) -> crate::RecordResult {
        (self.execute)(c, r, &self.data, f)
    }
}
impl<F, D, E> ComputeRecipe<F> for Retained<D, E>
where
    D: PassData,
    E: ComputeExecute<F, D>,
{
    fn call<'a>(
        &self,
        c: &mut ComputeCommands<'a>,
        r: &mut PassResourceResolver<'a>,
        f: &F,
    ) -> crate::RecordResult {
        (self.execute)(c, r, &self.data, f)
    }
}
impl<F, D, E> CopyRecipe<F> for Retained<D, E>
where
    D: PassData,
    E: CopyExecute<F, D>,
{
    fn call<'a>(
        &self,
        c: &mut CopyCommands<'a>,
        r: &mut PassResourceResolver<'a>,
        f: &F,
    ) -> crate::RecordResult {
        (self.execute)(c, r, &self.data, f)
    }
}

pub(crate) enum StoredExecute<F> {
    Raster(Arc<dyn RasterRecipe<F>>),
    Compute(Arc<dyn ComputeRecipe<F>>),
    Copy(Arc<dyn CopyRecipe<F>>),
}

pub(crate) fn retain_raster<F, D, E>(data: D, execute: E) -> StoredExecute<F>
where
    D: PassData,
    E: RasterExecute<F, D>,
{
    StoredExecute::Raster(Arc::new(Retained { data, execute }))
}
pub(crate) fn retain_compute<F, D, E>(data: D, execute: E) -> StoredExecute<F>
where
    D: PassData,
    E: ComputeExecute<F, D>,
{
    StoredExecute::Compute(Arc::new(Retained { data, execute }))
}
pub(crate) fn retain_copy<F, D, E>(data: D, execute: E) -> StoredExecute<F>
where
    D: PassData,
    E: CopyExecute<F, D>,
{
    StoredExecute::Copy(Arc::new(Retained { data, execute }))
}

impl<F> Clone for StoredExecute<F> {
    fn clone(&self) -> Self {
        match self {
            Self::Raster(value) => Self::Raster(Arc::clone(value)),
            Self::Compute(value) => Self::Compute(Arc::clone(value)),
            Self::Copy(value) => Self::Copy(Arc::clone(value)),
        }
    }
}
