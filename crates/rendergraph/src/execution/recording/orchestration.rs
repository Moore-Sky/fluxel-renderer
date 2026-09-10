//! Opens each planned pass, runs its callback in one resolver session, and closes it safely.
//!
//! Pass closing happens even when callback code panics, and backend recording failures
//! take precedence over portable callback errors before control returns to frame execution.

use std::{
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
};

use crate::{
    backend::{
        ExecutionBackend, ExecutionError, RasterColorAttachment, RasterDepthStencilAttachment,
        RasterPassDescriptor, RenderObjectProvider,
    },
    pass::{ComputeCommands, CopyCommands, PassKind, PassResourceResolver, RasterCommands},
};

use super::{
    RecordPassRequest, SessionObjects,
    compute::ComputeBridge,
    copy::CopyBridge,
    raster::{self, RasterBridge},
    resolver::ResolverBridge,
    shared::CommandBridge,
};

pub(crate) fn record_pass<B, O, F>(
    request: RecordPassRequest<'_, B, O, F>,
) -> Result<(), ExecutionError<B::Error>>
where
    B: ExecutionBackend,
    O: RenderObjectProvider<B>,
{
    let RecordPassRequest {
        backend,
        encoder,
        pass,
        planned,
        physical,
        objects,
        frame,
        retained,
    } = request;
    let session = SessionObjects::<B>::new();
    let accesses: HashMap<_, _> = pass
        .accesses
        .iter()
        .map(|access| (access.handle, access))
        .collect();

    match pass.kind {
        PassKind::Raster => {
            let raster = planned.raster.as_ref().expect("raster plan is retained");
            let colors: Vec<_> = raster
                .colors
                .iter()
                .map(|attachment| RasterColorAttachment {
                    index: attachment.descriptor.index,
                    texture: raster::texture(physical, attachment.resource),
                    range: attachment.descriptor.range,
                    operations: attachment.descriptor.operations,
                })
                .collect();
            let depth_stencil =
                raster
                    .depth_stencil
                    .as_ref()
                    .map(|attachment| RasterDepthStencilAttachment {
                        texture: raster::texture(physical, attachment.resource),
                        range: attachment.descriptor.range,
                        depth: attachment.descriptor.depth,
                        stencil: attachment.descriptor.stencil,
                    });
            backend
                .begin_raster(
                    encoder,
                    &RasterPassDescriptor {
                        label: &pass.name,
                        colors: &colors,
                        depth_stencil,
                    },
                )
                .map_err(ExecutionError::Backend)?;
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut resolver_bridge = ResolverBridge::new(
                    pass.id,
                    backend.device_identity(),
                    &accesses,
                    physical,
                    objects,
                    &session,
                );
                let mut command_bridge = RasterBridge::new(CommandBridge {
                    pass: pass.id,
                    device: backend.device_identity(),
                    backend,
                    encoder,
                    accesses: &accesses,
                    physical,
                    objects,
                    session: &session,
                    backend_error: None,
                });
                let mut commands = RasterCommands::new(&mut command_bridge);
                let mut resolver = PassResourceResolver::new(&mut resolver_bridge);
                let callback = match &pass.recipe {
                    crate::internal::StoredExecute::Raster(recipe) => {
                        recipe.call(&mut commands, &mut resolver, frame)
                    }
                    _ => unreachable!(),
                };
                command_bridge.finish(callback)
            }));
            let end = backend.end_raster(encoder).map_err(ExecutionError::Backend);
            finish_pass(end, result)?;
        }
        PassKind::Compute => {
            backend
                .begin_compute(encoder, &pass.name)
                .map_err(ExecutionError::Backend)?;
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut resolver_bridge = ResolverBridge::new(
                    pass.id,
                    backend.device_identity(),
                    &accesses,
                    physical,
                    objects,
                    &session,
                );
                let mut command_bridge = ComputeBridge::new(CommandBridge {
                    pass: pass.id,
                    device: backend.device_identity(),
                    backend,
                    encoder,
                    accesses: &accesses,
                    physical,
                    objects,
                    session: &session,
                    backend_error: None,
                });
                let mut commands = ComputeCommands::new(&mut command_bridge);
                let mut resolver = PassResourceResolver::new(&mut resolver_bridge);
                let callback = match &pass.recipe {
                    crate::internal::StoredExecute::Compute(recipe) => {
                        recipe.call(&mut commands, &mut resolver, frame)
                    }
                    _ => unreachable!(),
                };
                command_bridge.finish(callback)
            }));
            let end = backend
                .end_compute(encoder)
                .map_err(ExecutionError::Backend);
            finish_pass(end, result)?;
        }
        PassKind::Copy => {
            backend
                .begin_copy(encoder, &pass.name)
                .map_err(ExecutionError::Backend)?;
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut resolver_bridge = ResolverBridge::new(
                    pass.id,
                    backend.device_identity(),
                    &accesses,
                    physical,
                    objects,
                    &session,
                );
                let mut command_bridge = CopyBridge::new(CommandBridge {
                    pass: pass.id,
                    device: backend.device_identity(),
                    backend,
                    encoder,
                    accesses: &accesses,
                    physical,
                    objects,
                    session: &session,
                    backend_error: None,
                });
                let mut commands = CopyCommands::new(&mut command_bridge);
                let mut resolver = PassResourceResolver::new(&mut resolver_bridge);
                let callback = match &pass.recipe {
                    crate::internal::StoredExecute::Copy(recipe) => {
                        recipe.call(&mut commands, &mut resolver, frame)
                    }
                    _ => unreachable!(),
                };
                command_bridge.finish(callback)
            }));
            let end = backend.end_copy(encoder).map_err(ExecutionError::Backend);
            finish_pass(end, result)?;
        }
    }
    session.finish(retained);
    Ok(())
}

fn finish_pass<E>(
    end: Result<(), ExecutionError<E>>,
    recording: std::thread::Result<Result<(), ExecutionError<E>>>,
) -> Result<(), ExecutionError<E>> {
    match recording {
        Ok(recording) => end.and(recording),
        Err(payload) => {
            // A begun native pass must be closed before user panic propagation.
            // The original panic remains the observable failure.
            let _ = end;
            resume_unwind(payload)
        }
    }
}
