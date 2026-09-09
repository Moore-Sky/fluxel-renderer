//! Pass recording orchestration and per-pass recording sessions.

mod compute;
mod copy;
mod raster;
mod resolver;
mod shared;

use std::{
    cell::RefCell,
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    backend::{
        BoundBindings, ExecutionBackend, ExecutionError, RasterColorAttachment,
        RasterDepthStencilAttachment, RasterPassDescriptor, RenderObjectProvider,
    },
    internal::PassDecl,
    pass::{ComputeCommands, CopyCommands, PassKind, PassResourceResolver, RasterCommands},
    plan::PlannedPass,
};

use self::{
    compute::ComputeBridge, copy::CopyBridge, raster::RasterBridge, resolver::ResolverBridge,
    shared::CommandBridge,
};

pub(crate) enum PhysicalResource<B: ExecutionBackend> {
    Texture {
        physical: B::Texture,
        descriptor: crate::rhi::TextureDesc,
    },
    Buffer {
        physical: B::Buffer,
        descriptor: crate::rhi::BufferDesc,
    },
}

pub(crate) type PhysicalResources<B> = HashMap<crate::handles::ResourceId, PhysicalResource<B>>;

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

pub(super) struct SessionObjects<B: ExecutionBackend> {
    pub(super) session: u64,
    pub(super) bindings: RefCell<Vec<BoundBindings<B::Bindings, B::Lease>>>,
    pub(super) leases: RefCell<Vec<B::Lease>>,
}

impl<B: ExecutionBackend> SessionObjects<B> {
    fn new() -> Self {
        Self {
            session: NEXT_SESSION.fetch_add(1, Ordering::Relaxed),
            bindings: RefCell::new(Vec::new()),
            leases: RefCell::new(Vec::new()),
        }
    }

    fn finish(self, retained: &mut Vec<B::Lease>) {
        retained.extend(self.leases.into_inner());
        retained.extend(
            self.bindings
                .into_inner()
                .into_iter()
                .map(|binding| binding.lease),
        );
    }
}

pub(crate) struct RecordPassRequest<'a, B: ExecutionBackend, O, F> {
    pub backend: &'a mut B,
    pub encoder: &'a mut B::Encoder,
    pub pass: &'a PassDecl<F>,
    pub planned: &'a PlannedPass,
    pub physical: &'a PhysicalResources<B>,
    pub objects: &'a O,
    pub frame: &'a F,
    pub retained: &'a mut Vec<B::Lease>,
}

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
