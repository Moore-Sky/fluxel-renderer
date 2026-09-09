//! A typed render graph for dependency-driven GPU work planning.
//!
//! Pass setup declares versioned texture and buffer access. Compilation derives
//! dependencies, culling, capability validation, and future lowering inputs from
//! those declarations. Pass execution can use only the typed access handles its
//! own setup callback produced.
//!
//! The current release has a reusable single-queue execution plan, frame
//! resource and render-object providers, validated command recording,
//! completion retirement, a deterministic CPU-only `TestRhi`, and typed
//! resource-usage requirements. Real GPU command and surface execution remain
//! deliberate future boundaries. Start with the numbered examples and the
//! repository's `documents/design-rendergraph.md`.

#![deny(missing_docs)]

pub mod access;
pub mod backend;
pub mod compile;
pub mod error;
pub mod execution;
pub mod graph;
pub mod handles;
mod internal;
pub mod pass;
pub mod plan;
pub mod recipe;
pub mod resource;
pub mod rhi;
pub mod test_rhi;

pub use access::*;
pub use backend::*;
pub use compile::*;
pub use error::*;
pub use execution::*;
pub use graph::{DeclaredPass, ExplicitOrderReason, RenderGraph, SideEffectReason};
pub use handles::*;
pub use pass::{
    AttachmentOps, BindingResource, ColorAttachmentDesc, ComputeCommands, ComputePassBuilder,
    CopyCommands, CopyPassBuilder, DepthStencilAttachmentDesc, LoadOp, PassKind,
    PassResourceResolver, RasterCommands, RasterPassBuilder, ResolvedBindings, ScissorRect,
    StoreOp, Viewport,
};
pub use plan::*;
pub use recipe::*;
pub use resource::*;
pub use rhi::*;
