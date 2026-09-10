//! Closed mappings between the audited fixed-frame draw contracts and RHI recipes.

use super::*;

/// Vertex streams required by one closed fixed-frame raster contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum VertexRecipe {
    Position,
    PositionUv,
    PositionNormal,
    PositionColor,
}

/// Texture interpretation required by one closed fixed-frame raster contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TextureRecipe {
    None,
    LinearLoadMip0,
    LinearClampMip0,
    SrgbLinearClampMip0,
}

/// Graph shape token; it intentionally has no public combinator API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GraphRecipe {
    Camera,
    Textured,
    UvTextured,
    NormalLambert,
    VertexColor,
}

/// Binding topology token for a closed raster artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BindingRecipe {
    Uniform,
    TextureLoad,
    UvTextureLoad,
    UvLinearClamp,
    UvLinearClampSrgb,
    NormalLambert,
    VertexColor,
}

/// Mesh snapshot domain accepted by a closed raster artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum MeshDomain {
    Indexed,
    TexturedIndexed,
    NormalIndexed,
    VertexColorIndexed,
}

/// Texture snapshot domain accepted by a closed raster artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TextureDomain {
    None,
    Linear,
    Srgb,
}

/// Reservation topology held from draw acceptance to completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReservationRecipe {
    Mesh,
    MeshAndTexture,
}

/// Export topology produced by a closed graph declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ExportRecipe {
    Camera,
    Texture,
    UvTexture,
    NormalLambert,
    VertexColor,
}

/// One of exactly six audited fixed-frame contracts.
///
/// Fields are private so callers cannot construct a combination which the
/// graph, bindings, resource domains, and reservation topology never proved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RasterRecipe {
    kernel: RasterKernel,
    pipeline: u64,
    bindings: u64,
    vertex: VertexRecipe,
    texture: TextureRecipe,
    graph: GraphRecipe,
    binding: BindingRecipe,
    mesh_domain: MeshDomain,
    texture_domain: TextureDomain,
    reservation: ReservationRecipe,
    export: ExportRecipe,
}

macro_rules! fixed_recipe {
    ($kernel:expr, $pipeline:expr, $bindings:expr, $vertex:expr, $texture:expr, $graph:expr, $binding:expr, $mesh:expr, $domain:expr, $reservation:expr, $export:expr $(,)?) => {
        RasterRecipe {
            kernel: $kernel,
            pipeline: $pipeline,
            bindings: $bindings,
            vertex: $vertex,
            texture: $texture,
            graph: $graph,
            binding: $binding,
            mesh_domain: $mesh,
            texture_domain: $domain,
            reservation: $reservation,
            export: $export,
        }
    };
}

impl RasterRecipe {
    pub(super) const LEGACY_UNLIT: Self = fixed_recipe!(
        RasterKernel::IndexedPositionFloat32x3CameraMaterial,
        0x0220_0001,
        0x0220_0001,
        VertexRecipe::Position,
        TextureRecipe::None,
        GraphRecipe::Camera,
        BindingRecipe::Uniform,
        MeshDomain::Indexed,
        TextureDomain::None,
        ReservationRecipe::Mesh,
        ExportRecipe::Camera,
    );
    pub(super) const POSITION_TEXTURE_LOAD: Self = fixed_recipe!(
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture,
        0x0230_0001,
        0x0230_0001,
        VertexRecipe::Position,
        TextureRecipe::LinearLoadMip0,
        GraphRecipe::Textured,
        BindingRecipe::TextureLoad,
        MeshDomain::Indexed,
        TextureDomain::Linear,
        ReservationRecipe::MeshAndTexture,
        ExportRecipe::Texture,
    );
    pub(super) const UV_TEXTURE_LOAD: Self = fixed_recipe!(
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv,
        0x0240_0001,
        0x0240_0001,
        VertexRecipe::PositionUv,
        TextureRecipe::LinearLoadMip0,
        GraphRecipe::UvTextured,
        BindingRecipe::UvTextureLoad,
        MeshDomain::TexturedIndexed,
        TextureDomain::Linear,
        ReservationRecipe::MeshAndTexture,
        ExportRecipe::UvTexture,
    );
    pub(super) const UV_LINEAR_CLAMP_UNORM: Self = fixed_recipe!(
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp,
        0x0250_0001,
        0x0250_0001,
        VertexRecipe::PositionUv,
        TextureRecipe::LinearClampMip0,
        GraphRecipe::UvTextured,
        BindingRecipe::UvLinearClamp,
        MeshDomain::TexturedIndexed,
        TextureDomain::Linear,
        ReservationRecipe::MeshAndTexture,
        ExportRecipe::UvTexture,
    );
    pub(super) const UV_LINEAR_CLAMP_SRGB: Self = fixed_recipe!(
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb,
        0x0260_0001,
        0x0260_0001,
        VertexRecipe::PositionUv,
        TextureRecipe::SrgbLinearClampMip0,
        GraphRecipe::UvTextured,
        BindingRecipe::UvLinearClampSrgb,
        MeshDomain::TexturedIndexed,
        TextureDomain::Srgb,
        ReservationRecipe::MeshAndTexture,
        ExportRecipe::UvTexture,
    );
    pub(super) const NORMAL_LAMBERT: Self = fixed_recipe!(
        RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert,
        0x0270_0001,
        0x0270_0001,
        VertexRecipe::PositionNormal,
        TextureRecipe::None,
        GraphRecipe::NormalLambert,
        BindingRecipe::NormalLambert,
        MeshDomain::NormalIndexed,
        TextureDomain::None,
        ReservationRecipe::Mesh,
        ExportRecipe::NormalLambert,
    );
    pub(super) const VERTEX_COLOR: Self = fixed_recipe!(
        RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor,
        0x0280_0001,
        0x0280_0001,
        VertexRecipe::PositionColor,
        TextureRecipe::None,
        GraphRecipe::VertexColor,
        BindingRecipe::VertexColor,
        MeshDomain::VertexColorIndexed,
        TextureDomain::None,
        ReservationRecipe::Mesh,
        ExportRecipe::VertexColor,
    );

    pub(super) const fn kernel(self) -> RasterKernel {
        self.kernel
    }
    pub(super) fn pipeline(self) -> RasterPipelineId {
        RasterPipelineId::new(self.pipeline)
    }
    pub(super) fn bindings(self) -> BindingSetId {
        BindingSetId::new(self.bindings)
    }
    pub(super) const fn vertex(self) -> VertexRecipe {
        self.vertex
    }
    pub(super) const fn texture(self) -> TextureRecipe {
        self.texture
    }
    pub(super) const fn graph(self) -> GraphRecipe {
        self.graph
    }
    pub(super) const fn binding(self) -> BindingRecipe {
        self.binding
    }
    pub(super) const fn mesh_domain(self) -> MeshDomain {
        self.mesh_domain
    }
    pub(super) const fn texture_domain(self) -> TextureDomain {
        self.texture_domain
    }
    pub(super) const fn reservation(self) -> ReservationRecipe {
        self.reservation
    }
    pub(super) const fn export(self) -> ExportRecipe {
        self.export
    }

    /// Confirms that the explicit recipe and accepted private snapshots agree.
    pub(super) fn accepts(
        self,
        snapshot: &FrameMeshSnapshot,
        texture: Option<&FrameTextureSnapshot>,
    ) -> bool {
        matches!(
            (self.mesh_domain(), snapshot),
            (MeshDomain::Indexed, FrameMeshSnapshot::Indexed(_))
                | (
                    MeshDomain::TexturedIndexed,
                    FrameMeshSnapshot::TexturedUv(_)
                )
                | (MeshDomain::NormalIndexed, FrameMeshSnapshot::Normal(_))
                | (
                    MeshDomain::VertexColorIndexed,
                    FrameMeshSnapshot::VertexColor(_)
                )
        ) && matches!(
            (self.texture_domain(), texture),
            (TextureDomain::None, None)
                | (TextureDomain::Linear, Some(FrameTextureSnapshot::Linear(_)))
                | (TextureDomain::Srgb, Some(FrameTextureSnapshot::Srgb(_)))
        ) && matches!(
            (self.reservation(), texture),
            (ReservationRecipe::Mesh, None) | (ReservationRecipe::MeshAndTexture, Some(_))
        ) && matches!(
            (
                self.vertex(),
                self.texture(),
                self.graph(),
                self.binding(),
                self.export()
            ),
            (
                VertexRecipe::Position,
                TextureRecipe::None,
                GraphRecipe::Camera,
                BindingRecipe::Uniform,
                ExportRecipe::Camera
            ) | (
                VertexRecipe::Position,
                TextureRecipe::LinearLoadMip0,
                GraphRecipe::Textured,
                BindingRecipe::TextureLoad,
                ExportRecipe::Texture
            ) | (
                VertexRecipe::PositionUv,
                TextureRecipe::LinearLoadMip0,
                GraphRecipe::UvTextured,
                BindingRecipe::UvTextureLoad,
                ExportRecipe::UvTexture
            ) | (
                VertexRecipe::PositionUv,
                TextureRecipe::LinearClampMip0,
                GraphRecipe::UvTextured,
                BindingRecipe::UvLinearClamp,
                ExportRecipe::UvTexture
            ) | (
                VertexRecipe::PositionUv,
                TextureRecipe::SrgbLinearClampMip0,
                GraphRecipe::UvTextured,
                BindingRecipe::UvLinearClampSrgb,
                ExportRecipe::UvTexture
            ) | (
                VertexRecipe::PositionNormal,
                TextureRecipe::None,
                GraphRecipe::NormalLambert,
                BindingRecipe::NormalLambert,
                ExportRecipe::NormalLambert
            ) | (
                VertexRecipe::PositionColor,
                TextureRecipe::None,
                GraphRecipe::VertexColor,
                BindingRecipe::VertexColor,
                ExportRecipe::VertexColor
            )
        )
    }
}

#[cfg(test)]
mod tests;
