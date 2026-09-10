//! Freezes the six audited closed recipe mappings.

use super::*;

#[test]
fn six_closed_recipes_freeze_audited_kernel_and_topology() {
    let cases = [
        (
            RasterRecipe::LEGACY_UNLIT,
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            camera_pipeline(),
            camera_bindings(),
            VertexRecipe::Position,
            TextureRecipe::None,
            BindingRecipe::Uniform,
            MeshDomain::Indexed,
            TextureDomain::None,
            ReservationRecipe::Mesh,
            ExportRecipe::Camera,
        ),
        (
            RasterRecipe::POSITION_TEXTURE_LOAD,
            RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture,
            textured_pipeline(),
            textured_bindings(),
            VertexRecipe::Position,
            TextureRecipe::LinearLoadMip0,
            BindingRecipe::TextureLoad,
            MeshDomain::Indexed,
            TextureDomain::Linear,
            ReservationRecipe::MeshAndTexture,
            ExportRecipe::Texture,
        ),
        (
            RasterRecipe::UV_TEXTURE_LOAD,
            RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv,
            RasterPipelineId::new(0x0240_0001),
            BindingSetId::new(0x0240_0001),
            VertexRecipe::PositionUv,
            TextureRecipe::LinearLoadMip0,
            BindingRecipe::UvTextureLoad,
            MeshDomain::TexturedIndexed,
            TextureDomain::Linear,
            ReservationRecipe::MeshAndTexture,
            ExportRecipe::UvTexture,
        ),
        (
            RasterRecipe::UV_LINEAR_CLAMP_UNORM,
            RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp,
            RasterPipelineId::new(0x0250_0001),
            BindingSetId::new(0x0250_0001),
            VertexRecipe::PositionUv,
            TextureRecipe::LinearClampMip0,
            BindingRecipe::UvLinearClamp,
            MeshDomain::TexturedIndexed,
            TextureDomain::Linear,
            ReservationRecipe::MeshAndTexture,
            ExportRecipe::UvTexture,
        ),
        (
            RasterRecipe::UV_LINEAR_CLAMP_SRGB,
            RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb,
            RasterPipelineId::new(0x0260_0001),
            BindingSetId::new(0x0260_0001),
            VertexRecipe::PositionUv,
            TextureRecipe::SrgbLinearClampMip0,
            BindingRecipe::UvLinearClampSrgb,
            MeshDomain::TexturedIndexed,
            TextureDomain::Srgb,
            ReservationRecipe::MeshAndTexture,
            ExportRecipe::UvTexture,
        ),
        (
            RasterRecipe::NORMAL_LAMBERT,
            RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert,
            normal_lambert_pipeline(),
            normal_lambert_bindings(),
            VertexRecipe::PositionNormal,
            TextureRecipe::None,
            BindingRecipe::NormalLambert,
            MeshDomain::NormalIndexed,
            TextureDomain::None,
            ReservationRecipe::Mesh,
            ExportRecipe::NormalLambert,
        ),
    ];
    for (
        recipe,
        kernel,
        pipeline,
        bindings,
        vertex,
        texture,
        binding,
        mesh,
        domain,
        reservation,
        export,
    ) in cases
    {
        assert_eq!(recipe.kernel(), kernel);
        assert_eq!((recipe.pipeline(), recipe.bindings()), (pipeline, bindings));
        assert_eq!(
            (recipe.vertex(), recipe.texture(), recipe.binding()),
            (vertex, texture, binding)
        );
        assert_eq!(
            (recipe.mesh_domain(), recipe.texture_domain()),
            (mesh, domain)
        );
        assert_eq!(
            (recipe.reservation(), recipe.export()),
            (reservation, export)
        );
    }
}
