//! Tests separation of the fixed vertex and fragment shader source boundaries.

use super::*;

#[test]
fn basic_material_keeps_stage_specific_source_boundaries() {
    let modules = basic_material_modules();
    assert_eq!(modules[0].stage(), ShaderStage::Vertex);
    assert_eq!(modules[1].stage(), ShaderStage::Fragment);
    assert!(modules.iter().all(|module| !module.source().is_empty()));
}
