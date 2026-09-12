//! Per-frame GPU resource and fixed-pipeline construction for WebGPU.
//!
//! Resource ownership remains with the parent session's tickets; this module
//! only constructs and destroys the JavaScript objects it is handed.

use js_sys::{Array, Object, Reflect};
use wasm_bindgen::JsValue;

use super::js::{call0, call1, call2, call3, set_js, set_raw};
use super::{Objects, WebGpuCanvasFormat};

pub(super) struct FrameDrawResources {
    pub(super) position: JsValue,
    pub(super) index: JsValue,
    pub(super) uniform: JsValue,
    pub(super) bind_group: JsValue,
}

fn buffer(device: &JsValue, size: u32, usage: u32) -> Result<JsValue, JsValue> {
    let descriptor = Object::new();
    Reflect::set(&descriptor, &"size".into(), &size.into())?;
    Reflect::set(&descriptor, &"usage".into(), &usage.into())?;
    call1(device, "createBuffer", &descriptor)
}

fn bytes_rounded(bytes: usize) -> u32 {
    u32::try_from(bytes.max(4).next_multiple_of(4)).unwrap_or(u32::MAX)
}

pub(super) fn create_frame_resources(
    device: &JsValue,
    layout: &JsValue,
    position_bytes: usize,
    index_bytes: usize,
) -> Result<FrameDrawResources, JsValue> {
    let position = buffer(device, bytes_rounded(position_bytes * 4), 0x20 | 0x8)?;
    let index = match buffer(device, bytes_rounded(index_bytes * 4), 0x10 | 0x8) {
        Ok(value) => value,
        Err(error) => {
            let _ = call0(&position, "destroy");
            return Err(error);
        }
    };
    let uniform = match buffer(device, 256, 0x40 | 0x8) {
        Ok(value) => value,
        Err(error) => {
            let _ = call0(&position, "destroy");
            let _ = call0(&index, "destroy");
            return Err(error);
        }
    };
    let bind_group = match frame_bind_group(device, layout, &uniform) {
        Ok(value) => value,
        Err(error) => {
            let _ = call0(&position, "destroy");
            let _ = call0(&index, "destroy");
            let _ = call0(&uniform, "destroy");
            return Err(error);
        }
    };
    Ok(FrameDrawResources {
        position,
        index,
        uniform,
        bind_group,
    })
}

fn frame_bind_group(
    device: &JsValue,
    layout: &JsValue,
    uniform: &JsValue,
) -> Result<JsValue, JsValue> {
    let resource = Object::new();
    set_raw(&resource, "buffer", uniform.clone())?;
    let entry = Object::new();
    set_raw(&entry, "binding", 0)?;
    set_raw(&entry, "resource", resource)?;
    let entries = Array::new();
    entries.push(&entry);
    let descriptor = Object::new();
    set_raw(&descriptor, "layout", layout.clone())?;
    set_raw(&descriptor, "entries", entries)?;
    call1(device, "createBindGroup", &descriptor)
}

pub(super) fn destroy_frame_resources(resources: Vec<FrameDrawResources>) {
    for resource in resources {
        let _ = call0(&resource.position, "destroy");
        let _ = call0(&resource.index, "destroy");
        let _ = call0(&resource.uniform, "destroy");
        drop(resource.bind_group);
    }
}

pub(super) fn unregister_uncaptured_error(objects: &Objects) {
    let _ = call2(
        &objects.device,
        "removeEventListener",
        &"uncapturederror".into(),
        objects.uncaptured.as_ref(),
    );
}

pub(super) fn pipeline(
    device: &JsValue,
    format: WebGpuCanvasFormat,
) -> Result<(JsValue, JsValue), JsValue> {
    let module = Object::new();
    set_raw(
        &module,
        "code",
        "struct U { pvm: mat4x4<f32>, color: vec4<f32> }; @group(0) @binding(0) var<uniform> u: U; struct O { @builtin(position) p: vec4<f32> }; @vertex fn vs(@location(0) p: vec3<f32>) -> O { var o: O; o.p = u.pvm * vec4<f32>(p,1.0); return o; } @fragment fn fs() -> @location(0) vec4<f32> { return u.color; }",
    )?;
    let shader = call1(device, "createShaderModule", &module)?;
    let bind = Object::new();
    set_raw(&bind, "binding", 0)?;
    // The fixed uniform is read by both vertex PVM and fragment color.
    set_raw(&bind, "visibility", 3)?;
    let buffer = Object::new();
    set_raw(&buffer, "type", "uniform")?;
    set_js(&bind, "buffer", &buffer)?;
    let layout = Object::new();
    let entries = Array::new();
    entries.push(&bind);
    set_js(&layout, "entries", &entries)?;
    let bgl = call1(device, "createBindGroupLayout", &layout)?;
    let pipeline_layout = Object::new();
    let layouts = Array::new();
    layouts.push(&bgl);
    set_js(&pipeline_layout, "bindGroupLayouts", &layouts)?;
    let pipeline_layout = call1(device, "createPipelineLayout", &pipeline_layout)?;
    let vertex = Object::new();
    set_js(&vertex, "module", &shader)?;
    set_raw(&vertex, "entryPoint", "vs")?;
    let attribute = Object::new();
    set_raw(&attribute, "shaderLocation", 0)?;
    set_raw(&attribute, "offset", 0)?;
    set_raw(&attribute, "format", "float32x3")?;
    let attributes = Array::new();
    attributes.push(&attribute);
    let vertex_buffer = Object::new();
    set_raw(&vertex_buffer, "arrayStride", 12)?;
    set_js(&vertex_buffer, "attributes", &attributes)?;
    let vertex_buffers = Array::new();
    vertex_buffers.push(&vertex_buffer);
    set_js(&vertex, "buffers", &vertex_buffers)?;
    let fragment = Object::new();
    set_js(&fragment, "module", &shader)?;
    set_raw(&fragment, "entryPoint", "fs")?;
    let target = Object::new();
    set_raw(&target, "format", format.as_str())?;
    let targets = Array::new();
    targets.push(&target);
    set_js(&fragment, "targets", &targets)?;
    let primitive = Object::new();
    set_raw(&primitive, "topology", "triangle-list")?;
    let descriptor = Object::new();
    set_js(&descriptor, "layout", &pipeline_layout)?;
    set_js(&descriptor, "vertex", &vertex)?;
    set_js(&descriptor, "fragment", &fragment)?;
    set_js(&descriptor, "primitive", &primitive)?;
    Ok((call1(device, "createRenderPipeline", &descriptor)?, bgl))
}

pub(super) fn write_buffer(
    queue: &JsValue,
    buffer: &JsValue,
    data: &JsValue,
) -> Result<(), JsValue> {
    call3(queue, "writeBuffer", buffer, &0.into(), data).map(|_| ())
}
