# Fluxel patch for wgpu-hal 30.0.1

This directory is the crates.io `wgpu-hal` 30.0.1 source under its original
MIT/Apache-2.0 licenses. Fluxel carries one DX12 fix from upstream commit
`7d1314d537216a2c84b6a4bbadd4df7b87fc0a6b` / gfx-rs/wgpu#10221:

- lower `SamplerDescriptor::compare == None` to
  `D3D12_COMPARISON_FUNC_NONE`, not the default `ALWAYS` value.

The crates.io implementation triggers D3D12 validation error #1361 when a
standard filtering sampler is created. Remove this patch and the workspace
`[patch.crates-io]` entry after a Rust-1.87-compatible upstream release contains
the fix.
