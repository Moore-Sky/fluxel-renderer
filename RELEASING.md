# Releasing Fluxel Renderer

Each workspace release uses one package version and one annotated tag. For
example, package version `0.7.0` is released as `v0.7.0`; the tag points to the
same commit consumed by all three Git dependencies.

Before creating that tag, commit the release candidate and make sure the
checkout is clean. On a Windows `x86_64-pc-windows-msvc` machine with both
DX12 and Vulkan validation available, run:

```powershell
./scripts/conformance.ps1
```

The script is deliberately fail-closed: it rejects a dirty checkout, a GNU or
other non-Microsoft host/target, an invalid `HEAD`, and a conflicting
`CARGO_BUILD_TARGET`. It runs all workspace ignored fixtures, stores the exact
SHA and test command in `target/conformance/<sha>/manifest.json`, and keeps
the combined Cargo log there even if a fixture fails. Inspect the manifest and
log before tagging. They stay out of the source commit so that the tested SHA
does not change, but both files must be uploaded as assets of the GitHub Release
for the matching tag. A release is incomplete until those durable asset URLs
exist and identify the tagged commit.

After the conformance gate and the required platform checks pass, create an
annotated tag, push the branch and tag, then verify that `origin/main` and the
remote tag resolve to the same release commit. Create the GitHub Release and
attach that commit's `manifest.json` and `cargo.log`. Do not create or push a
release tag when the hardware gate has failed or could not run.
