# GQL Aura Kernel Manifest

`Kernel.aur` is a versioned, human-readable declaration of the intended GQL runtime lifecycle, modules, security posture, and future PSC extension point.

## Current status

This is an **architecture manifest**, not an executable kernel. The current service remains the existing Python runtime under `gpu/runtime/`, with HTTP and GraphQL API handling, a job registry, hardware diagnostics, and opt-in trusted workers. Adding this manifest does not change those code paths.

## Design boundaries

- Existing GQL APIs and worker behavior remain authoritative.
- Inference workers stay disabled unless explicitly enabled by configuration.
- The manifest does not authorize arbitrary script, shell, or native-code execution.
- PSC files (`Config.psc` and `Script.psc`) are a planned extension only. No PSC parser or interpreter is claimed to exist.
- The resource limits in `Kernel.aur` are proposed policy values; they are not enforced until a runtime implementation and tests are added.
- CPU, CUDA, and ROCm availability depends on the deployed image, dependencies, host drivers, and hardware.

## Planned implementation sequence

1. Define and version the Aura manifest grammar.
2. Add a parser and schema validation with rejection of unknown or unsafe directives.
3. Map only supported manifest fields into typed GQL configuration; do not dynamically execute manifest content.
4. Implement PSC parsing separately with a restricted AST/interpreter, explicit capabilities, and enforced step, memory, and wall-clock limits.
5. Add unit tests for valid manifests, malformed input, denied capabilities, limit enforcement, and backward compatibility.
6. Integrate behind an opt-in feature flag only after tests and security review pass.

## Review checklist

- [ ] Grammar and version compatibility specified
- [ ] Parser rejects malformed and unknown security-sensitive directives
- [ ] No shell/eval/native execution from manifest text
- [ ] Policy defaults are deny-by-default
- [ ] Resource limits are enforced by runtime, not merely documented
- [ ] Existing API, GraphQL, job lifecycle, and trusted-worker tests pass
- [ ] CI validates the manifest and PSC fixtures

## Example

See the root-level `Kernel.aur` file. Its declarations are illustrative until the parser and runtime are implemented.
