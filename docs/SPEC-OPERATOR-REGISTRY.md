# SPEC: Kairo Operator Registry

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-13

---

## Overview

The **Operator Registry** is the single source of truth for all operations in Kairo. It defines:

1. **Operator signatures** — Inputs, outputs, parameters with types/units
2. **Determinism metadata** — Tier (strict/repro/live) and behavior
3. **Numeric properties** — Order, symplectic, conservative, etc.
4. **Transform metadata** — Domain changes (time→frequency, etc.)
5. **Lowering hints** — MLIR tiling, vectorization, memory patterns
6. **Profile overrides** — Per-profile behavior customization

**Design Principle:** If it's in the registry, it's documented, validated, and ready for codegen. Adding new operators becomes trivial.

---

## Registry Schema (JSON)

### Top-Level Structure

```json
{
  "version": "1.0",
  "operators": [
    {
      /* Operator definition */
    }
  ]
}
```

---

## Operator Definition

### Minimal Example

```json
{
  "name": "sine",
  "category": "oscillator",
  "description": "Sine wave oscillator",
  "inputs": [],
  "outputs": [
    {"name": "out", "type": "Stream<f32,time,audio>", "description": "Audio output"}
  ],
  "params": [
    {
      "name": "freq",
      "type": "f32<Hz>",
      "default": "440Hz",
      "description": "Frequency in Hertz"
    },
    {
      "name": "phase",
      "type": "f32<rad>",
      "default": "0rad",
      "description": "Initial phase"
    }
  ],
  "determinism": "strict",
  "rate": "audio"
}
```

---

### Full Operator Schema

```json
{
  "name": "string (required)",
  "category": "string (required)",
  "description": "string (required)",

  "inputs": [
    {
      "name": "string (required)",
      "type": "string (required)",
      "description": "string (optional)",
      "optional": "bool (default: false)"
    }
  ],

  "outputs": [
    {
      "name": "string (required)",
      "type": "string (required)",
      "description": "string (optional)"
    }
  ],

  "params": [
    {
      "name": "string (required)",
      "type": "string (required)",
      "default": "string (optional)",
      "description": "string (optional)",
      "range": "[min, max] (optional)",
      "enum": "['val1', 'val2', ...] (optional)"
    }
  ],

  "determinism": "strict | repro | live (required)",
  "rate": "audio | control | visual | sim (required)",

  "numeric_properties": {
    "order": "int (optional)",
    "symplectic": "bool (default: false)",
    "conservative": "bool (default: false)",
    "reversible": "bool (default: false)"
  },

  "transform_metadata": {
    "input_domain": "string (optional)",
    "output_domain": "string (optional)",
    "transform_type": "string (optional)"
  },

  "lowering_hints": {
    "tile_sizes": "[int, int, ...] (optional)",
    "vectorize": "bool (default: true)",
    "parallelize": "bool (default: true)",
    "memory_pattern": "string (optional)"
  },

  "profile_overrides": {
    "strict": {/* Profile-specific settings */},
    "repro": {/* Profile-specific settings */},
    "live": {/* Profile-specific settings */}
  },

  "implementation": {
    "python": "string (module path)",
    "mlir": "string (dialect.op)",
    "lowering_template": "string (optional)"
  },

  "tests": [
    {
      "name": "string",
      "inputs": {},
      "params": {},
      "expected_outputs": {},
      "tolerance": "float (default: 0)"
    }
  ]
}
```

---

## Categories

Operators are organized into **categories**:

| Category | Description | Examples |
|----------|-------------|----------|
| `oscillator` | Waveform generators | sine, saw, square, triangle, noise |
| `filter` | Frequency filters | lpf, hpf, bpf, notch, allpass |
| `envelope` | Amplitude envelopes | adsr, ar, exp_decay |
| `effect` | Audio effects | reverb, delay, chorus, flanger |
| `transform` | Domain transforms | fft, ifft, stft, istft, dct |
| `math` | Mathematical ops | add, mul, sin, cos, exp, log |
| `field` | Spatial field ops | advect, diffuse, project, laplacian |
| `agent` | Particle/agent ops | spawn, force_sum, integrate |
| `visual` | Rendering ops | colorize, render, blend |
| `control` | Control flow | gate, switch, seq |
| `utility` | Utilities | resample, delay, mix |

---

## Type System Integration

### Input/Output Types

Types follow the **SPEC-TYPE-SYSTEM.md** definitions:

```json
{
  "inputs": [
    {"name": "in", "type": "Stream<f32,time,audio>"}
  ],
  "outputs": [
    {"name": "out", "type": "Stream<f32,time,audio>"}
  ]
}
```

**Supported Types:**
- `Stream<T,Domain,Rate>` — Time-varying signals
- `Field<T,Domain>` — Spatial fields
- `Evt<A>` — Event streams
- Scalar types: `f32`, `f64`, `i32`, `bool`, etc.
- Vector types: `Vec2<f32>`, `Vec3<f32>`, etc.
- Complex types: `Complex<f32>`, `Complex<f64>`

---

### Parameter Types with Units

Parameters must include **unit annotations**:

```json
{
  "params": [
    {"name": "freq", "type": "f32<Hz>", "default": "440Hz"},
    {"name": "cutoff", "type": "f32<Hz>", "default": "2kHz"},
    {"name": "time", "type": "f32<s>", "default": "0.5s"},
    {"name": "gain", "type": "f32<dB>", "default": "-6dB"},
    {"name": "phase", "type": "f32<rad>", "default": "0rad"},
    {"name": "ratio", "type": "f32", "default": "0.5"}  // Unitless
  ]
}
```

**Validation:**
```python
def validate_param_value(param_def, value):
    """Validate parameter value against definition."""
    value_numeric, value_unit = parse_unit(value)
    param_type, param_unit = parse_type_unit(param_def["type"])

    if param_unit and value_unit != param_unit:
        raise ValueError(f"Unit mismatch: {value_unit} != {param_unit}")

    if "range" in param_def:
        min_val, max_val = param_def["range"]
        if not (min_val <= value_numeric <= max_val):
            raise ValueError(f"Value {value_numeric} out of range [{min_val}, {max_val}]")
```

---

## Determinism Metadata

Every operator declares its **determinism tier**:

```json
{
  "determinism": "strict",
  "determinism_rationale": "Uses Philox RNG with explicit seed",
  "profile_overrides": {
    "strict": {
      "rng": "philox",
      "precision": "f64"
    },
    "repro": {
      "rng": "philox",
      "precision": "f32"
    },
    "live": {
      "rng": "philox_fast",
      "precision": "f32"
    }
  }
}
```

**Determinism Tiers:**
- **strict** — Bit-exact across devices/runs
- **repro** — Deterministic within FP precision
- **live** — Replayable but not bit-exact

---

## Numeric Properties

For numerical algorithms (integrators, solvers), declare **numeric properties**:

```json
{
  "name": "rk4",
  "category": "integrator",
  "numeric_properties": {
    "order": 4,              // 4th-order accurate
    "symplectic": false,     // Not symplectic
    "conservative": false,   // Not energy-conserving
    "reversible": false,     // Not time-reversible
    "explicit": true,        // Explicit method
    "adaptive": false        // Fixed timestep
  }
}
```

**Use Cases:**
- Inform users about algorithm properties
- Enable validation (e.g., "use symplectic for Hamiltonian systems")
- Guide optimizer selection

---

## Transform Metadata

Operators that change domains (FFT, STFT, etc.) declare **transform metadata**:

```json
{
  "name": "fft",
  "category": "transform",
  "transform_metadata": {
    "input_domain": "time",
    "output_domain": "frequency",
    "transform_type": "fourier",
    "invertible": true,
    "inverse_op": "ifft"
  },
  "params": [
    {
      "name": "window",
      "type": "string",
      "default": "hann",
      "enum": ["hann", "hamming", "blackman", "kaiser", "rectangular"]
    },
    {
      "name": "norm",
      "type": "string",
      "default": "ortho",
      "enum": ["ortho", "forward", "backward"]
    }
  ]
}
```

**Validation:**
```python
def validate_transform(op, input_type):
    """Validate transform is legal for input type."""
    if input_type.domain != op.transform_metadata["input_domain"]:
        raise ValueError(
            f"Transform {op.name} expects domain={op.transform_metadata['input_domain']}, "
            f"got {input_type.domain}"
        )
```

---

## Lowering Hints

Operators provide **lowering hints** to guide MLIR code generation:

```json
{
  "name": "convolution",
  "category": "effect",
  "lowering_hints": {
    "tile_sizes": [16, 16],        // Tile spatial dims
    "vectorize": true,             // Enable vectorization
    "parallelize": true,           // Enable parallelization
    "memory_pattern": "streaming", // "streaming", "random", "stencil"
    "prefer_fft": true,            // Use FFT for large IRs
    "partition_size": 8192         // FFT partition size
  }
}
```

**Memory Patterns:**
- `streaming` — Sequential access (enable prefetch)
- `random` — Random access (disable prefetch)
- `stencil` — Neighborhood access (tile for cache locality)

---

## Profile Overrides

Operators can customize behavior per profile:

```json
{
  "name": "reverb",
  "profile_overrides": {
    "strict": {
      "fft_provider": "reference",
      "ir_cache": false,
      "precision": "f64"
    },
    "repro": {
      "fft_provider": "fftw",
      "ir_cache": true,
      "precision": "f32"
    },
    "live": {
      "fft_provider": "fastest",
      "ir_cache": true,
      "precision": "f32",
      "partition_size": 2048  // Smaller for low latency
    }
  }
}
```

---

## Implementation References

### Python Implementation

```json
{
  "implementation": {
    "python": "kairo.stdlib.oscillators.sine"
  }
}
```

**Python function signature:**
```python
def sine(freq: f32<Hz>, phase: f32<rad> = 0.0) -> Stream<f32, time, audio>:
    """Sine wave oscillator."""
    ...
```

---

### MLIR Dialect

```json
{
  "implementation": {
    "mlir": "kairo.signal.sine"
  }
}
```

**MLIR operation:**
```mlir
%out = kairo.signal.sine %freq, %phase : (f32, f32) -> !kairo.stream<f32>
```

---

### Lowering Template

For custom lowering logic:

```json
{
  "implementation": {
    "lowering_template": "templates/fft_lowering.mlir.j2"
  }
}
```

**Template (Jinja2):**
```mlir
func.func @fft_{{op.id}}(%input: tensor<{{size}}xf32>) -> tensor<{{size}}xcomplex<f32>> {
  // FFT-specific lowering
  {{#if profile.strict}}
    %result = fft.reference %input : tensor<{{size}}xf32> -> tensor<{{size}}xcomplex<f32>>
  {{else}}
    %result = fft.vendor %input : tensor<{{size}}xf32> -> tensor<{{size}}xcomplex<f32>>
  {{/if}}
  return %result : tensor<{{size}}xcomplex<f32>>
}
```

---

## Golden Test Vectors

Every operator must include **golden test vectors** for validation:

```json
{
  "tests": [
    {
      "name": "sine_440hz_strict",
      "params": {"freq": "440Hz", "phase": "0rad"},
      "duration": "1s",
      "sample_rate": 48000,
      "profile": "strict",
      "expected_output_hash": "sha256:abc123...",
      "expected_output_samples": [0.0, 0.0574, 0.1144, ...],  // First 10 samples
      "tolerance": 0.0  // Bit-exact
    },
    {
      "name": "sine_440hz_repro",
      "params": {"freq": "440Hz", "phase": "0rad"},
      "duration": "1s",
      "sample_rate": 48000,
      "profile": "repro",
      "expected_output_hash": "sha256:def456...",
      "tolerance": 1e-7  // Within FP precision
    }
  ]
}
```

**Validation:**
```python
def run_golden_test(op_def, test):
    """Run golden test for operator."""
    op = instantiate_operator(op_def, test["params"])
    output = execute_operator(op, test["duration"], test["sample_rate"], test["profile"])

    if test["tolerance"] == 0:
        # Bit-exact comparison
        expected = np.array(test["expected_output_samples"])
        assert np.array_equal(output[:len(expected)], expected)
    else:
        # Floating-point comparison
        expected = np.array(test["expected_output_samples"])
        assert np.allclose(output[:len(expected)], expected, rtol=test["tolerance"])
```

---

## Registry Operations

### Loading Registry

```python
def load_registry(path="kairo/registry/operators.json"):
    """Load operator registry from JSON."""
    with open(path) as f:
        data = json.load(f)

    registry = OperatorRegistry(version=data["version"])

    for op_def in data["operators"]:
        op = OperatorDefinition.from_dict(op_def)
        registry.register(op)

    return registry
```

---

### Querying Registry

```python
# Get operator by name
op = registry.get("sine")

# Get operators by category
oscillators = registry.get_by_category("oscillator")

# Get operators by determinism tier
strict_ops = registry.get_by_determinism("strict")

# Search operators
results = registry.search("filter", category="filter")
```

---

### Validating Operator Invocation

```python
def validate_operator_call(op_def, params, inputs):
    """Validate operator call against registry definition."""

    # Check all required params provided
    for param_def in op_def.params:
        if "default" not in param_def and param_def["name"] not in params:
            raise ValueError(f"Missing required parameter: {param_def['name']}")

    # Check parameter types and units
    for param_name, param_value in params.items():
        param_def = op_def.get_param(param_name)
        validate_param_value(param_def, param_value)

    # Check input types
    for input_def in op_def.inputs:
        if input_def["name"] in inputs:
            input_value = inputs[input_def["name"]]
            if not types_compatible(input_value.type, input_def["type"]):
                raise TypeError(
                    f"Input type mismatch: {input_value.type} != {input_def['type']}"
                )
```

---

## Codegen from Registry

### Generate Python Stubs

```python
def generate_python_stub(op_def):
    """Generate Python function stub from operator definition."""

    params_sig = ", ".join(
        f"{p['name']}: {p['type']}" + (f" = {p['default']}" if "default" in p else "")
        for p in op_def.params
    )

    inputs_sig = ", ".join(f"{i['name']}: {i['type']}" for i in op_def.inputs)

    outputs_sig = ", ".join(o["type"] for o in op_def.outputs)
    if len(op_def.outputs) == 1:
        return_type = outputs_sig
    else:
        return_type = f"Tuple[{outputs_sig}]"

    all_params = (inputs_sig + ", " + params_sig) if inputs_sig and params_sig else (inputs_sig or params_sig)

    return f"""
def {op_def.name}({all_params}) -> {return_type}:
    \"\"\"{op_def.description}\"\"\"
    # Implementation here
    pass
"""
```

---

### Generate MLIR Stubs

```python
def generate_mlir_stub(op_def):
    """Generate MLIR operation stub from operator definition."""

    inputs_mlir = ", ".join(
        f"%{i['name']}: {mlir_type(i['type'])}" for i in op_def.inputs
    )

    params_mlir = ", ".join(
        f"{p['name']}: {mlir_type(p['type'])}" for p in op_def.params
    )

    outputs_mlir = ", ".join(mlir_type(o["type"]) for o in op_def.outputs)

    return f"""
def {op_def.category}.{op_def.name}({inputs_mlir}, {params_mlir}) -> ({outputs_mlir})
"""
```

---

### Generate Documentation

```python
def generate_markdown_docs(registry):
    """Generate markdown documentation from registry."""

    md = "# Kairo Operator Reference\n\n"

    for category in registry.categories:
        md += f"## {category.title()}\n\n"

        for op in registry.get_by_category(category):
            md += f"### {op.name}\n\n"
            md += f"{op.description}\n\n"

            # Parameters table
            md += "**Parameters:**\n\n"
            md += "| Name | Type | Default | Description |\n"
            md += "|------|------|---------|-------------|\n"
            for param in op.params:
                default = param.get("default", "—")
                desc = param.get("description", "")
                md += f"| {param['name']} | {param['type']} | {default} | {desc} |\n"
            md += "\n"

            # Example
            md += "**Example:**\n\n"
            md += f"```kairo\nlet output = {op.name}("
            md += ", ".join(f"{p['name']}={p.get('default', '...')}" for p in op.params)
            md += ")\n```\n\n"

    return md
```

---

## Example: Complete Operator Definition

```json
{
  "name": "lpf",
  "category": "filter",
  "description": "Second-order lowpass filter (Butterworth)",

  "inputs": [
    {"name": "in", "type": "Stream<f32,time,audio>", "description": "Input signal"}
  ],

  "outputs": [
    {"name": "out", "type": "Stream<f32,time,audio>", "description": "Filtered output"}
  ],

  "params": [
    {
      "name": "cutoff",
      "type": "f32<Hz>",
      "default": "1kHz",
      "description": "Cutoff frequency",
      "range": [20, 20000]
    },
    {
      "name": "q",
      "type": "f32",
      "default": "0.707",
      "description": "Resonance (Q factor)",
      "range": [0.1, 10.0]
    }
  ],

  "determinism": "repro",
  "rate": "audio",

  "numeric_properties": {
    "order": 2,
    "stable": true
  },

  "lowering_hints": {
    "vectorize": true,
    "memory_pattern": "streaming"
  },

  "profile_overrides": {
    "strict": {"precision": "f64"},
    "repro": {"precision": "f32"},
    "live": {"precision": "f32"}
  },

  "implementation": {
    "python": "kairo.stdlib.filters.lpf",
    "mlir": "kairo.signal.lpf"
  },

  "tests": [
    {
      "name": "lpf_1khz_sine",
      "params": {"cutoff": "1kHz", "q": "0.707"},
      "input": "sine(440Hz)",
      "duration": "1s",
      "sample_rate": 48000,
      "profile": "repro",
      "expected_magnitude_at_440hz": 0.95,
      "expected_magnitude_at_2khz": 0.25,
      "tolerance": 0.05
    }
  ]
}
```

---

## Implementation Checklist

### Phase 1: Registry Parser
- [ ] JSON schema definition
- [ ] Registry loader (JSON → Python objects)
- [ ] Validation (required fields, type checking)

### Phase 2: Operator Definitions
- [ ] Define 50+ core operators (oscillators, filters, effects, etc.)
- [ ] Add golden test vectors for each operator
- [ ] Validate all definitions

### Phase 3: Codegen
- [ ] Python stub generator
- [ ] MLIR stub generator
- [ ] Documentation generator (markdown)

### Phase 4: Runtime Integration
- [ ] Operator instantiation from registry
- [ ] Parameter validation
- [ ] Type checking

---

## Summary

The Operator Registry provides:

✅ **Single source of truth** — All operators defined in one place
✅ **Type-safe definitions** — Inputs, outputs, params with units
✅ **Determinism metadata** — Explicit tiers and guarantees
✅ **Lowering hints** — Guide MLIR code generation
✅ **Golden tests** — Validate correctness
✅ **Codegen-ready** — Generate stubs, docs, validation

This makes **adding new operators trivial** and ensures consistency across frontends, kernel, and docs.

---

## References

- `SPEC-TYPE-SYSTEM.md` — Type definitions used in registry
- `SPEC-PROFILES.md` — Profile overrides
- `SPEC-TRANSFORM.md` — Transform metadata
- `SPEC-GRAPH-IR.md` — Graph IR uses operator names from registry
