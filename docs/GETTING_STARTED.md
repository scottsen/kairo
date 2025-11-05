# Getting Started with Creative Computation DSL

Welcome to Creative Computation DSL! This guide will help you get up and running in under 30 minutes.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Install from Source

```bash
# Clone the repository
git clone <repository-url>
cd tia-projects

# Install the package
pip install -e .
```

This will install Creative Computation DSL and its dependencies:
- **numpy** - For numerical operations
- **pillow** - For image output

### Verify Installation

```bash
# Check version
ccdsl version

# You should see:
# Creative Computation DSL v0.2.2
# A typed, semantics-first DSL for expressive, deterministic simulations
```

## Your First Simulation

Let's create a simple heat diffusion simulation to understand the basics.

### Example: Heat Diffusion

Create a new file called `heat.py`:

```python
from creative_computation.stdlib.field import field
from creative_computation.stdlib.visual import visual

# Create a random temperature field (64x64 grid)
temperature = field.random((64, 64), seed=42, low=0.0, high=1.0)

# Apply diffusion over 20 iterations
temperature = field.diffuse(temperature, rate=0.3, dt=0.1, iterations=20)

# Apply boundary conditions (heat reflects at edges)
temperature = field.boundary(temperature, spec="reflect")

# Visualize the result
vis = visual.colorize(temperature, palette="fire")
visual.output(vis, path="heat_diffusion.png")

print("Simulation complete! Check heat_diffusion.png")
```

Run it:

```bash
python heat.py
```

You should see a `heat_diffusion.png` file showing the smoothed temperature field with a fire color palette.

## Core Concepts

### 1. Fields

Fields are dense 2D grids that store scalar or vector values. They're the foundation for simulations.

**Create a field:**
```python
# Random field
field = field.random((128, 128), seed=0)

# Allocated field (all zeros)
field = field.alloc((256, 256), fill_value=0.0)
```

**Field operations:**
```python
# Advection (move values along velocity field)
scalar = field.advect(scalar, velocity, dt=0.01)

# Diffusion (smooth values)
field = field.diffuse(field, rate=0.1, dt=0.01, iterations=20)

# Projection (make velocity divergence-free)
velocity = field.project(velocity, iterations=20)

# Element-wise operations
result = field.combine(field_a, field_b, operation="add")

# Apply function to each element
squared = field.map(field, func="square")

# Boundary conditions
field = field.boundary(field, spec="reflect")  # or "periodic"
```

### 2. Visualization

The visual module helps you see your simulation results.

**Color Palettes:**
- `grayscale` - Black to white
- `fire` - Black → red → orange → yellow → white
- `viridis` - Perceptually uniform, colorblind-friendly
- `coolwarm` - Blue → white → red

**Example:**
```python
vis = visual.colorize(field, palette="viridis")
visual.output(vis, path="output.png")
```

### 3. Determinism

All operations with the same seed produce identical results:

```python
# These will produce identical fields
field1 = field.random((100, 100), seed=42)
field2 = field.random((100, 100), seed=42)
# field1.data == field2.data  → True
```

## Complete Examples

### Example 1: Reaction-Diffusion (Gray-Scott)

```python
from creative_computation.stdlib.field import field
from creative_computation.stdlib.visual import visual

# Initialize chemical concentrations
u = field.random((256, 256), seed=10, low=0.9, high=1.0)
v = field.random((256, 256), seed=20, low=0.0, high=0.1)

# Parameters
F = 0.055  # Feed rate
k = 0.062  # Kill rate
Du = 0.16  # Diffusion rate for U
Dv = 0.08  # Diffusion rate for V
dt = 1.0

# Simulate 100 steps
for step in range(100):
    # Diffusion
    u_diffused = field.diffuse(u, rate=Du, dt=dt, iterations=10)
    v_diffused = field.diffuse(v, rate=Dv, dt=dt, iterations=10)

    # Reaction (simplified - for MVP we'd need custom functions)
    # In full DSL: u = u - u*v*v + F*(1-u)
    # For MVP demo, just use diffused fields
    u = u_diffused
    v = v_diffused

    # Boundaries
    u = field.boundary(u, spec="periodic")
    v = field.boundary(v, spec="periodic")

    if step % 20 == 0:
        print(f"Step {step}/100")

# Visualize
vis = visual.colorize(v, palette="viridis")
visual.output(vis, path="reaction_diffusion.png")
print("Complete! See reaction_diffusion.png")
```

### Example 2: Velocity Field Smoothing

```python
from creative_computation.stdlib.field import field
from creative_computation.stdlib.visual import visual
import numpy as np

# Create a velocity field (2-channel: vx, vy)
h, w = 128, 128
vx = field.random((h, w), seed=1, low=-1.0, high=1.0)
vy = field.random((h, w), seed=2, low=-1.0, high=1.0)

# Stack into velocity field
velocity_data = np.stack([vx.data, vy.data], axis=-1)
velocity = field.Field2D(velocity_data, dx=1.0, dy=1.0)

# Make divergence-free (incompressible flow)
velocity = field.project(velocity, iterations=30)

# Visualize velocity magnitude
vx_proj = velocity.data[:, :, 0]
vy_proj = velocity.data[:, :, 1]
magnitude = np.sqrt(vx_proj**2 + vy_proj**2)
mag_field = field.Field2D(magnitude)

vis = visual.colorize(mag_field, palette="coolwarm")
visual.output(vis, path="velocity_magnitude.png")
print("Velocity field projected and visualized!")
```

## API Quick Reference

### Field Operations

| Operation | Purpose | Key Parameters |
|-----------|---------|----------------|
| `field.alloc` | Create new field | `shape`, `fill_value` |
| `field.random` | Random initialization | `shape`, `seed`, `low`, `high` |
| `field.advect` | Transport by velocity | `field`, `velocity`, `dt` |
| `field.diffuse` | Smooth/blur | `rate`, `dt`, `iterations` |
| `field.project` | Remove divergence | `iterations` |
| `field.combine` | Element-wise ops | `operation` ("add", "mul", etc.) |
| `field.map` | Apply function | `func` ("abs", "sin", "square", etc.) |
| `field.boundary` | Edge conditions | `spec` ("reflect", "periodic") |

### Visual Operations

| Operation | Purpose | Key Parameters |
|-----------|---------|----------------|
| `visual.colorize` | Map values to colors | `palette`, `vmin`, `vmax` |
| `visual.output` | Save to file | `path`, `format` |

## Common Patterns

### Pattern 1: Smooth Random Noise

```python
# Create and smooth noise
noise = field.random((200, 200), seed=123)
smooth = field.diffuse(noise, rate=1.0, dt=0.1, iterations=30)
```

### Pattern 2: Iterative Refinement

```python
# Iteratively improve a solution
field = field.random((128, 128), seed=0)

for i in range(10):
    field = field.diffuse(field, rate=0.2, dt=0.1, iterations=5)
    field = field.boundary(field, spec="reflect")
```

### Pattern 3: Multi-Scale Visualization

```python
# Visualize at different value ranges
vis_full = visual.colorize(field, palette="viridis")  # Auto range
vis_detail = visual.colorize(field, palette="viridis", vmin=0.3, vmax=0.7)

visual.output(vis_full, path="full_range.png")
visual.output(vis_detail, path="detail.png")
```

## Performance Tips

1. **Field Size**: Start with 64×64 or 128×128 for experimentation. Larger fields (512×512+) take more time.

2. **Iteration Count**: For diffusion and projection:
   - **Quick preview**: 10 iterations
   - **Good quality**: 20 iterations
   - **High accuracy**: 40+ iterations

3. **Deterministic Seeds**: Always use fixed seeds for reproducible results:
   ```python
   field = field.random((100, 100), seed=42)  # ✓ Reproducible
   ```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Reinstall with dependencies
pip install -e .
```

### No Output Image

Check that:
1. You called `visual.output()`
2. The path is writable
3. Pillow is installed: `pip install pillow`

### Simulation Too Slow

- Reduce field size: `(256, 256)` → `(128, 128)`
- Reduce iterations: `iterations=40` → `iterations=20`
- Profile your code: Focus on operations inside loops

## Next Steps

1. **Explore Examples**: Check `examples/` directory for more simulations
2. **Read Language Reference**: See `LANGUAGE_REFERENCE.md` for full language spec
3. **Understand Architecture**: See `docs/architecture.md` for implementation details
4. **Write Tests**: Add your own examples and verify determinism

## Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: Browse `examples/` for working code
- **Issues**: Report bugs or request features via GitHub issues

## MVP Limitations

The current MVP (v0.2.2) focuses on field operations. Not yet implemented:

- ❌ Agent-based systems
- ❌ Signal processing / audio
- ❌ Full DSL parser (tuple syntax, complex expressions)
- ❌ MLIR compilation (using NumPy interpreter)
- ❌ Real-time rendering
- ❌ GPU acceleration

These features are planned for future releases. See `MVP_ROADMAP.md` for details.

---

**Congratulations!** You're now ready to create your own simulations with Creative Computation DSL. Happy coding! 🎨
