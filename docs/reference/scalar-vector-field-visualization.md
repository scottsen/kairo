# Scalar + Vector Field Visualization Guide

## Overview

A common challenge in scientific visualization is displaying a **scalar field** (e.g., density, temperature, pressure) that lives on top of a **vector field** (e.g., velocity, force). This guide provides comprehensive techniques for visualizing both fields simultaneously without creating visual clutter.

**Key Principle:** Show both fields without making a mess.

**See Also:**
- [Visualization Ideas by Domain](visualization-ideas-by-domain.md) - Domain-specific visualization examples and cross-domain compositions
- [Interactive Visualization Guide](../architecture/interactive-visualization.md) - Real-time visualization API and controls

---

## Table of Contents

1. [2D Flow Visualization](#2d-flow-visualization)
2. [3D Flow Visualization](#3d-flow-visualization)
3. [Temporal Visualization](#temporal-visualization-showing-change-over-time)
4. [Experimental Methods](#experimental-methods-non-computational)
5. [Practical Tools](#practical-tools-for-implementation)
6. [Implementation Recipes](#implementation-recipes)
7. [Morphogen Implementation Status](#morphogen-implementation-status)

---

## 2D Flow Visualization

For 2D flows, you have several clear approaches depending on what aspects you want to emphasize.

### A. Color Maps (Heatmaps)

**Best for:** Seeing where density is changing

Show density as color on a 2D plane.

**Method:**
- Take a slice (or your full 2D domain)
- Plot `ρ(x, y)` as an image
  - Bright ↔ high density, dark ↔ low density (or vice versa)
  - **Use perceptually uniform colormaps** (e.g., viridis), not "rainbow"
  - Morphogen supports: grayscale, fire, viridis, coolwarm

**For changing density in time:**
- Create an animation (frames over time)
- Use an interactive slider over time steps

**Morphogen Status:** ✅ **Fully Implemented**
```python
# Current implementation
density = field.random((128, 128), seed=42, low=0.0, high=1.0)
vis = visual.colorize(density, palette="viridis", vmin=0.0, vmax=1.0)
visual.output(vis, "density_field.png")
```

**Why this works:** Usually the clearest way to see where density is changing.

---

### B. Contours / Filled Contours

**Best for:** Exact density levels (shock thickness, mixing layer extent)

**Method:**
- Draw **contour lines** of constant density (like elevation lines on a map)
- Or **filled contours** (regions between contour values filled with color)

**Overlay options:**
- Velocity vectors (quiver plot)
- Streamlines

**Example composite:** Streamlines over colored density field

**Morphogen Status:** 📋 **Planned** (needs contour extraction)
```python
# Future API
scalar_field = field.gaussian_bump((256, 256), center=(128, 128), sigma=30)
contours = field.contours(scalar_field, levels=10)

scene = scene.create("contours")
for i, contour in enumerate(contours):
    curve = geo.curve_from_points(contour.points)
    curve.set_color(palette.get("terrain", i/10))
    scene.add(curve)
```

---

### C. Velocity + Density Together

**Best for:** Seeing how dense/less-dense fluid is transported

#### Option 1: Quiver + Colormap

**Setup:**
- Background image: density (as colormap)
- Arrows: velocity vectors (sparse, to avoid clutter)

**Morphogen Status:** 🚧 **Partially Available** (has density colorization, needs vector arrow rendering)

```python
# Current capability
density_vis = visual.colorize(density, palette="viridis")

# Future: overlay velocity arrows
velocity_arrows = geo.vector_field(velocity, bounds, spacing=10)
composite = visual.composite(density_vis, velocity_arrows, mode="over")
```

#### Option 2: Streamlines Colored by Density

**Setup:**
- Use streamlines to show trajectories
- Color each point of the streamline by:
  - Local density at that point
  - Or density of the starting point

**Morphogen Status:** 📋 **Planned** (needs streamline computation)

```python
# Future API
velocity_field = field.vector(lambda pos: rotation_field(pos))
seeds = [(i, j) for i in range(0, 10, 2) for j in range(0, 10, 2)]

scene = scene.create("streamlines")
for seed in seeds:
    curve = field.integral_curve(velocity_field, start=seed, steps=100)

    # Color by density at each point
    densities = [density_field(p) for p in curve.points]
    curve.color_by_values(densities, palette="viridis")

    scene.add(curve)
```

**Why this works:** Great for seeing how dense/less-dense fluid is transported.

---

## 3D Flow Visualization

In 3D you can't show everything at once, so you use clever strategies:

### A. Slices

**Best for:** Most practical option for 3D visualization

**Method:**
- Take planes (e.g., x-y at fixed z; y-z at fixed x)
- On each plane, use 2D techniques:
  - Color map
  - Contours
  - Vectors
- Add a few slices in different orientations for context

**Morphogen Status:** 🚧 **2D Complete, 3D Planned**

```python
# Future 3D slicing API
field_3d = field.random_3d((64, 64, 64), seed=42)
velocity_3d = field.vector_3d(...)

# Extract slices
xy_slice = field.slice_3d(field_3d, axis='z', index=32)
xz_slice = field.slice_3d(field_3d, axis='y', index=32)

# Visualize each slice with 2D methods
vis_xy = visual.colorize(xy_slice, palette="viridis")
vis_xz = visual.colorize(xz_slice, palette="viridis")
```

---

### B. Isosurfaces of Density

**Best for:**
- Shock surfaces
- Jets or plumes with a given density threshold
- Interfaces between fluids

**Method:**
- Pick a few density values ρ = constant
- Render each as a semi-transparent surface
- Can color the isosurface by another quantity (e.g., velocity magnitude)

**Morphogen Status:** 💡 **Concept** (needs 3D field operations and mesh rendering)

```python
# Future 3D isosurface API
field_3d = field.random_3d((64, 64, 64), seed=42)

# Extract isosurface at value=0.5
vertices, faces = field.isosurface(field_3d, isovalue=0.5, method="marching_cubes")
mesh = geo.mesh(vertices, faces)

# Color by gradient magnitude
gradient = field.gradient_3d(field_3d)
grad_mag = field.magnitude(gradient)
mesh.color_by_field(grad_mag, palette="viridis")

scene.add(mesh)
```

---

### C. Volume Rendering

**Best for:** Beautiful visualization when properly tuned

**Method (requires ParaView, VisIt, etc.):**
- Treat density as a volumetric dataset
- Map density to color and opacity in 3D
  - High density = more opaque/brighter
  - Low density = transparent

**Morphogen Status:** 💡 **Long-term Goal** (GPU-accelerated volume rendering)

**Note:** Looks great but can be tricky to tune parameters.

---

### D. Particle / Pathline Visualizations

**Best for:** Seeing mixing of "heavy" vs "light" fluid parcels

**Method:**
- Seed virtual particles in the flow
- Integrate trajectories (pathlines / streaklines)
- Color the particles or paths by density:
  - Instantaneous local ρ
  - Or initial ρ

**Morphogen Status:** 🚧 **Agent System Available** (needs field-agent coupling)

```python
# Current capability with agents
def particle_flow_viz():
    particles = agents.create(n=1000, properties=['density', 'velocity'])
    velocity_field = field.vector(...)

    while True:
        # Move particles along velocity field
        particles = agents.move_by_field(particles, velocity_field, dt=0.016)

        # Sample density at particle positions
        # Future: particles.sample_field(density_field, property='density')

        # Visualize colored by density
        vis = visual.agents(
            particles,
            color_property='density',
            size_property='speed',
            palette='viridis',
            trail=True,
            trail_length=20
        )
        yield vis
```

---

## Temporal Visualization (Showing Change Over Time)

Since you specifically mentioned **changing densities**, temporal visualization is critical:

### A. Animations

**Method:**
- Create a movie where each frame is density at time *t*
- **Fix the color scale across all frames** so you can compare visually

**Morphogen Status:** ✅ **Fully Implemented**

```python
def density_evolution():
    density = field.random((128, 128), seed=42)

    while True:
        # Evolve density (advection, diffusion, etc.)
        density = field.advect(density, velocity_field, dt=0.1)
        density = field.diffuse(density, rate=0.1, dt=0.1)

        # Fixed color scale for temporal comparison
        yield visual.colorize(density, palette="viridis", vmin=0.0, vmax=1.0)

# Interactive display
visual.display(density_evolution, target_fps=30)

# Or video export
visual.video(density_evolution, "density_evolution.mp4", duration=10.0, fps=30)
```

---

### B. Δρ Plots (Difference Maps)

**Best for:** Highlighting regions of change

**Method:**
- Plot `ρ(t₂) - ρ(t₁)` to see where density increased or decreased
- Positive changes in one color, negative in another
- Use diverging colormap (e.g., red/blue, coolwarm)

**Morphogen Status:** ✅ **Can Implement Now**

```python
# Compute difference between time steps
density_t1 = field.random((128, 128), seed=42)
# ... evolve to t2 ...
density_t2 = field.diffuse(density_t1, rate=0.1, dt=1.0)

# Difference map
delta_density = density_t2 - density_t1

# Visualize with diverging colormap
vis = visual.colorize(delta_density, palette="coolwarm", vmin=-0.5, vmax=0.5)
visual.output(vis, "density_change.png")
```

---

### C. Time Series at Points

**Best for:** Quantifying what you see in animations

**Method:**
- Pick some locations
- Plot ρ(t) vs time
- Helps validate and measure dynamics

**Morphogen Status:** 📋 **Planned** (needs time-series plotting)

```python
# Future API
def density_timeseries():
    density = field.random((128, 128), seed=42)
    probe_points = [(32, 32), (64, 64), (96, 96)]

    timeseries = {pt: [] for pt in probe_points}

    for t in range(1000):
        density = field.diffuse(density, rate=0.1, dt=0.1)

        # Sample at probe points
        for pt in probe_points:
            timeseries[pt].append(density[pt])

    # Plot time series
    plot = visual.line_plot(timeseries, title="Density Evolution at Probe Points")
    return plot
```

---

## Experimental Methods (Non-Computational)

For **real physical flows** with density variation:

### A. Schlieren / Shadowgraph

**Best for:** Gas flows, shock waves, buoyant plumes

**Method:**
- Visualizes refractive index gradients
- Correlates with density gradients
- No tracer particles needed

**Applications:**
- Shock waves
- Combustion
- Convection

---

### B. Laser-Induced Fluorescence (LIF)

**Best for:** Precise scalar concentration measurements

**Method:**
- Add a tracer dye
- Fluorescence intensity corresponds to concentration/density
- Can be calibrated for quantitative measurements

---

### C. PIV + Scalar Field

**Best for:** Simultaneous velocity and scalar measurements

**Method:**
- Particle Image Velocimetry (PIV) to get velocity
- Simultaneously capture scalar concentration/density with color images
- Provides both vector and scalar fields experimentally

---

## Practical Tools for Implementation

### Numerical Simulation Tools

**For Big 2D/3D CFD Datasets:**
- **ParaView** - Open source, powerful, cross-platform
- **VisIt** - Scientific visualization, designed for HPC
- **Tecplot** - Commercial, widely used in CFD

**Morphogen's Approach:**
- NumPy + SciPy for computation
- Custom visualization pipeline
- Interactive display via Pygame
- Export to PNG, MP4, GIF

**For Quick 2D Visualization (Python):**
- **matplotlib + numpy** - Quick slices, contours, animations
- **Morphogen visual.* API** - Designed for this use case

**For Interactive Web Visualizations:**
- **Plotly** - Interactive plots, 3D support
- **Bokeh** - Real-time streaming data

---

## Implementation Recipes

### Recipe 1: Simple 2D CFD Visualization

**Goal:** Visualize density field with velocity overlay

```python
from creative_computation.stdlib.field import field
from creative_computation.stdlib.visual import visual

# Load your data
density = field.random((256, 256), seed=42, low=0.0, high=1.0)  # Replace with your ρ(x,y,t)
velocity = field.vector_zeros((256, 256))  # Replace with your v(x,y,t)

# 1. Visualize density as colormap
density_vis = visual.colorize(density, palette="viridis", vmin=0.0, vmax=1.0)

# 2. Overlay velocity vectors (future: quiver plot)
# velocity_vis = visual.quiver(velocity, spacing=10, scale=0.5)
# composite = visual.composite(density_vis, velocity_vis, mode="over")

# 3. Output
visual.output(density_vis, "density_snapshot.png")
```

---

### Recipe 2: Animated Density Evolution

**Goal:** Create animation showing density changes over time

```python
def density_animation():
    """Generator yielding frames over time."""
    density = field.random((128, 128), seed=42, low=0.0, high=1.0)
    velocity = field.vector(lambda pos: [0.1, 0.0])  # Constant flow to right

    for timestep in range(200):
        # Evolve density
        density = field.advect(density, velocity, dt=0.1)
        density = field.diffuse(density, rate=0.05, dt=0.1, iterations=20)
        density = field.boundary(density, spec="periodic")

        # Fixed color scale for temporal comparison
        yield visual.colorize(density, palette="fire", vmin=0.0, vmax=1.0)

# Interactive display
gen = density_animation()
visual.display(lambda: next(gen), title="Density Evolution", target_fps=30)

# Or export to video
gen = density_animation()
visual.video(gen, "density_flow.mp4", duration=6.67, fps=30)  # 200 frames / 30 fps
```

---

### Recipe 3: Streamlines Colored by Scalar

**Goal:** Show flow trajectories colored by density

```python
# Future implementation (when streamlines are available)
def streamlines_with_density():
    velocity = field.vector(lambda pos: rotation_field(pos))
    density = field.gaussian_bump((256, 256), center=(128, 128), sigma=40)

    # Seed points in grid
    seeds = [(i, j) for i in range(20, 240, 20) for j in range(20, 240, 20)]

    scene = scene.create("streamlines")
    for seed in seeds:
        # Compute integral curve
        curve = field.integral_curve(velocity, start=seed, steps=100, dt=0.1)

        # Sample density along curve
        densities = [density[int(p[1]), int(p[0])] for p in curve.points]

        # Color by density
        curve.color_by_values(densities, palette="viridis")
        scene.add(curve)

    return scene.render()
```

**Current Workaround:** Use agent particles to approximate streamlines

```python
def particle_streamlines():
    """Use particles as streamline approximation."""
    velocity = field.vector(lambda pos: [np.sin(pos[1]*0.1), np.cos(pos[0]*0.1)])
    density = field.gaussian_bump((256, 256), center=(128, 128), sigma=40)

    # Seed particles
    particles = agents.create([
        agents.particle(pos=[i, j], vel=[0, 0])
        for i in range(20, 240, 20)
        for j in range(20, 240, 20)
    ])

    while True:
        # Move particles along velocity field
        particles = agents.move_by_field(particles, velocity, dt=0.1)

        # Color by local density (future: auto-sampling)
        # For now, manually sample density at particle positions

        vis = visual.agents(
            particles,
            color_property='speed',  # Will be 'density' when sampling works
            trail=True,
            trail_length=50,
            palette='viridis',
            blend_mode='normal'
        )
        yield vis
```

---

### Recipe 4: Comparative Time Steps

**Goal:** Show before/after or difference between time steps

```python
# Initialize
density_t0 = field.random((256, 256), seed=42, low=0.0, high=1.0)

# Evolve to t1
density_t1 = density_t0
for _ in range(100):
    density_t1 = field.diffuse(density_t1, rate=0.1, dt=0.1)

# Compute difference
delta = density_t1 - density_t0

# Create composite visualization
vis_t0 = visual.colorize(density_t0, palette="viridis", vmin=0.0, vmax=1.0)
vis_t1 = visual.colorize(density_t1, palette="viridis", vmin=0.0, vmax=1.0)
vis_delta = visual.colorize(delta, palette="coolwarm", vmin=-0.5, vmax=0.5)

# Output side by side (future: multi-panel layout)
visual.output(vis_t0, "density_t0.png")
visual.output(vis_t1, "density_t1.png")
visual.output(vis_delta, "density_change.png")
```

---

## Morphogen Implementation Status

### ✅ Currently Available (v0.6.0+)

**Scalar Field Visualization:**
- ✅ 2D color maps (heatmaps) with 4 palettes
- ✅ Custom value ranges (vmin, vmax)
- ✅ Field operations: diffusion, advection, gradients
- ✅ Boundary conditions: reflect, periodic, clamp

**Temporal Visualization:**
- ✅ Interactive display with controls (pause, step, speed)
- ✅ Video export (MP4, GIF)
- ✅ Animation loops

**Agent-Based Approximations:**
- ✅ Particle systems with trails
- ✅ Property-based coloring and sizing
- ✅ Blend modes (additive, normal, etc.)

**Composite Visualization:**
- ✅ Layer composition (5 blend modes)
- ✅ Multi-field visualization

---

### 🚧 Partially Available

**Vector Field Visualization:**
- 🚧 Vector field operations (divergence, curl, gradient)
- ⏳ Quiver plots (arrow glyphs) - **needs implementation**
- ⏳ Streamline computation - **needs implementation**

**Field-Agent Coupling:**
- ✅ Agent movement by forces
- ⏳ Agent sampling from fields - **needs implementation**
- ⏳ Agent deposition to fields - **needs implementation**

---

### 📋 Planned Features

**2D Visualization:**
- 📋 Contour extraction and rendering
- 📋 Line Integral Convolution (LIC)
- 📋 Vector arrow/glyph rendering
- 📋 Streamline integration and rendering

**3D Visualization (Phase 6):**
- 📋 3D field operations
- 📋 Isosurface extraction (marching cubes)
- 📋 Volume rendering
- 📋 3D mesh rendering
- 📋 Camera paths and orbits

**Advanced Features:**
- 📋 Multi-panel layouts
- 📋 Time-series plotting
- 📋 Histogram rendering
- 📋 Text and label rendering

---

## Key Recommendations for Morphogen

### High Priority (Immediate Impact)

1. **Vector arrow/glyph rendering**
   - Enable velocity field overlay on scalar fields
   - Critical for showing flow direction

2. **Streamline computation and rendering**
   - Most powerful way to show vector fields
   - Can be colored by scalar quantities

3. **Field-agent coupling operators**
   - `agents.sample_field()` - read field values at particle positions
   - `agents.deposit_to_field()` - write particle properties to field
   - Enables particle-based flow visualization

4. **Contour extraction**
   - Important for scientific visualization
   - Shows iso-values clearly

### Medium Priority

1. **Line Integral Convolution (LIC)**
   - Beautiful texture-based vector field visualization
   - Standard in CFD visualization

2. **Multi-panel layouts**
   - Show multiple time steps or fields side-by-side
   - Essential for comparative visualization

3. **3D slicing from 3D fields**
   - Extract 2D planes from 3D data
   - Use existing 2D visualization tools

### Future Enhancements

1. **3D isosurface rendering**
   - Industry-standard for volumetric data
   - Requires 3D mesh rendering pipeline

2. **Volume rendering**
   - Advanced 3D visualization
   - GPU-accelerated via MLIR

3. **Interactive probes and measurements**
   - Click to sample values
   - Draw regions for integration/statistics

---

## References and Further Reading

### Academic Papers
- Cabral, B., & Leedom, L.C. (1993). "Imaging vector fields using line integral convolution"
- Helman, J., & Hesselink, L. (1991). "Visualizing vector field topology in fluid flows"

### Books
- Tufte, E. (1990). "Envisioning Information"
- Ware, C. (2012). "Information Visualization: Perception for Design"

### Online Resources
- ParaView Guide: https://www.paraview.org/paraview-guide/
- VisIt User Manual: https://visit-sphinx-github-user-manual.readthedocs.io/
- SciVis Color Maps: https://sciviscolor.org/

### CFD Visualization Best Practices
- NASA Visualization Guidelines
- AIAA Visualization Standards

---

## Contributing to This Guide

This document captures visualization strategies for scalar + vector field combinations. As Morphogen's visualization capabilities expand, please:

1. Update implementation status (✅ 🚧 📋 💡)
2. Add working code examples as features become available
3. Document performance considerations
4. Share example outputs and case studies

**Last Updated:** 2025-11-20
**Morphogen Version:** v0.10.0+
**Status:** Living Document
