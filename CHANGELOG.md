# Changelog

All notable changes to Kairo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.1] - 2025-11-14

### Added
- **Ecosystem Map Documentation** (`ECOSYSTEM_MAP.md`) - Comprehensive map of all Kairo domains, modules, and libraries
- **Documentation Accuracy Improvements**:
  - Complete rewrite of `STATUS.md` with honest assessment of what's implemented
  - Clear distinction between production-ready, experimental, and planned features
  - Accurate MLIR status (text-based IR, not real MLIR bindings)
- **Version Consistency**: Fixed `kairo/__init__.py` to match setup.py version (0.3.1)
- **Branding**: Consistent "Kairo" naming throughout (replaced "Creative Computation DSL" remnants)

### Changed
- **STATUS.md**: Complete rewrite with factual, accurate status of all components
- **CHANGELOG.md**: Rewritten to reflect actual development history accurately

### Documentation
- Clarified that MLIR implementation is text-based IR generation, not production MLIR
- Explicitly noted that Audio and Agent dialects are specifications only (no implementation)
- Accurate test counts (247 tests) and coverage details
- Honest assessment of what works vs what's planned

---

## [0.3.0] - 2025-11-06

### Added - Complete v0.3.0 Syntax Features
- **Function definitions** with typed parameters: `fn add(a: f32, b: f32) -> f32 { return a + b }`
- **Lambda expressions** with full closure support: `let f = |x| x * 2`
- **If/else expressions** returning values: `if condition then value else other`
- **Enhanced flow blocks** with dt, steps, and substeps parameters
- **Struct definitions**: `struct Point { x: f32, y: f32 }`
- **Struct literals**: `Point { x: 3.0, y: 4.0 }`
- **Return statements** with early exit
- **Recursion support**
- **Higher-order functions**
- **Physical unit type annotations**

### Implementation
- Parser support for all v0.3.0 syntax features
- Runtime interpreter support for functions, lambdas, structs, if/else
- MLIR text-based IR generation for basic operations
- Comprehensive test coverage for new features

### Testing
- **Parser tests**: Full v0.3.0 syntax coverage
- **Runtime tests**: All v0.3.0 features tested
- **MLIR tests**: Text IR generation tests (72 tests)
- **Integration tests**: End-to-end examples working

### Documentation
- `SPECIFICATION.md` updated with v0.3.0 features
- `ARCHITECTURE.md` - Finalized Kairo Stack architecture
- Architecture specifications for all core components

---

## [0.2.2] - 2025-11-05

### Added - MVP Completion: Working Field Simulations

#### Field Operations (Production-Ready)
- **`field.alloc(shape, fill_value)`** - Field allocation
- **`field.random(shape, seed, low, high)`** - Deterministic random initialization
- **`field.advect(field, velocity, dt)`** - Semi-Lagrangian advection
- **`field.diffuse(field, rate, dt, iterations)`** - Jacobi diffusion solver (20 iterations default)
- **`field.project(velocity, iterations)`** - Pressure projection for divergence-free velocity
- **`field.combine(a, b, operation)`** - Element-wise operations (add, mul, sub, div, min, max)
- **`field.map(field, func)`** - Apply functions (abs, sin, cos, sqrt, square, exp, log)
- **`field.boundary(field, spec)`** - Boundary conditions (reflect, periodic)
- **`field.laplacian(field)`** - 5-point stencil Laplacian
- **`field.gradient(field)`** - Central difference gradient
- **`field.divergence(field)`** - Divergence operator

#### Visualization (Production-Ready)
- **`visual.colorize(field, palette, vmin, vmax)`** - Scalar field to RGB with 4 palettes:
  - `grayscale` - Black → white
  - `fire` - Black → red → orange → yellow → white
  - `viridis` - Perceptually uniform, colorblind-friendly
  - `coolwarm` - Blue → white → red
- **`visual.output(visual, path, format)`** - PNG and JPEG export with Pillow
- **`visual.display(visual)`** - Interactive Pygame window
- **sRGB gamma correction** for proper display
- **Custom value range mapping** (vmin/vmax)

#### Runtime Execution
- **ExecutionContext** class with double-buffering support
- **Runtime** interpreter with full expression evaluation
- **Step-by-step execution model**
- **CLI integration**: `kairo run <file>` command working
- **Deterministic RNG** with seeding

#### Testing
- **27 field operation tests** - All passing, determinism verified
- **23 visual operation tests** - All passing
- **Integration tests** - End-to-end pipeline working
- **Total**: 66 tests passing (100% pass rate)

#### Documentation
- **`docs/GETTING_STARTED.md`** - Complete user guide (350+ lines)
  - Installation instructions
  - First simulation walkthrough
  - Core concepts explained
  - 3 complete working examples
  - API quick reference
  - Performance tips
- **`docs/TROUBLESHOOTING.md`** - Comprehensive troubleshooting guide (400+ lines)
  - Installation issues
  - Runtime errors with solutions
  - Visualization problems
  - Performance optimization
  - Known limitations

#### Examples
- Heat diffusion simulation
- Reaction-diffusion (Gray-Scott patterns)
- Velocity field projection
- Python test demonstrating full pipeline

### Implementation Details
- **`kairo/runtime/runtime.py`** - 398 lines of production runtime code
- **`kairo/stdlib/field.py`** - 369 lines of NumPy-backed field operations
- **`kairo/stdlib/visual.py`** - 217 lines of visualization code
- **NumPy backend** for all field operations
- **Pillow** for image export
- **Pygame** for interactive display

### Performance
- Field operations scale to 512×512 grids
- Parse + type-check: <100ms for typical programs
- Field operations: <1s per frame for 256×256 grid
- Jacobi solver: 20 iterations sufficient for good quality

### Determinism
- ✅ Random fields bit-identical with same seed
- ✅ All operations reproducible across runs
- ✅ No external sources of randomness
- ✅ Verified through automated tests

---

## [0.2.0] - 2025-01 (Early Development)

### Added - Language Frontend Foundation

#### Lexer
- **60+ token types** (numbers, strings, identifiers, keywords, operators)
- **Physical unit annotations**: `[m]`, `[m/s]`, `[Hz]`, `[K]`, etc.
- **Decorator syntax**: `@state`, `@param`, `@double_buffer`
- **Comment handling** (single-line with `#`)
- **Source location tracking** for error messages
- **Error reporting** with line and column numbers

#### Parser
- **Recursive descent parser** building complete AST
- **Expression parsing**: literals, identifiers, binary/unary ops, function calls, field access
- **Statement parsing**: assignments, functions, flow blocks
- **Type annotations** with physical units: `Field2D<f32 [K]>`
- **Operator precedence** (PEMDAS)
- **Error recovery** and reporting

#### Type System
- **Scalar types**: `f32`, `f64`, `i32`, `u64`, `bool`
- **Vector types**: `Vec2<f32>`, `Vec3<f32>` with physical units
- **Field types**: `Field2D<T>`, `Field3D<T>`
- **Signal types**: `Signal<T>`
- **Agent types**: `Agents<Record>`
- **Visual type**
- **Type compatibility checking**
- **Unit compatibility** (annotations only, not enforced)

#### AST
- **Expression nodes**: Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess
- **Statement nodes**: Assignment, Flow, Function, Struct
- **Type annotation nodes**
- **Decorator nodes**
- **Visitor pattern** for traversal
- **AST printer** for debugging

#### Type Checker
- **Type inference** for expressions
- **Symbol table management** with scoping
- **Type compatibility validation**
- **Unit checking** (annotation-level)
- **Error collection and reporting**

### Testing
- **18 lexer and parser tests** - All passing
- **Full coverage** of frontend components

### Project Structure
- **Package setup**: `setup.py` and `pyproject.toml`
- **CLI framework**: `kairo` command with subcommands
- **Directory organization**: `kairo/lexer/`, `kairo/parser/`, `kairo/ast/`
- **Test infrastructure**: `tests/` with pytest configuration

### Documentation
- **`README.md`** - Project overview
- **`SPECIFICATION.md`** - Complete language specification (~47KB)
- **`LANGUAGE_REFERENCE.md`** - Quick reference guide
- **`docs/architecture.md`** - Architecture overview

---

## [0.1.0] - 2025-01 (Initial Concept)

### Added - Project Initialization
- **Project concept**: Typed, deterministic DSL for creative computation
- **Initial design**: Unifying simulation, sound, visualization, and procedural design
- **Core principles**:
  - Determinism by default
  - Explicit temporal model
  - Declarative state management
  - Physical units in type system
  - Multi-domain support
- **Repository setup**: GitHub repository, license (MIT), initial documentation

---

## Upcoming (Planned)

### [0.4.0] - Agent Dialect Implementation (3-4 months)
- Implement `Agents<T>` type and data structure
- Agent operations: `map`, `filter`, `reduce`, `spawn`
- Force calculations: gravity, springs, collision
- Field-agent coupling
- Boids, flocking, and particle system examples

### [0.5.0] - Audio Dialect Implementation (6-8 months)
- Implement Kairo.Audio operations from specification
- Oscillators: sine, saw, tri, square, noise
- Filters: lpf, hpf, bpf, notch, allpass
- Envelopes: ADSR, AR, exponential
- Physical modeling: waveguides, resonant bodies
- Audio I/O and real-time rendering
- Example compositions and instruments

### [0.6.0] - Real MLIR Integration (12+ months)
- Integrate actual `mlir-python-bindings`
- Implement real MLIR dialects (not text-based)
- LLVM lowering and optimization passes
- Native code generation
- GPU compilation pipeline
- Performance benchmarking vs Python interpreter

### [1.0.0] - Production Release (18-24 months)
- All dialects complete and production-ready
- Physical unit dimensional analysis enforced at runtime
- Hot-reload implementation
- Performance optimization and tuning
- Production-ready tooling and CLI
- Comprehensive examples and tutorials
- Video documentation and courses
- Community contributions and ecosystem

---

## Release Philosophy

### Version Numbering
- **Major (X.0.0)**: Significant new dialects or breaking changes
- **Minor (0.X.0)**: New features, dialect additions, significant improvements
- **Patch (0.0.X)**: Bug fixes, documentation, minor improvements

### Development Status
- **v0.1-0.3**: Language foundation (lexer, parser, type system, runtime)
- **v0.4-0.6**: Domain libraries (agents, audio, real MLIR)
- **v0.7-0.9**: Production readiness (optimization, tooling, polish)
- **v1.0+**: Stable, production-ready platform

### Honesty in Releases
All changelog entries reflect **actual implemented features**, not aspirational roadmap items. Features are marked as:
- ✅ **Production-Ready**: Fully implemented, tested, documented
- 🚧 **Experimental**: Working but not production-quality
- 📋 **Planned**: Designed but not yet implemented
- ❌ **Not Implemented**: Specification exists, no code

---

## Notes

### Rebranding (0.3.0)
- Project renamed from "Creative Computation DSL" to "Kairo"
- More memorable, unique branding
- Reflects evolution from DSL to full creative computation platform

### MLIR Clarification (0.3.1)
- MLIR implementation is **text-based IR generation**, not real MLIR bindings
- Designed for development and validation without full LLVM build
- Real MLIR integration planned for v0.6.0

### Audio/Agent Status (0.3.1)
- **Specifications complete**: Full design documents exist
- **No implementation**: Zero code for Audio or Agent dialects
- **Intentional**: Focus on solid foundation first (fields, runtime, visualization)
- **Timeline**: Audio planned for v0.5.0, Agents for v0.4.0

---

**For detailed status of all components:** See [STATUS.md](STATUS.md)
**For architecture overview:** See [ARCHITECTURE.md](ARCHITECTURE.md)
**For ecosystem roadmap:** See [ECOSYSTEM_MAP.md](ECOSYSTEM_MAP.md)
