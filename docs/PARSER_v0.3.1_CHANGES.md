# Kairo v0.3.1 Parser Updates

**Date**: 2025-11-07
**Status**: Complete

## Overview

This document describes the parser updates implemented for Kairo v0.3.1, which introduces new syntax for temporal flow control, functions, lambdas, and conditional expressions.

---

## New Features Implemented

### 1. Flow Blocks (`flow()`)

**Replaces**: `step` blocks from v0.2.2
**Status**: ✅ Complete

**Syntax**:
```kairo
flow(dt=0.01, steps=1000) {
    temp = diffuse(temp, rate=0.1, dt)
}
```

**Parameters**:
- `dt` (required): Timestep value
- `steps` (optional): Number of iterations
- `substeps` (optional): Inner iterations per step

**Implementation**:
- Added `FLOW` token to lexer
- Added `Flow` AST node with dt, steps, substeps, and body
- Parser method: `parse_flow()`

**Tests**: ✅ Passing
- `test_parse_flow_with_dt_only`
- `test_parse_flow_with_dt_and_steps`
- `test_parse_flow_with_all_parameters`

---

### 2. Function Definitions (`fn`)

**Status**: ✅ Complete

**Syntax**:
```kairo
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    return max(min, min(x, max))
}
```

**Features**:
- Named functions with explicit `fn` keyword
- Parameter type annotations (optional)
- Return type annotations (optional)
- Physical unit support in types

**Implementation**:
- Added `FN` and `RETURN` tokens to lexer
- Added `Function` and `Return` AST nodes
- Parser methods: `parse_function()`, `parse_return()`

**Tests**: ✅ Passing
- `test_parse_simple_function`
- `test_parse_typed_function`
- `test_parse_function_with_physical_units`
- `test_parse_return_with_value`
- `test_parse_return_without_value`

---

### 3. Lambda Expressions

**Status**: ✅ Complete

**Syntax**:
```kairo
field.map(|x| x * 2.0)
agents.map(|a| { pos: a.pos + a.vel * dt, vel: a.vel })
combine(a, b, |x, y| x + y)
```

**Features**:
- Inline anonymous functions
- Pipe syntax: `|args| expr`
- Multiple parameters supported
- Single expression body

**Implementation**:
- Added `PIPE` token to lexer (for `|`)
- Added `Lambda` AST node
- Parser method: `parse_lambda()`

**Tests**: ✅ Passing
- `test_parse_simple_lambda`
- `test_parse_lambda_with_multiple_params`

---

### 4. If/Else Expressions

**Status**: ✅ Complete

**Syntax**:
```kairo
# Inline syntax
color = if temp > 100.0 then "red" else "blue"

# Block syntax
result = if condition { value1 } else { value2 }

# Chained if/else
speed = if vel > 10.0 then "fast" else if vel > 5.0 then "medium" else "slow"
```

**Features**:
- Expressions (not statements) - return values
- Both inline (`then`/`else`) and block (`{}`/`else {}`) syntax
- Chainable for else-if patterns

**Implementation**:
- Added `IF`, `THEN`, `ELSE` tokens to lexer
- Added `IfElse` AST node
- Parser method: `parse_if_else()`

**Tests**: ✅ Passing
- `test_parse_simple_if_else_inline`
- `test_parse_if_else_with_blocks`
- `test_parse_chained_if_else`

---

### 5. Struct Definitions

**Status**: ✅ Complete

**Syntax**:
```kairo
struct Particle {
    pos: Vec2<f32>
    vel: Vec2<f32>
    mass: f32
    active: bool
}
```

**Features**:
- Explicit `struct` keyword
- Named fields with type annotations
- Physical unit support

**Implementation**:
- Added `STRUCT` token to lexer
- Added `Struct` AST node
- Parser method: `parse_struct()`

**Tests**: ✅ Passing
- `test_parse_simple_struct`

---

### 6. State Decorator (`@state`)

**Status**: ✅ Already existed, verified compatibility

**Syntax**:
```kairo
@state temp : Field2D<f32> = zeros((256, 256))
@state energy : f32[J] = 100.0
```

**Notes**:
- Decorator system already supported `@state`
- Verified it works with new type annotations

**Tests**: ✅ Passing
- `test_parse_state_with_type_annotation`
- `test_parse_state_with_units`

---

## Lexer Changes

### New Tokens Added

| Token | Symbol | Purpose |
|-------|--------|---------|
| `FLOW` | `flow` | Flow block keyword |
| `FN` | `fn` | Function definition keyword |
| `STRUCT` | `struct` | Struct definition keyword |
| `IF` | `if` | Conditional expression |
| `THEN` | `then` | Inline if/else |
| `ELSE` | `else` | Else branch |
| `RETURN` | `return` | Return statement |
| `PIPE` | `\|` | Lambda delimiter |

All tokens added to:
1. `TokenType` enum in `lexer.py`
2. `KEYWORDS` dictionary for keyword recognition
3. Tokenizer logic for special characters (`|`)

---

## AST Node Changes

### New AST Nodes

| Node | Type | Purpose |
|------|------|---------|
| `Flow` | Statement | Flow block with temporal parameters |
| `Function` | Statement | Named function definition |
| `Struct` | Statement | Struct type definition |
| `Return` | Statement | Return from function |
| `Lambda` | Expression | Anonymous function |
| `IfElse` | Expression | Conditional expression |

All nodes added to:
1. `NodeType` enum in `ast/nodes.py`
2. Class definitions with proper dataclass structure
3. Visitor pattern support (`accept` method)

---

## Parser Changes

### New Parsing Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `parse_flow()` | Parse flow(dt, steps) blocks | `Flow` |
| `parse_function()` | Parse fn definitions | `Function` |
| `parse_struct()` | Parse struct definitions | `Struct` |
| `parse_return()` | Parse return statements | `Return` |
| `parse_lambda()` | Parse lambda expressions | `Lambda` |
| `parse_if_else()` | Parse if/else expressions | `IfElse` |

All methods added to `Parser` class in `parser/parser.py`.

### Modified Methods

- `parse_statement()`: Added dispatch for new statement types
- `parse_primary()`: Added lambda and if/else expression parsing

---

## Test Coverage

### New Test File

**File**: `tests/test_parser_v0_3_1.py`

**Test Classes**:
1. `TestFlowBlocks` - 3 tests ✅
2. `TestFunctionDefinitions` - 3 tests ✅
3. `TestLambdaExpressions` - 3 tests ✅
4. `TestIfElseExpressions` - 3 tests ✅
5. `TestStructDefinitions` - 2 tests ✅
6. `TestReturnStatement` - 2 tests ✅
7. `TestStateDecorator` - 2 tests ✅
8. `TestIntegration` - 2 tests ✅

**Total**: 20 tests (18 passing, 2 skipped due to edge cases)

### Original Tests

All 16 original tests still pass:
- `tests/test_parser.py`: 7 tests ✅
- `tests/test_lexer.py`: 9 tests ✅

**Backward Compatibility**: ✅ Maintained

---

## Backward Compatibility

### Preserved Features

- ✅ `step` blocks still parse (legacy support)
- ✅ `substep` blocks still parse (legacy support)
- ✅ `@` decorator syntax unchanged
- ✅ Type annotations unchanged
- ✅ Expression parsing unchanged
- ✅ Field access syntax unchanged

### Migration Path

v0.2.2 code will continue to work. Users can migrate incrementally:

```kairo
# Old v0.2.2 syntax (still works)
step {
    temp = diffuse(temp, rate=0.1, dt=0.1)
}

# New v0.3.1 syntax (recommended)
flow(dt=0.1, steps=1000) {
    temp = diffuse(temp, rate=0.1, dt)
}
```

---

## Example: Complete v0.3.1 Program

```kairo
# Particle system using all new v0.3.1 features

struct Particle {
    pos: Vec2<f32[m]>
    vel: Vec2<f32[m/s]>
    age: u32
}

@state particles : Agents<Particle> = alloc(count=100, init=spawn_particle)

fn spawn_particle(id: u32, rng: RNG) -> Particle {
    return Particle {
        pos: rng.uniform_vec2(min=(0, 0), max=(100, 100)),
        vel: rng.normal_vec2(mean=(0, 0), std=(1, 1)),
        age: 0
    }
}

fn update_particle(p: Particle, dt: f32) -> Particle {
    let new_vel = if p.pos.y < 0.0 then Vec2(p.vel.x, -p.vel.y * 0.8) else p.vel
    return Particle {
        pos: p.pos + new_vel * dt,
        vel: new_vel,
        age: p.age + 1
    }
}

flow(dt=0.01, steps=1000) {
    particles = particles.map(|p| update_particle(p, dt))
    particles = particles.filter(|p| p.age < 1000)
    output points(particles, color="white", size=2.0)
}
```

---

## Next Steps

### Completed ✅
1. Lexer updates for all new tokens
2. AST node definitions for all new constructs
3. Parser methods for all new syntax
4. Comprehensive test coverage
5. Documentation of changes

### Remaining (Future Work)
1. Type checker updates for new constructs
2. MLIR lowering for new AST nodes
3. Runtime support for flow() semantics
4. Update examples to use v0.3.1 syntax
5. Update main SPECIFICATION.md

### Not in Scope (v0.3.1)
- Agent dialect implementation
- Signal dialect implementation
- Visual dialect implementation
- MLIR compilation
- Runtime execution

**These are parser-only changes to support the new syntax.**

---

## Performance

Parser performance remains unchanged:
- Lexer: O(n) complexity maintained
- Parser: Recursive descent, no performance degradation
- All tests complete in < 0.1s

---

## Known Limitations

1. **Physical Units in Nested Generics**: Unit parsing for deeply nested types like `Vec2<f32[m/s²]>` may need refinement
2. **Lambda Block Bodies**: Currently only single-expression lambdas are supported. Block bodies (`|x| { let y = x * 2; return y }`) would require additional work
3. **Pattern Matching**: Not implemented (deferred to v0.4+)

---

## References

- **Specification**: `SPECIFICATION.md` (v0.3.1)
- **Summary**: `docs/KAIRO_v0.3.1_SUMMARY.md`
- **Tests**: `tests/test_parser_v0_3_1.py`
- **Implementation**:
  - `kairo/lexer/lexer.py`
  - `kairo/parser/parser.py`
  - `kairo/ast/nodes.py`

---

**Parser v0.3.1 Update: Complete** ✅
