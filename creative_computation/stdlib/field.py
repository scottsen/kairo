"""Field operations implementation using NumPy backend.

This module provides NumPy-based implementations of all core field operations
for the MVP, including advection, diffusion, projection, and boundary conditions.
"""

from typing import Callable, Optional, Tuple, Union
import numpy as np


class Field2D:
    """2D field with NumPy backend.

    Represents a dense 2D grid with scalar or vector values.
    """

    def __init__(self, data: np.ndarray, dx: float = 1.0, dy: float = 1.0):
        """Initialize field.

        Args:
            data: NumPy array of field values (shape: (height, width) or (height, width, channels))
            dx: Grid spacing in x direction
            dy: Grid spacing in y direction
        """
        self.data = data
        self.dx = dx
        self.dy = dy
        self.shape = data.shape[:2]  # (height, width)

    @property
    def height(self) -> int:
        """Get field height."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Get field width."""
        return self.shape[1]

    def copy(self) -> 'Field2D':
        """Create a copy of this field."""
        return Field2D(self.data.copy(), self.dx, self.dy)

    def __repr__(self) -> str:
        return f"Field2D(shape={self.shape}, dtype={self.data.dtype})"


class FieldOperations:
    """Namespace for field operations (accessed as 'field' in DSL)."""

    @staticmethod
    def alloc(shape: Tuple[int, int], dtype: type = np.float32,
              fill_value: float = 0.0, dx: float = 1.0, dy: float = 1.0) -> Field2D:
        """Allocate a new field.

        Args:
            shape: Field shape (height, width)
            dtype: Data type
            fill_value: Initial value
            dx: Grid spacing in x
            dy: Grid spacing in y

        Returns:
            New field filled with fill_value
        """
        data = np.full(shape, fill_value, dtype=dtype)
        return Field2D(data, dx, dy)

    @staticmethod
    def advect(field: Field2D, velocity: Field2D, dt: float,
               method: str = "semi_lagrangian") -> Field2D:
        """Advect field by velocity field.

        Uses semi-Lagrangian advection (backward trace + interpolation).

        Args:
            field: Field to advect
            velocity: Velocity field (2-channel: vx, vy)
            dt: Timestep
            method: Advection method ("semi_lagrangian" only for MVP)

        Returns:
            Advected field
        """
        if method != "semi_lagrangian":
            raise NotImplementedError(f"Advection method '{method}' not implemented in MVP")

        h, w = field.shape
        result = field.copy()

        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)

        # Backward trace
        if len(velocity.data.shape) == 3 and velocity.data.shape[2] == 2:
            # Vector velocity field
            vx = velocity.data[:, :, 0]
            vy = velocity.data[:, :, 1]
        else:
            raise ValueError("Velocity field must be 2-channel (vx, vy)")

        # Trace back to source positions
        src_x = x - vx * dt / field.dx
        src_y = y - vy * dt / field.dy

        # Clamp to field boundaries
        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)

        # Bilinear interpolation
        # Get integer and fractional parts
        x0 = np.floor(src_x).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y0 = np.floor(src_y).astype(int)
        y1 = np.minimum(y0 + 1, h - 1)

        fx = src_x - x0
        fy = src_y - y0

        # Interpolate
        if len(field.data.shape) == 2:
            # Scalar field
            result.data = (
                field.data[y0, x0] * (1 - fx) * (1 - fy) +
                field.data[y0, x1] * fx * (1 - fy) +
                field.data[y1, x0] * (1 - fx) * fy +
                field.data[y1, x1] * fx * fy
            )
        else:
            # Vector field
            for c in range(field.data.shape[2]):
                result.data[:, :, c] = (
                    field.data[y0, x0, c] * (1 - fx) * (1 - fy) +
                    field.data[y0, x1, c] * fx * (1 - fy) +
                    field.data[y1, x0, c] * (1 - fx) * fy +
                    field.data[y1, x1, c] * fx * fy
                )

        return result

    @staticmethod
    def diffuse(field: Field2D, rate: float, dt: float,
                method: str = "jacobi", iterations: int = 20) -> Field2D:
        """Diffuse field using implicit solver.

        Solves: (I - α∇²) x = x₀
        where α = rate * dt

        Args:
            field: Field to diffuse
            rate: Diffusion rate
            dt: Timestep
            method: Solver method ("jacobi" only for MVP)
            iterations: Number of solver iterations

        Returns:
            Diffused field
        """
        if method != "jacobi":
            raise NotImplementedError(f"Diffusion method '{method}' not implemented in MVP")

        alpha = rate * dt
        h, w = field.shape

        # Jacobi iteration: x^(k+1) = (x₀ + α * neighbors) / (1 + 4α)
        result = field.copy()
        x0 = field.data.copy()

        for _ in range(iterations):
            # Get neighbors (with boundary handling)
            left = np.roll(result.data, 1, axis=1)
            right = np.roll(result.data, -1, axis=1)
            up = np.roll(result.data, 1, axis=0)
            down = np.roll(result.data, -1, axis=0)

            # Jacobi update
            result.data = (x0 + alpha * (left + right + up + down)) / (1 + 4 * alpha)

        return result

    @staticmethod
    def project(velocity: Field2D, method: str = "jacobi",
                iterations: int = 20, tolerance: float = 1e-4) -> Field2D:
        """Make velocity field divergence-free (pressure projection).

        Solves for pressure p: ∇²p = ∇·v
        Then updates velocity: v = v - ∇p

        Args:
            velocity: Velocity field to project
            method: Solver method ("jacobi" only for MVP)
            iterations: Number of solver iterations
            tolerance: Convergence tolerance (not used in MVP)

        Returns:
            Divergence-free velocity field
        """
        if method != "jacobi":
            raise NotImplementedError(f"Projection method '{method}' not implemented in MVP")

        if velocity.data.shape[2] != 2:
            raise ValueError("Projection requires 2-channel velocity field")

        h, w = velocity.shape
        vx = velocity.data[:, :, 0]
        vy = velocity.data[:, :, 1]

        # Compute divergence
        div = np.zeros((h, w), dtype=np.float32)
        div[1:-1, 1:-1] = (
            (vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * velocity.dx) +
            (vy[2:, 1:-1] - vy[:-2, 1:-1]) / (2 * velocity.dy)
        )

        # Solve for pressure: ∇²p = div
        pressure = np.zeros((h, w), dtype=np.float32)

        for _ in range(iterations):
            # Jacobi iteration for Poisson equation
            left = np.roll(pressure, 1, axis=1)
            right = np.roll(pressure, -1, axis=1)
            up = np.roll(pressure, 1, axis=0)
            down = np.roll(pressure, -1, axis=0)

            pressure[1:-1, 1:-1] = (left[1:-1, 1:-1] + right[1:-1, 1:-1] +
                                    up[1:-1, 1:-1] + down[1:-1, 1:-1] - div[1:-1, 1:-1]) / 4

        # Compute pressure gradient
        grad_px = np.zeros((h, w), dtype=np.float32)
        grad_py = np.zeros((h, w), dtype=np.float32)

        grad_px[:, 1:-1] = (pressure[:, 2:] - pressure[:, :-2]) / (2 * velocity.dx)
        grad_py[1:-1, :] = (pressure[2:, :] - pressure[:-2, :]) / (2 * velocity.dy)

        # Subtract gradient from velocity
        result = velocity.copy()
        result.data[:, :, 0] = vx - grad_px
        result.data[:, :, 1] = vy - grad_py

        return result

    @staticmethod
    def combine(field_a: Field2D, field_b: Field2D,
                operation: Union[str, Callable] = "add") -> Field2D:
        """Combine two fields element-wise.

        Args:
            field_a: First field
            field_b: Second field
            operation: Operation ("add", "sub", "mul", "div", "min", "max") or callable

        Returns:
            Combined field
        """
        if field_a.shape != field_b.shape:
            raise ValueError(f"Field shapes must match: {field_a.shape} vs {field_b.shape}")

        result = field_a.copy()

        if callable(operation):
            result.data = operation(field_a.data, field_b.data)
        elif operation == "add":
            result.data = field_a.data + field_b.data
        elif operation == "sub":
            result.data = field_a.data - field_b.data
        elif operation == "mul":
            result.data = field_a.data * field_b.data
        elif operation == "div":
            result.data = field_a.data / (field_b.data + 1e-10)  # avoid division by zero
        elif operation == "min":
            result.data = np.minimum(field_a.data, field_b.data)
        elif operation == "max":
            result.data = np.maximum(field_a.data, field_b.data)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        return result

    @staticmethod
    def map(field: Field2D, func: Union[str, Callable]) -> Field2D:
        """Apply function to each element of field.

        Args:
            field: Input field
            func: Function to apply (callable or string name like "abs", "sin", "cos")

        Returns:
            Mapped field
        """
        result = field.copy()

        if callable(func):
            result.data = func(field.data)
        elif func == "abs":
            result.data = np.abs(field.data)
        elif func == "sin":
            result.data = np.sin(field.data)
        elif func == "cos":
            result.data = np.cos(field.data)
        elif func == "sqrt":
            result.data = np.sqrt(np.maximum(field.data, 0))
        elif func == "square":
            result.data = field.data ** 2
        elif func == "exp":
            result.data = np.exp(field.data)
        elif func == "log":
            result.data = np.log(np.maximum(field.data, 1e-10))
        else:
            raise ValueError(f"Unknown function: {func}")

        return result

    @staticmethod
    def boundary(field: Field2D, spec: str = "reflect") -> Field2D:
        """Apply boundary conditions.

        Args:
            field: Field to apply boundaries to
            spec: Boundary specification ("reflect" or "periodic")

        Returns:
            Field with boundaries applied
        """
        result = field.copy()

        if spec == "reflect":
            # Mirror boundaries (Neumann)
            result.data[0, :] = result.data[1, :]     # Top
            result.data[-1, :] = result.data[-2, :]   # Bottom
            result.data[:, 0] = result.data[:, 1]     # Left
            result.data[:, -1] = result.data[:, -2]   # Right

        elif spec == "periodic":
            # Wrap boundaries
            result.data[0, :] = result.data[-2, :]    # Top = Bottom-1
            result.data[-1, :] = result.data[1, :]    # Bottom = Top+1
            result.data[:, 0] = result.data[:, -2]    # Left = Right-1
            result.data[:, -1] = result.data[:, 1]    # Right = Left+1

        else:
            raise ValueError(f"Unknown boundary spec: {spec}")

        return result

    @staticmethod
    def random(shape: Tuple[int, int], seed: int = 0,
               low: float = 0.0, high: float = 1.0) -> Field2D:
        """Create field with random values.

        Args:
            shape: Field shape (height, width)
            seed: Random seed for determinism
            low: Minimum value
            high: Maximum value

        Returns:
            Field with random values
        """
        rng = np.random.RandomState(seed)
        data = rng.uniform(low, high, size=shape).astype(np.float32)
        return Field2D(data)


# Create singleton instance for use as 'field' namespace
field = FieldOperations()
