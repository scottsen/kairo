"""Visual operations implementation.

This module provides visualization capabilities including field colorization
and PNG output for the MVP.
"""

from typing import Optional, Union
import numpy as np


class Visual:
    """Opaque visual representation (linear RGB).

    Stores rendered image data ready for output.
    """

    def __init__(self, data: np.ndarray):
        """Initialize visual.

        Args:
            data: RGB image data (shape: (height, width, 3), dtype: float32, range: [0, 1])
        """
        if len(data.shape) != 3 or data.shape[2] != 3:
            raise ValueError(f"Visual data must be (height, width, 3), got {data.shape}")

        self.data = np.clip(data, 0.0, 1.0).astype(np.float32)
        self.shape = data.shape[:2]

    @property
    def height(self) -> int:
        """Get image height."""
        return self.shape[0]

    @property
    def width(self) -> int:
        """Get image width."""
        return self.shape[1]

    def copy(self) -> 'Visual':
        """Create a copy of this visual."""
        return Visual(self.data.copy())

    def __repr__(self) -> str:
        return f"Visual(shape={self.shape})"


class VisualOperations:
    """Namespace for visual operations (accessed as 'visual' in DSL)."""

    # Color palettes (linear RGB values)
    PALETTES = {
        "grayscale": [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ],
        "fire": [
            (0.0, 0.0, 0.0),      # Black
            (0.5, 0.0, 0.0),      # Dark red
            (1.0, 0.0, 0.0),      # Red
            (1.0, 0.5, 0.0),      # Orange
            (1.0, 1.0, 0.0),      # Yellow
            (1.0, 1.0, 1.0),      # White
        ],
        "viridis": [
            (0.267, 0.005, 0.329),  # Dark purple
            (0.283, 0.141, 0.458),  # Purple
            (0.254, 0.266, 0.530),  # Blue-purple
            (0.207, 0.372, 0.554),  # Blue
            (0.164, 0.471, 0.558),  # Cyan-blue
            (0.135, 0.568, 0.551),  # Cyan
            (0.196, 0.664, 0.523),  # Green-cyan
            (0.395, 0.762, 0.420),  # Green
            (0.671, 0.867, 0.253),  # Yellow-green
            (0.993, 0.906, 0.144),  # Yellow
        ],
        "coolwarm": [
            (0.23, 0.30, 0.75),     # Cool blue
            (0.57, 0.77, 0.87),     # Light blue
            (0.87, 0.87, 0.87),     # White
            (0.96, 0.68, 0.52),     # Light orange
            (0.71, 0.02, 0.15),     # Warm red
        ],
    }

    @staticmethod
    def colorize(field, palette: str = "grayscale",
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None) -> Visual:
        """Map scalar field to colors using a palette.

        Args:
            field: Field2D to colorize
            palette: Palette name ("grayscale", "fire", "viridis", "coolwarm")
            vmin: Minimum value for mapping (default: field min)
            vmax: Maximum value for mapping (default: field max)

        Returns:
            Visual representation of the field
        """
        from .field import Field2D

        if not isinstance(field, Field2D):
            raise TypeError(f"Expected Field2D, got {type(field)}")

        # Get field data
        data = field.data

        # Handle multi-channel fields (use magnitude)
        if len(data.shape) == 3:
            data = np.linalg.norm(data, axis=2)

        # Normalize to [0, 1]
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)

        # Avoid division by zero
        if vmax - vmin < 1e-10:
            normalized = np.zeros_like(data)
        else:
            normalized = (data - vmin) / (vmax - vmin)

        normalized = np.clip(normalized, 0.0, 1.0)

        # Get palette colors
        if palette not in VisualOperations.PALETTES:
            raise ValueError(f"Unknown palette: {palette}. Available: {list(VisualOperations.PALETTES.keys())}")

        palette_colors = np.array(VisualOperations.PALETTES[palette])
        n_colors = len(palette_colors)

        # Map normalized values to palette indices
        indices = normalized * (n_colors - 1)
        idx_low = np.floor(indices).astype(int)
        idx_high = np.minimum(idx_low + 1, n_colors - 1)
        frac = indices - idx_low

        # Interpolate between palette colors
        h, w = normalized.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        for c in range(3):
            rgb[:, :, c] = (
                palette_colors[idx_low, c] * (1 - frac) +
                palette_colors[idx_high, c] * frac
            )

        return Visual(rgb)

    @staticmethod
    def output(visual: Visual, path: str, format: str = "auto") -> None:
        """Save visual to file.

        Args:
            visual: Visual to save
            path: Output file path
            format: Output format ("auto", "png", "jpg") - auto infers from extension

        Raises:
            ImportError: If PIL/Pillow is not installed
        """
        if not isinstance(visual, Visual):
            raise TypeError(f"Expected Visual, got {type(visual)}")

        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "PIL/Pillow is required for visual output. "
                "Install with: pip install Pillow"
            )

        # Infer format from path if auto
        if format == "auto":
            if path.endswith(".png"):
                format = "png"
            elif path.endswith(".jpg") or path.endswith(".jpeg"):
                format = "jpeg"
            else:
                format = "png"  # Default

        # Normalize format for PIL
        format_map = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG"
        }
        pil_format = format_map.get(format.lower(), "PNG")

        # Convert linear RGB to sRGB (gamma correction)
        srgb = VisualOperations._linear_to_srgb(visual.data)

        # Convert to 8-bit
        rgb_8bit = (srgb * 255).astype(np.uint8)

        # Save image
        img = Image.fromarray(rgb_8bit, mode="RGB")
        img.save(path, pil_format)

        print(f"Saved visual to: {path}")

    @staticmethod
    def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
        """Convert linear RGB to sRGB with gamma correction.

        Args:
            linear: Linear RGB values in [0, 1]

        Returns:
            sRGB values in [0, 1]
        """
        # sRGB gamma correction
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055
        )
        return np.clip(srgb, 0.0, 1.0)


# Create singleton instance for use as 'visual' namespace
visual = VisualOperations()
