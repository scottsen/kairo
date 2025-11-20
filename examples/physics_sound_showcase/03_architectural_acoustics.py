"""Architectural Acoustics - Physics-to-Sound Showcase

This example demonstrates Morphogen's capability to simulate room acoustics,
enabling architects and acousticians to design spaces before construction.

APPLICATION: Architectural acoustic design and optimization

PHYSICS → ACOUSTICS → AUDIO PIPELINE:
1. Room geometry (dimensions, shape, volume)
2. Material properties (absorption, diffusion, impedance)
3. Wave-based acoustic propagation (reflections, modes, diffraction)
4. Reverberation modeling (RT60, early reflections, late field)
5. Audio rendering (impulse responses, auralization)

ARCHITECTURAL USE CASES:
- Concert hall design (optimize RT60, clarity, warmth)
- Recording studio acoustic treatment
- Classroom speech intelligibility
- Home theater optimization
- Restaurant noise control
- Office acoustic comfort

REAL-WORLD VALUE:
- Architects: Design better spaces before building ($$$)
- Acousticians: Predict acoustic performance accurately
- Audio engineers: Virtual room modeling for recording/mixing
- Builders: Specify acoustic treatments correctly

Industry tools like CATT-Acoustic, EASE, Odeon cost $5k-20k.
This open-source physics-based approach is unique.
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from morphogen.stdlib import audio, field, visual, palette, io_storage
from morphogen.stdlib.audio import AudioBuffer


# ============================================================================
# ROOM GEOMETRY AND MATERIALS
# ============================================================================

@dataclass
class AcousticMaterial:
    """Acoustic material with frequency-dependent absorption."""
    name: str
    absorption_125hz: float  # Absorption coefficient at 125 Hz
    absorption_250hz: float  # at 250 Hz
    absorption_500hz: float  # at 500 Hz
    absorption_1khz: float   # at 1 kHz
    absorption_2khz: float   # at 2 kHz
    absorption_4khz: float   # at 4 kHz

    def get_absorption(self, frequency: float) -> float:
        """Interpolate absorption coefficient at given frequency."""
        freq_points = [125, 250, 500, 1000, 2000, 4000]
        abs_points = [
            self.absorption_125hz,
            self.absorption_250hz,
            self.absorption_500hz,
            self.absorption_1khz,
            self.absorption_2khz,
            self.absorption_4khz
        ]

        # Linear interpolation
        return np.interp(frequency, freq_points, abs_points)

    @property
    def average_absorption(self) -> float:
        """Average absorption coefficient (500-2k Hz)."""
        return (self.absorption_500hz + self.absorption_1khz + self.absorption_2khz) / 3.0


# Common architectural materials (octave band absorption coefficients)
CONCRETE = AcousticMaterial("Concrete", 0.01, 0.01, 0.02, 0.02, 0.02, 0.03)
WOOD_PANEL = AcousticMaterial("Wood Panel", 0.15, 0.20, 0.10, 0.08, 0.10, 0.10)
ACOUSTIC_PANEL = AcousticMaterial("Acoustic Panel", 0.30, 0.70, 0.90, 0.95, 0.90, 0.85)
CARPET = AcousticMaterial("Carpet (thick)", 0.05, 0.10, 0.20, 0.45, 0.65, 0.70)
AUDIENCE = AcousticMaterial("Seated Audience", 0.60, 0.75, 0.85, 0.95, 0.95, 0.90)
GLASS = AcousticMaterial("Glass Window", 0.15, 0.05, 0.03, 0.03, 0.02, 0.02)
GYPSUM = AcousticMaterial("Gypsum Board", 0.05, 0.08, 0.09, 0.10, 0.11, 0.04)
BRICK = AcousticMaterial("Brick (painted)", 0.01, 0.01, 0.02, 0.02, 0.02, 0.03)


@dataclass
class Room:
    """Room geometry and material specification."""
    name: str
    length: float  # meters (x-dimension)
    width: float   # meters (y-dimension)
    height: float  # meters (z-dimension)

    # Surface materials
    floor_material: AcousticMaterial
    ceiling_material: AcousticMaterial
    walls_material: AcousticMaterial

    @property
    def volume(self) -> float:
        """Room volume in cubic meters."""
        return self.length * self.width * self.height

    @property
    def surface_area(self) -> float:
        """Total surface area in square meters."""
        floor_ceiling = 2 * self.length * self.width
        side_walls = 2 * self.length * self.height
        end_walls = 2 * self.width * self.height
        return floor_ceiling + side_walls + end_walls


# ============================================================================
# REVERBERATION TIME (SABINE EQUATION)
# ============================================================================

def calculate_rt60(room: Room, frequency: float = 1000.0) -> float:
    """Calculate RT60 (reverberation time) using Sabine equation.

    Sabine equation:
    RT60 = 0.161 * V / A

    where:
        V = room volume (m³)
        A = total absorption (m² Sabine)
        A = Σ(S_i * α_i)  where S_i is surface area, α_i is absorption

    This is the time for sound to decay by 60 dB.

    Args:
        room: Room specification
        frequency: Frequency to calculate RT60 (Hz)

    Returns:
        RT60 in seconds
    """
    V = room.volume

    # Calculate total absorption
    # Floor
    S_floor = room.length * room.width
    alpha_floor = room.floor_material.get_absorption(frequency)
    A_floor = S_floor * alpha_floor

    # Ceiling
    S_ceiling = room.length * room.width
    alpha_ceiling = room.ceiling_material.get_absorption(frequency)
    A_ceiling = S_ceiling * alpha_ceiling

    # Walls (all four walls)
    S_walls = 2 * room.length * room.height + 2 * room.width * room.height
    alpha_walls = room.walls_material.get_absorption(frequency)
    A_walls = S_walls * alpha_walls

    # Total absorption
    A_total = A_floor + A_ceiling + A_walls

    # Sabine equation
    # Add small constant to avoid division by zero
    RT60 = 0.161 * V / (A_total + 1e-6)

    return RT60


def calculate_rt60_spectrum(room: Room) -> Dict[float, float]:
    """Calculate RT60 across frequency spectrum.

    Args:
        room: Room specification

    Returns:
        Dictionary mapping frequency (Hz) to RT60 (seconds)
    """
    frequencies = [125, 250, 500, 1000, 2000, 4000]

    rt60_spectrum = {}
    for freq in frequencies:
        rt60 = calculate_rt60(room, freq)
        rt60_spectrum[freq] = rt60

    return rt60_spectrum


# ============================================================================
# EARLY REFLECTIONS
# ============================================================================

def calculate_first_order_reflections(
    room: Room,
    source_position: Tuple[float, float, float],  # (x, y, z)
    listener_position: Tuple[float, float, float]  # (x, y, z)
) -> List[Tuple[float, float, str]]:
    """Calculate first-order reflections (6 surfaces).

    Returns:
        List of (delay_time, amplitude, surface_name) tuples
    """
    c = 343.0  # Speed of sound (m/s)

    sx, sy, sz = source_position
    lx, ly, lz = listener_position

    reflections = []

    # Floor reflection
    mirror_z = -sz
    dist = np.sqrt((lx - sx)**2 + (ly - sy)**2 + (lz - mirror_z)**2)
    delay = dist / c
    amplitude = 1.0 - room.floor_material.average_absorption
    reflections.append((delay, amplitude, "Floor"))

    # Ceiling reflection
    mirror_z = 2 * room.height - sz
    dist = np.sqrt((lx - sx)**2 + (ly - sy)**2 + (lz - mirror_z)**2)
    delay = dist / c
    amplitude = 1.0 - room.ceiling_material.average_absorption
    reflections.append((delay, amplitude, "Ceiling"))

    # Left wall (x=0)
    mirror_x = -sx
    dist = np.sqrt((lx - mirror_x)**2 + (ly - sy)**2 + (lz - sz)**2)
    delay = dist / c
    amplitude = 1.0 - room.walls_material.average_absorption
    reflections.append((delay, amplitude, "Left Wall"))

    # Right wall (x=length)
    mirror_x = 2 * room.length - sx
    dist = np.sqrt((lx - mirror_x)**2 + (ly - sy)**2 + (lz - sz)**2)
    delay = dist / c
    amplitude = 1.0 - room.walls_material.average_absorption
    reflections.append((delay, amplitude, "Right Wall"))

    # Front wall (y=0)
    mirror_y = -sy
    dist = np.sqrt((lx - sx)**2 + (ly - mirror_y)**2 + (lz - sz)**2)
    delay = dist / c
    amplitude = 1.0 - room.walls_material.average_absorption
    reflections.append((delay, amplitude, "Front Wall"))

    # Back wall (y=width)
    mirror_y = 2 * room.width - sy
    dist = np.sqrt((lx - sx)**2 + (ly - mirror_y)**2 + (lz - sz)**2)
    delay = dist / c
    amplitude = 1.0 - room.walls_material.average_absorption
    reflections.append((delay, amplitude, "Back Wall"))

    return reflections


# ============================================================================
# IMPULSE RESPONSE GENERATION
# ============================================================================

def generate_room_impulse_response(
    room: Room,
    source_position: Tuple[float, float, float],
    listener_position: Tuple[float, float, float],
    sample_rate: int = 44100,
    duration: float = 3.0
) -> AudioBuffer:
    """Generate room impulse response (RIR).

    The RIR contains:
    1. Direct sound (delta at t=0)
    2. Early reflections (first 50-80ms)
    3. Late reverb (exponential decay)

    Args:
        room: Room specification
        source_position: Source (x, y, z) position
        listener_position: Listener (x, y, z) position
        sample_rate: Audio sample rate
        duration: IR duration in seconds

    Returns:
        AudioBuffer with impulse response
    """
    c = 343.0  # Speed of sound
    n_samples = int(duration * sample_rate)
    ir = np.zeros(n_samples)

    # === 1. Direct sound ===
    sx, sy, sz = source_position
    lx, ly, lz = listener_position

    direct_dist = np.sqrt((lx - sx)**2 + (ly - sy)**2 + (lz - sz)**2)
    direct_delay = direct_dist / c
    direct_sample = int(direct_delay * sample_rate)

    if direct_sample < n_samples:
        # Amplitude: 1/r law (inverse distance)
        direct_amplitude = 1.0 / (direct_dist + 0.1)
        ir[direct_sample] = direct_amplitude

    # === 2. Early reflections (first-order) ===
    reflections = calculate_first_order_reflections(room, source_position, listener_position)

    for delay, amplitude, surface in reflections:
        sample_idx = int(delay * sample_rate)
        if sample_idx < n_samples:
            # Distance attenuation
            refl_amplitude = amplitude / (delay * c + 0.1)
            ir[sample_idx] += refl_amplitude * 0.5  # Scale down reflections

    # === 3. Late reverberation (diffuse field) ===
    # Exponential decay starting after early reflections (~80ms)
    early_time = 0.08  # seconds
    early_sample = int(early_time * sample_rate)

    # RT60 determines decay rate
    rt60 = calculate_rt60(room, frequency=1000.0)

    # Decay constant: -60 dB in RT60 seconds
    # Amplitude decay: A(t) = A0 * 10^(-3*t/RT60)
    # In linear: A(t) = A0 * exp(-6.91 * t / RT60)
    decay_constant = 6.91 / rt60

    t = np.arange(early_sample, n_samples) / sample_rate - early_time

    # Generate dense late reflections (noise-like)
    late_reverb = np.random.randn(n_samples - early_sample) * 0.1

    # Apply exponential decay
    envelope = np.exp(-decay_constant * t)
    late_reverb *= envelope

    ir[early_sample:] += late_reverb

    # Normalize
    if np.max(np.abs(ir)) > 0:
        ir = ir / np.max(np.abs(ir))

    return AudioBuffer(data=ir, sample_rate=sample_rate)


def auralize_sound_in_room(
    dry_sound: AudioBuffer,
    room_ir: AudioBuffer
) -> AudioBuffer:
    """Apply room impulse response to dry sound (convolution).

    This creates the "auralized" sound - what it would sound like in the room.

    Args:
        dry_sound: Dry (anechoic) audio
        room_ir: Room impulse response

    Returns:
        Wet (auralized) audio
    """
    # Convolve dry sound with room IR
    wet = np.convolve(dry_sound.data, room_ir.data, mode='same')

    # Normalize
    if np.max(np.abs(wet)) > 0:
        wet = wet / np.max(np.abs(wet))

    return AudioBuffer(data=wet, sample_rate=dry_sound.sample_rate)


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

def demo_room_types(output_dir: Path):
    """Compare different room types and their RT60."""

    print("\n" + "="*70)
    print("ROOM TYPE COMPARISON")
    print("="*70)

    rooms = {
        "Small Recording Studio": Room(
            name="Recording Studio",
            length=5.0, width=4.0, height=2.8,
            floor_material=CARPET,
            ceiling_material=ACOUSTIC_PANEL,
            walls_material=ACOUSTIC_PANEL
        ),

        "Living Room": Room(
            name="Living Room",
            length=6.0, width=5.0, height=2.7,
            floor_material=WOOD_PANEL,
            ceiling_material=GYPSUM,
            walls_material=GYPSUM
        ),

        "Concert Hall": Room(
            name="Concert Hall",
            length=30.0, width=20.0, height=12.0,
            floor_material=WOOD_PANEL,
            ceiling_material=WOOD_PANEL,
            walls_material=ACOUSTIC_PANEL
        ),

        "Lecture Hall": Room(
            name="Lecture Hall",
            length=15.0, width=12.0, height=4.0,
            floor_material=CARPET,
            ceiling_material=ACOUSTIC_PANEL,
            walls_material=GYPSUM
        ),

        "Bathroom (Very Reverberant)": Room(
            name="Bathroom",
            length=3.0, width=2.5, height=2.5,
            floor_material=CONCRETE,
            ceiling_material=CONCRETE,
            walls_material=CONCRETE
        )
    }

    for room_name, room in rooms.items():
        print(f"\n{room_name}:")
        print(f"  Dimensions: {room.length}m × {room.width}m × {room.height}m")
        print(f"  Volume: {room.volume:.1f} m³")

        rt60_spectrum = calculate_rt60_spectrum(room)

        print(f"  RT60 (1kHz): {rt60_spectrum[1000]:.2f} seconds")
        print(f"  RT60 spectrum:")
        for freq, rt60 in sorted(rt60_spectrum.items()):
            print(f"    {freq:5d} Hz: {rt60:.2f} s")


def demo_acoustic_treatment(output_dir: Path):
    """Demonstrate effect of acoustic treatment on RT60."""

    print("\n" + "="*70)
    print("ACOUSTIC TREATMENT COMPARISON")
    print("="*70)

    # Untreated room
    untreated = Room(
        name="Untreated Room",
        length=5.0, width=4.0, height=2.8,
        floor_material=WOOD_PANEL,
        ceiling_material=GYPSUM,
        walls_material=GYPSUM
    )

    # Treated room (acoustic panels on walls/ceiling)
    treated = Room(
        name="Treated Room",
        length=5.0, width=4.0, height=2.8,
        floor_material=CARPET,
        ceiling_material=ACOUSTIC_PANEL,
        walls_material=ACOUSTIC_PANEL
    )

    print(f"\nUNTREATED ROOM:")
    rt60_untreated = calculate_rt60(untreated, 1000.0)
    print(f"  RT60 (1kHz): {rt60_untreated:.2f} seconds")

    print(f"\nTREATED ROOM:")
    rt60_treated = calculate_rt60(treated, 1000.0)
    print(f"  RT60 (1kHz): {rt60_treated:.2f} seconds")

    improvement = ((rt60_untreated - rt60_treated) / rt60_untreated) * 100
    print(f"\n  Improvement: {improvement:.1f}% reduction in RT60")
    print(f"  Treatment makes the room {rt60_untreated/rt60_treated:.1f}x less reverberant")


def demo_room_auralization(output_dir: Path):
    """Generate impulse responses and auralized examples."""

    print("\n" + "="*70)
    print("ROOM AURALIZATION")
    print("="*70)

    # Create a few room types
    rooms_to_test = {
        "studio": Room(
            name="Studio",
            length=5.0, width=4.0, height=2.8,
            floor_material=CARPET,
            ceiling_material=ACOUSTIC_PANEL,
            walls_material=ACOUSTIC_PANEL
        ),

        "hall": Room(
            name="Hall",
            length=20.0, width=15.0, height=8.0,
            floor_material=WOOD_PANEL,
            ceiling_material=WOOD_PANEL,
            walls_material=ACOUSTIC_PANEL
        ),

        "bathroom": Room(
            name="Bathroom",
            length=3.0, width=2.5, height=2.5,
            floor_material=CONCRETE,
            ceiling_material=CONCRETE,
            walls_material=CONCRETE
        )
    }

    # Source and listener positions
    for room_name, room in rooms_to_test.items():
        print(f"\n{room.name}:")

        # Source at 1/4 room, listener at 3/4
        source_pos = (room.length * 0.25, room.width * 0.5, room.height * 0.5)
        listener_pos = (room.length * 0.75, room.width * 0.5, room.height * 0.5)

        # Generate impulse response
        ir = generate_room_impulse_response(
            room, source_pos, listener_pos,
            sample_rate=44100,
            duration=2.0
        )

        rt60 = calculate_rt60(room, 1000.0)
        print(f"  RT60: {rt60:.2f} seconds")
        print(f"  IR length: {len(ir.data) / ir.sample_rate:.2f} seconds")

        # Save impulse response
        ir_filename = output_dir / f"ir_{room_name}.wav"
        io_storage.save_audio(str(ir_filename), ir.data, ir.sample_rate)
        print(f"  Saved IR: {ir_filename}")

    # Create dry sound (handclap)
    print("\nGenerating dry sound (handclap)...")
    sample_rate = 44100
    dry_sound = generate_handclap(sample_rate)

    dry_filename = output_dir / "dry_handclap.wav"
    io_storage.save_audio(str(dry_filename), dry_sound.data, dry_sound.sample_rate)
    print(f"Saved dry sound: {dry_filename}")

    # Auralize in each room
    print("\nAuralizing handclap in each room...")
    for room_name, room in rooms_to_test.items():
        source_pos = (room.length * 0.25, room.width * 0.5, room.height * 0.5)
        listener_pos = (room.length * 0.75, room.width * 0.5, room.height * 0.5)

        ir = generate_room_impulse_response(
            room, source_pos, listener_pos,
            sample_rate=sample_rate,
            duration=2.0
        )

        wet_sound = auralize_sound_in_room(dry_sound, ir)

        wet_filename = output_dir / f"wet_handclap_{room_name}.wav"
        io_storage.save_audio(str(wet_filename), wet_sound.data, wet_sound.sample_rate)
        print(f"  {room.name}: {wet_filename}")


def generate_handclap(sample_rate: int = 44100) -> AudioBuffer:
    """Generate a simple handclap sound (dry).

    A handclap is a short broadband impulse - perfect for demonstrating reverb.
    """
    duration = 0.1  # 100ms
    n_samples = int(duration * sample_rate)

    # White noise burst with exponential envelope
    noise = np.random.randn(n_samples)

    # Fast attack, fast decay
    t = np.arange(n_samples) / sample_rate
    envelope = np.exp(-t / 0.01)  # 10ms decay

    handclap = noise * envelope

    # Normalize
    handclap = handclap / np.max(np.abs(handclap))

    return AudioBuffer(data=handclap, sample_rate=sample_rate)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run comprehensive architectural acoustics demo."""

    print("\n" + "="*70)
    print("MORPHOGEN ARCHITECTURAL ACOUSTICS SHOWCASE")
    print("Room Geometry → Material Physics → Audio Rendering")
    print("="*70)

    # Create output directory
    output_dir = Path(__file__).parent / "output" / "architectural"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # === DEMO 1: Room type comparison ===
    demo_room_types(output_dir)

    # === DEMO 2: Acoustic treatment ===
    demo_acoustic_treatment(output_dir)

    # === DEMO 3: Auralization ===
    demo_room_auralization(output_dir)

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(list(output_dir.glob('*.wav')))} audio files")
    print("\nKEY INSIGHTS:")
    print("  • RT60 depends on volume and surface absorption")
    print("  • Different materials dramatically affect room sound")
    print("  • Early reflections define spatial perception")
    print("  • Late reverb creates sense of space and envelopment")
    print("\nThis physics-based approach enables:")
    print("  ✓ Virtual acoustic design before construction")
    print("  ✓ Material selection optimization")
    print("  ✓ Acoustic treatment planning")
    print("  ✓ Immersive auralization for clients")
    print("\n🏛️ Architects & Acousticians: Design perfect spaces! 🏛️\n")


if __name__ == "__main__":
    main()
