import numpy as np
import math
from typing import List, Optional, Tuple
from enum import Enum

class SpinorLevel(Enum):
    GROUND = 0      # s-orbital (8 bits base)
    FIRST = 1       # p-orbital (3 orientations)
    SECOND = 2      # d-orbital (5 orientations)
    THIRD = 3       # f-orbital (7 orientations)

class TorusSpinorByteWord:
    """
    A ByteWord encoded as a multilevel spinor on a torus topology
    
    Each ByteWord lives in a specific 'orbital' around the torus, with:
    - Church winding number (topological integer)
    - Spinor orientation (quantum-like state)
    - Energy level (determines orbital)
    - Hole complement (what's missing from 8-bit encoding)
    """
    
    def __init__(self, value: int, level: SpinorLevel = SpinorLevel.GROUND):
        self.value = value & 0xFF  # 8-bit constraint
        self.level = level
        self.winding_number = self._compute_church_winding()
        self.spinor_state = self._compute_spinor_orientation()
        self.hole = 256 - self.value  # Topological complement
        self.neighbors: List['TorusSpinorByteWord'] = []
        
    def _compute_church_winding(self) -> int:
        """Church numeral as topological winding around torus"""
        # Number of times we've wound around the 8-bit torus
        return sum(int(bit) for bit in format(self.value, '08b'))
    
    def _compute_spinor_orientation(self) -> Tuple[float, float]:
        """Spinor orientation on the torus surface"""
        # Map 8 bits to angular coordinates on torus
        theta = (self.value & 0x0F) * (2 * math.pi / 16)  # Major circle
        phi = ((self.value & 0xF0) >> 4) * (2 * math.pi / 16)  # Minor circle
        return (theta, phi)
    
    def orbital_capacity(self) -> int:
        """How many ByteWords can fit in this orbital level"""
        capacities = {
            SpinorLevel.GROUND: 2,    # s orbital: 2 electrons
            SpinorLevel.FIRST: 6,     # p orbital: 6 electrons  
            SpinorLevel.SECOND: 10,   # d orbital: 10 electrons
            SpinorLevel.THIRD: 14     # f orbital: 14 electrons
        }
        return capacities[self.level]
    
    def church_successor(self) -> 'TorusSpinorByteWord':
        """f(f(f(...))) - wind around torus one more time"""
        new_value = (self.value + 1) % 256
        if new_value == 0:  # Completed full winding
            # Promote to next orbital level
            next_level = SpinorLevel((self.level.value + 1) % 4)
            return TorusSpinorByteWord(new_value, next_level)
        return TorusSpinorByteWord(new_value, self.level)
    
    def compose(self, other: 'TorusSpinorByteWord') -> 'TorusSpinorByteWord':
        """
        Pacman-world composition with gravitational energy transfer
        """
        # Check if we can absorb energy at current level
        if len(self.neighbors) < self.orbital_capacity():
            # Direct composition - wind together
            new_winding = (self.winding_number + other.winding_number) % 256
            result = TorusSpinorByteWord(new_winding, self.level)
            result.neighbors = self.neighbors + [other]
            return result
        else:
            # Orbital full - pass to high-energy neighbor
            if self.neighbors:
                return self.neighbors[0].receive_gravitational_energy(self, other)
            else:
                # Only two bodies - bounce back through torus hole
                return self._bounce_through_hole(other)
    
    def receive_gravitational_energy(self, sender: 'TorusSpinorByteWord', 
                                   payload: 'TorusSpinorByteWord') -> 'TorusSpinorByteWord':
        """Receive energy from gravitational neighbor"""
        # Combine spinor orientations
        theta1, phi1 = sender.spinor_state
        theta2, phi2 = payload.spinor_state
        
        # Spinor composition (simplified quaternion-like)
        new_theta = (theta1 + theta2) % (2 * math.pi)
        new_phi = (phi1 + phi2) % (2 * math.pi)
        
        # Convert back to 8-bit value
        theta_bits = int((new_theta / (2 * math.pi)) * 16) & 0x0F
        phi_bits = int((new_phi / (2 * math.pi)) * 16) & 0x0F
        new_value = theta_bits | (phi_bits << 4)
        
        return TorusSpinorByteWord(new_value, self.level)
    
    def _bounce_through_hole(self, other: 'TorusSpinorByteWord') -> 'TorusSpinorByteWord':
        """
        When only two bodies exist, bounce through the topological hole
        This is the non-associative magic!
        """
        # The hole is what's not encoded in our 8 bits
        hole_value = (self.hole + other.hole) % 256
        
        # Bounce creates quantum interference
        interference_value = self.value ^ other.value  # XOR for quantum-like behavior
        
        # Final result winds through the hole
        result_value = (hole_value + interference_value) % 256
        
        # Result exists at elevated energy level
        elevated_level = SpinorLevel((max(self.level.value, other.level.value) + 1) % 4)
        
        return TorusSpinorByteWord(result_value, elevated_level)
    
    def propagate(self, steps: int = 1) -> List['TorusSpinorByteWord']:
        """
        Evolve through torus topology for multiple steps
        """
        states = [self]
        current = self
        
        for step in range(steps):
            # Each step is a church successor around the torus
            current = current.church_successor()
            
            # Spinor precession - orientation evolves
            theta, phi = current.spinor_state
            theta += 0.1 * step  # Precession rate
            phi += 0.05 * step   # Different precession rate
            
            # Update spinor with precessed values
            theta_bits = int((theta / (2 * math.pi)) * 16) & 0x0F
            phi_bits = int((phi / (2 * math.pi)) * 16) & 0x0F
            current.value = theta_bits | (phi_bits << 4)
            current.spinor_state = (theta % (2 * math.pi), phi % (2 * math.pi))
            
            states.append(current)
        
        return states
    
    def homotopy_group(self) -> int:
        """
        Compute the homotopy/winding group classification
        This determines the topological invariant of the ByteWord
        """
        # π₁(T²) = Z × Z for torus fundamental group
        major_winding = self.winding_number % 16  # Major circle winding
        minor_winding = (self.winding_number // 16) % 16  # Minor circle winding
        return (major_winding, minor_winding)
    
    def is_topologically_equivalent(self, other: 'TorusSpinorByteWord') -> bool:
        """Two ByteWords are equivalent if they have same homotopy class"""
        return self.homotopy_group() == other.homotopy_group()
    
    def to_float(self) -> float:
        """Observation collapses spinor to classical value"""
        theta, phi = self.spinor_state
        # Spinor amplitude on torus surface
        amplitude = math.cos(theta/2) * math.cos(phi/2)
        return amplitude * (self.value / 256.0)
    
    def __repr__(self):
        theta, phi = self.spinor_state
        return (f"TorusSpinor(value={self.value:08b}, level={self.level.name}, "
                f"winding={self.winding_number}, θ={theta:.2f}, φ={phi:.2f}, "
                f"hole={self.hole})")

# Example Usage and Demo
if __name__ == "__main__":
    print("=== Torus Spinor ByteWord Demo ===\n")
    
    # Create two ByteWords in pacman world
    word1 = TorusSpinorByteWord(0b10101010, SpinorLevel.GROUND)
    word2 = TorusSpinorByteWord(0b01010101, SpinorLevel.GROUND)
    
    print(f"Word 1: {word1}")
    print(f"Word 2: {word2}")
    print(f"Homotopy groups: {word1.homotopy_group()}, {word2.homotopy_group()}")
    print()
    
    # Church winding succession
    print("=== Church Winding Evolution ===")
    successor = word1.church_successor()
    print(f"Successor: {successor}")
    print()
    
    # Gravitational composition
    print("=== Gravitational Composition ===")
    composed = word1.compose(word2)
    print(f"Composed: {composed}")
    print(f"Topologically equivalent to word1? {composed.is_topologically_equivalent(word1)}")
    print()
    
    # Propagation through torus
    print("=== Torus Propagation ===")
    evolution = word1.propagate(steps=5)
    for i, state in enumerate(evolution):
        print(f"Step {i}: {state}")
    print()
    
    # Observation collapse
    print("=== Quantum Observation ===")
    classical_value = composed.to_float()
    print(f"Collapsed to classical: {classical_value}")