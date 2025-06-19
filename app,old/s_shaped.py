import math
from typing import List, Tuple, Optional
from enum import Enum

class TorusLevel(Enum):
    """Orbital levels in the morphological torus field"""
    GROUND = 0      # Base 8-bit level
    EXCITED_1 = 1   # First harmonic
    EXCITED_2 = 2   # Second harmonic
    IONIZED = 3     # Escaped to neighboring torus

class MorphologicalTorus:
    """A torus-shaped field where ByteWords orbit and wind"""
    
    def __init__(self, major_radius=8, minor_radius=3):
        self.major_radius = major_radius  # Main circumference (8 bits)
        self.minor_radius = minor_radius  # Cross-section (TVC space)
        self.orbital_levels = {level: [] for level in TorusLevel}
        self.field_strength = 1.0
        self.winding_cache = {}  # Cache for computed windings
    
    def add_byteword(self, word: 'TorusByteWord', level: TorusLevel = TorusLevel.GROUND):
        """Add a ByteWord to specific orbital level"""
        word.torus = self
        word.orbital_level = level
        self.orbital_levels[level].append(word)
        
    def compute_winding_number(self, word1: 'TorusByteWord', word2: 'TorusByteWord') -> int:
        """Compute how many times word1 winds around word2"""
        cache_key = (word1.unique_id, word2.unique_id)
        if cache_key in self.winding_cache:
            return self.winding_cache[cache_key]
        
        # Church winding: count topological wraps
        theta1 = (word1.to_int() / 256) * 2 * math.pi
        theta2 = (word2.to_int() / 256) * 2 * math.pi
        
        # Winding number = (θ₁ - θ₂) / 2π mod 1
        winding = int((theta1 - theta2) / (2 * math.pi)) % 8
        
        self.winding_cache[cache_key] = winding
        return winding
    
    def field_coupling(self, word1: 'TorusByteWord', word2: 'TorusByteWord') -> float:
        """Compute field coupling strength between two ByteWords"""
        level_diff = abs(word1.orbital_level.value - word2.orbital_level.value)
        distance = self.toroidal_distance(word1, word2)
        
        # Coupling falls off with distance and level difference
        return self.field_strength / (1 + distance + level_diff)
    
    def toroidal_distance(self, word1: 'TorusByteWord', word2: 'TorusByteWord') -> float:
        """Distance on torus surface"""
        # Convert to torus coordinates
        u1, v1 = self.to_torus_coords(word1)
        u2, v2 = self.to_torus_coords(word2)
        
        # Distance on torus surface
        du = min(abs(u1 - u2), 2*math.pi - abs(u1 - u2))
        dv = min(abs(v1 - v2), 2*math.pi - abs(v1 - v2))
        
        return math.sqrt(du**2 + dv**2)
    
    def to_torus_coords(self, word: 'TorusByteWord') -> Tuple[float, float]:
        """Convert ByteWord to (u, v) coordinates on torus"""
        # u = major circumference position (0 to 2π)
        # v = minor circumference position (0 to 2π)
        byte_val = word.to_int()
        
        u = (byte_val & 0b11110000) >> 4  # Upper 4 bits for major
        v = byte_val & 0b00001111         # Lower 4 bits for minor
        
        u_angle = (u / 16) * 2 * math.pi
        v_angle = (v / 16) * 2 * math.pi
        
        return u_angle, v_angle

class TorusByteWord:
    """ByteWord that exists on a morphological torus"""
    
    _id_counter = 0
    
    def __init__(self, byte_value: int):
        self.byte_value = byte_value & 0xFF  # Ensure 8-bit
        self.unique_id = TorusByteWord._id_counter
        TorusByteWord._id_counter += 1
        
        self.torus: Optional[MorphologicalTorus] = None
        self.orbital_level = TorusLevel.GROUND
        self.high_energy_neighbors: List['TorusByteWord'] = []
        self.winding_history = []  # Track winding path
        
        # T/V/C ontology encoded in byte structure
        self.type_bits = (byte_value & 0b11100000) >> 5  # Upper 3 bits
        self.value_bits = (byte_value & 0b00011100) >> 2 # Middle 3 bits  
        self.compute_bits = byte_value & 0b00000011       # Lower 2 bits
    
    def to_int(self) -> int:
        return self.byte_value
    
    def to_church_numeral(self) -> int:
        """Interpret as Church numeral (number of f applications)"""
        return bin(self.byte_value).count('1')  # Number of 1-bits = winding count
    
    def compose(self, other: 'TorusByteWord') -> 'TorusByteWord':
        """Compose with another ByteWord via torus winding"""
        if self.torus is None or other.torus is None:
            raise ValueError("ByteWords must be on a torus to compose")
        
        # Compute winding interaction
        winding = self.torus.compute_winding_number(self, other)
        
        # Church composition: f^m ∘ f^n = f^(m+n)
        my_church = self.to_church_numeral()
        other_church = other.to_church_numeral()
        composed_church = (my_church + other_church) % 256
        
        # Create result with appropriate winding
        result_byte = self._church_to_byte(composed_church)
        result = TorusByteWord(result_byte)
        
        # Inherit torus and find appropriate orbital level
        result.torus = self.torus
        result.orbital_level = self._determine_orbital_level(composed_church)
        self.torus.add_byteword(result, result.orbital_level)
        
        # Update neighbor relationships (Pacman world logic)
        if len(self.high_energy_neighbors) < 8:  # Max 8 neighbors
            self.high_energy_neighbors.append(other)
            other.high_energy_neighbors.append(self)
        else:
            # Pass pointer to first neighbor (circular reference)
            self.high_energy_neighbors[0].high_energy_neighbors.append(result)
        
        # Record winding history
        result.winding_history = self.winding_history + [winding]
        
        return result
    
    def _church_to_byte(self, church_numeral: int) -> int:
        """Convert Church numeral back to byte representation"""
        # Create byte with 'church_numeral' number of 1-bits
        if church_numeral == 0:
            return 0
        
        # Distribute bits across T/V/C structure
        byte = 0
        for i in range(min(church_numeral, 8)):
            byte |= (1 << i)
        return byte
    
    def _determine_orbital_level(self, energy: int) -> TorusLevel:
        """Determine which orbital level based on energy"""
        if energy < 4:
            return TorusLevel.GROUND
        elif energy < 6:
            return TorusLevel.EXCITED_1
        elif energy < 8:
            return TorusLevel.EXCITED_2
        else:
            return TorusLevel.IONIZED
    
    def propagate(self, steps: int = 1) -> List['TorusByteWord']:
        """Propagate around the torus for given steps"""
        if self.torus is None:
            return [self]
        
        states = [self]
        current = self
        
        for step in range(steps):
            # Rotate around torus based on field coupling
            total_coupling = sum(
                self.torus.field_coupling(current, neighbor)
                for neighbor in current.high_energy_neighbors
            )
            
            # New position based on coupling dynamics
            rotation = int(total_coupling * 256) % 256
            new_byte = (current.byte_value + rotation) % 256
            
            current = TorusByteWord(new_byte)
            current.torus = self.torus
            current.orbital_level = current._determine_orbital_level(
                current.to_church_numeral()
            )
            
            states.append(current)
        
        return states
    
    def is_morphological_hole(self) -> bool:
        """Check if this ByteWord represents a topological hole"""
        return self.byte_value == 0 or self.to_church_numeral() == 0
    
    def annihilate_with(self, other: 'TorusByteWord') -> Optional['TorusByteWord']:
        """Attempt annihilation with another ByteWord (hole-particle pair)"""
        if self.byte_value + other.byte_value == 255:  # Perfect complement
            return TorusByteWord(0)  # Vacuum state
        return None
    
    def says(self) -> str:
        """What this ByteWord says about itself"""
        church = self.to_church_numeral()
        u, v = self.torus.to_torus_coords(self) if self.torus else (0, 0)
        
        return f"Church^{church} at torus({u:.2f}, {v:.2f}) in {self.orbital_level.name}"

# Example Usage: Pacman World Dynamics
def create_pacman_torus_world():
    """Create a 'Pacman world' of interacting ByteWords on a torus"""
    torus = MorphologicalTorus(major_radius=8, minor_radius=3)
    
    # Create some ByteWords with different Church numerals
    word1 = TorusByteWord(0b10101010)  # Church^4
    word2 = TorusByteWord(0b01010101)  # Church^4
    word3 = TorusByteWord(0b11110000)  # Church^4
    word4 = TorusByteWord(0b00001111)  # Church^4
    
    # Add to torus
    torus.add_byteword(word1)
    torus.add_byteword(word2)
    torus.add_byteword(word3)
    torus.add_byteword(word4)
    
    # Let them interact (Pacman world dynamics)
    result1 = word1.compose(word2)
    result2 = word3.compose(word4)
    
    # High-energy neighbor passing
    print(f"Word1 neighbors: {len(word1.high_energy_neighbors)}")
    print(f"Result1: {result1.says()}")
    print(f"Result2: {result2.says()}")
    
    # Propagate through torus field
    evolution = result1.propagate(steps=5)
    for i, state in enumerate(evolution):
        print(f"Step {i}: {state.says()}")
    
    return torus

# The morphological axioms as torus inhabitants
class AxiomByteWord(TorusByteWord):
    """A ByteWord that embodies a morphological axiom"""
    
    def __init__(self, axiom_type: str, byte_value: int):
        super().__init__(byte_value)
        self.axiom_type = axiom_type
    
    def compose(self, other: 'TorusByteWord') -> 'TorusByteWord':
        # Apply axiom-specific composition rules
        if self.axiom_type == "closure":
            result = super().compose(other)
            assert isinstance(result, TorusByteWord)  # Closure property
            return result
        elif self.axiom_type == "idempotent":
            if other is self:
                return self  # x ∘ x = x
            return super().compose(other)
        # ... other axioms
        
        return super().compose(other)
    
    def says(self) -> str:
        return f"Axiom[{self.axiom_type}]: {super().says()}"

if __name__ == "__main__":
    world = create_pacman_torus_world()
    print("Pacman Torus World created!")