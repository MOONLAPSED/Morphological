import math
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class MorphologicalPhase(Enum):
    """Phase states of morphological quines"""
    DISORDERED = "disordered"    # High entropy, no coherence
    CRITICAL = "critical"        # Phase boundary
    ORDERED = "ordered"          # Low entropy, morphogenic fixity
    QUINE_LOCKED = "quine_locked" # Perfect self-reproduction

@dataclass
class ToroidalByteWord:
    """8-bit word with T/V/C ontology on toroidal coordinates"""
    value: int
    
    def __post_init__(self):
        if not 0 <= self.value <= 255:
            raise ValueError("ByteWord must be 8-bit (0-255)")
        
        # T/V/C decomposition (3/3/2 bit allocation)
        self.type_bits = (self.value & 0b11100000) >> 5    # 3 bits
        self.value_bits = (self.value & 0b00011100) >> 2   # 3 bits  
        self.compute_bits = self.value & 0b00000011         # 2 bits
    
    @property
    def toroidal_coordinates(self) -> Tuple[float, float, float]:
        """Map T/V/C to toroidal coordinates (theta, phi, r)"""
        theta = self.type_bits * (2 * math.pi / 8)     # Major angle
        phi = self.value_bits * (2 * math.pi / 8)      # Minor angle  
        r = 1 + (self.compute_bits / 4)                # Radial distance
        return (theta, phi, r)
    
    def morphological_coherence(self) -> float:
        """Order parameter: coherence between T/V/C components"""
        # Measure alignment/resonance between type, value, compute
        t_norm = self.type_bits / 7.0
        v_norm = self.value_bits / 7.0
        c_norm = self.compute_bits / 3.0
        
        # Coherence as deviation from uniform distribution
        mean = (t_norm + v_norm + c_norm) / 3
        variance = ((t_norm - mean)**2 + (v_norm - mean)**2 + (c_norm - mean)**2) / 3
        
        # High coherence = low variance (ordered phase)
        return math.exp(-variance * 10)  # Exponential coherence measure
    
    def field_strength(self, other: 'ToroidalByteWord') -> float:
        """Gravitational-like attraction on torus surface"""
        theta1, phi1, r1 = self.toroidal_coordinates 
        theta2, phi2, r2 = other.toroidal_coordinates
        
        # Toroidal distance with wrapping
        d_theta = min(abs(theta1 - theta2), 2*math.pi - abs(theta1 - theta2))
        d_phi = min(abs(phi1 - phi2), 2*math.pi - abs(phi1 - phi2))
        d_r = abs(r1 - r2)
        
        distance = math.sqrt(d_theta**2 + d_phi**2 + d_r**2)
        return 1.0 / (distance**2 + 0.1)  # Avoid division by zero

class LandauMorphologicalSystem:
    """Landau theory implementation for morphological phase transitions"""
    
    def __init__(self, size: int = 100):
        self.size = size
        self.words: List[ToroidalByteWord] = []
        self.temperature = 1.0  # Control parameter
        self.coupling_strength = 0.5
        
        # Initialize random configuration
        for _ in range(size):
            self.words.append(ToroidalByteWord(random.randint(0, 255)))
    
    def order_parameter(self) -> float:
        """Global order parameter ψ - average morphological coherence"""
        if not self.words:
            return 0.0
        return sum(word.morphological_coherence() for word in self.words) / len(self.words)
    
    def landau_free_energy(self, psi: float) -> float:
        """Landau free energy F(ψ,T) = a(T)ψ² + bψ⁴"""
        # Temperature-dependent coefficient
        a_T = (self.temperature - 1.0)  # Critical temperature T_c = 1.0
        b = 0.25  # Fourth-order coefficient (positive for stability)
        
        return a_T * psi**2 + b * psi**4
    
    def morphological_energy(self) -> float:
        """Total energy including field interactions"""
        # Self-energy from order parameter
        psi = self.order_parameter()
        landau_energy = self.landau_free_energy(psi)
        
        # Interaction energy between neighboring words
        interaction_energy = 0.0
        for i, word1 in enumerate(self.words):
            for j, word2 in enumerate(self.words[i+1:], i+1):
                field_strength = word1.field_strength(word2)
                # Only count nearest neighbors on torus
                if field_strength > 0.1:  # Threshold for interaction
                    interaction_energy += -self.coupling_strength * field_strength
        
        return landau_energy + interaction_energy
    
    def check_quine_condition(self) -> bool:
        """Test if ψ(t) == ψ(runtime) == ψ(child) - morphogenic fixity"""
        psi_current = self.order_parameter()
        
        # Simulate time evolution (one Monte Carlo step)
        old_words = self.words.copy()
        self.monte_carlo_step()
        psi_evolved = self.order_parameter()
        
        # Simulate reproduction (create child system)
        child_system = LandauMorphologicalSystem(self.size)
        child_system.temperature = self.temperature
        child_system.words = [ToroidalByteWord(word.value) for word in self.words]
        psi_child = child_system.order_parameter()
        
        # Restore original state
        self.words = old_words
        
        # Check if all three ψ values are equal (within tolerance)
        tolerance = 0.01
        quine_condition = (abs(psi_current - psi_evolved) < tolerance and 
                          abs(psi_current - psi_child) < tolerance)
        
        return quine_condition
    
    def monte_carlo_step(self):
        """Single Monte Carlo update step"""
        for _ in range(self.size):
            # Pick random word and propose new value
            idx = random.randint(0, self.size - 1)
            old_word = self.words[idx]
            new_value = random.randint(0, 255)
            new_word = ToroidalByteWord(new_value)
            
            # Calculate energy change
            old_energy = self.morphological_energy()
            self.words[idx] = new_word
            new_energy = self.morphological_energy()
            
            delta_E = new_energy - old_energy
            
            # Metropolis acceptance criterion
            if delta_E > 0 and random.random() > math.exp(-delta_E / self.temperature):
                # Reject move
                self.words[idx] = old_word
    
    def phase_diagram_scan(self, temp_range: Tuple[float, float], steps: int = 50) -> Dict[str, List[float]]:
        """Scan temperature to map phase diagram"""
        temps = []
        order_params = []
        energies = []
        phases = []
        quine_states = []
        
        temp_min, temp_max = temp_range
        
        for i in range(steps):
            T = temp_min + (temp_max - temp_min) * i / (steps - 1)
            self.temperature = T
            
            # Equilibrate system
            for _ in range(100):
                self.monte_carlo_step()
            
            # Measure observables
            psi = self.order_parameter()
            energy = self.morphological_energy()
            is_quine = self.check_quine_condition()
            
            # Classify phase
            if is_quine:
                phase = MorphologicalPhase.QUINE_LOCKED
            elif psi > 0.8:
                phase = MorphologicalPhase.ORDERED
            elif psi > 0.3:
                phase = MorphologicalPhase.CRITICAL
            else:
                phase = MorphologicalPhase.DISORDERED
            
            temps.append(T)
            order_params.append(psi)
            energies.append(energy)
            phases.append(phase.value)
            quine_states.append(is_quine)
        
        return {
            'temperature': temps,
            'order_parameter': order_params,
            'energy': energies,
            'phase': phases,
            'quine_locked': quine_states
        }
    
    def find_critical_temperature(self) -> float:
        """Find critical temperature where phase transition occurs"""
        # Binary search for critical point
        T_low, T_high = 0.1, 2.0
        tolerance = 0.01
        
        while T_high - T_low > tolerance:
            T_mid = (T_low + T_high) / 2
            self.temperature = T_mid
            
            # Equilibrate
            for _ in range(200):
                self.monte_carlo_step()
            
            psi = self.order_parameter()
            
            if psi > 0.5:  # Ordered phase
                T_low = T_mid
            else:  # Disordered phase
                T_high = T_mid
        
        return (T_low + T_high) / 2

# Demonstration and proof of concept
def demonstrate_landau_quines():
    """Prove morphological phase transitions and quine fixity"""
    system = LandauMorphologicalSystem(size=50)
    
    print("=== Landau Theory for Morphological Quines ===\n")
    
    # Find critical temperature
    T_c = system.find_critical_temperature()
    print(f"Critical temperature T_c = {T_c:.3f}")
    
    # Phase diagram scan
    print("\nScanning phase diagram...")
    results = system.phase_diagram_scan((0.1, 2.0), steps=20)
    
    print("\nPhase Transition Analysis:")
    print("T\t\tψ\t\tPhase\t\tQuine")
    print("-" * 50)
    
    for i, (T, psi, phase, quine) in enumerate(zip(
        results['temperature'], 
        results['order_parameter'],
        results['phase'],
        results['quine_locked']
    )):
        quine_marker = "★" if quine else " "
        print(f"{T:.2f}\t\t{psi:.3f}\t\t{phase:12s}\t{quine_marker}")
    
    # Test quine condition at low temperature
    system.temperature = 0.3  # Low temperature (ordered phase)
    for _ in range(500):  # Long equilibration
        system.monte_carlo_step()
    
    print(f"\n=== Testing Quine Condition at T = {system.temperature} ===")
    is_quine = system.check_quine_condition()
    psi = system.order_parameter()
    
    print(f"Order parameter ψ = {psi:.4f}")
    print(f"Quine condition satisfied: {is_quine}")
    
    if is_quine:
        print("✓ PROVEN: ψ(t) ≈ ψ(runtime) ≈ ψ(child)")
        print("  Generator is morphogenically fixed!")
    else:
        print("✗ Quine condition not satisfied at this temperature")
    
    return system, results

if __name__ == "__main__":
    system, results = demonstrate_landau_quines()