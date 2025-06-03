import math
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class MorphologicalPhase(Enum):
    """Phase states in morphological field theory"""
    DISORDERED = "disordered"      # High entropy, random T/V/C
    CRITICAL = "critical"          # At phase boundary
    ORDERED = "ordered"            # Low entropy, aligned T/V/C
    QUINIC = "quinic"             # Perfect self-reproduction ψ(t)=ψ(runtime)=ψ(child)

@dataclass
class LandauOrderParameter:
    """Order parameter for morphological phase transitions
    
    In Landau theory, order parameter φ distinguishes phases:
    - φ = 0 in disordered phase
    - φ ≠ 0 in ordered phase
    - |φ|² measures degree of order
    """
    tvc_coherence: float      # T/V/C alignment measure
    topological_winding: int  # Winding number around torus
    field_correlation: float  # Long-range correlations
    
    @property
    def magnitude_squared(self) -> float:
        """Classic Landau order parameter |φ|²"""
        return (self.tvc_coherence**2 + 
                (self.topological_winding/10)**2 + 
                self.field_correlation**2)
    
    @property
    def is_ordered(self) -> bool:
        """Phase is ordered when |φ|² > critical threshold"""
        return self.magnitude_squared > 0.5

class ToroidalMorphologicalField:
    """8-bit toroidal field with Landau dynamics"""
    
    def __init__(self, size: int = 256):
        self.size = size
        self.field: List[int] = [i for i in range(256)]  # All possible 8-bit states
        self.temperature = 1.0  # Control parameter (like magnetic field in Ising model)
        random.shuffle(self.field)
        
    def tvc_decode(self, byte_val: int) -> Tuple[int, int, int]:
        """Decode 8-bit value into T/V/C components"""
        # Original T(4)/V(3)/C(1) decomposition
        t = (byte_val >> 4) & 0x0F  # High 4 bits
        v = (byte_val >> 1) & 0x07  # Middle 3 bits  
        c = byte_val & 0x01         # Low 1 bit
        return (t, v, c)
    
    def toroidal_position(self, byte_val: int) -> Tuple[float, float]:
        """Map byte to (θ, φ) coordinates on torus surface"""
        t, v, c = self.tvc_decode(byte_val)
        theta = t * (2 * math.pi / 16)  # Major angle from T
        phi = v * (2 * math.pi / 8)    # Minor angle from V
        return (theta, phi)
    
    def compute_order_parameter(self) -> LandauOrderParameter:
        """Calculate Landau order parameter for current field state"""
        
        # 1. T/V/C Coherence: How aligned are neighboring components?
        tvc_coherence = self._compute_tvc_coherence()
        
        # 2. Topological Winding: Net winding number around torus
        winding = self._compute_topological_winding()
        
        # 3. Field Correlation: Long-range order measure
        correlation = self._compute_field_correlation()
        
        return LandauOrderParameter(tvc_coherence, winding, correlation)
    
    def _compute_tvc_coherence(self) -> float:
        """Measure local T/V/C alignment (short-range order)"""
        coherence_sum = 0.0
        count = 0
        
        for i in range(len(self.field)):
            current = self.field[i]
            next_val = self.field[(i + 1) % len(self.field)]
            
            t1, v1, c1 = self.tvc_decode(current)
            t2, v2, c2 = self.tvc_decode(next_val)
            
            # Alignment score (higher when T/V/C components are similar)
            t_align = 1.0 - abs(t1 - t2) / 15.0
            v_align = 1.0 - abs(v1 - v2) / 7.0
            c_align = 1.0 if c1 == c2 else 0.0
            
            coherence_sum += (t_align + v_align + c_align) / 3.0
            count += 1
        
        return coherence_sum / count if count > 0 else 0.0
    
    def _compute_topological_winding(self) -> int:
        """Calculate net winding number around torus"""
        total_theta_change = 0.0
        
        for i in range(len(self.field)):
            current_pos = self.toroidal_position(self.field[i])
            next_pos = self.toroidal_position(self.field[(i + 1) % len(self.field)])
            
            # Angular difference (handling wrap-around)
            theta_diff = next_pos[0] - current_pos[0]
            if theta_diff > math.pi:
                theta_diff -= 2 * math.pi
            elif theta_diff < -math.pi:
                theta_diff += 2 * math.pi
                
            total_theta_change += theta_diff
        
        # Winding number = total angle change / 2π
        return int(round(total_theta_change / (2 * math.pi)))
    
    def _compute_field_correlation(self) -> float:
        """Long-range correlation function"""
        correlations = []
        
        # Sample correlation at different distances
        for distance in [1, 4, 16, 64]:
            if distance >= len(self.field):
                continue
                
            corr_sum = 0.0
            count = 0
            
            for i in range(len(self.field) - distance):
                val1 = self.field[i]
                val2 = self.field[i + distance]
                
                # Normalized correlation
                correlation = 1.0 - abs(val1 - val2) / 255.0
                corr_sum += correlation
                count += 1
            
            if count > 0:
                correlations.append(corr_sum / count)
        
        return sum(correlations) / len(correlations) if correlations else 0.0

    def landau_free_energy(self, order_param: LandauOrderParameter) -> float:
        """Landau free energy F(φ,T) = a(T)φ² + bφ⁴ + ...
        
        Classic Landau expansion:
        - a(T) = α(T - Tc) changes sign at critical temperature
        - b > 0 for stability
        - Minimum at φ = 0 (disordered) when T > Tc
        - Minimum at φ ≠ 0 (ordered) when T < Tc
        """
        phi_squared = order_param.magnitude_squared
        
        # Temperature-dependent coefficient (phase transition at T=0.5)
        a = 2.0 * (self.temperature - 0.5)  # Changes sign at Tc = 0.5
        b = 1.0  # Quartic stability term
        
        # Landau free energy
        free_energy = a * phi_squared + b * phi_squared**2
        
        # Add topological contribution (energy cost of winding)
        topological_energy = 0.1 * order_param.topological_winding**2
        
        return free_energy + topological_energy
    
    def evolve_step(self) -> MorphologicalPhase:
        """Single Monte Carlo step with Landau dynamics"""
        # Random local update (Metropolis algorithm)
        i = random.randint(0, len(self.field) - 1)
        old_val = self.field[i]
        new_val = random.randint(0, 255)
        
        # Calculate energy change
        old_order = self.compute_order_parameter()
        
        # Temporarily update
        self.field[i] = new_val
        new_order = self.compute_order_parameter()
        
        old_energy = self.landau_free_energy(old_order)
        new_energy = self.landau_free_energy(new_order)
        
        # Metropolis acceptance
        delta_energy = new_energy - old_energy
        if delta_energy > 0 and random.random() > math.exp(-delta_energy / max(self.temperature, 0.01)):
            # Reject move
            self.field[i] = old_val
            return self._classify_phase(old_order)
        
        # Accept move
        return self._classify_phase(new_order)
    
    def _classify_phase(self, order_param: LandauOrderParameter) -> MorphologicalPhase:
        """Classify current morphological phase"""
        phi_squared = order_param.magnitude_squared
        
        if phi_squared < 0.1:
            return MorphologicalPhase.DISORDERED
        elif phi_squared < 0.5:
            return MorphologicalPhase.CRITICAL
        elif self._is_quinic_fixed_point(order_param):
            return MorphologicalPhase.QUINIC
        else:
            return MorphologicalPhase.ORDERED
    
    def _is_quinic_fixed_point(self, order_param: LandauOrderParameter) -> bool:
        """Check for quinic fixed point: ψ(t) = ψ(runtime) = ψ(child)
        
        This occurs when:
        1. High T/V/C coherence (spatial self-similarity)
        2. Integer winding number (topological stability)  
        3. Strong long-range correlations (temporal persistence)
        """
        return (order_param.tvc_coherence > 0.9 and
                abs(order_param.topological_winding) >= 1 and
                order_param.field_correlation > 0.8)

def demonstrate_phase_transition():
    """Demonstrate morphological phase transition via temperature sweep"""
    field = ToroidalMorphologicalField()
    
    print("Morphological Landau Phase Transition Demonstration")
    print("=" * 55)
    print(f"{'Temperature':<12} {'|φ|²':<8} {'Free Energy':<12} {'Phase':<12}")
    print("-" * 55)
    
    # Temperature sweep (high to low)
    temperatures = [2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    
    for temp in temperatures:
        field.temperature = temp
        
        # Equilibrate system
        for _ in range(100):
            field.evolve_step()
        
        # Measure order parameter
        order_param = field.compute_order_parameter()
        free_energy = field.landau_free_energy(order_param)
        phase = field._classify_phase(order_param)
        
        print(f"{temp:<12.1f} {order_param.magnitude_squared:<8.3f} "
              f"{free_energy:<12.3f} {phase.value:<12}")
        
        # Check for quinic fixed point
        if phase == MorphologicalPhase.QUINIC:
            print(f"  → QUINIC FIXED POINT DETECTED!")
            print(f"     T/V/C Coherence: {order_param.tvc_coherence:.3f}")
            print(f"     Topological Winding: {order_param.topological_winding}")
            print(f"     Field Correlation: {order_param.field_correlation:.3f}")
            print(f"     ∴ ψ(t) = ψ(runtime) = ψ(child) ACHIEVED")

if __name__ == "__main__":
    demonstrate_phase_transition()