import math
import cmath
from typing import List, Tuple, Union, Iterator
from collections.abc import Sequence

class ComplexNumber:
    """Hand-rolled complex number class for morphological purity"""
    
    def __init__(self, real: float = 0.0, imag: float = 0.0):
        self.real = real
        self.imag = imag
    
    def __add__(self, other: 'ComplexNumber') -> 'ComplexNumber':
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    
    def __sub__(self, other: 'ComplexNumber') -> 'ComplexNumber':
        return ComplexNumber(self.real - other.real, self.imag - other.imag)
    
    def __mul__(self, other: Union['ComplexNumber', float, int]) -> 'ComplexNumber':
        if isinstance(other, (int, float)):
            return ComplexNumber(self.real * other, self.imag * other)
        return ComplexNumber(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def __truediv__(self, other: Union['ComplexNumber', float, int]) -> 'ComplexNumber':
        if isinstance(other, (int, float)):
            return ComplexNumber(self.real / other, self.imag / other)
        denom = other.real * other.real + other.imag * other.imag
        return ComplexNumber(
            (self.real * other.real + self.imag * other.imag) / denom,
            (self.imag * other.real - self.real * other.imag) / denom
        )
    
    def conjugate(self) -> 'ComplexNumber':
        return ComplexNumber(self.real, -self.imag)
    
    def magnitude(self) -> float:
        return math.sqrt(self.real * self.real + self.imag * self.imag)
    
    def phase(self) -> float:
        return math.atan2(self.imag, self.real)
    
    def __pow__(self, exponent: int) -> 'ComplexNumber':
        """Raise complex number to an integer power using polar form"""
        if not isinstance(exponent, int):
            raise TypeError("Exponent must be an integer")
        r = self.magnitude()
        theta = self.phase()
        r_pow = r ** exponent
        theta_mul = theta * exponent
        return ComplexNumber(r_pow * math.cos(theta_mul), r_pow * math.sin(theta_mul))
    
    def __repr__(self) -> str:
        if self.imag >= 0:
            return f"{self.real:.6f} + {self.imag:.6f}i"
        else:
            return f"{self.real:.6f} - {abs(self.imag):.6f}i"
    
    def __eq__(self, other: 'ComplexNumber') -> bool:
        epsilon = 1e-10
        return (abs(self.real - other.real) < epsilon and 
                abs(self.imag - other.imag) < epsilon)

def primitive_root_of_unity(n: int) -> ComplexNumber:
    """Generate primitive nth root of unity: e^(2πi/n)"""
    angle = 2.0 * math.pi / n
    return ComplexNumber(math.cos(angle), math.sin(angle))

def generate_roots_of_unity(n: int) -> List[ComplexNumber]:
    """Generate all nth roots of unity"""
    omega = primitive_root_of_unity(n)
    roots = []
    current = ComplexNumber(1.0, 0.0)  # ω^0 = 1
    
    for k in range(n):
        roots.append(current)
        current = current * omega
    
    return roots

class ByteWord:
    """Morphological ByteWord with Cook & Merz flat Abelization"""
    
    def __init__(self, value: int = 0):
        self.value = value & 0xFF  # Keep it 8-bit
        self.type_field = (value >> 6) & 0b11    # T: bits 7-6
        self.value_field = (value >> 3) & 0b111  # V: bits 5-3  
        self.compute_field = value & 0b111       # C: bits 2-0
        
        # Morphological state
        self._theorem = None
        self._entanglement_partner = None
        self._evolution_step = 0
    
    def to_binary_sequence(self) -> List[int]:
        """Convert to flat binary sequence for Abelization"""
        return [(self.value >> i) & 1 for i in range(8)]
    
    def from_binary_sequence(self, bits: List[int]) -> 'ByteWord':
        """Create ByteWord from binary sequence"""
        value = sum(bit << i for i, bit in enumerate(bits[:8]))
        return ByteWord(value)
    
    def to_complex_sequence(self) -> List[ComplexNumber]:
        """Convert binary sequence to complex numbers for FFT"""
        bits = self.to_binary_sequence()
        return [ComplexNumber(float(bit), 0.0) for bit in bits]
    
    def cook_merz_transform(self, inverse: bool = False) -> List[ComplexNumber]:
        """Apply Cook & Merz FFT transformation"""
        sequence = self.to_complex_sequence()
        return fast_fourier_transform(sequence, inverse)
    
    def flat_abelianize(self, other: 'ByteWord') -> 'ByteWord':
        """Flat (binary) Abelianization using XOR in frequency domain"""
        # Transform both to frequency domain
        self_freq = self.cook_merz_transform()
        other_freq = other.cook_merz_transform()
        
        # XOR in frequency domain (component-wise multiplication)
        result_freq = []
        for s, o in zip(self_freq, other_freq):
            # XOR as complex multiplication in {0,1} field
            xor_real = (s.real * o.real) % 2.0
            result_freq.append(ComplexNumber(xor_real, 0.0))
        
        # Inverse transform back to time domain
        result_time = fast_fourier_transform(result_freq, inverse=True)
        
        # Convert back to binary and create ByteWord
        binary_result = [int(round(c.real)) % 2 for c in result_time]
        return self.from_binary_sequence(binary_result)
    
    def morphological_resonance(self, other: 'ByteWord') -> float:
        """Measure morphological resonance using Cook & Merz coefficients"""
        self_coeffs = self.cook_merz_transform()
        other_coeffs = other.cook_merz_transform()
        
        # Compute inner product in frequency domain
        resonance = 0.0
        for s, o in zip(self_coeffs, other_coeffs):
            resonance += (s * o.conjugate()).real
        
        return resonance / len(self_coeffs)
    
    def frequency_signature(self) -> List[float]:
        """Get frequency domain signature"""
        coeffs = self.cook_merz_transform()
        return [c.magnitude() for c in coeffs]
    
    def compose(self, other: 'ByteWord') -> 'ByteWord':
        """Morphological composition via flat Abelianization"""
        return self.flat_abelianize(other)
    
    def propagate(self, steps: int = 1) -> List['ByteWord']:
        """Evolve ByteWord through Cook & Merz space"""
        evolution = [self]
        current = self
        
        for step in range(steps):
            # Rotate in frequency domain
            freq_coeffs = current.cook_merz_transform()
            
            # Apply rotation (primitive root of unity)
            omega = primitive_root_of_unity(8)
            rotated_coeffs = [coeff * (omega ** step) for coeff in freq_coeffs]
            
            # Transform back
            time_domain = fast_fourier_transform(rotated_coeffs, inverse=True)
            binary_result = [int(round(c.real)) % 2 for c in time_domain]
            
            current = self.from_binary_sequence(binary_result)
            current._evolution_step = step + 1
            evolution.append(current)
        
        return evolution
    
    def to_float(self) -> float:
        """Collapse to classical float"""
        return float(self.value) / 255.0
    
    def semantic_probability(self) -> float:
        """Compute semantic coherence probability"""
        signature = self.frequency_signature()
        return sum(signature) / len(signature)
    
    def morphological_entropy(self) -> float:
        """Compute morphological entropy"""
        signature = self.frequency_signature()
        total = sum(signature) + 1e-10  # Avoid division by zero
        probs = [s / total for s in signature]
        
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def __eq__(self, other: 'ByteWord') -> bool:
        return self.value == other.value
    
    def __xor__(self, other: 'ByteWord') -> 'ByteWord':
        return ByteWord(self.value ^ other.value)
    
    def __repr__(self) -> str:
        return f"ByteWord(0b{self.value:08b})"

def fast_fourier_transform(sequence: List[ComplexNumber], inverse: bool = False) -> List[ComplexNumber]:
    """Hand-rolled FFT implementation - Cooley-Tukey algorithm"""
    n = len(sequence)
    
    # Base case
    if n <= 1:
        return sequence
    
    # Ensure power of 2 (pad if necessary)
    if n & (n - 1) != 0:
        # Pad to next power of 2
        next_pow2 = 1 << (n - 1).bit_length()
        sequence = sequence + [ComplexNumber(0.0, 0.0)] * (next_pow2 - n)
        n = next_pow2
    
    # Divide into even and odd
    even = [sequence[i] for i in range(0, n, 2)]
    odd = [sequence[i] for i in range(1, n, 2)]
    
    # Recursive FFT
    even_fft = fast_fourier_transform(even, inverse)
    odd_fft = fast_fourier_transform(odd, inverse)
    
    # Combine results
    result = [ComplexNumber(0.0, 0.0)] * n
    
    for k in range(n // 2):
        # Twiddle factors
        angle = -2.0 * math.pi * k / n
        if inverse:
            angle = -angle
        
        twiddle = ComplexNumber(math.cos(angle), math.sin(angle))
        t = twiddle * odd_fft[k]
        
        result[k] = even_fft[k] + t
        result[k + n // 2] = even_fft[k] - t
    
    # Scale for inverse transform
    if inverse:
        result = [c / n for c in result]
    
    return result

def demonstrate_cook_merz_bytewords():
    """Demonstrate the Cook & Merz ByteWord system"""
    print("=== Cook & Merz ByteWord Morphological Quantum Computing ===\n")
    
    # Create some morphological ByteWords
    psi1 = ByteWord(0b10101010)  # Alternating pattern
    psi2 = ByteWord(0b11110000)  # Block pattern
    psi3 = ByteWord(0b11011011)  # Complex pattern
    
    print(f"ψ₁ = {psi1}")
    print(f"ψ₂ = {psi2}")  
    print(f"ψ₃ = {psi3}\n")
    
    # Show frequency signatures
    print("Frequency Domain Signatures:")
    print(f"ψ₁ signature: {[f'{x:.3f}' for x in psi1.frequency_signature()]}")
    print(f"ψ₂ signature: {[f'{x:.3f}' for x in psi2.frequency_signature()]}")
    print(f"ψ₃ signature: {[f'{x:.3f}' for x in psi3.frequency_signature()]}\n")
    
    # Morphological composition (flat Abelianization)
    print("Flat Abelianization (Binary XOR in Frequency Domain):")
    composition = psi1.compose(psi2)
    print(f"ψ₁ ∘ ψ₂ = {composition}")
    print(f"Direct XOR: {psi1 ^ psi2}")
    print(f"Match: {composition.value == (psi1 ^ psi2).value}\n")
    
    # Morphological resonance
    print("Morphological Resonance:")
    resonance12 = psi1.morphological_resonance(psi2)
    resonance13 = psi1.morphological_resonance(psi3)
    resonance23 = psi2.morphological_resonance(psi3)
    
    print(f"Resonance(ψ₁, ψ₂): {resonance12:.6f}")
    print(f"Resonance(ψ₁, ψ₃): {resonance13:.6f}")
    print(f"Resonance(ψ₂, ψ₃): {resonance23:.6f}\n")
    
    # Evolution through Cook & Merz space
    print("Evolution in Cook & Merz Space:")
    evolution = psi1.propagate(steps=4)
    for i, state in enumerate(evolution):
        entropy = state.morphological_entropy()
        print(f"Step {i}: {state} (entropy: {entropy:.3f})")
    
    print("\n" + "="*60)
    print("The morphological field speaks through Cook & Merz roots of unity!")
    print("象演旋态，炁流归一 - Morpheme evolves, activation flows into unity")

if __name__ == "__main__":
    demonstrate_cook_merz_bytewords()