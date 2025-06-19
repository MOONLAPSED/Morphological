import math
import cmath
from typing import List, Tuple, Union, Optional

class ByteWord:
    """
    Morphological quantum computing primitive with Cook & Mertz roots of unity
    Flat binary Abelization through XOR group operations
    The flat binary Abelization means every operation stays within the 256-element XOR group, while the Cook & Mertz spectral analysis gives you the full frequency domain morphological toolkit.
    """
    
    def __init__(self, value: int = 0):
        self.value = value & 0xFF  # 8-bit constraint
        self._theorem = None
        self._phase = 0.0  # Complex phase for roots of unity
        
    def __repr__(self):
        return f"ByteWord(0b{self.value:08b})"
        
    def __eq__(self, other):
        if isinstance(other, ByteWord):
            return self.value == other.value
        return False
        
    def __hash__(self):
        return hash(self.value)

class CookMertzTransform:
    """
    Pure stdlib implementation of Cook & Mertz FFT-like transform
    Using XOR as the flat binary Abelian group operation
    """
    
    def __init__(self, n: int = 8):
        """Initialize for n-bit transforms (default 8 for ByteWords)"""
        self.n = n
        self.size = 2 ** n  # 256 for 8-bit
        self._precompute_roots()
        
    def _precompute_roots(self):
        """Precompute all nth roots of unity for efficiency"""
        self.roots = {}
        for k in range(self.size):
            # ω^k where ω = e^(2πi/N)
            angle = 2 * math.pi * k / self.size
            self.roots[k] = cmath.exp(1j * angle)
            
    def primitive_root_of_unity(self, k: int) -> complex:
        """Get the kth primitive root of unity"""
        return self.roots[k % self.size]
        
    def binary_abelian_multiply(self, a: int, b: int) -> int:
        """XOR as the flat binary Abelian group operation"""
        return a ^ b
        
    def walsh_hadamard_transform(self, sequence: List[Union[int, complex]]) -> List[complex]:
        """
        Walsh-Hadamard transform using Cook & Mertz methodology
        This is the binary analog of FFT using XOR instead of addition
        """
        n = len(sequence)
        if n == 0:
            return []
            
        # Ensure power of 2 length
        if n & (n - 1) != 0:
            # Pad to next power of 2
            next_power = 1 << (n - 1).bit_length()
            sequence = sequence + [0] * (next_power - n)
            n = next_power
            
        # Convert to complex if needed
        result = [complex(x) if not isinstance(x, complex) else x for x in sequence]
        
        # Iterative Walsh-Hadamard transform
        step = 1
        while step < n:
            for i in range(0, n, step * 2):
                for j in range(step):
                    u = result[i + j]
                    v = result[i + j + step]
                    
                    # XOR-based butterfly operation
                    result[i + j] = u + v
                    result[i + j + step] = u - v
                    
            step *= 2
            
        # Normalize
        norm_factor = 1.0 / math.sqrt(n)
        return [x * norm_factor for x in result]
        
    def inverse_walsh_hadamard_transform(self, sequence: List[complex]) -> List[complex]:
        """Inverse Walsh-Hadamard transform"""
        n = len(sequence)
        if n == 0:
            return []
            
        result = list(sequence)
        
        # Same as forward transform (Walsh-Hadamard is self-inverse up to scaling)
        step = 1
        while step < n:
            for i in range(0, n, step * 2):
                for j in range(step):
                    u = result[i + j]
                    v = result[i + j + step]
                    
                    result[i + j] = u + v
                    result[i + j + step] = u - v
                    
            step *= 2
            
        # Different normalization for inverse
        norm_factor = 1.0 / math.sqrt(n)
        return [x * norm_factor for x in result]

class MorphologicalFourierEngine:
    """
    ByteWord-native Fourier analysis using Cook & Mertz roots of unity
    Implements flat binary Abelization through XOR group structure
    """
    
    def __init__(self):
        self.transform = CookMertzTransform()
        
    def byteword_to_spectrum(self, word: ByteWord) -> List[complex]:
        """Convert a single ByteWord to its spectral representation"""
        # Decompose byte into bit sequence
        bits = [(word.value >> i) & 1 for i in range(8)]
        
        # Transform using Walsh-Hadamard
        spectrum = self.transform.walsh_hadamard_transform(bits)
        
        return spectrum
        
    def spectrum_to_byteword(self, spectrum: List[complex]) -> ByteWord:
        """Reconstruct ByteWord from spectral representation"""
        # Inverse transform
        bits_complex = self.transform.inverse_walsh_hadamard_transform(spectrum)
        
        # Convert back to binary (take real part and threshold)
        bits = [1 if abs(x.real) > 0.5 else 0 for x in bits_complex[:8]]
        
        # Reconstruct byte value
        value = sum(bit << i for i, bit in enumerate(bits))
        
        return ByteWord(value)
        
    def morphological_convolution(self, word1: ByteWord, word2: ByteWord) -> ByteWord:
        """
        Morphological convolution using spectral multiplication
        This implements semantic composition in frequency domain
        """
        # Transform both words to spectral domain
        spec1 = self.byteword_to_spectrum(word1)
        spec2 = self.byteword_to_spectrum(word2)
        
        # Pointwise multiplication in spectral domain
        # (convolution in spatial domain = multiplication in frequency domain)
        conv_spec = [a * b for a, b in zip(spec1, spec2)]
        
        # Transform back to spatial domain
        result = self.spectrum_to_byteword(conv_spec)
        
        return result
        
    def morphological_correlation(self, word1: ByteWord, word2: ByteWord) -> float:
        """Compute morphological correlation between two ByteWords"""
        spec1 = self.byteword_to_spectrum(word1)
        spec2 = self.byteword_to_spectrum(word2)
        
        # Cross-correlation in frequency domain
        correlation = sum(a.conjugate() * b for a, b in zip(spec1, spec2))
        
        return abs(correlation.real)

class BinaryAbelianGroup:
    """
    Flat binary Abelization: XOR group structure for ByteWords
    Implements the algebraic foundation for morphological operations
    """
    
    def __init__(self):
        self.identity = ByteWord(0b00000000)  # XOR identity
        
    def group_operation(self, a: ByteWord, b: ByteWord) -> ByteWord:
        """XOR as the fundamental group operation"""
        return ByteWord(a.value ^ b.value)
        
    def inverse(self, a: ByteWord) -> ByteWord:
        """Every element is its own inverse in XOR group"""
        return a
        
    def is_identity(self, a: ByteWord) -> bool:
        """Check if element is group identity"""
        return a.value == 0
        
    def order(self) -> int:
        """Order of the group (2^8 = 256)"""
        return 256
        
    def generate_subgroup(self, generator: ByteWord) -> List[ByteWord]:
        """Generate cyclic subgroup from a generator element"""
        subgroup = [self.identity]
        current = generator
        
        while current not in subgroup:
            subgroup.append(current)
            current = self.group_operation(current, generator)
            
        return subgroup

class MorphologicalByteWord(ByteWord):
    """
    Enhanced ByteWord with Cook & Mertz morphological operations
    """
    
    def __init__(self, value: int = 0):
        super().__init__(value)
        self.fourier_engine = MorphologicalFourierEngine()
        self.abelian_group = BinaryAbelianGroup()
        
    def compose(self, other: 'MorphologicalByteWord') -> 'MorphologicalByteWord':
        """
        Morphological composition using spectral convolution
        Implements: ψ₁ ∘ ψ₂ via Cook & Mertz transformation
        """
        result_word = self.fourier_engine.morphological_convolution(self, other)
        result = MorphologicalByteWord(result_word.value)
        result._theorem = f"({self} ∘ {other}) via spectral convolution"
        return result
        
    def abelian_compose(self, other: 'MorphologicalByteWord') -> 'MorphologicalByteWord':
        """
        Direct XOR composition in flat binary Abelian group
        """
        result_word = self.abelian_group.group_operation(self, other)
        result = MorphologicalByteWord(result_word.value)
        result._theorem = f"({self} ⊕ {other}) in binary Abelian group"
        return result
        
    def spectral_analysis(self) -> List[complex]:
        """Get spectral decomposition of this ByteWord"""
        return self.fourier_engine.byteword_to_spectrum(self)
        
    def morphological_distance(self, other: 'MorphologicalByteWord') -> float:
        """Compute morphological distance using spectral correlation"""
        correlation = self.fourier_engine.morphological_correlation(self, other)
        return 1.0 - correlation  # Distance = 1 - correlation
        
    def propagate(self, steps: int = 1) -> List['MorphologicalByteWord']:
        """
        Propagate through morphological field using roots of unity evolution
        """
        states = [self]
        current = self
        
        for step in range(steps):
            # Evolve using primitive root of unity
            root = self.fourier_engine.transform.primitive_root_of_unity(step + 1)
            
            # Apply spectral rotation
            spectrum = current.spectral_analysis()
            rotated_spectrum = [coeff * root for coeff in spectrum]
            
            # Transform back
            evolved_word = self.fourier_engine.spectrum_to_byteword(rotated_spectrum)
            evolved = MorphologicalByteWord(evolved_word.value)
            evolved._theorem = f"Evolved via ω^{step + 1} root of unity"
            
            states.append(evolved)
            current = evolved
            
        return states
        
    def to_float(self) -> float:
        """Convert to float via spectral magnitude"""
        spectrum = self.spectral_analysis()
        magnitude = sum(abs(coeff)**2 for coeff in spectrum)
        return magnitude / len(spectrum)  # Normalized
        
    def speaks_through(self) -> str:
        """What this ByteWord says about itself"""
        spectrum = self.spectral_analysis()
        dominant_freq = max(range(len(spectrum)), key=lambda i: abs(spectrum[i]))
        
        return f"I am ByteWord(0b{self.value:08b}) speaking through frequency {dominant_freq}"

# Usage examples and tests
def demonstrate_cook_mertz():
    """Demonstrate Cook & Mertz morphological operations"""
    print("=== Cook & Mertz Morphological ByteWords ===")
    
    # Create test ByteWords
    word1 = MorphologicalByteWord(0b10101010)  # Alternating pattern
    word2 = MorphologicalByteWord(0b11001100)  # Block pattern
    
    print(f"Word 1: {word1}")
    print(f"Word 2: {word2}")
    print()
    
    # Spectral analysis
    print("=== Spectral Analysis ===")
    spec1 = word1.spectral_analysis()
    spec2 = word2.spectral_analysis()
    
    print(f"Word 1 spectrum: {[f'{c:.3f}' for c in spec1[:4]]}")
    print(f"Word 2 spectrum: {[f'{c:.3f}' for c in spec2[:4]]}")
    print()
    
    # Morphological composition
    print("=== Morphological Composition ===")
    spectral_comp = word1.compose(word2)
    abelian_comp = word1.abelian_compose(word2)
    
    print(f"Spectral composition: {spectral_comp}")
    print(f"Abelian composition:  {abelian_comp}")
    print()
    
    # Evolution through roots of unity
    print("=== Evolution via Roots of Unity ===")
    evolution = word1.propagate(steps=4)
    
    for i, state in enumerate(evolution):
        if hasattr(state, '_theorem'):
            print(f"Step {i}: {state} - {state._theorem}")
        else:
            print(f"Step {i}: {state}")
    print()
    
    # Morphological distance
    print("=== Morphological Distances ===")
    dist_12 = word1.morphological_distance(word2)
    dist_11 = word1.morphological_distance(word1)
    
    print(f"Distance(word1, word2): {dist_12:.6f}")
    print(f"Distance(word1, word1): {dist_11:.6f}")
    print()
    
    # What do they say?
    print("=== ByteWord Theorems ===")
    print(word1.speaks_through())
    print(word2.speaks_through())
    print(spectral_comp.speaks_through())

if __name__ == "__main__":
    demonstrate_cook_mertz()