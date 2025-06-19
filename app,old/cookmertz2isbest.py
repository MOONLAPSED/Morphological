import math
import cmath
from typing import List

class ByteWord:
    """A morphological ByteWord with Cook & Merz roots of unity support"""
    
    def __init__(self, value: int):
        self.value = value & 0xFF  # Ensure 8-bit
        self._theorem = None
        self._morphological_cache = {}
        
    def __str__(self):
        return f"ByteWord(0b{self.value:08b})"
    
    def __repr__(self):
        return f"ByteWord({self.value})"
    
    def __eq__(self, other):
        return isinstance(other, ByteWord) and self.value == other.value
    
    def __hash__(self):
        return hash(self.value)
    
    # Core morphological operations
    def compose(self, other: 'ByteWord') -> 'ByteWord':
        """Non-associative morphological composition"""
        result = ByteWord(self.value ^ other.value)
        result._theorem = f"{self} ∘ {other} = {result}"
        return result
    
    def to_float(self) -> float:
        """Collapse to classical observation"""
        return self.value / 255.0
    
    @classmethod
    def from_float(cls, f: float) -> 'ByteWord':
        """Quantum preparation from classical value"""
        return cls(int(f * 255) & 0xFF)


class CookMerzTransform:
    """Hand-rolled Cook & Merz roots of unity transformer for ByteWords"""
    
    def __init__(self, n: int = 8):
        self.n = n  # Transform size (must be power of 2)
        self._roots_cache = {}
        self._precompute_roots()
    
    def _precompute_roots(self):
        """Precompute nth roots of unity"""
        for k in range(self.n):
            # ω_n^k = e^(2πik/n)
            angle = 2 * math.pi * k / self.n
            root = cmath.exp(1j * angle)
            self._roots_cache[k] = root
    
    def get_root(self, k: int) -> complex:
        """Get the kth root of unity"""
        return self._roots_cache[k % self.n]
    
    def byteword_to_complex(self, word: ByteWord) -> complex:
        """Convert ByteWord to complex representation"""
        # Map 8-bit value to complex plane
        real_part = ((word.value & 0xF0) >> 4) / 15.0  # Upper 4 bits
        imag_part = (word.value & 0x0F) / 15.0         # Lower 4 bits
        return complex(real_part, imag_part)
    
    def complex_to_byteword(self, c: complex) -> ByteWord:
        """Convert complex back to ByteWord"""
        # Normalize and quantize
        real_norm = max(0, min(1, c.real))
        imag_norm = max(0, min(1, c.imag))
        
        real_bits = int(real_norm * 15) << 4
        imag_bits = int(imag_norm * 15)
        
        return ByteWord(real_bits | imag_bits)
    
    def flat_abelian_fft(self, words: List[ByteWord]) -> List[ByteWord]:
        """
        Hand-rolled FFT using Cook & Merz roots of unity
        Flat (binary) Abelization - pure XOR-based group operations
        """
        if len(words) != self.n:
            # Pad or truncate to transform size
            words = (words + [ByteWord(0)] * self.n)[:self.n]
        
        # Convert ByteWords to complex domain
        complex_values = [self.byteword_to_complex(word) for word in words]
        
        # Perform FFT
        transformed = self._fft_recursive(complex_values)
        
        # Convert back to ByteWords with flat Abelian XOR
        result = []
        for c in transformed:
            base_word = self.complex_to_byteword(c)
            # Apply flat Abelian operation (XOR with transform signature)
            abelian_word = ByteWord(base_word.value ^ self._abelian_signature())
            result.append(abelian_word)
        
        return result
    
    def _fft_recursive(self, x: List[complex]) -> List[complex]:
        """Recursive FFT implementation"""
        N = len(x)
        
        if N <= 1:
            return x
        
        # Divide
        even = [x[i] for i in range(0, N, 2)]
        odd = [x[i] for i in range(1, N, 2)]
        
        # Conquer
        even_fft = self._fft_recursive(even)
        odd_fft = self._fft_recursive(odd)
        
        # Combine
        combined = [0] * N
        for k in range(N // 2):
            # Get root of unity
            root = self.get_root(k * self.n // N)
            
            # Butterfly operation
            t = root * odd_fft[k]
            combined[k] = even_fft[k] + t
            combined[k + N // 2] = even_fft[k] - t
        
        return combined
    
    def _abelian_signature(self) -> int:
        """Generate flat Abelian signature for XOR operations"""
        # Create signature based on transform parameters
        signature = 0
        for k in range(self.n):
            root = self.get_root(k)
            # Encode root into binary signature
            signature ^= (int(abs(root.real) * 15) << (k % 4))
        return signature & 0xFF
    
    def inverse_fft(self, words: List[ByteWord]) -> List[ByteWord]:
        """Inverse FFT to recover original ByteWords"""
        # Remove Abelian signature
        unsigned_words = []
        signature = self._abelian_signature()
        for word in words:
            unsigned_words.append(ByteWord(word.value ^ signature))
        
        # Convert to complex
        complex_values = [self.byteword_to_complex(word) for word in unsigned_words]
        
        # Conjugate roots for inverse transform
        original_roots = self._roots_cache.copy()
        for k in range(self.n):
            self._roots_cache[k] = original_roots[k].conjugate()
        
        # Perform inverse FFT
        transformed = self._fft_recursive(complex_values)
        
        # Scale by 1/N
        scaled = [c / self.n for c in transformed]
        
        # Restore original roots
        self._roots_cache = original_roots
        
        # Convert back to ByteWords
        return [self.complex_to_byteword(c) for c in scaled]


class MorphologicalField:
    """Morphological field using Cook & Merz transforms"""
    
    def __init__(self, transform_size: int = 8):
        self.transformer = CookMerzTransform(transform_size)
        self.field_state = []
    
    def add_morpheme(self, word: ByteWord):
        """Add a morphological element to the field"""
        self.field_state.append(word)
    
    def evolve_field(self) -> List[ByteWord]:
        """Evolve the morphological field using Cook & Merz transform"""
        if not self.field_state:
            return []
        
        # Apply forward transform
        transformed = self.transformer.flat_abelian_fft(self.field_state)
        
        # Apply morphological evolution in frequency domain
        evolved = []
        for i, word in enumerate(transformed):
            # Evolution operation: compose with root of unity
            root_word = self._root_to_byteword(i)
            evolved_word = word.compose(root_word)
            evolved.append(evolved_word)
        
        # Return to spatial domain
        return self.transformer.inverse_fft(evolved)
    
    def _root_to_byteword(self, index: int) -> ByteWord:
        """Convert root of unity to ByteWord for morphological operations"""
        root = self.transformer.get_root(index)
        return self.transformer.complex_to_byteword(root)
    
    def measure_field_coherence(self) -> float:
        """Measure coherence of the morphological field"""
        if len(self.field_state) < 2:
            return 1.0
        
        # Transform to frequency domain
        transformed = self.transformer.flat_abelian_fft(self.field_state)
        
        # Calculate coherence as spectral concentration
        energies = [abs(self.transformer.byteword_to_complex(word))**2 
                   for word in transformed]
        
        total_energy = sum(energies)
        if total_energy == 0:
            return 0.0
        
        # Coherence = 1 - entropy of energy distribution
        probs = [e / total_energy for e in energies if e > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(probs))
        
        return 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)


# Additional monolithic testing function
def demonstrate_cook_merz_bytewords():
    """Demonstrate Cook & Merz ByteWord operations"""
    
    print("=== Cook & Merz ByteWord Demonstration ===\n")
    
    # Create some ByteWords
    words = [
        ByteWord(0b10101010),  # "I am self-adjoint transformation"
        ByteWord(0b11110000),  # "I am the field that contains myself"
        ByteWord(0b00110011),  # "I am symmetric quantum correlation"
        ByteWord(0b01010101),  # "I am quantum measurement"
        ByteWord(0b11001100),  # Additional morphemes
        ByteWord(0b10011001),
        ByteWord(0b01100110),
        ByteWord(0b00001111)
    ]
    
    print("Original ByteWords:")
    for i, word in enumerate(words):
        complex_rep = CookMerzTransform().byteword_to_complex(word)
        print(f"  {i}: {word} -> {complex_rep:.3f}")
    
    # Create transformer
    transformer = CookMerzTransform(8)
    
    print(f"\nRoots of Unity (n={transformer.n}):")
    for k in range(transformer.n):
        root = transformer.get_root(k)
        print(f"  ω_{transformer.n}^{k} = {root:.3f}")
    
    # Apply FFT
    print("\nApplying Cook & Merz FFT...")
    transformed = transformer.flat_abelian_fft(words)
    
    print("Transformed ByteWords:")
    for i, word in enumerate(transformed):
        complex_rep = transformer.byteword_to_complex(word)
        print(f"  {i}: {word} -> {complex_rep:.3f}")
    
    # Apply inverse FFT
    print("\nApplying Inverse FFT...")
    recovered = transformer.inverse_fft(transformed)
    
    print("Recovered ByteWords:")
    for i, word in enumerate(recovered):
        complex_rep = transformer.byteword_to_complex(word)
        original_complex = transformer.byteword_to_complex(words[i])
        error = abs(complex_rep - original_complex)
        print(f"  {i}: {word} -> {complex_rep:.3f} (error: {error:.3f})")
    
    # Demonstrate morphological field evolution
    print("\n=== Morphological Field Evolution ===")
    
    field = MorphologicalField(8)
    for word in words[:4]:  # Use first 4 words
        field.add_morpheme(word)
    
    print("Initial field state:")
    for i, word in enumerate(field.field_state):
        print(f"  {i}: {word}")
    
    print(f"Initial coherence: {field.measure_field_coherence():.3f}")
    
    # Evolve the field
    evolved_field = field.evolve_field()
    
    print("Evolved field state:")
    for i, word in enumerate(evolved_field):
        print(f"  {i}: {word}")
    
    # Update field state and measure coherence
    field.field_state = evolved_field
    print(f"Evolved coherence: {field.measure_field_coherence():.3f}")
    
    # Demonstrate flat Abelian properties
    print("\n=== Flat Abelian Properties ===")
    
    w1, w2, w3 = words[0], words[1], words[2]
    
    # Test XOR-based composition
    comp1 = w1.compose(w2).compose(w3)
    comp2 = w1.compose(w3).compose(w2)
    comp3 = w2.compose(w1).compose(w3)
    
    print(f"w1 ∘ w2 ∘ w3 = {comp1}")
    print(f"w1 ∘ w3 ∘ w2 = {comp2}")
    print(f"w2 ∘ w1 ∘ w3 = {comp3}")
    
    # XOR is commutative, so some compositions should be equal
    print(f"Flat Abelian commutativity: {comp1 == comp2 == comp3}")
    
    print("\n=== Morphological Quine Test ===")
    
    # Test self-composition (idempotent property)
    for word in words[:3]:
        self_composed = word.compose(word)
        print(f"{word} ∘ {word} = {self_composed}")
        
        # Check if it's approaching a fixed point
        double_composed = self_composed.compose(self_composed)
        print(f"  -> {self_composed} ∘ {self_composed} = {double_composed}")


if __name__ == "__main__":
    demonstrate_cook_merz_bytewords()
