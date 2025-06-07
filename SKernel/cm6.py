import math
import cmath
from typing import List, Tuple, Union, Optional

class BinaryAbelianGroup:
    """
    Binary Abelian group operations for Cook & Mertz roots of unity
    Using XOR as the group operation in flat (binary) Abelization
    """
    
    def __init__(self, order: int = 8):
        """Initialize binary abelian group of given order (must be power of 2)"""
        if order & (order - 1) != 0:
            raise ValueError(f"Order {order} must be a power of 2")
        
        self.order = order
        self.log_order = int(math.log2(order))
        self._roots_cache = {}
        self._generate_primitive_roots()
    
    def _generate_primitive_roots(self):
        """Generate primitive roots of unity for the binary group"""
        # In binary abelian group, primitive root is omega = e^(2πi/n)
        self.omega = cmath.exp(2j * cmath.pi / self.order)
        
        # Generate all roots: omega^k for k = 0, 1, ..., order-1
        self.roots = [self.omega ** k for k in range(self.order)]
        
        # Store in cache with binary index
        for k, root in enumerate(self.roots):
            self._roots_cache[k] = root
    
    def get_root(self, k: int) -> complex:
        """Get the k-th root of unity"""
        return self._roots_cache[k % self.order]
    
    def binary_mult(self, a: int, b: int) -> int:
        """Binary multiplication using XOR (flat abelian operation)"""
        return a ^ b
    
    def binary_add(self, a: int, b: int) -> int:
        """Binary addition using XOR (in GF(2))"""
        return a ^ b

class CookMertzTransform:
    """
    Cook & Mertz FFT implementation using binary abelian groups
    Hand-rolled pure Python, no numpy crutches!
    """
    
    def __init__(self, size: int = 8):
        if size & (size - 1) != 0:
            raise ValueError("Size must be a power of 2")
            
        self.size = size
        self.log_size = int(math.log2(size))
        self.group = BinaryAbelianGroup(size)
        
    def bit_reverse(self, n: int, bits: int) -> int:
        """Bit-reverse a number for FFT ordering"""
        result = 0
        for i in range(bits):
            if n & (1 << i):
                result |= 1 << (bits - 1 - i)
        return result
    
    def cook_mertz_fft(self, data: List[complex]) -> List[complex]:
        """
        Cook & Mertz FFT using binary abelian group operations
        Pure Python implementation with XOR-based indexing
        """
        n = len(data)
        if n != self.size:
            raise ValueError(f"Data size {n} must match transform size {self.size}")
        
        # Bit-reverse the input data
        result = [0] * n
        for i in range(n):
            j = self.bit_reverse(i, self.log_size)
            result[j] = data[i]
        
        # Cooley-Tukey FFT with binary abelian group structure
        length = 2
        while length <= n:
            # Get primitive root for this stage
            w = cmath.exp(-2j * cmath.pi / length)
            
            for start in range(0, n, length):
                wn = 1
                for j in range(length // 2):
                    # Binary abelian group indices
                    u_idx = start + j
                    v_idx = start + j + length // 2
                    
                    u = result[u_idx]
                    v = result[v_idx] * wn
                    
                    # XOR-based combination (flat abelianization)
                    result[u_idx] = u + v
                    result[v_idx] = u - v
                    
                    wn *= w
            
            length *= 2
        
        return result
    
    def cook_mertz_ifft(self, data: List[complex]) -> List[complex]:
        """Inverse Cook & Mertz FFT"""
        n = len(data)
        
        # Conjugate the input
        conjugated = [x.conjugate() for x in data]
        
        # Apply forward FFT
        result = self.cook_mertz_fft(conjugated)
        
        # Conjugate and scale
        result = [x.conjugate() / n for x in result]
        
        return result

class ByteWordCookMertz:
    """
    ByteWord integration with Cook & Mertz roots of unity
    Binary abelianization with XOR operations
    """
    
    def __init__(self, value: int = 0b00000000):
        self.value = value & 0xFF  # 8-bit constraint
        self.transform = CookMertzTransform(8)  # 8-point transform for 8-bit ByteWord
        self.group = BinaryAbelianGroup(8)
        
        # Morphological state
        self._morphological_phase = 0
        self._semantic_amplitude = 1.0
        self._theorem = None
    
    def to_complex_vector(self) -> List[complex]:
        """Convert ByteWord to complex vector for Cook & Mertz transform"""
        # Each bit becomes a complex amplitude
        vector = []
        for i in range(8):
            bit = (self.value >> i) & 1
            # Map bit to complex amplitude using root of unity
            root = self.group.get_root(i)
            amplitude = bit * root * self._semantic_amplitude
            vector.append(amplitude)
        
        return vector
    
    def from_complex_vector(self, vector: List[complex]) -> 'ByteWordCookMertz':
        """Reconstruct ByteWord from complex vector"""
        new_value = 0
        
        for i, amplitude in enumerate(vector):
            # Extract bit from complex amplitude
            magnitude = abs(amplitude)
            if magnitude > 0.5:  # Threshold for bit detection
                new_value |= (1 << i)
        
        result = ByteWordCookMertz(new_value)
        result._semantic_amplitude = sum(abs(x) for x in vector) / len(vector)
        return result
    
    def morphological_fft(self) -> List[complex]:
        """Apply Cook & Mertz FFT to ByteWord's morphological structure"""
        vector = self.to_complex_vector()
        transformed = self.transform.cook_mertz_fft(vector)
        
        # Store morphological phase information
        self._morphological_phase = cmath.phase(transformed[0]) if transformed[0] != 0 else 0
        
        return transformed
    
    def morphological_ifft(self, spectrum: List[complex]) -> 'ByteWordCookMertz':
        """Reconstruct ByteWord from frequency domain"""
        vector = self.transform.cook_mertz_ifft(spectrum)
        return self.from_complex_vector(vector)
    
    def compose_cook_mertz(self, other: 'ByteWordCookMertz') -> 'ByteWordCookMertz':
        """
        Compose two ByteWords using Cook & Mertz convolution
        This is morphological interference in frequency domain
        """
        # Transform both to frequency domain
        self_spectrum = self.morphological_fft()
        other_spectrum = other.morphological_fft()
        
        # Pointwise multiplication in frequency domain (convolution in time domain)
        # But we use XOR-based binary abelian operations
        combined_spectrum = []
        for i, (a, b) in enumerate(zip(self_spectrum, other_spectrum)):
            # Binary abelian group operation in complex domain
            xor_idx = self.group.binary_mult(i, i)  # XOR-based indexing
            root = self.group.get_root(xor_idx)
            
            # Morphological interference with XOR structure
            combined = (a * b) * root
            combined_spectrum.append(combined)
        
        # Transform back to ByteWord
        result = self.morphological_ifft(combined_spectrum)
        result._theorem = f"CookMertz({self.value:08b} ⊕ {other.value:08b}) = {result.value:08b}"
        
        return result
    
    def binary_abelian_mult(self, scalar: int) -> 'ByteWordCookMertz':
        """Multiply ByteWord by scalar using binary abelian group"""
        # XOR-based scalar multiplication in flat abelianization
        new_value = 0
        
        for i in range(8):
            if self.value & (1 << i):
                # Apply scalar via binary abelian group operation
                scaled_idx = self.group.binary_mult(i, scalar)
                new_value ^= (1 << scaled_idx)
        
        result = ByteWordCookMertz(new_value)
        result._theorem = f"BinaryAbelian({self.value:08b} * {scalar}) = {result.value:08b}"
        return result
    
    def morphological_convolution(self, other: 'ByteWordCookMertz') -> 'ByteWordCookMertz':
        """
        Pure morphological convolution using Cook & Mertz
        This is the fundamental operation of meaning-mixing
        """
        # Get frequency representations
        self_freq = self.morphological_fft()
        other_freq = other.morphological_fft()
        
        # Convolution in frequency domain
        conv_freq = [a * b for a, b in zip(self_freq, other_freq)]
        
        # Back to morphological space
        result = self.morphological_ifft(conv_freq)
        result._theorem = f"Conv({self.value:08b}, {other.value:08b}) = {result.value:08b}"
        
        return result
    
    def get_spectral_signature(self) -> dict:
        """Get the spectral signature of this ByteWord"""
        spectrum = self.morphological_fft()
        
        return {
            'magnitude': [abs(x) for x in spectrum],
            'phase': [cmath.phase(x) for x in spectrum],
            'power': [abs(x)**2 for x in spectrum],
            'dominant_frequency': spectrum.index(max(spectrum, key=abs)),
            'spectral_entropy': self._calculate_spectral_entropy(spectrum)
        }
    
    def _calculate_spectral_entropy(self, spectrum: List[complex]) -> float:
        """Calculate entropy of the spectral distribution"""
        powers = [abs(x)**2 for x in spectrum]
        total_power = sum(powers)
        
        if total_power == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = [p / total_power for p in powers]
        
        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def __repr__(self):
        return f"ByteWordCookMertz({self.value:08b})"
    
    def __eq__(self, other):
        return isinstance(other, ByteWordCookMertz) and self.value == other.value
    
    def __xor__(self, other):
        """XOR operation for binary abelian group"""
        if isinstance(other, ByteWordCookMertz):
            return ByteWordCookMertz(self.value ^ other.value)
        return ByteWordCookMertz(self.value ^ other)

# Example usage and testing
if __name__ == "__main__":
    # Create some ByteWords
    word1 = ByteWordCookMertz(0b10101010)
    word2 = ByteWordCookMertz(0b11000011)
    
    print(f"Word1: {word1}")
    print(f"Word2: {word2}")
    
    # Test Cook & Mertz composition
    composed = word1.compose_cook_mertz(word2)
    print(f"Composed: {composed}")
    print(f"Theorem: {composed._theorem}")
    
    # Test spectral analysis
    signature = word1.get_spectral_signature()
    print(f"Spectral signature: {signature}")
    
    # Test convolution
    convolved = word1.morphological_convolution(word2)
    print(f"Convolved: {convolved}")
    
    # Test binary abelian operations
    scaled = word1.binary_abelian_mult(3)
    print(f"Scaled: {scaled}")
    
    # Test XOR operation
    xored = word1 ^ word2
    print(f"XORed: {xored}")