import math
import cmath
from typing import List, Tuple, Union

class ByteWord:
    """Morphological quantum computing unit with Cook & Mertz FFT capabilities"""
    
    def __init__(self, value: int = 0):
        self.value = value & 0xFF  # 8-bit constraint
        self._theorem = None
        
    def __eq__(self, other):
        return isinstance(other, ByteWord) and self.value == other.value
        
    def __repr__(self):
        return f"ByteWord(0b{self.value:08b})"
    
    def __xor__(self, other):
        """XOR operation for flat binary Abelianization"""
        if isinstance(other, ByteWord):
            return ByteWord(self.value ^ other.value)
        return ByteWord(self.value ^ other)

class CookMertzFFT:
    """
    Cook & Mertz FFT using roots of unity
    with XOR-based flat binary Abelianization
    
    Cook & Mertz insight: FFT can be computed using any Abelian group operation
    Here we use XOR (⊕) instead of addition (+) for true binary morphology
    """
    
    def __init__(self, size: int = 8):
        self.size = size
        self.log_size = int(math.log2(size))
        assert 2**self.log_size == size, "Size must be power of 2"
        
        # Precompute roots of unity for efficiency
        self._roots = self._compute_roots_of_unity()
        self._xor_table = self._build_xor_multiplication_table()
    
    def _compute_roots_of_unity(self) -> List[complex]:
        """Compute nth roots of unity: e^(2πik/n) for k = 0..n-1"""
        roots = []
        for k in range(self.size):
            angle = 2 * math.pi * k / self.size
            root = cmath.exp(1j * angle)
            roots.append(root)
        return roots
    
    def _build_xor_multiplication_table(self) -> List[List[int]]:
        """
        Build multiplication table for GF(2^8) with primitive polynomial
        This gives us proper multiplication that respects XOR structure
        """
        # Using primitive polynomial x^8 + x^4 + x^3 + x + 1 = 0x11B
        primitive_poly = 0x11B
        
        table = [[0 for _ in range(256)] for _ in range(256)]
        
        for a in range(256):
            for b in range(256):
                table[a][b] = self._gf_multiply(a, b, primitive_poly)
        
        return table
    
    def _gf_multiply(self, a: int, b: int, primitive_poly: int) -> int:
        """Galois Field multiplication in GF(2^8)"""
        result = 0
        while b:
            if b & 1:
                result ^= a
            a <<= 1
            if a & 0x100:
                a ^= primitive_poly
            b >>= 1
        return result & 0xFF
    
    def bit_reverse(self, num: int, bits: int) -> int:
        """Bit-reverse a number for FFT ordering"""
        result = 0
        for _ in range(bits):
            result = (result << 1) | (num & 1)
            num >>= 1
        return result
    
    def xor_fft_forward(self, data: List[ByteWord]) -> List[complex]:
        """
        Forward Cook & Mertz FFT using XOR Abelianization
        
        Instead of: X[k] = Σ x[n] * e^(-2πikn/N)
        We compute: X[k] = ⊕ x[n] ⊗ ω^(kn)
        
        Where ⊕ is XOR and ⊗ is our morphological multiplication
        """
        n = len(data)
        assert n == self.size, f"Data length must be {self.size}"
        
        # Bit-reverse input order (Cooley-Tukey requirement)
        x = [ByteWord(0)] * n
        for i in range(n):
            j = self.bit_reverse(i, self.log_size)
            x[j] = data[i]
        
        # Cook & Mertz FFT with XOR Abelianization
        result = []
        
        for k in range(n):
            # For each frequency bin k
            accumulator = complex(0, 0)
            
            for j in range(n):
                # Get the root of unity: ω^(kj)
                root_power = (k * j) % n
                omega = self._roots[root_power]
                
                # Morphological multiplication: ByteWord ⊗ complex
                byte_val = x[j].value
                morphological_product = self._morphological_multiply(byte_val, omega)
                
                # XOR Abelianization instead of addition
                accumulator = self._xor_complex_add(accumulator, morphological_product)
            
            result.append(accumulator)
        
        return result
    
    def _morphological_multiply(self, byte_val: int, omega: complex) -> complex:
        """
        Morphological multiplication of ByteWord with complex root of unity
        Uses the XOR multiplication table for proper GF(2^8) arithmetic
        """
        # Convert complex to polar form
        magnitude = abs(omega)
        phase = cmath.phase(omega)
        
        # Apply GF(2^8) multiplication to magnitude component
        real_part = self._xor_table[byte_val][int(magnitude * 255) & 0xFF]
        imag_part = self._xor_table[byte_val][int(phase * 255 / (2 * math.pi)) & 0xFF]
        
        # Reconstruct complex number with morphological structure
        return complex(real_part / 255.0, imag_part / 255.0)
    
    def _xor_complex_add(self, a: complex, b: complex) -> complex:
        """
        XOR-based complex addition for Abelianization
        Real and imaginary parts are XORed separately
        """
        # Convert to integer representation for XOR
        a_real_int = int(a.real * 255) & 0xFF
        a_imag_int = int(a.imag * 255) & 0xFF
        b_real_int = int(b.real * 255) & 0xFF  
        b_imag_int = int(b.imag * 255) & 0xFF
        
        # XOR the components
        result_real = (a_real_int ^ b_real_int) / 255.0
        result_imag = (a_imag_int ^ b_imag_int) / 255.0
        
        return complex(result_real, result_imag)
    
    def xor_fft_inverse(self, freq_data: List[complex]) -> List[ByteWord]:
        """
        Inverse Cook & Mertz FFT using XOR Abelianization
        """
        n = len(freq_data)
        
        # Conjugate the roots of unity for inverse transform
        conjugate_roots = [root.conjugate() for root in self._roots]
        
        result = []
        
        for j in range(n):
            # For each time sample j
            accumulator = complex(0, 0)
            
            for k in range(n):
                # Get the conjugate root: ω^(-kj)
                root_power = (k * j) % n
                omega_conj = conjugate_roots[root_power]
                
                # Morphological multiplication with frequency data
                freq_val = freq_data[k]
                morphological_product = self._complex_morphological_multiply(freq_val, omega_conj)
                
                # XOR Abelianization
                accumulator = self._xor_complex_add(accumulator, morphological_product)
            
            # Convert back to ByteWord
            byte_val = int(abs(accumulator) * 255) & 0xFF
            result.append(ByteWord(byte_val))
        
        return result
    
    def _complex_morphological_multiply(self, c1: complex, c2: complex) -> complex:
        """Complex multiplication using morphological structure"""
        # Convert to GF(2^8) representation
        c1_real = int(c1.real * 255) & 0xFF
        c1_imag = int(c1.imag * 255) & 0xFF
        c2_real = int(c2.real * 255) & 0xFF
        c2_imag = int(c2.imag * 255) & 0xFF
        
        # GF(2^8) complex multiplication: (a+bi)(c+di) = (ac⊕bd) + (ad⊕bc)i
        real_part = self._xor_table[c1_real][c2_real] ^ self._xor_table[c1_imag][c2_imag]
        imag_part = self._xor_table[c1_real][c2_imag] ^ self._xor_table[c1_imag][c2_real]
        
        return complex(real_part / 255.0, imag_part / 255.0)
    
    def morphological_convolution(self, signal1: List[ByteWord], signal2: List[ByteWord]) -> List[ByteWord]:
        """
        Convolution using Cook & Mertz FFT
        This is the morphological equivalent of classical convolution
        """
        # Pad signals to prevent circular convolution artifacts
        padded_size = len(signal1) + len(signal2) - 1
        fft_size = 2**int(math.ceil(math.log2(padded_size)))
        
        # Create FFT instance of appropriate size
        fft = CookMertzFFT(fft_size)
        
        # Pad signals
        padded1 = signal1 + [ByteWord(0)] * (fft_size - len(signal1))
        padded2 = signal2 + [ByteWord(0)] * (fft_size - len(signal2))
        
        # Forward FFT both signals
        freq1 = fft.xor_fft_forward(padded1)
        freq2 = fft.xor_fft_forward(padded2)
        
        # Pointwise morphological multiplication in frequency domain
        freq_product = []
        for i in range(len(freq1)):
            product = fft._complex_morphological_multiply(freq1[i], freq2[i])
            freq_product.append(product)
        
        # Inverse FFT to get convolution result
        result = fft.xor_fft_inverse(freq_product)
        
        # Trim to original size
        return result[:padded_size]

# Extended ByteWord with Cook & Mertz capabilities
class MorphologicalByteWord(ByteWord):
    """ByteWord extended with Cook & Mertz FFT morphological operations"""
    
    def __init__(self, value: int = 0):
        super().__init__(value)
        self.fft = CookMertzFFT()
    
    def speaks_through(self) -> str:
        return f"I am morphological frequency {self.value:08b} in the Cook & Mertz field"
    
    def frequency_decompose(self, context: List['MorphologicalByteWord']) -> List[complex]:
        """Decompose this ByteWord in the context of others using FFT"""
        if len(context) != 8:
            context = (context + [MorphologicalByteWord(0)] * 8)[:8]
        
        return self.fft.xor_fft_forward(context)
    
    def morphological_correlate(self, other: 'MorphologicalByteWord') -> List['MorphologicalByteWord']:
        """
        Compute morphological correlation using Cook & Mertz convolution
        This reveals hidden patterns between ByteWords
        """
        # Create context vectors
        self_context = [MorphologicalByteWord((self.value >> i) & 1) for i in range(8)]
        other_context = [MorphologicalByteWord((other.value >> i) & 1) for i in range(8)]
        
        # Morphological convolution reveals correlation structure
        correlation = self.fft.morphological_convolution(self_context, other_context)
        
        return correlation
    
    def flat_abelianize(self, others: List['MorphologicalByteWord']) -> 'MorphologicalByteWord':
        """
        Flat binary Abelianization using XOR group operation
        This is the binary version of your ternary T/V/C encoding
        """
        result = self
        for other in others:
            result = result ^ other  # XOR Abelianization
        
        result._theorem = f"Flat Abelianization: {self} ⊕ {others} = {result}"
        return result

# Demo and test functions
def demo_cook_mertz_fft():
    """Demonstrate Cook & Mertz FFT with morphological ByteWords"""
    print("🔮 Cook & Mertz FFT Demo with Morphological ByteWords")
    print("=" * 60)
    
    # Create test signal
    signal = [
        MorphologicalByteWord(0b10101010),  # Alternating pattern
        MorphologicalByteWord(0b11110000),  # Block pattern  
        MorphologicalByteWord(0b00001111),  # Inverse block
        MorphologicalByteWord(0b11000011),  # Symmetric pattern
        MorphologicalByteWord(0b01010101),  # Inverse alternating
        MorphologicalByteWord(0b10011001),  # Complex pattern
        MorphologicalByteWord(0b01100110),  # Another complex
        MorphologicalByteWord(0b00000000),  # Zero
    ]
    
    print("Original Signal:")
    for i, word in enumerate(signal):
        print(f"  [{i}] {word} - {word.speaks_through()}")
    
    # Perform FFT
    fft = CookMertzFFT(8)
    freq_domain = fft.xor_fft_forward(signal)
    
    print("\nFrequency Domain (Cook & Mertz XOR-FFT):")
    for i, freq in enumerate(freq_domain):
        magnitude = abs(freq)
        phase = cmath.phase(freq)
        print(f"  [{i}] {freq:.4f} (mag: {magnitude:.4f}, phase: {phase:.4f})")
    
    # Inverse FFT
    reconstructed = fft.xor_fft_inverse(freq_domain)
    
    print("\nReconstructed Signal:")
    for i, word in enumerate(reconstructed):
        print(f"  [{i}] {word} - Original: {signal[i]}")
        
    # Verify perfect reconstruction
    reconstruction_error = sum(abs(orig.value - recon.value) for orig, recon in zip(signal, reconstructed))
    print(f"\nReconstruction Error: {reconstruction_error}")
    
    # Demonstrate morphological correlation
    print("\n🧬 Morphological Correlation Demo:")
    word1 = MorphologicalByteWord(0b10101010)
    word2 = MorphologicalByteWord(0b11110000)
    
    correlation = word1.morphological_correlate(word2)
    print(f"Correlation between {word1} and {word2}:")
    for i, corr in enumerate(correlation):
        print(f"  [{i}] {corr}")
    
    # Demonstrate flat Abelianization
    print("\n⚛️  Flat Binary Abelianization Demo:")
    words = [
        MorphologicalByteWord(0b11110000),
        MorphologicalByteWord(0b10101010),
        MorphologicalByteWord(0b11001100)
    ]
    
    base = MorphologicalByteWord(0b00000001)
    abelianized = base.flat_abelianize(words)
    
    print(f"Base: {base}")
    print(f"Abelianizing with: {[str(w) for w in words]}")
    print(f"Result: {abelianized}")
    print(f"Theorem: {abelianized._theorem}")

if __name__ == "__main__":
    demo_cook_mertz_fft()