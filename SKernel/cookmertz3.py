import math
import cmath
from typing import List, Tuple, Optional, Union

class ComplexNumber:
    """Hand-rolled complex numbers because we don't sin with numpy"""
    def __init__(self, real: float, imag: float = 0.0):
        self.real = real
        self.imag = imag
    
    def __add__(self, other: 'ComplexNumber') -> 'ComplexNumber':
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other: 'ComplexNumber') -> 'ComplexNumber':
        # (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        real = self.real * other.real - self.imag * other.imag
        imag = self.real * other.imag + self.imag * other.real
        return ComplexNumber(real, imag)
    
    def __pow__(self, n: int) -> 'ComplexNumber':
        """Hand-rolled complex exponentiation"""
        if n == 0:
            return ComplexNumber(1.0, 0.0)
        if n == 1:
            return ComplexNumber(self.real, self.imag)
        
        # Use repeated squaring for efficiency
        result = ComplexNumber(1.0, 0.0)
        base = ComplexNumber(self.real, self.imag)
        
        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2
        
        return result
    
    def __eq__(self, other: 'ComplexNumber') -> bool:
        return abs(self.real - other.real) < 1e-10 and abs(self.imag - other.imag) < 1e-10
    
    def __repr__(self) -> str:
        if abs(self.imag) < 1e-10:
            return f"{self.real:.6f}"
        elif abs(self.real) < 1e-10:
            return f"{self.imag:.6f}i"
        else:
            sign = "+" if self.imag >= 0 else "-"
            return f"{self.real:.6f}{sign}{abs(self.imag):.6f}i"
    
    def magnitude(self) -> float:
        return math.sqrt(self.real * self.real + self.imag * self.imag)
    
    def phase(self) -> float:
        return math.atan2(self.imag, self.real)

class CookMerzRoots:
    """Hand-rolled Cook & Merz roots of unity for flat binary Abelization"""
    
    def __init__(self, n: int = 8):
        """Initialize nth roots of unity for n-bit operations"""
        self.n = n
        self.roots = self._generate_roots()
        
    def _generate_roots(self) -> List[ComplexNumber]:
        """Generate all nth roots of unity: e^(2πik/n) for k=0,1,...,n-1"""
        roots = []
        for k in range(self.n):
            angle = 2.0 * math.pi * k / self.n
            real = math.cos(angle)
            imag = math.sin(angle)
            roots.append(ComplexNumber(real, imag))
        return roots
    
    def get_primitive_root(self) -> ComplexNumber:
        """Get the primitive nth root of unity: e^(2πi/n)"""
        return self.roots[1]
    
    def get_root(self, k: int) -> ComplexNumber:
        """Get the kth root of unity"""
        return self.roots[k % self.n]
    
    def __getitem__(self, k: int) -> ComplexNumber:
        return self.get_root(k)

class BinaryAbelianByteWord:
    """ByteWord with Cook & Merz roots of unity and flat binary Abelization"""
    
    def __init__(self, value: int = 0):
        self.value = value & 0xFF  # Keep it 8-bit
        self.roots = CookMerzRoots(8)  # 8th roots of unity for 8-bit
        self._theorem = None
        self._abelian_encoding = self._compute_abelian_encoding()
    
    def _compute_abelian_encoding(self) -> List[ComplexNumber]:
        """Encode the byte value as a sum of roots of unity"""
        encoding = []
        for bit_pos in range(8):
            bit = (self.value >> bit_pos) & 1
            if bit:
                # Use the bit position to select which root of unity
                root = self.roots[bit_pos]
                encoding.append(root)
            else:
                # Zero contribution
                encoding.append(ComplexNumber(0.0, 0.0))
        return encoding
    
    def get_abelian_sum(self) -> ComplexNumber:
        """Get the sum of all roots of unity for this ByteWord"""
        total = ComplexNumber(0.0, 0.0)
        for root in self._abelian_encoding:
            total = total + root
        return total
    
    def xor_abelian_compose(self, other: 'BinaryAbelianByteWord') -> 'BinaryAbelianByteWord':
        """XOR composition in flat binary Abelian group"""
        # XOR the values
        new_value = self.value ^ other.value
        result = BinaryAbelianByteWord(new_value)
        
        # The theorem of XOR composition
        result._theorem = f"({self.value:08b} ⊕ {other.value:08b}) = {new_value:08b} in ℤ₂⁸"
        
        return result
    
    def abelian_multiply(self, other: 'BinaryAbelianByteWord') -> ComplexNumber:
        """Multiply in the abelian group using roots of unity"""
        self_sum = self.get_abelian_sum()
        other_sum = other.get_abelian_sum()
        return self_sum * other_sum
    
    def cook_merz_transform(self) -> List[ComplexNumber]:
        """Apply Cook & Merz discrete transform using roots of unity"""
        transform = []
        
        for k in range(8):
            # Compute X[k] = sum(x[n] * ω^(kn)) for n=0 to 7
            X_k = ComplexNumber(0.0, 0.0)
            
            for n in range(8):
                bit_n = (self.value >> n) & 1
                if bit_n:
                    # ω^(kn) where ω is primitive 8th root of unity
                    omega_kn = self.roots[k] ** n
                    X_k = X_k + omega_kn
            
            transform.append(X_k)
        
        return transform
    
    def inverse_cook_merz_transform(self, transform: List[ComplexNumber]) -> 'BinaryAbelianByteWord':
        """Inverse Cook & Merz transform"""
        # Hand-rolled inverse DFT
        reconstructed_bits = []
        
        for n in range(8):
            x_n = ComplexNumber(0.0, 0.0)
            
            for k in range(8):
                # Use conjugate of root for inverse
                omega_minus_kn = self.roots[(-k * n) % 8]
                x_n = x_n + (transform[k] * omega_minus_kn)
            
            # Normalize by 1/N
            x_n = ComplexNumber(x_n.real / 8.0, x_n.imag / 8.0)
            
            # Threshold to binary
            bit = 1 if x_n.real > 0.5 else 0
            reconstructed_bits.append(bit)
        
        # Reconstruct byte value
        reconstructed_value = 0
        for i, bit in enumerate(reconstructed_bits):
            reconstructed_value |= (bit << i)
        
        return BinaryAbelianByteWord(reconstructed_value)
    
    def abelian_group_order(self) -> int:
        """Find the order of this element in the abelian group"""
        current = BinaryAbelianByteWord(self.value)
        identity = BinaryAbelianByteWord(0)  # Identity element
        order = 1
        
        while not (current.value == identity.value and order > 1):
            current = current.xor_abelian_compose(self)
            order += 1
            if order > 256:  # Safety check
                break
                
        return order
    
    def generate_abelian_subgroup(self) -> List['BinaryAbelianByteWord']:
        """Generate the cyclic subgroup generated by this element"""
        subgroup = []
        current = BinaryAbelianByteWord(0)  # Start with identity
        
        for _ in range(self.abelian_group_order()):
            subgroup.append(BinaryAbelianByteWord(current.value))
            current = current.xor_abelian_compose(self)
        
        return subgroup
    
    def flat_binary_abelization(self) -> dict:
        """Complete flat binary Abelization analysis"""
        return {
            'value': self.value,
            'binary': f"{self.value:08b}",
            'abelian_sum': self.get_abelian_sum(),
            'group_order': self.abelian_group_order(),
            'cook_merz_transform': self.cook_merz_transform(),
            'subgroup_size': len(self.generate_abelian_subgroup()),
            'roots_of_unity': [self.roots[i] for i in range(8)]
        }
    
    def speaks_through(self) -> str:
        """The theorem this ByteWord embodies"""
        if self._theorem:
            return self._theorem
        
        abelian_sum = self.get_abelian_sum()
        return f"ByteWord({self.value:08b}) = Σ(ω^k) = {abelian_sum} in flat binary ℤ₂⁸"
    
    def compose(self, other: 'BinaryAbelianByteWord') -> 'BinaryAbelianByteWord':
        """Primary composition using XOR in flat binary abelian group"""
        return self.xor_abelian_compose(other)
    
    def to_float(self) -> float:
        """Collapse to float via abelian sum magnitude"""
        abelian_sum = self.get_abelian_sum()
        return abelian_sum.magnitude()
    
    def propagate(self, steps: int = 1) -> List['BinaryAbelianByteWord']:
        """Propagate through Cook & Merz transform space"""
        states = [BinaryAbelianByteWord(self.value)]
        current = self
        
        for step in range(steps):
            # Transform, rotate in complex plane, and inverse transform
            transform = current.cook_merz_transform()
            
            # Rotate each component by a small angle
            rotation_angle = 2 * math.pi * step / (8 * steps)
            rotated_transform = []
            
            for component in transform:
                # Rotate by multiplying with e^(iθ)
                rotation = ComplexNumber(math.cos(rotation_angle), math.sin(rotation_angle))
                rotated = component * rotation
                rotated_transform.append(rotated)
            
            # Inverse transform back to ByteWord
            next_state = current.inverse_cook_merz_transform(rotated_transform)
            states.append(next_state)
            current = next_state
        
        return states
    
    def __eq__(self, other: 'BinaryAbelianByteWord') -> bool:
        return self.value == other.value
    
    def __repr__(self) -> str:
        return f"BinaryAbelianByteWord({self.value:08b} = {self.value})"
    
    def __str__(self) -> str:
        abelian_sum = self.get_abelian_sum()
        return f"B({self.value:08b}) → {abelian_sum}"

# Demonstration of the flat binary Abelization
def demonstrate_binary_abelian_bytewords():
    """Show off the hand-rolled Cook & Merz roots of unity magic"""
    
    print("🔥 FLAT BINARY ABELIZATION WITH COOK & MERZ ROOTS OF UNITY 🔥")
    print("=" * 60)
    
    # Create some ByteWords
    word1 = BinaryAbelianByteWord(0b10101010)  # 170
    word2 = BinaryAbelianByteWord(0b01010101)  # 85
    word3 = BinaryAbelianByteWord(0b11110000)  # 240
    
    print(f"Word1: {word1}")
    print(f"Word2: {word2}")  
    print(f"Word3: {word3}")
    print()
    
    # Show abelian composition
    composed = word1.compose(word2)
    print(f"Composition (XOR): {word1.value:08b} ⊕ {word2.value:08b} = {composed}")
    print(f"Theorem: {composed.speaks_through()}")
    print()
    
    # Show Cook & Merz transforms
    print("Cook & Merz Transforms:")
    transform1 = word1.cook_merz_transform()
    for i, coeff in enumerate(transform1):
        print(f"  X[{i}] = {coeff}")
    print()
    
    # Show abelian group properties
    print("Abelian Group Analysis:")
    analysis = word1.flat_binary_abelization()
    print(f"  Group order: {analysis['group_order']}")
    print(f"  Subgroup size: {analysis['subgroup_size']}")
    print(f"  Abelian sum: {analysis['abelian_sum']}")
    print()
    
    # Show propagation through transform space
    print("Propagation through Cook & Merz space:")
    propagated = word1.propagate(steps=3)
    for i, state in enumerate(propagated):
        print(f"  Step {i}: {state}")
    print()
    
    # Show roots of unity
    print("8th Roots of Unity (hand-rolled, no numpy!):")
    roots = CookMerzRoots(8)
    for i in range(8):
        root = roots[i]
        print(f"  ω^{i} = {root}")
    print()
    
    # Verify group properties
    print("Verifying Abelian Group Properties:")
    
    # Identity element
    identity = BinaryAbelianByteWord(0)
    print(f"Identity: {identity}")
    
    # Closure test
    test_composed = word1.compose(word2).compose(word3)
    print(f"Closure test: ((word1 ⊕ word2) ⊕ word3) = {test_composed}")
    
    # Associativity test
    left_assoc = word1.compose(word2.compose(word3))
    right_assoc = word1.compose(word2).compose(word3)
    print(f"Associativity: {left_assoc.value == right_assoc.value}")
    
    # Commutativity test
    forward = word1.compose(word2)
    backward = word2.compose(word1)
    print(f"Commutativity: {forward.value == backward.value}")
    
    print("\n🎯 PURE PYTHON STDLIB ROOTS OF UNITY ACHIEVED! 🎯")

if __name__ == "__main__":
    demonstrate_binary_abelian_bytewords()