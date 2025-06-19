from typing import ClassVar

class QuantumXnorMorphogen:
    """
    A self-aware “quantum XNOR” morphogen for 8-bit ByteWords, with built-in
    intuition about Python’s bitwise NOT (~) operator and masking requirements.
    
    This class:
      - Takes a 4-bit type (T), 3-bit value (V), and 1-bit control (C).
      - Applies layered XNOR gates, carefully masking after each NOT (~) to
        avoid Python’s infinite-width signed integer behavior.
      - Emits a single 8-bit “quantum state” byte.
    """
    # Masks for each stage
    T_MASK: ClassVar[int]    = 0b1111     # 4 bits
    V_MASK: ClassVar[int]    = 0b111      # 3 bits
    M1_MASK: ClassVar[int]   = 0b1111     # Stage‐1 output (4 bits)
    M2_MASK: ClassVar[int]   = 0b11       # Stage‐2 output (2 bits)
    M3_MASK: ClassVar[int]   = 0b1        # Final bit
    
    @staticmethod
    def _masked_not(x: int, mask: int) -> int:
        """
        Bitwise NOT with masking:
        
          Python’s ~x produces infinite-width two’s-complement. To restrict
          the result to N bits, mask with (2**N-1).
        """
        return (~x) & mask
    
    @classmethod
    def compute(cls, t: int, v: int, c: int) -> int:
        """
        Compute the 8-bit quantum XNOR state:
        
        Args:
          t (int): 4-bit Topology code (0–15)
          v (int): 3-bit Value winding code (0–7)
          c (int): 1-bit Control flag (0 or 1)
        
        Returns:
          int: 8-bit byte representing the “aligned” morphic state.
        """
        # Validate inputs
        if not (0 <= t < 16):
            raise ValueError(f"T out of range: {t}")
        if not (0 <= v < 8):
            raise ValueError(f"V out of range: {v}")
        if not (0 <= c < 2):
            raise ValueError(f"C out of range: {c}")

        # Stage-1: XNOR on full nibbles (4 bits)
        m1 = cls._masked_not(t & cls.T_MASK, cls.M1_MASK) ^ (v & cls.V_MASK)

        # Stage-2: XNOR on high-order bits of T and V (2 bits)
        t_high = (t >> 2) & cls.M2_MASK
        v_high = (v >> 1) & cls.M2_MASK
        m2 = cls._masked_not(t_high, cls.M2_MASK) ^ v_high

        # Stage-3: XNOR combining stage-1 & stage-2 with control bit (1 bit)
        m1_low2 = m1 & cls.M2_MASK
        m12_and = m1_low2 & m2
        m3 = cls._masked_not(m12_and, cls.M3_MASK) ^ (c & cls.M3_MASK)

        # Assemble into 8 bits: [ m1(4) | m2(2) | m3(1) ]
        quantum_state = ( (m1 & cls.M1_MASK) << 4 ) | ( (m2 & cls.M2_MASK) << 1 ) | (m3 & cls.M3_MASK)
        return quantum_state & 0xFF

    def __repr__(self) -> str:
        return f"<QuantumXnorMorphogen>"

# -------------------------
# Example usage / self-test
# -------------------------
if __name__ == "__main__":
    # Example inputs
    T = 0b1010  # 4 bits
    V = 0b101   # 3 bits
    C = 1       # 1 bit

    state = QuantumXnorMorphogen.compute(T, V, C)
    print(f"T={T:04b}, V={V:03b}, C={C}: quantum_state={state:08b}")
    # Expected: an 8-bit pattern representing layered XNOR gates,
    # correctly masked at each stage to avoid Python’s ~ pitfalls.
