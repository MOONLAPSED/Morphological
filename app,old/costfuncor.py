#!/usr/bin/env python3
from __future__ import annotations
import math, functools
from dataclasses import dataclass
from typing import Tuple
import random

# ---- Landauer cost ----
k_B   = 1.380649e-23        # J/K
T_ENV = 300.0               # K
LN2   = math.log(2.0)

def landauer(bits: int) -> float:
    return bits * k_B * T_ENV * LN2

# ---- ToroidalByteWord ----
@dataclass(frozen=True)
class ToroidalByteWord:
    winding: Tuple[int, int]    # (w1, w2) mod N
    orientation: int            # 0..3 bits
    
    @classmethod
    def random(cls, N: int = 256) -> ToroidalByteWord:
        return cls((random.randrange(N), random.randrange(N)),
                   random.randrange(4))
    
    def collapse(self, choose: Tuple[int,int] = None) -> Tuple[ToroidalByteWord, float]:
        """
        Collapse latent winding to a definite one.
        Pays Landauer cost = log2(#states) bits.
        """
        # Number of possible states = N^2 * 4
        N = 256
        bits = math.log2(N*N*4)
        cost = landauer(int(bits))
        if choose is None:
            # pick uniformly
            w1 = random.randrange(N)
            w2 = random.randrange(N)
        else:
            w1, w2 = choose
        # orientation can also collapse
        ori = random.randrange(4)
        return ToroidalByteWord((w1, w2), ori), cost
    
    def compose(self, other: ToroidalByteWord) -> ToroidalByteWord:
        """
        Non-associative composition: only if windings match exactly.
        Orientation must also resonate (xor==0).
        """
        if self.winding != other.winding:
            raise ValueError("Winding mismatch: resonance failed")
        new_ori = self.orientation ^ other.orientation
        return ToroidalByteWord(self.winding, new_ori)

# ---- Example usage ----
if __name__ == "__main__":
    latent = ToroidalByteWord.random()
    active, heat = latent.collapse()
    print("Latent:", latent)
    print("Active:", active, f"(heat paid: {heat:.3e} J)")
    # Compose two copies of the same attractor
    child = active.compose(active)
    print("Child:", child)
