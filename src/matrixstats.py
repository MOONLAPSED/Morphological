from decimal import Decimal, getcontext
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar, List, Tuple, Union, Optional, Dict, Any
from functools import reduce
from itertools import product
import math
import cmath

# Set high precision for quantum calculations
getcontext().prec = 50

T = TypeVar('T')
U = TypeVar('U')

#############################################
# Enhanced ComplexDecimal from your framework
#############################################

class ComplexDecimal:
    """High-precision complex number using Decimal arithmetic."""
    
    def __init__(self, real, imag=Decimal('0')):
        self.real = Decimal(str(real))
        self.imag = Decimal(str(imag))
    
    def __add__(self, other):
        if isinstance(other, (int, float, Decimal)):
            return ComplexDecimal(self.real + Decimal(str(other)), self.imag)
        return ComplexDecimal(self.real + other.real, self.imag + other.imag)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float, Decimal)):
            return ComplexDecimal(self.real - Decimal(str(other)), self.imag)
        return ComplexDecimal(self.real - other.real, self.imag - other.imag)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float, Decimal)):
            return ComplexDecimal(Decimal(str(other)) - self.real, -self.imag)
        return ComplexDecimal(other.real - self.real, other.imag - self.imag)
    
    def __mul__(self, other):
        if isinstance(other, (int, float, Decimal)):
            scalar = Decimal(str(other))
            return ComplexDecimal(self.real * scalar, self.imag * scalar)
        return ComplexDecimal(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float, Decimal)):
            scalar = Decimal(str(other))
            return ComplexDecimal(self.real / scalar, self.imag / scalar)
        denom = other.real * other.real + other.imag * other.imag
        num = self * other.conjugate()
        return ComplexDecimal(num.real / denom, num.imag / denom)
    
    def conjugate(self):
        return ComplexDecimal(self.real, -self.imag)
    
    def abs(self):
        return getcontext().sqrt(self.real * self.real + self.imag * self.imag)
    
    def phase(self):
        """Return the phase angle of the complex number."""
        if self.real == 0 and self.imag == 0:
            return Decimal('0')
        # Use atan2 approximation for high precision
        return self._atan2(self.imag, self.real)
    
    def _atan2(self, y, x):
        """High precision atan2 using series expansion."""
        if x > 0:
            return self._atan(y / x)
        elif x < 0 and y >= 0:
            return self._atan(y / x) + Decimal('3.14159265358979323846264338327950288')
        elif x < 0 and y < 0:
            return self._atan(y / x) - Decimal('3.14159265358979323846264338327950288')
        elif x == 0 and y > 0:
            return Decimal('1.57079632679489661923132169163975144')
        elif x == 0 and y < 0:
            return -Decimal('1.57079632679489661923132169163975144')
        else:
            return Decimal('0')  # x == 0 and y == 0
    
    def _atan(self, x):
        """Arctangent using Taylor series."""
        if abs(x) > 1:
            return Decimal('1.57079632679489661923132169163975144') - self._atan(1/x)
        
        result = Decimal('0')
        power = x
        for n in range(1, 100, 2):
            term = power / Decimal(n)
            if n % 4 == 3:
                term = -term
            result += term
            power *= x * x
            if abs(term) < Decimal('1e-40'):
                break
        return result
    
    def __pow__(self, exponent):
        """Complex exponentiation using De Moivre's formula."""
        if isinstance(exponent, (int, float, Decimal)):
            exp = Decimal(str(exponent))
            r = self.abs()
            theta = self.phase()
            new_r = r ** exp
            new_theta = theta * exp
            return ComplexDecimal(
                new_r * self._cos(new_theta),
                new_r * self._sin(new_theta)
            )
        raise NotImplementedError("Complex exponent not implemented")
    
    def _cos(self, x):
        """Cosine using Taylor series."""
        result = Decimal('0')
        sign = Decimal('1')
        x_power = Decimal('1')
        factorial = Decimal('1')
        for n in range(0, 100, 2):
            if n > 0:
                factorial *= Decimal(n) * Decimal(n-1)
            result += sign * x_power / factorial
            sign *= -1
            x_power *= x * x
            if abs(x_power / factorial) < Decimal('1e-40'):
                break
        return result
    
    def _sin(self, x):
        """Sine using Taylor series."""
        result = Decimal('0')
        sign = Decimal('1')
        x_power = x
        factorial = Decimal('1')
        for n in range(1, 100, 2):
            if n > 1:
                factorial *= Decimal(n) * Decimal(n-1)
            result += sign * x_power / factorial
            sign *= -1
            x_power *= x * x
            if abs(x_power / factorial) < Decimal('1e-40'):
                break
        return result
    
    def __abs__(self):
        return float(self.abs())
    
    def __eq__(self, other):
        if isinstance(other, ComplexDecimal):
            return self.real == other.real and self.imag == other.imag
        return False
    
    def __repr__(self):
        if self.imag >= 0:
            return f"({self.real}+{self.imag}j)"
        else:
            return f"({self.real}{self.imag}j)"

#############################################
# Quantum State Vectors and Operations
#############################################

@dataclass(frozen=True)
class Ket(Generic[T]):
    """
    |ψ⟩ - A quantum state vector (ket) in Dirac notation.
    Represents a column vector in the quantum state space.
    """
    amplitudes: Tuple[ComplexDecimal, ...]
    label: Optional[str] = None
    
    def __post_init__(self):
        if not self.amplitudes:
            raise ValueError("Ket must have at least one amplitude")
    
    @property
    def dimension(self) -> int:
        return len(self.amplitudes)
    
    def __add__(self, other: 'Ket') -> 'Ket':
        """Linear superposition of quantum states."""
        if self.dimension != other.dimension:
            raise ValueError("Kets must have same dimension for addition")
        new_amps = tuple(a + b for a, b in zip(self.amplitudes, other.amplitudes))
        return Ket(new_amps, f"({self.label or '?'} + {other.label or '?'})")
    
    def __mul__(self, scalar: Union[ComplexDecimal, int, float, Decimal]) -> 'Ket':
        """Scalar multiplication of quantum state."""
        if isinstance(scalar, (int, float, Decimal)):
            scalar = ComplexDecimal(scalar)
        new_amps = tuple(scalar * amp for amp in self.amplitudes)
        return Ket(new_amps, f"{scalar}*{self.label or '?'}")
    
    def __rmul__(self, scalar: Union[ComplexDecimal, int, float, Decimal]) -> 'Ket':
        return self.__mul__(scalar)
    
    def normalize(self) -> 'Ket':
        """Normalize the quantum state to unit length."""  
        norm_squared = sum(amp.abs() ** 2 for amp in self.amplitudes)
        if norm_squared == 0:
            raise ValueError("Cannot normalize zero state")
        norm = getcontext().sqrt(norm_squared)
        normalized_amps = tuple(amp / norm for amp in self.amplitudes)
        return Ket(normalized_amps, f"norm({self.label or '?'})")
    
    def norm(self) -> Decimal:
        """Calculate the norm (length) of the state vector."""  
        return getcontext().sqrt(sum(amp.abs() ** 2 for amp in self.amplitudes))
    
    def probability(self, index: int) -> Decimal:
        """Probability of measuring the state in basis state |index⟩."""
        if index >= self.dimension:
            raise IndexError("Index out of range")
        return self.amplitudes[index].abs() ** 2
    
    def probabilities(self) -> List[Decimal]:
        """All measurement probabilities."""
        return [amp.abs() ** 2 for amp in self.amplitudes]
    
    def dagger(self) -> 'Bra':
        """Hermitian conjugate: |ψ⟩† = ⟨ψ|"""
        return Bra(tuple(amp.conjugate() for amp in self.amplitudes), self.label)
    
    def __repr__(self):
        label = self.label or "ψ"
        return f"|{label}⟩"

@dataclass(frozen=True)  
class Bra(Generic[T]):
    """
    ⟨φ| - A quantum state covector (bra) in Dirac notation.
    Represents a row vector in the dual space.
    """
    amplitudes: Tuple[ComplexDecimal, ...]
    label: Optional[str] = None
    
    @property
    def dimension(self) -> int:
        return len(self.amplitudes)
    
    def __mul__(self, ket: Ket) -> ComplexDecimal:
        """Inner product ⟨φ|ψ⟩ - probability amplitude."""
        if self.dimension != ket.dimension:
            raise ValueError("Bra and Ket must have same dimension")
        return sum(b_amp * k_amp for b_amp, k_amp in zip(self.amplitudes, ket.amplitudes))
    
    def __matmul__(self, ket: Ket) -> ComplexDecimal:
        """Alternative inner product notation using @."""
        return self.__mul__(ket)
    
    def outer(self, ket: Ket) -> 'Operator':
        """Outer product ⟨φ|ψ⟩ creating an operator."""
        matrix = []
        for bra_amp in self.amplitudes:
            row = [bra_amp * ket_amp for ket_amp in ket.amplitudes]
            matrix.append(tuple(row))
        return Operator(tuple(matrix), f"|{ket.label or '?'}⟩⟨{self.label or '?'}|")
    
    def dagger(self) -> Ket:
        """Hermitian conjugate: ⟨φ|† = |φ⟩"""
        return Ket(tuple(amp.conjugate() for amp in self.amplitudes), self.label)
    
    def __repr__(self):
        label = self.label or "φ"
        return f"⟨{label}|"

#############################################
# Quantum Operators and Matrices
#############################################

@dataclass(frozen=True)
class Operator:
    """
    Quantum operator represented as a matrix acting on quantum states.
    """
    matrix: Tuple[Tuple[ComplexDecimal, ...], ...]
    label: Optional[str] = None
    
    def __post_init__(self):
        if not self.matrix or not self.matrix[0]:
            raise ValueError("Operator matrix cannot be empty")
        # Check if matrix is square
        rows = len(self.matrix)
        for row in self.matrix:
            if len(row) != rows:
                raise ValueError("Operator matrix must be square")
    
    @property
    def dimension(self) -> int:
        return len(self.matrix)
    
    def __mul__(self, other: Union[Ket, 'Operator']) -> Union[Ket, 'Operator']:
        """Matrix multiplication with kets or other operators."""
        if isinstance(other, Ket):
            if self.dimension != other.dimension:
                raise ValueError("Operator and Ket dimensions must match")
            new_amps = []
            for row in self.matrix:
                amp = sum(matrix_elem * ket_amp 
                         for matrix_elem, ket_amp in zip(row, other.amplitudes))
                new_amps.append(amp)
            return Ket(tuple(new_amps), f"{self.label or 'Op'}|{other.label or '?'}⟩")
        
        elif isinstance(other, Operator):
            if self.dimension != other.dimension:
                raise ValueError("Operators must have same dimension")
            new_matrix = []
            for i in range(self.dimension):
                new_row = []
                for j in range(self.dimension):
                    elem = sum(self.matrix[i][k] * other.matrix[k][j] 
                             for k in range(self.dimension))
                    new_row.append(elem)
                new_matrix.append(tuple(new_row))
            return Operator(tuple(new_matrix), f"({self.label or 'Op1'})({other.label or 'Op2'})")
        
        else:
            raise TypeError("Can only multiply Operator with Ket or Operator")
    
    def __add__(self, other: 'Operator') -> 'Operator':
        """Add two operators."""
        if self.dimension != other.dimension:
            raise ValueError("Operators must have same dimension for addition")
        new_matrix = []
        for i in range(self.dimension):
            new_row = []
            for j in range(self.dimension):
                new_row.append(self.matrix[i][j] + other.matrix[i][j])
            new_matrix.append(tuple(new_row))
        return Operator(tuple(new_matrix), f"({self.label or 'Op1'} + {other.label or 'Op2'})")
    
    def dagger(self) -> 'Operator':
        """Hermitian conjugate (adjoint) of the operator."""
        new_matrix = []
        for j in range(self.dimension):
            new_row = []
            for i in range(self.dimension):
                new_row.append(self.matrix[i][j].conjugate())
            new_matrix.append(tuple(new_row))
        return Operator(tuple(new_matrix), f"({self.label or 'Op'})†")
    
    def trace(self) -> ComplexDecimal:
        """Trace of the operator matrix."""
        return sum(self.matrix[i][i] for i in range(self.dimension))
    
    def is_hermitian(self) -> bool:
        """Check if operator is Hermitian (self-adjoint)."""
        dagger = self.dagger()
        for i in range(self.dimension):
            for j in range(self.dimension):
                if self.matrix[i][j] != dagger.matrix[i][j]:
                    return False
        return True
    
    def is_unitary(self) -> bool:
        """Check if operator is unitary (U†U = I)."""
        product = self.dagger() * self
        # Check if it's the identity matrix
        for i in range(self.dimension):
            for j in range(self.dimension):
                expected = ComplexDecimal(1) if i == j else ComplexDecimal(0)
                if abs(float(product.matrix[i][j].real - expected.real)) > 1e-10:
                    return False
                if abs(float(product.matrix[i][j].imag - expected.imag)) > 1e-10:
                    return False
        return True
    
    def expectation_value(self, state: Ket) -> ComplexDecimal:
        """Expected value ⟨ψ|A|ψ⟩ of operator A in state |ψ⟩."""
        return state.dagger() * (self * state)
    
    def __repr__(self):
        return f"Operator({self.label or 'Op'}, {self.dimension}x{self.dimension})"

#############################################
# Pauli Matrices and Common Quantum Gates
#############################################

class PauliMatrices:
    """Standard Pauli matrices for spin-1/2 systems."""
    
    @staticmethod
    def I() -> Operator:
        """Identity matrix."""
        return Operator((
            (ComplexDecimal(1), ComplexDecimal(0)),
            (ComplexDecimal(0), ComplexDecimal(1))
        ), "I")
    
    @staticmethod
    def X() -> Operator:
        """Pauli-X (bit flip) matrix."""
        return Operator((
            (ComplexDecimal(0), ComplexDecimal(1)),
            (ComplexDecimal(1), ComplexDecimal(0))
        ), "σₓ")
    
    @staticmethod
    def Y() -> Operator:
        """Pauli-Y matrix."""
        return Operator((
            (ComplexDecimal(0), ComplexDecimal(0, -1)),
            (ComplexDecimal(0, 1), ComplexDecimal(0))
        ), "σᵧ")
    
    @staticmethod
    def Z() -> Operator:
        """Pauli-Z (phase flip) matrix."""
        return Operator((
            (ComplexDecimal(1), ComplexDecimal(0)),
            (ComplexDecimal(0), ComplexDecimal(-1))
        ), "σᵤ")

class QuantumGates:
    """Common quantum gates."""
    
    @staticmethod
    def Hadamard() -> Operator:
        """Hadamard gate - creates superposition."""
        sqrt2_inv = ComplexDecimal(1) / ComplexDecimal(getcontext().sqrt(Decimal('2')))
        return Operator((
            (sqrt2_inv, sqrt2_inv),
            (sqrt2_inv, sqrt2_inv * ComplexDecimal(-1))
        ), "H")
    
    @staticmethod
    def CNOT() -> Operator:
        """Controlled-NOT gate for 2-qubit systems."""
        return Operator((
            (ComplexDecimal(1), ComplexDecimal(0), ComplexDecimal(0), ComplexDecimal(0)),
            (ComplexDecimal(0), ComplexDecimal(1), ComplexDecimal(0), ComplexDecimal(0)),
            (ComplexDecimal(0), ComplexDecimal(0), ComplexDecimal(0), ComplexDecimal(1)),
            (ComplexDecimal(0), ComplexDecimal(0), ComplexDecimal(1), ComplexDecimal(0))
        ), "CNOT")
    
    @staticmethod
    def phase_gate(phi: Union[float, Decimal]) -> Operator:
        """Phase gate with arbitrary phase."""
        phase = ComplexDecimal(0, phi)
        exp_phase = ComplexDecimal(1).__pow__(phase)  # e^(iφ)
        return Operator((
            (ComplexDecimal(1), ComplexDecimal(0)),
            (ComplexDecimal(0), exp_phase)
        ), f"P({phi})")

#############################################
# Standard Quantum States
#############################################

def display_ket(ket: Ket) -> str:
    return " + ".join(
        f"({amp})|{i}⟩" for i, amp in enumerate(ket.amplitudes) if amp.real != 0 or amp.imag != 0
    )

def tensor_product(ket1: Ket, ket2: Ket) -> Ket:
    amps = tuple(a * b for a in ket1.amplitudes for b in ket2.amplitudes)
    return Ket(amps, f"{ket1.label or '?'}⊗{ket2.label or '?'}")

def tensor_product_operator(op1: Operator, op2: Operator) -> Operator:
    matrix = []
    for row1 in op1.matrix:
        for row2 in op2.matrix:
            row = []
            for elem1 in row1:
                row.extend([elem1 * elem2 for elem2 in row2])
            matrix.append(tuple(row))
    return Operator(tuple(matrix), f"{op1.label or '?'}⊗{op2.label or '?'}")

@dataclass(frozen=True)
class DensityMatrix:
    matrix: Tuple[Tuple[ComplexDecimal, ...], ...]
    
    def trace(self) -> ComplexDecimal:
        return sum(self.matrix[i][i] for i in range(len(self.matrix)))

    def purity(self) -> Decimal:
        """Tr(ρ²) indicates whether state is pure."""
        squared = Operator(self.matrix) * Operator(self.matrix)
        return squared.trace().abs()
    
    def is_pure(self) -> bool:
        return abs(self.purity() - Decimal(1)) < Decimal('1e-10')

def to_density_matrix(ket: Ket) -> DensityMatrix:
    outer = ket.dagger().outer(ket)
    return DensityMatrix(outer.matrix)

def compose(*ops: Operator) -> Operator:
    return reduce(lambda a, b: a * b, ops)

class StandardStates:
    """Standard quantum states."""
    
    @staticmethod
    def zero() -> Ket:
        """Ground state |0⟩."""
        return Ket((ComplexDecimal(1), ComplexDecimal(0)), "0")
    
    @staticmethod
    def one() -> Ket:
        """Excited state |1⟩."""
        return Ket((ComplexDecimal(0), ComplexDecimal(1)), "1")
    
    @staticmethod
    def plus() -> Ket:
        """Superposition state |+⟩ = (|0⟩ + |1⟩)/√2."""
        sqrt2_inv = ComplexDecimal(1) / ComplexDecimal(getcontext().sqrt(Decimal('2')))
        return Ket((sqrt2_inv, sqrt2_inv), "+")
    
    @staticmethod
    def minus() -> Ket:
        """Superposition state |-⟩ = (|0⟩ - |1⟩)/√2."""
        sqrt2_inv = ComplexDecimal(1) / ComplexDecimal(getcontext().sqrt(Decimal('2')))
        return Ket((sqrt2_inv, sqrt2_inv * ComplexDecimal(-1)), "-")
    
    @staticmethod
    def bell_state_00() -> Ket:
        """Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2."""
        sqrt2_inv = ComplexDecimal(1) / ComplexDecimal(getcontext().sqrt(Decimal('2')))
        return Ket((sqrt2_inv, ComplexDecimal(0), ComplexDecimal(0), sqrt2_inv), "Φ⁺")

@dataclass(frozen=True)
class WaveFunction(Generic[T]):
    """
    Enhanced wave function with full quantum mechanical operations.
    Acts as a functor and monad for quantum state transformations.
    """
    state: T
    
    def map(self, func: Callable[[T], U]) -> 'WaveFunction[U]':
        """Functor map: apply function to quantum state."""
        return WaveFunction(func(self.state))
    
    def flat_map(self, func: Callable[[T], 'WaveFunction[U]']) -> 'WaveFunction[U]':
        """Monadic bind: compose quantum operations."""
        return func(self.state)
    
    def apply_operator(self, operator: Operator) -> 'WaveFunction':
        """Apply quantum operator to the wave function."""
        if isinstance(self.state, Ket):
            return WaveFunction(operator * self.state)
        raise TypeError("Can only apply operators to Ket states")
    
    def measure(self, observable: Operator) -> Tuple[ComplexDecimal, 'WaveFunction']:
        """Measure observable and return expectation value and collapsed state."""
        if isinstance(self.state, Ket):
            expectation = observable.expectation_value(self.state)
            # For simplicity, return the same state (full collapse would require eigenvalue decomposition)
            return expectation, self
        raise TypeError("Can only measure Ket states")
    
    def evolve(self, hamiltonian: Operator, time: Union[float, Decimal]) -> 'WaveFunction':
        """Time evolution under Hamiltonian: |ψ(t)⟩ = e^(-iHt/ℏ)|ψ(0)⟩."""
        # Simplified evolution for demonstration (proper implementation would use matrix exponentiation)
        if isinstance(self.state, Ket):
            # For now, just apply the Hamiltonian (this is a simplified approximation)
            evolved_state = hamiltonian * self.state
            return WaveFunction(evolved_state)
        raise TypeError("Can only evolve Ket states")
    
    def entangle_with(self, other: 'WaveFunction') -> 'WaveFunction':
        """Create entangled state (tensor product)."""
        if isinstance(self.state, Ket) and isinstance(other.state, Ket):
            # Tensor product of two kets
            new_amps = []
            for amp1 in self.state.amplitudes:
                for amp2 in other.state.amplitudes:
                    new_amps.append(amp1 * amp2)
            entangled_state = Ket(tuple(new_amps), f"{self.state.label or '?'}⊗{other.state.label or '?'}")
            return WaveFunction(entangled_state)
        raise TypeError("Can only entangle Ket states")

#############################################
# Quantum Measurement and Probability
#############################################

class QuantumMeasurement:
    """Quantum measurement operations and probability calculations."""
    
    @staticmethod
    def born_rule(state: Ket, basis_state: Ket) -> Decimal:
        """Born rule: P(outcome) = |⟨basis|state⟩|²."""
        amplitude = basis_state.dagger() * state
        return amplitude.abs() ** 2
    
    @staticmethod
    def measurement_probabilities(state: Ket, basis_states: List[Ket]) -> List[Decimal]:
        """Calculate measurement probabilities for all basis states."""
        return [QuantumMeasurement.born_rule(state, basis) for basis in basis_states]
    
    @staticmethod
    def expectation_value(state: Ket, observable: Operator) -> ComplexDecimal:
        """Calculate expectation value ⟨ψ|A|ψ⟩."""
        return observable.expectation_value(state)
    
    @staticmethod
    def variance(state: Ket, observable: Operator) -> Decimal:
        """Calculate variance Var(A) = ⟨A²⟩ - ⟨A⟩²."""
        expectation_A = QuantumMeasurement.expectation_value(state, observable)
        expectation_A2 = QuantumMeasurement.expectation_value(state, observable * observable)
        return (expectation_A2 - expectation_A * expectation_A).real

#############################################
# Example Usage and Testing
#############################################

def demonstrate_quantum_mechanics():
    """Demonstrate the quantum mechanics library."""
    print("=== Quantum Mechanics Library Demonstration ===\n")
    
    # Create basic states
    zero = StandardStates.zero()
    one = StandardStates.one()
    plus = StandardStates.plus()
    
    print(f"Ground state: {zero}")
    print(f"Excited state: {one}")
    print(f"Superposition state: {plus}")
    print(f"Plus state amplitudes: {[str(amp) for amp in plus.amplitudes]}")
    print()
    
    # Test inner products
    print("=== Inner Products (Bra-Ket) ===")
    print(f"⟨0|0⟩ = {zero.dagger() * zero}")
    print(f"⟨1|1⟩ = {one.dagger() * one}")
    print(f"⟨0|1⟩ = {zero.dagger() * one}")
    print(f"⟨+|0⟩ = {plus.dagger() * zero}")
    print()
    
    # Test Pauli matrices
    print("=== Pauli Matrices ===")
    sigma_x = PauliMatrices.X()
    sigma_y = PauliMatrices.Y()
    sigma_z = PauliMatrices.Z()
    
    print(f"σₓ|0⟩ = {sigma_x * zero}")
    print(f"σᵧ|0⟩ = {sigma_y * zero}")
    print(f"σᵤ|0⟩ = {sigma_z * zero}")
    print(f"σₓ is Hermitian: {sigma_x.is_hermitian()}")
    print(f"σₓ is Unitary: {sigma_x.is_unitary()}")
    print()
    
    # Test Hadamard gate
    print("=== Quantum Gates ===")
    hadamard = QuantumGates.Hadamard()
    h_zero = hadamard * zero
    print(f"H|0⟩ = {h_zero}")
    print(f"H|0⟩ amplitudes: {[str(amp) for amp in h_zero.amplitudes]}")
    print(f"Measurement probabilities: {h_zero.probabilities()}")
    print()
    
    # Test expectation values
    print("=== Expectation Values ===")
    exp_z_zero = QuantumMeasurement.expectation_value(zero, sigma_z)
    exp_z_one = QuantumMeasurement.expectation_value(one, sigma_z)
    exp_z_plus = QuantumMeasurement.expectation_value(plus, sigma_z)
    
    print(f"⟨0|σᵤ|0⟩ = {exp_z_zero}")
    print(f"⟨1|σᵤ|1⟩ = {exp_z_one}")
    print(f"⟨+|σᵤ|+⟩ = {exp_z_plus}")
    print()
    
    # Test wave function evolution
    print("=== Wave Function Operations ===")
    wf = WaveFunction(zero)
    wf_evolved = wf.apply_operator(hadamard)
    print(f"Original wave function state: {wf.state}")
    print(f"After Hadamard: {wf_evolved.state}")
    
    # Test entanglement
    wf1 = WaveFunction(zero)
    wf2 = WaveFunction(one)
    entangled = wf1.entangle_with(wf2)
    print(f"Entangled state: {entangled.state}")
    print(f"Entangled amplitudes: {[str(amp) for amp in entangled.state.amplitudes]}")
    print()
    
    # Test Bell state
    print("=== Bell States ===")
    bell = StandardStates.bell_state_00()
    print(f"Bell state |Φ⁺⟩: {bell}")
    print(f"Bell state amplitudes: {[str(amp) for amp in bell.amplitudes]}")
    print(f"Bell state probabilities: {bell.probabilities()}")

if __name__ == "__main__":
    demonstrate_quantum_mechanics()