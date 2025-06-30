from typing import TypeVar, Generic, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import ctypes, weakref, math
from pathlib import Path

T = TypeVar('T')  # Type structure (static/potential)
V = TypeVar('V')  # Value space (measured/actual)
C = TypeVar('C')  # Computation space (transformative)

class QuantumState(Enum):
    SUPERPOSITION = auto()  # handle-only, uncollapsed
    ENTANGLED     = auto()  # referenced, not fully materialized
    COLLAPSED     = auto()  # fully materialized Python object
    DECOHERENT    = auto()  # garbage collected

class OperatorType(Enum):
    COMPOSITION = auto()  # function composition (>>)
    TENSOR      = auto()  # tensor product (⊗)
    DIRECT_SUM  = auto()  # direct sum (⊕)
    OUTER       = auto()  # outer product (|ψ⟩⟨φ|)
    ADJOINT     = auto()  # Hermitian adjoint (†)
    MEASUREMENT = auto()  # quantum measurement

@dataclass(frozen=True)
class ModuleMetadata:
    original_path: Path
    module_name: str
    is_python: bool
    file_size: int
    mtime: float
    content_hash: str

@dataclass
class Particle(Generic[T, V, C]):
    state_vector: complex
    phase: float
    type_structure: T
    value_space: V
    compute_space: C
    probability_amplitude: complex = field(default_factory=lambda: complex(1.0, 0.0))

    def __matmul__(self, other: "Particle[T,V,C]") -> "Particle[T,V,C]":
        # Tensor product ⊗
        return Particle(
            state_vector=self.state_vector * other.state_vector,
            phase=(self.phase + other.phase) % (2 * math.pi),
            type_structure=(self.type_structure, other.type_structure),
            value_space=(self.value_space, other.value_space),
            compute_space=lambda x: self.compute_space(other.compute_space(x))  # type: ignore
        )  # type: ignore

    def compose(self, other: "Particle[T,V,C]") -> "Particle[T,V,C]":
        # Function composition >>
        return Particle(
            state_vector=self.state_vector * other.state_vector,
            phase=self.phase,
            type_structure=other.type_structure,
            value_space=other.value_space,
            compute_space=lambda x: other.compute_space(self.compute_space(x))  # type: ignore
        )  # type: ignore

class PyObjectBridge:
    class CPyObject(ctypes.Structure):
        _fields_ = [
            ("ob_refcnt", ctypes.c_ssize_t),
            ("ob_type", ctypes.c_void_p)
        ]
    @staticmethod
    def get_refcount(obj: Any) -> int:
        return ctypes.cast(id(obj), ctypes.POINTER(PyObjectBridge.CPyObject)).contents.ob_refcnt

@dataclass
class StateVector:
    amplitude: complex
    phase: float
    data: Any
    timestamp: float
    def interfere(self, other: "StateVector") -> "StateVector":
        new_amp   = self.amplitude * other.amplitude
        new_phase = (self.phase + other.phase) % (2 * math.pi)
        return StateVector(new_amp, new_phase, self.data, self.timestamp)

# Frame / Field / Space for bridging CPython ↔ MSC holospaces

class Frame(Generic[T, V, C], metaclass=ABCMeta):  # type: ignore
    def __init__(self):
        self._handle = id(self)
        self._state = QuantumState.SUPERPOSITION
        self._type_space: Optional[T] = None  # type: ignore
        self._value_space: Optional[V] = None  # type: ignore
        self._compute_space: Optional[C] = None  # type: ignore
        self._observers: weakref.WeakSet = weakref.WeakSet()
    @property
    def state(self) -> QuantumState:
        return self._state
    def collapse(self) -> V:
        if self._state == QuantumState.SUPERPOSITION:
            self._materialize()  # type: ignore
        return self._value_space
    # … (abstract _materialize_type, _collapse_value, _create_compute_space) …

class Field(Frame[T, V, C], ABC):  # type: ignore
    def __init__(self):
        super().__init__()
        self._degrees: weakref.WeakSet = weakref.WeakSet()
    @abstractmethod  # type: ignore
    def transform(self, operator: Callable[[V], V]) -> None:
        pass

class Space(Field[T, V, C], ABC):  # type: ignore
    def __init__(self):
        super().__init__()
        self.fields: dict[str, Field] = {}
    def compose(self, other: "Space[T,V,C]") -> "Space[T,V,C]":
        new_space = Space()
        for h, f in self.fields.items():
            if h in other.fields:
                nf = new_space.fields.setdefault(h, Field())
                nf.entangle(f); nf.entangle(other.fields[h])
        return new_space
