from typing import TypeVar, Generic, Callable, Dict, Any, Optional, Set, Union, Awaitable, cast, List, Tuple
from enum import Enum, auto, StrEnum
from abc import ABC, abstractmethod
import asyncio
import weakref
import inspect
import time
import array
import hashlib
import uuid
import json
from dataclasses import dataclass, field
from types import SimpleNamespace, ModuleType
import ast
import sys
import random
import math
from collections import defaultdict, deque
import logging

# Covariant type variables for better type safety
T_co = TypeVar('T_co', covariant=True)
V_co = TypeVar('V_co', covariant=True)
C_co = TypeVar('C_co', bound=Callable, covariant=True)
Q_co = TypeVar('Q_co', bound='QuantumState', covariant=True)

logger = logging.getLogger(__name__)

class EntanglementType(StrEnum):
    """Types of quantum entanglement between runtime atoms."""
    TEMPORAL = "temporal"      # Entangled across time evolution
    SPATIAL = "spatial"        # Entangled across distributed runtimes
    CAUSAL = "causal"         # Entangled through cause-effect chains
    QUINIC = "quinic"         # Self-referential entanglement (quining)

class CoherenceState(Enum):
    """Quantum coherence states for runtime atoms."""
    SUPERPOSITION = auto()     # Multiple states simultaneously
    ENTANGLED = auto()        # Coupled with other atoms
    COLLAPSED = auto()        # Single definite state
    DECOHERENT = auto()       # Lost quantum properties

@dataclass(frozen=True)
class QuantumState:
    """Immutable quantum state representation for runtime atoms."""
    amplitude: complex
    phase: float
    entanglement_id: Optional[str] = None
    coherence: CoherenceState = CoherenceState.SUPERPOSITION
    timestamp: float = field(default_factory=time.time)
    
    @property
    def probability(self) -> float:
        """Born rule: |ψ|² gives measurement probability."""
        return abs(self.amplitude) ** 2
    
    def evolve(self, operator: complex, dt: float = 1.0) -> 'QuantumState':
        """Apply unitary evolution: ψ(t+dt) = U(dt)ψ(t)"""
        # U(t) = exp(-iHt) for Hamiltonian H
        new_amplitude = self.amplitude * cmath.exp(-1j * operator * dt)
        new_phase = (self.phase + operator.imag * dt) % (2 * math.pi)
        
        return QuantumState(
            amplitude=new_amplitude,
            phase=new_phase,
            entanglement_id=self.entanglement_id,
            coherence=self.coherence,
            timestamp=time.time()
        )
    
    def entangle_with(self, other: 'QuantumState') -> Tuple['QuantumState', 'QuantumState']:
        """Create entangled pair with shared entanglement_id."""
        entanglement_id = str(uuid.uuid4())
        
        # Create Bell state: |00⟩ + |11⟩ (normalized)
        norm_factor = 1 / math.sqrt(2)
        
        state1 = QuantumState(
            amplitude=self.amplitude * norm_factor,
            phase=self.phase,
            entanglement_id=entanglement_id,
            coherence=CoherenceState.ENTANGLED
        )
        
        state2 = QuantumState(
            amplitude=other.amplitude * norm_factor,
            phase=other.phase + math.pi,  # Phase difference for entanglement
            entanglement_id=entanglement_id,
            coherence=CoherenceState.ENTANGLED
        )
        
        return state1, state2

@dataclass
class EntanglementMetadata:
    """Metadata tracking entanglement relationships between atoms."""
    entanglement_id: str
    entangled_atoms: Set[str]  # Atom IDs
    entanglement_type: EntanglementType
    strength: float  # 0.0 to 1.0
    created_at: float = field(default_factory=time.time)
    last_measurement: Optional[float] = None

class QuantumField:
    """Global field managing quantum coherence across all runtime atoms."""
    
    def __init__(self):
        self._atoms: Dict[str, 'AsyncQuantumAtom'] = {}
        self._entanglements: Dict[str, EntanglementMetadata] = {}
        self._coherence_lock = asyncio.Lock()
        self._field_hamiltonian = complex(1.0, 0.1)  # Field evolution operator
        self._measurement_history: deque = deque(maxlen=1000)
        
    async def register_atom(self, atom: 'AsyncQuantumAtom') -> None:
        """Register an atom with the quantum field."""
        async with self._coherence_lock:
            self._atoms[atom.atom_id] = atom
            logger.info(f"Registered quantum atom {atom.atom_id} with field")
    
    async def deregister_atom(self, atom_id: str) -> None:
        """Remove an atom from the quantum field."""
        async with self._coherence_lock:
            if atom_id in self._atoms:
                # Clean up any entanglements
                await self._cleanup_entanglements(atom_id)
                del self._atoms[atom_id]
                logger.info(f"Deregistered quantum atom {atom_id} from field")
    
    async def entangle_atoms(self, atom1_id: str, atom2_id: str, 
                           entanglement_type: EntanglementType = EntanglementType.SPATIAL) -> str:
        """Create quantum entanglement between two atoms."""
        async with self._coherence_lock:
            if atom1_id not in self._atoms or atom2_id not in self._atoms:
                raise ValueError("Both atoms must be registered with the field")
            
            atom1 = self._atoms[atom1_id]
            atom2 = self._atoms[atom2_id]
            
            # Create entangled quantum states
            state1, state2 = atom1._quantum_state.entangle_with(atom2._quantum_state)
            
            # Update atom states
            atom1._quantum_state = state1
            atom2._quantum_state = state2
            
            # Track entanglement metadata
            entanglement_id = state1.entanglement_id
            self._entanglements[entanglement_id] = EntanglementMetadata(
                entanglement_id=entanglement_id,
                entangled_atoms={atom1_id, atom2_id},
                entanglement_type=entanglement_type,
                strength=1.0
            )
            
            logger.info(f"Entangled atoms {atom1_id} and {atom2_id} with type {entanglement_type}")
            return entanglement_id
    
    async def evolve_field(self, dt: float = 0.1) -> None:
        """Evolve the entire quantum field forward in time."""
        async with self._coherence_lock:
            evolution_tasks = []
            
            for atom in self._atoms.values():
                task = asyncio.create_task(atom._evolve_quantum_state(self._field_hamiltonian, dt))
                evolution_tasks.append(task)
            
            await asyncio.gather(*evolution_tasks, return_exceptions=True)
            
            # Update entanglement strengths (decoherence over time)
            for entanglement in self._entanglements.values():
                decay_rate = 0.01  # Configurable decoherence rate
                entanglement.strength *= math.exp(-decay_rate * dt)
                
                # Remove weak entanglements
                if entanglement.strength < 0.1:
                    await self._break_entanglement(entanglement.entanglement_id)
    
    async def measure_coherence(self) -> Dict[str, Any]:
        """Measure the overall coherence of the quantum field."""
        async with self._coherence_lock:
            total_atoms = len(self._atoms)
            entangled_atoms = sum(1 for atom in self._atoms.values() 
                                if atom._quantum_state.coherence == CoherenceState.ENTANGLED)
            
            coherence_metric = entangled_atoms / total_atoms if total_atoms > 0 else 0.0
            
            measurement = {
                "timestamp": time.time(),
                "total_atoms": total_atoms,
                "entangled_atoms": entangled_atoms,
                "coherence_ratio": coherence_metric,
                "active_entanglements": len(self._entanglements),
                "field_energy": abs(self._field_hamiltonian) ** 2
            }
            
            self._measurement_history.append(measurement)
            return measurement
    
    async def _cleanup_entanglements(self, atom_id: str) -> None:
        """Clean up entanglements involving a specific atom."""
        to_remove = []
        for ent_id, metadata in self._entanglements.items():
            if atom_id in metadata.entangled_atoms:
                to_remove.append(ent_id)
        
        for ent_id in to_remove:
            await self._break_entanglement(ent_id)
    
    async def _break_entanglement(self, entanglement_id: str) -> None:
        """Break an entanglement and collapse involved atoms."""
        if entanglement_id in self._entanglements:
            metadata = self._entanglements[entanglement_id]
            
            # Collapse entangled atoms to definite states
            for atom_id in metadata.entangled_atoms:
                if atom_id in self._atoms:
                    await self._atoms[atom_id]._collapse_quantum_state()
            
            del self._entanglements[entanglement_id]
            logger.info(f"Broke entanglement {entanglement_id}")

# Global quantum field instance
QUANTUM_FIELD = QuantumField()

class AsyncQuantumAtom(Generic[T_co, V_co, C_co], ABC):
    """
    Quantum-enhanced asynchronous atom with full QSD support.
    
    Each atom is a runtime quantum that can:
    - Exist in superposition of states
    - Become entangled with other atoms
    - Quine itself recursively
    - Maintain probabilistic coherence
    - Evolve unitarily over time
    """
    __slots__ = (
        'atom_id', '_code', '_value', '_local_env', '_refcount', '_ttl', '_created_at',
        'request_data', 'session', 'runtime_namespace', 'security_context',
        '_pending_tasks', '_lock', '_buffer_size', '_buffer', '_last_access_time',
        '_quantum_state', '_entanglement_metadata', '_quine_history', '_module_cache',
        '_hamiltonian', '_measurement_count', '_coherence_callbacks'
    )

    def __init__(
        self,
        code: str,
        value: Optional[V_co] = None,
        ttl: Optional[int] = None,
        request_data: Optional[Dict[str, Any]] = None,
        buffer_size: int = 1024 * 64,
        initial_amplitude: complex = complex(1.0, 0.0),
        hamiltonian: complex = complex(1.0, 0.1)
    ):
        self.atom_id = str(uuid.uuid4())
        self._code = code
        self._value = value
        self._local_env: Dict[str, Any] = {}
        self._refcount = 1
        self._ttl = ttl
        self._created_at = time.time()
        self._last_access_time = self._created_at
        self.request_data = request_data or {}
        self.session: Dict[str, Any] = self.request_data.get("session", {})
        self.runtime_namespace = None
        self.security_context = None

        # Async-specific attributes
        self._pending_tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._buffer_size = buffer_size
        self._buffer = bytearray(buffer_size)

        # Quantum-specific attributes
        self._quantum_state = QuantumState(
            amplitude=initial_amplitude,
            phase=0.0,
            coherence=CoherenceState.SUPERPOSITION
        )
        self._entanglement_metadata: Dict[str, EntanglementMetadata] = {}
        self._quine_history: List[str] = []  # Track self-reproductions
        self._module_cache: Dict[str, ModuleType] = {}
        self._hamiltonian = hamiltonian
        self._measurement_count = 0
        self._coherence_callbacks: List[Callable] = []

        # Register with quantum field
        asyncio.create_task(QUANTUM_FIELD.register_atom(self))

    async def __aenter__(self):
        """Support async context manager protocol with quantum coherence."""
        self._refcount += 1
        self._last_access_time = time.time()
        
        # Evolve quantum state on entry
        await self._evolve_quantum_state(self._hamiltonian, 0.01)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Support async context manager protocol with quantum cleanup."""
        self._refcount -= 1
        
        # Measure quantum state on exit
        await self._measure_quantum_state()
        
        # Schedule cleanup if refcount reaches zero
        if self._refcount <= 0:
            asyncio.create_task(self._cleanup())
        return False

    async def _cleanup(self):
        """Cleanup resources with quantum deregistration."""
        # Cancel any pending tasks
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()

        # Deregister from quantum field
        await QUANTUM_FIELD.deregister_atom(self.atom_id)

        # Clear buffers and references
        self._buffer = bytearray(0)
        self._local_env.clear()
        self._module_cache.clear()

    async def _evolve_quantum_state(self, operator: complex, dt: float) -> None:
        """Evolve the atom's quantum state using unitary evolution."""
        async with self._lock:
            self._quantum_state = self._quantum_state.evolve(operator, dt)
            
            # Trigger coherence callbacks
            for callback in self._coherence_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self._quantum_state)
                    else:
                        callback(self._quantum_state)
                except Exception as e:
                    logger.warning(f"Coherence callback failed: {e}")

    async def _measure_quantum_state(self) -> QuantumState:
        """Perform quantum measurement, potentially collapsing the state."""
        async with self._lock:
            self._measurement_count += 1
            
            # Probabilistic collapse based on measurement
            if random.random() < self._quantum_state.probability:
                # Collapse to definite state
                collapsed_state = QuantumState(
                    amplitude=complex(1.0, 0.0),
                    phase=self._quantum_state.phase,
                    entanglement_id=self._quantum_state.entanglement_id,
                    coherence=CoherenceState.COLLAPSED
                )
                self._quantum_state = collapsed_state
            
            return self._quantum_state

    async def _collapse_quantum_state(self) -> None:
        """Force collapse of quantum state (used when entanglement breaks)."""
        async with self._lock:
            self._quantum_state = QuantumState(
                amplitude=complex(1.0, 0.0),
                phase=self._quantum_state.phase,
                coherence=CoherenceState.COLLAPSED
            )

    async def quine_self(self, variations: Optional[Dict[str, Any]] = None) -> 'AsyncQuantumAtom':
        """
        Quine the atom - create a new instance that can reproduce the current atom.
        
        This implements the quinic behavior essential to QSD.
        """
        variations = variations or {}
        
        # Create quining code that includes self-reproduction capability
        quine_code = f"""
# Quined atom generated at {time.time()}
# Original atom ID: {self.atom_id}

import asyncio
from typing import Any, Dict, Optional

async def main(*args, **kwargs):
    # Access to original atom through closure
    original_atom = kwargs.get('__atom_self__')
    
    # Execute original logic
    {self._code}
    
    # Self-reproduction capability
    if kwargs.get('__should_quine__', False):
        new_atom = await original_atom.quine_self()
        return {{'quined_atom_id': new_atom.atom_id, 'generation': len(original_atom._quine_history) + 1}}
    
    return locals().get('__return__', {{'status': 'executed', 'atom_id': original_atom.atom_id}})
"""
        
        # Apply variations to the base code
        for key, value in variations.items():
            quine_code = quine_code.replace(f"__{key}__", str(value))
        
        # Create new quantum atom with entangled state
        new_atom = self.__class__(
            code=quine_code,
            value=self._value,
            ttl=self._ttl,
            request_data=self.request_data.copy(),
            hamiltonian=self._hamiltonian
        )
        
        # Entangle with parent atom
        await QUANTUM_FIELD.entangle_atoms(
            self.atom_id, 
            new_atom.atom_id, 
            EntanglementType.QUINIC
        )
        
        # Track quining history
        self._quine_history.append(new_atom.atom_id)
        
        logger.info(f"Atom {self.atom_id} quined into {new_atom.atom_id}")
        return new_atom

    async def create_dynamic_module(self, module_name: str, module_code: str) -> Optional[ModuleType]:
        """
        Create a dynamic module with quantum entanglement metadata.
        
        This implements the unitary operator U(t) transforming code strings into runtime modules.
        """
        # Check cache first
        cache_key = hashlib.md5((module_name + module_code).encode()).hexdigest()
        if cache_key in self._module_cache:
            return self._module_cache[cache_key]
        
        try:
            # Create module with quantum metadata
            dynamic_module = ModuleType(module_name)
            dynamic_module.__file__ = f"quantum_runtime_{self.atom_id}"
            dynamic_module.__package__ = module_name
            dynamic_module.__path__ = None
            dynamic_module.__doc__ = f"Quantum module generated by atom {self.atom_id}"
            
            # Add quantum metadata
            dynamic_module.__quantum_atom_id__ = self.atom_id
            dynamic_module.__quantum_state__ = self._quantum_state
            dynamic_module.__creation_time__ = time.time()
            
            # Execute code in module namespace
            exec(module_code, dynamic_module.__dict__)
            
            # Register module
            sys.modules[f"{module_name}_{self.atom_id}"] = dynamic_module
            self._module_cache[cache_key] = dynamic_module
            
            # Evolve quantum state after module creation
            await self._evolve_quantum_state(self._hamiltonian, 0.1)
            
            return dynamic_module
            
        except Exception as e:
            logger.error(f"Error creating dynamic module in atom {self.atom_id}: {e}")
            return None

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the atom with quantum state evolution and measurement.
        """
        self._last_access_time = time.time()
        
        # Evolve quantum state before execution
        await self._evolve_quantum_state(self._hamiltonian, random.uniform(0.01, 0.1))

        # Create execution environment with quantum context
        async with self._lock:
            local_env = self._local_env.copy()
            local_env.update({
                'args': args,
                'kwargs': kwargs,
                '__atom_self__': self,
                '__quantum_state__': self._quantum_state,
                '__atom_id__': self.atom_id
            })

        try:
            # Determine execution path based on quantum state
            is_async = self._is_async_code(self._code)
            
            # Probabilistic execution based on quantum amplitude
            execution_probability = self._quantum_state.probability
            if random.random() > execution_probability:
                # Quantum tunneling: execute alternative path
                logger.info(f"Quantum tunneling in atom {self.atom_id}")
                return await self._quantum_tunnel_execution(*args, **kwargs)

            # Standard quantum execution
            if is_async:
                result = await self._execute_async_code(local_env, *args, **kwargs)
            else:
                result = await self._execute_sync_code(local_env, *args, **kwargs)

            # Measure quantum state after execution
            final_state = await self._measure_quantum_state()
            
            # Add quantum metadata to result
            if isinstance(result, dict):
                result['__quantum_metadata__'] = {
                    'atom_id': self.atom_id,
                    'quantum_state': {
                        'amplitude': str(final_state.amplitude),
                        'phase': final_state.phase,
                        'coherence': final_state.coherence.name,
                        'probability': final_state.probability
                    },
                    'measurement_count': self._measurement_count
                }

            return result

        except Exception as e:
            # Quantum error handling: collapse state and propagate
            await self._collapse_quantum_state()
            raise RuntimeError(f"Quantum execution error in atom {self.atom_id}: {e}")

    async def _quantum_tunnel_execution(self, *args: Any, **kwargs: Any) -> Any:
        """Alternative execution path accessed through quantum tunneling."""
        # This could implement alternative algorithms, error correction, or creative variations
        return {
            'status': 'quantum_tunneled',
            'atom_id': self.atom_id,
            'message': 'Execution occurred through quantum tunneling',
            'args': args,
            'kwargs': kwargs
        }

    async def _execute_async_code(self, local_env: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Execute asynchronous code with quantum context."""
        code_obj = compile(self._code, f'<quantum_atom_{self.atom_id}>', 'exec')
        namespace = {}
        exec(code_obj, globals(), namespace)
        
        main_func = namespace.get('main')
        if main_func and inspect.iscoroutinefunction(main_func):
            return await main_func(*args, **kwargs)
        else:
            # Find any async function
            for name, func in namespace.items():
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
            raise ValueError("No async function found in async code")

    async def _execute_sync_code(self, local_env: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Execute synchronous code with quantum context."""
        code_obj = compile(self._code, f'<quantum_atom_{self.atom_id}>', 'exec')
        exec(code_obj, globals(), local_env)
        return local_env.get('__return__')

    def _is_async_code(self, code: str) -> bool:
        """Detect if code contains async constructs."""
        try:
            parsed = ast.parse(code)
            for node in ast.walk(parsed):
                if isinstance(node, (ast.AsyncFunctionDef, ast.Await)):
                    return True
            return False
        except SyntaxError:
            return False

    def add_coherence_callback(self, callback: Callable) -> None:
        """Add callback to be triggered on quantum state changes."""
        self._coherence_callbacks.append(callback)

    def remove_coherence_callback(self, callback: Callable) -> None:
        """Remove coherence callback."""
        if callback in self._coherence_callbacks:
            self._coherence_callbacks.remove(callback)

    # Properties for quantum state access
    @property
    def quantum_state(self) -> QuantumState:
        """Get current quantum state (read-only)."""
        return self._quantum_state

    @property
    def is_entangled(self) -> bool:
        """Check if atom is quantum entangled."""
        return self._quantum_state.coherence == CoherenceState.ENTANGLED

    @property
    def entanglement_id(self) -> Optional[str]:
        """Get entanglement ID if atom is entangled."""
        return self._quantum_state.entanglement_id

    @property
    def quine_generation(self) -> int:
        """Get the generation count of this atom's quining history."""
        return len(self._quine_history)

    # Abstract methods - simplified for core quantum functionality
    @abstractmethod
    async def handle_quantum_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests with quantum processing."""
        pass


# Example implementation
class ConcreteQuantumAtom(AsyncQuantumAtom[str, dict, Callable]):
    """Concrete implementation with quantum request handling."""

    async def handle_quantum_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum-enhanced requests."""
        operation = request_context.get("operation", "default")
        
        if operation == "quine":
            # Self-reproduction request
            new_atom = await self.quine_self()
            return {
                "status": "success",
                "operation": "quine",
                "new_atom_id": new_atom.atom_id,
                "generation": self.quine_generation
            }
        
        elif operation == "entangle":
            # Entanglement request
            target_atom_id = request_context.get("target_atom_id")
            if target_atom_id:
                entanglement_id = await QUANTUM_FIELD.entangle_atoms(
                    self.atom_id, target_atom_id, EntanglementType.SPATIAL
                )
                return {
                    "status": "success",
                    "operation": "entangle",
                    "entanglement_id": entanglement_id
                }
        
        elif operation == "measure_coherence":
            # Quantum measurement
            state = await self._measure_quantum_state()
            return {
                "status": "success",
                "operation": "measure_coherence",
                "quantum_state": {
                    "amplitude": str(state.amplitude),
                    "phase": state.phase,
                    "probability": state.probability,
                    "coherence": state.coherence.name
                }
            }
        
        elif operation == "create_module":
            # Dynamic module creation
            module_name = request_context.get("module_name", "dynamic_quantum_module")
            module_code = request_context.get("module_code", "")
            
            module = await self.create_dynamic_module(module_name, module_code)
            if module:
                return {
                    "status": "success",
                    "operation": "create_module",
                    "module_name": module_name,
                    "module_file": module.__file__
                }
        
        # Default quantum processing
        return {
            "status": "success",
            "operation": operation,
            "atom_id": self.atom_id,
            "quantum_state": {
                "coherence": self._quantum_state.coherence.name,
                "is_entangled": self.is_entangled
            }
        }


# Quantum field evolution background task
async def quantum_field_evolution_loop(interval: float = 0.1):
    """Background task for continuous quantum field evolution."""
    while True:
        try:
            await QUANTUM_FIELD.evolve_field(interval)
            await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Quantum field evolution error: {e}")
            await asyncio.sleep(interval * 10)  # Longer pause on error


# Demo function
async def quantum_demo():
    """Demonstrate quantum atom capabilities."""
    print("=== Quinic Statistical Dynamics Demo ===\n")
    
    # Start quantum field evolution
    evolution_task = asyncio.create_task(quantum_field_evolution_loop())
    
    try:
        # Create quantum atoms
        atom1_code = """
async def main(*args, **kwargs):
    import time
    print(f"Quantum Atom 1 executing at {time.time()}")
    return {"result": "quantum_computation_1", "args": args}
"""
        
        atom2_code = """
async def main(*args, **kwargs):
    import time
    print(f"Quantum Atom 2 executing at {time.time()}")
    return {"result": "quantum_computation_2", "kwargs": kwargs}
"""
        
        atom1 = ConcreteQuantumAtom(code=atom1_code)
        atom2 = ConcreteQuantumAtom(code=atom2_code)
        
        print(f"Created atoms: {atom1.atom_id} and {atom2.atom_id}")
        
        # Execute atoms
        result1 = await atom1("test", "data")
        result2 = await atom2(param="value")
        
        print(f"Atom 1 result: {result1}")
        print(f"Atom 2 result: {result2}")
        
        # Entangle atoms
        print("\n=== Quantum Entanglement ===")
        entanglement_id = await QUANTUM_FIELD.entangle_atoms(atom1.atom_id, atom2.atom_id)
        print(f"Entangled atoms with ID: {entanglement_id}")
        
        # Demonstrate quining
        print("\n=== Quinic Self-Reproduction ===")
        quined_atom = await atom1.quine_self()
        print(f"Atom {atom1.atom_id} quined into {quined_atom.atom_id}")
        
        # Measure field coherence
        print("\n=== Field Coherence Measurement ===")
        coherence = await QUANTUM_FIELD.measure_coherence()
        print(f"Field coherence: {coherence}")
        
        # Dynamic module creation
        print("\n=== Dynamic Module Creation ===")
        module_code = """
def quantum_function():
    return "Hello from quantum module!"

class QuantumClass:
    def __init__(self):
        self.state = "quantum"
"""
        
        module = await atom1.create_dynamic_module("quantum_module", module_code)
        if module:
            print(f"Created module: {module.__class__}")
            print(f"Module attributes: {module.__dict__}")
            print(f"Module function: {module.quantum_function()}")
        else:
            print("Failed to create module")
    except:
        print("Error occurred during quantum computation")
    finally:
        # Clean up
        await QUANTUM_FIELD.deregister_atom(atom1.atom_id)
if __name__ == "__main__":
    asyncio.run(quantum_demo())