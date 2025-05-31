from __future__ import annotations
import sys
import time
import hashlib
from types import ModuleType, SimpleNamespace
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

"""
Quinic Statistical Dynamics (QSD) Runtime Quanta Generator

This implements the fundamental U(t) operator that transforms source code ψ₀ 
into executing runtime quanta ψ(t), maintaining entanglement metadata across 
distributed instantiation for probabilistic coherence.
"""

@dataclass
class EntanglementMetadata:
    """Maintains quantum-like entanglement information across runtime instances"""
    lineage_hash: str
    generation: int
    parent_runtime_id: Optional[str] = None
    entangled_siblings: List[str] = field(default_factory=list)
    coherence_timestamp: float = field(default_factory=time.time)
    probabilistic_state: Dict[str, Any] = field(default_factory=dict)
    
    def update_coherence(self, new_state: Dict[str, Any]) -> None:
        """Update probabilistic state while maintaining temporal coherence"""
        self.probabilistic_state.update(new_state)
        self.coherence_timestamp = time.time()

@dataclass
class RuntimeQuantum:
    """A single runtime quantum with full QSD capabilities"""
    quantum_id: str
    module: ModuleType
    entanglement: EntanglementMetadata
    source_code: str
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    def observe(self, observable: str) -> Any:
        """Quantum observation of runtime state"""
        return getattr(self.module, observable, None)
    
    def quine(self, new_quantum_id: str) -> 'RuntimeQuantum':
        """Self-reproduction with entanglement preservation"""
        child_entanglement = EntanglementMetadata(
            lineage_hash=self.entanglement.lineage_hash,
            generation=self.entanglement.generation + 1,
            parent_runtime_id=self.quantum_id,
            entangled_siblings=[],  # Will be populated by QSD field
            probabilistic_state=self.entanglement.probabilistic_state.copy()
        )
        
        return QSDField.get_instance().apply_unitary_operator(
            self.source_code, new_quantum_id, child_entanglement
        )

class QSDField:
    """Singleton field managing distributed runtime quanta with statistical coherence"""
    
    _instance: Optional['QSDField'] = None
    
    def __init__(self):
        if QSDField._instance is not None:
            raise RuntimeError("QSDField is a singleton. Use get_instance().")
        
        self.active_quanta: Dict[str, RuntimeQuantum] = {}
        self.entanglement_graph: Dict[str, List[str]] = defaultdict(list)
        self.coherence_observers: List[Callable] = []
        self.field_state: Dict[str, Any] = {}
        
    @classmethod 
    def get_instance(cls) -> 'QSDField':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def apply_unitary_operator(
        self, 
        source_psi_0: str, 
        quantum_id: str,
        entanglement: Optional[EntanglementMetadata] = None
    ) -> RuntimeQuantum:
        """
        Apply U(t) = exp(-iOt) to transform ψ₀ (source) into ψ(t) (runtime)
        
        This is the core QSD transformation where:
        - ψ₀ is the source code string
        - U(t) is the unitary evolution operator (this method)
        - ψ(t) is the resulting runtime quantum
        """
        
        # Generate entanglement metadata if not provided
        if entanglement is None:
            lineage_hash = hashlib.sha256(source_psi_0.encode()).hexdigest()[:16]
            entanglement = EntanglementMetadata(
                lineage_hash=lineage_hash,
                generation=0
            )
        
        # Create the runtime module (the quantum collapse)
        dynamic_module = ModuleType(quantum_id)
        dynamic_module.__file__ = f"qsd_quantum_{quantum_id}"
        dynamic_module.__package__ = quantum_id
        dynamic_module.__path__ = None
        dynamic_module.__doc__ = f"QSD Runtime Quantum: {quantum_id}"
        
        # Inject QSD runtime capabilities
        dynamic_module.__qsd_quantum_id__ = quantum_id
        dynamic_module.__qsd_field__ = self
        dynamic_module.__qsd_entanglement__ = entanglement
        
        try:
            # Execute the source code in the module namespace
            exec(source_psi_0, dynamic_module.__dict__)
            
            # Register in sys.modules for full runtime integration
            sys.modules[quantum_id] = dynamic_module
            
            # Create the runtime quantum
            runtime_quantum = RuntimeQuantum(
                quantum_id=quantum_id,
                module=dynamic_module,
                entanglement=entanglement,
                source_code=source_psi_0
            )
            
            # Register in the QSD field
            self.active_quanta[quantum_id] = runtime_quantum
            
            # Update entanglement graph
            if entanglement.parent_runtime_id:
                self.entanglement_graph[entanglement.parent_runtime_id].append(quantum_id)
                # Update sibling entanglements
                siblings = self.entanglement_graph[entanglement.parent_runtime_id]
                for sibling_id in siblings:
                    if sibling_id != quantum_id and sibling_id in self.active_quanta:
                        self.active_quanta[sibling_id].entanglement.entangled_siblings.append(quantum_id)
                        runtime_quantum.entanglement.entangled_siblings.append(sibling_id)
            
            # Notify coherence observers
            self._notify_coherence_observers(runtime_quantum)
            
            return runtime_quantum
            
        except Exception as e:
            print(f"Error in unitary transformation for quantum {quantum_id}: {e}")
            return None
    
    def measure_field_coherence(self) -> Dict[str, Any]:
        """Calculate statistical coherence across all active quanta"""
        coherence_metrics = {
            'total_quanta': len(self.active_quanta),
            'entanglement_density': len(self.entanglement_graph),
            'generation_distribution': defaultdict(int),
            'lineage_families': defaultdict(int)
        }
        
        for quantum in self.active_quanta.values():
            coherence_metrics['generation_distribution'][quantum.entanglement.generation] += 1
            coherence_metrics['lineage_families'][quantum.entanglement.lineage_hash] += 1
            
        return coherence_metrics
    
    def add_coherence_observer(self, observer: Callable[[RuntimeQuantum], None]) -> None:
        """Add observer for quantum field state changes"""
        self.coherence_observers.append(observer)
    
    def _notify_coherence_observers(self, quantum: RuntimeQuantum) -> None:
        """Notify all observers of quantum state changes"""
        for observer in self.coherence_observers:
            try:
                observer(quantum)
            except Exception as e:
                print(f"Error in coherence observer: {e}")

# Legacy compatibility wrapper
def create_module(module_name: str, module_code: str, main_module_path: str) -> ModuleType | None:
    """Legacy wrapper - now creates proper QSD runtime quantum"""
    field = QSDField.get_instance()
    quantum = field.apply_unitary_operator(module_code, module_name)
    return quantum.module if quantum else None

# Example QSD usage demonstrating quantum behavior
if __name__ == "__main__":
    # Initialize the QSD field
    field = QSDField.get_instance()
    
    # Define a quantum that can self-replicate
    quinic_source = '''
def greet(name="Unknown"):
    print(f"Hello from quantum {__qsd_quantum_id__}, generation {__qsd_entanglement__.generation}!")
    return f"Quantum {__qsd_quantum_id__} active"

def reproduce(child_id):
    """Demonstrate quinic self-reproduction"""
    current_quantum = __qsd_field__.active_quanta[__qsd_quantum_id__]
    child = current_quantum.quine(child_id)
    return child

def observe_field():
    """Observe the current field state"""
    return __qsd_field__.measure_field_coherence()
'''
    
    # Apply the unitary operator to create the first quantum
    alpha_quantum = field.apply_unitary_operator(quinic_source, "alpha_quantum")
    
    if alpha_quantum:
        # Demonstrate quantum observation
        greeting = alpha_quantum.observe('greet')
        if greeting:
            result = greeting("QSD")
            print(f"Observation result: {result}")
        
        # Demonstrate quinic reproduction
        reproduce_func = alpha_quantum.observe('reproduce')
        if reproduce_func:
            beta_quantum = reproduce_func("beta_quantum")
            gamma_quantum = reproduce_func("gamma_quantum")
            
            # Show field coherence
            observe_func = alpha_quantum.observe('observe_field')
            if observe_func:
                coherence = observe_func()
                print(f"Field coherence metrics: {coherence}")
                
                # Demonstrate entanglement
                print(f"Alpha siblings: {alpha_quantum.entanglement.entangled_siblings}")
                if beta_quantum:
                    print(f"Beta parent: {beta_quantum.entanglement.parent_runtime_id}")