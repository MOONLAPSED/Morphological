import sys
import os
import re
import mmap
import pickle
import hashlib
import inspect
import importlib
import traceback
import threading
from pathlib import Path
from types import ModuleType
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar, Union, cast, Callable, Set
import math
import random

T = TypeVar('T')
V = TypeVar('V')
C = TypeVar('C')
T_co = TypeVar('T_co', covariant=True)
V_co = TypeVar('V_co', covariant=True)
C_co = TypeVar('C_co', covariant=True)
T_anti = TypeVar('T_anti', contravariant=True)
V_anti = TypeVar('V_anti', contravariant=True)
C_anti = TypeVar('C_anti', contravariant=True)
U = TypeVar('U')  # For composition

def hash_state(value: Any) -> int:
    """Hash a state value in a deterministic way"""
    if isinstance(value, int):
        return value * 2654435761 % 2**32  # Knuth's multiplicative hash
    elif isinstance(value, str):
        return sum(ord(c) * (31 ** i) for i, c in enumerate(value)) % 2**32
    else:
        return hash(str(value)) % 2**32

class MorphicComplex:
    """Represents a complex number with morphic properties."""
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag
    
    def conjugate(self) -> 'MorphicComplex':
        """Return the complex conjugate."""
        return MorphicComplex(self.real, -self.imag)
    
    def __add__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(self.real + other.real, self.imag + other.imag)
    
    def __sub__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(self.real - other.real, self.imag - other.imag)
    
    def __mul__(self, other: Union['MorphicComplex', float, int]) -> 'MorphicComplex':
        if isinstance(other, (int, float)):
            return MorphicComplex(self.real * other, self.imag * other)
        return MorphicComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def __rmul__(self, other: Union[float, int]) -> 'MorphicComplex':
        return self.__mul__(other)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, MorphicComplex):
            return False
        return (abs(self.real - other.real) < 1e-10 and 
                abs(self.imag - other.imag) < 1e-10)
    
    def __hash__(self) -> int:
        return hash((self.real, self.imag))
    
    def __repr__(self) -> str:
        if self.imag >= 0:
            return f"{self.real} + {self.imag}i"
        return f"{self.real} - {abs(self.imag)}i"

class Matrix:
    """Simple matrix implementation using standard Python"""
    def __init__(self, data: List[List[Any]]):
        if not data:
            raise ValueError("Matrix data cannot be empty")
        
        # Verify all rows have the same length
        cols = len(data[0])
        if any(len(row) != cols for row in data):
            raise ValueError("All rows must have the same length")
        
        self.data = data
        self.rows = len(data)
        self.cols = cols
    
    def __getitem__(self, idx: Tuple[int, int]) -> Any:
        i, j = idx
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise IndexError(f"Matrix indices {i},{j} out of range")
        return self.data[i][j]
    
    def __setitem__(self, idx: Tuple[int, int], value: Any) -> None:
        i, j = idx
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise IndexError(f"Matrix indices {i},{j} out of range")
        self.data[i][j] = value
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Matrix):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False
        return all(self.data[i][j] == other.data[i][j] 
                  for i in range(self.rows) 
                  for j in range(self.cols))
    
    def __matmul__(self, other: Union['Matrix', List[Any]]) -> Union['Matrix', List[Any]]:
        """Matrix multiplication operator @"""
        if isinstance(other, list):
            # Matrix @ vector
            if len(other) != self.cols:
                raise ValueError(f"Dimensions don't match for matrix-vector multiplication: "
                                f"matrix cols={self.cols}, vector length={len(other)}")
            return [sum(self.data[i][j] * other[j] for j in range(self.cols)) 
                    for i in range(self.rows)]
        else:
            # Matrix @ Matrix
            if self.cols != other.rows:
                raise ValueError(f"Dimensions don't match for matrix multiplication: "
                                f"first matrix cols={self.cols}, second matrix rows={other.rows}")
            result = [[sum(self.data[i][k] * other.data[k][j] 
                          for k in range(self.cols))
                      for j in range(other.cols)]
                      for i in range(self.rows)]
            return Matrix(result)
    
    def trace(self) -> Any:
        """Calculate the trace of the matrix"""
        if self.rows != self.cols:
            raise ValueError("Trace is only defined for square matrices")
        return sum(self.data[i][i] for i in range(self.rows))
    
    def transpose(self) -> 'Matrix':
        """Return the transpose of this matrix"""
        return Matrix([[self.data[j][i] for j in range(self.rows)] 
                      for i in range(self.cols)])
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """Create a matrix of zeros"""
        if rows <= 0 or cols <= 0:
            raise ValueError("Matrix dimensions must be positive")
        return Matrix([[0 for _ in range(cols)] for _ in range(rows)])
    
    @staticmethod
    def identity(n: int) -> 'Matrix':
        """Create an n×n identity matrix"""
        if n <= 0:
            raise ValueError("Matrix dimension must be positive")
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    def __repr__(self) -> str:
        return "\n".join([str(row) for row in self.data])

@dataclass
class MorphologicalBasis(Generic[T, V, C]):
    """Defines a structured basis with symmetry evolution."""
    type_structure: T  # Topological/Type representation
    value_space: V     # State space (e.g., physical degrees of freedom)
    compute_space: C   # Operator space (e.g., Lie Algebra of transformations)
    
    def evolve(self, generator: Matrix, time: float) -> 'MorphologicalBasis[T, V, C]':
        """Evolves the basis using a symmetry generator over time."""
        new_compute_space = self._transform_compute_space(generator, time)
        return MorphologicalBasis(
            self.type_structure, 
            self.value_space, 
            new_compute_space
        )
    
    def _transform_compute_space(self, generator: Matrix, time: float) -> C:
        """Transform the compute space using the generator"""
        # This would depend on the specific implementation of C
        # For demonstration, assuming C is a Matrix:
        if isinstance(self.compute_space, Matrix) and isinstance(generator, Matrix):
            # Simple time evolution using matrix exponential approximation
            # exp(tA) ≈ I + tA + (tA)²/2! + ...
            identity = Matrix.zeros(generator.rows, generator.cols)
            for i in range(identity.rows):
                identity.data[i][i] = 1
                
            scaled_gen = Matrix([[generator[i, j] * time for j in range(generator.cols)] 
                               for i in range(generator.rows)])
            
            # First-order approximation: I + tA
            result = identity
            for i in range(result.rows):
                for j in range(result.cols):
                    result.data[i][j] += scaled_gen.data[i][j]
                    
            return cast(C, result @ self.compute_space)
        
        return self.compute_space  # Default fallback

class Category(Generic[T_co, V_co, C_co]):
    """
    Represents a mathematical category with objects and morphisms.
    """
    def __init__(self, name: str):
        self.name = name
        self.objects: List[T_co] = []
        self.morphisms: Dict[Tuple[T_co, T_co], List[C_co]] = {}
    
    def add_object(self, obj: T_co) -> None:
        """Add an object to the category."""
        if obj not in self.objects:
            self.objects.append(obj)
    
    def add_morphism(self, source: T_co, target: T_co, morphism: C_co) -> None:
        """Add a morphism between objects."""
        if source not in self.objects:
            self.add_object(source)
        if target not in self.objects:
            self.add_object(target)
            
        key = (source, target)
        if key not in self.morphisms:
            self.morphisms[key] = []
        self.morphisms[key].append(morphism)
    
    def compose(self, f: C_co, g: C_co) -> C_co:
        """
        Compose two morphisms.
        For morphisms f: A → B and g: B → C, returns g ∘ f: A → C
        """
        def composed(x):
            return g(f(x))
        return cast(C_co, composed)

    def find_morphisms(self, source: T_co, target: T_co) -> List[C_co]:
        """Find all morphisms between two objects."""
        return self.morphisms.get((source, target), [])
    
    def is_functor_to(self, target_category: 'Category', object_map: Dict[T_co, Any], morphism_map: Dict[C_co, Any]) -> bool:
        """
        Check if the given maps form a functor from this category to the target category.
        A functor is a structure-preserving map between categories.
        """
        # Check that all objects are mapped
        if not all(obj in object_map for obj in self.objects):
            return False
            
        # Check that all morphisms are mapped
        all_morphisms = [m for morphs in self.morphisms.values() for m in morphs]
        if not all(m in morphism_map for m in all_morphisms):
            return False
            
        # Check that the functor preserves composition
        for src, tgt in self.morphisms:
            for f in self.morphisms[(src, tgt)]:
                for mid in self.objects:
                    g_list = self.find_morphisms(tgt, mid)
                    for g in g_list:
                        # Check if g ∘ f maps to morphism_map[g] ∘ morphism_map[f]
                        composed = self.compose(f, g)
                        if composed not in morphism_map:
                            return False
                        
                        # Check that the composition is preserved
                        target_f = morphism_map[f]
                        target_g = morphism_map[g]
                        target_composed = target_category.compose(target_f, target_g)
                        if morphism_map[composed] != target_composed:
                            return False
                            
        return True

class Morphism(Generic[T_co, T_anti]):
    """Abstract morphism between type structures"""
    
    @abstractmethod
    def apply(self, source: T_anti) -> T_co:
        """Apply this morphism to transform source into target"""
        pass
    
    def __call__(self, source: T_anti) -> T_co:
        return self.apply(source)
    
    def compose(self, other: 'Morphism[U, T_co]') -> 'Morphism[U, T_anti]':
        """Compose this morphism with another (this ∘ other)"""
        # Type U is implied here
        original_self = self
        original_other = other
        
        class ComposedMorphism(Morphism[T_co, T_anti]):  # type: ignore
            def apply(self, source: T_anti) -> T_co:
                return original_self.apply(original_other.apply(source))
                
        return ComposedMorphism()

class HermitianMorphism(Generic[T, V, C, T_anti, V_anti, C_anti]):
    """
    Represents a morphism with a Hermitian adjoint relationship between
    covariant and contravariant types.
    """
    def __init__(self, 
                 forward: Callable[[T, V], C],
                 adjoint: Callable[[T_anti, V_anti], C_anti]):
        self.forward = forward
        self.adjoint = adjoint
        self.domain = None  # Will be set dynamically
        self.codomain = None  # Will be set dynamically
        
    def apply(self, source: T, value: V) -> C:
        """Apply the forward morphism"""
        return self.forward(source, value)
        
    def apply_adjoint(self, source: T_anti, value: V_anti) -> C_anti:
        """Apply the adjoint (contravariant) morphism"""
        return self.adjoint(source, value)
        
    def get_adjoint(self) -> 'HermitianMorphism[V_anti, T_anti, C_anti, V, T, C]':
        """
        Create the Hermitian adjoint (contravariant dual) of this morphism.
        The adjoint reverses the morphism direction and applies the conjugate operation.
        """
        return HermitianMorphism(self.adjoint, self.forward)
    
    def __call__(self, source: T, value: V) -> C:
        """Make the morphism callable directly"""
        return self.apply(source, value)


# Define a MorphologicalBasis with simple matrices
basis = MorphologicalBasis(
    type_structure="TopologyA",
    value_space=[1, 2, 3],    # Could be a vector or state list
    compute_space=Matrix.identity(3)
)

# A generator matrix representing an infinitesimal symmetry
generator = Matrix([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 0]
])

# Evolve the basis for time t=0.1
new_basis = basis.evolve(generator, time=0.1)
print(new_basis.compute_space)


@dataclass
class TVCEntity(Generic[T, V, C], ABC):
    """
    Base class carrying:
      - T: the *type-structure* / archetype  
      - V: the *value-space* instantiation  
      - C: the *context* (entanglement lineage, metadata, etc.)
    """
    T: T
    V: V
    C: C

    @abstractmethod
    def quine(self) -> str:
        """
        Emit self-reproducing source code with injected T, V, C metadata.
        """
        ...

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable view of (T, V, C).
        """
        ...

# === Core Data Structures ===

@dataclass(frozen=True)
class ModuleMetadata:
    """Comprehensive metadata for module tracking and lazy loading."""
    original_path: Path
    module_name: str
    is_python: bool
    file_size: int
    mtime: float
    content_hash: str
    extension: str = ""
    created_time: float = 0.0
    
    @classmethod
    def from_path(cls, path: Path, module_name: str = None) -> 'ModuleMetadata':
        """Create metadata from a file path."""
        stat = path.stat()
        module_name = module_name or path.stem
        
        # Compute content hash
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        
        return cls(
            original_path=path,
            module_name=module_name,
            is_python=path.suffix == '.py',
            file_size=stat.st_size,
            mtime=stat.st_mtime,
            content_hash=hasher.hexdigest(),
            extension=path.suffix,
            created_time=stat.st_ctime
        )


@dataclass
class RuntimeConfig:
    """Configuration for the runtime system."""
    base_dir: Path
    max_cache_size: int = 1000
    max_workers: int = 4
    chunk_size: int = 1024 * 1024
    excluded_dirs: Set[str] = None
    hash_algorithm: str = 'sha256'
    use_mmap: bool = True
    max_scan_depth: int = 3
    
    def __post_init__(self):
        if self.excluded_dirs is None:
            self.excluded_dirs = {'.git', '__pycache__', 'venv', '.env', 'node_modules'}
        self.base_dir = Path(self.base_dir)


# === Module Management ===

class ModuleIndex:
    """Thread-safe index for module metadata with LRU caching."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.index: Dict[str, ModuleMetadata] = {}
        self.cache: OrderedDict[str, ModuleType] = OrderedDict()
        self.max_cache_size = max_cache_size
        self.lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
    
    def add(self, module_name: str, metadata: ModuleMetadata) -> None:
        """Add module metadata to the index."""
        with self.lock:
            self.index[module_name] = metadata
    
    def get(self, module_name: str) -> Optional[ModuleMetadata]:
        """Retrieve module metadata."""
        with self.lock:
            return self.index.get(module_name)

    def remove(self, module_name: str) -> bool:
        """Remove module from index and cache."""
        with self.lock:
            removed_from_index = self.index.pop(module_name, None) is not None
            removed_from_cache = self.cache.pop(module_name, None) is not None
            
            # Also remove from sys.modules if present
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            return removed_from_index or removed_from_cache

    def cache_module(self, module_name: str, module: ModuleType) -> None:
        """Cache a loaded module with LRU eviction."""
        with self.lock:
            # Remove from cache if already present
            if module_name in self.cache:
                del self.cache[module_name]
            
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_cache_size:
                oldest_name, oldest_module = self.cache.popitem(last=False)
                if hasattr(oldest_module, '__name__') and oldest_module.__name__ in sys.modules:
                    del sys.modules[oldest_module.__name__]
            
            self.cache[module_name] = module
            self._hit_count += 1
    
    def get_cached_module(self, module_name: str) -> Optional[ModuleType]:
        """Retrieve a cached module."""
        with self.lock:
            if module_name in self.cache:
                # Move to end (most recently used)
                module = self.cache.pop(module_name)
                self.cache[module_name] = module
                self._hit_count += 1
                return module
            self._miss_count += 1
            return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            return {
                'indexed_modules': len(self.index),
                'cached_modules': len(self.cache),
                'cache_hits': self._hit_count,
                'cache_misses': self._miss_count,
                'hit_ratio': self._hit_count / max(1, self._hit_count + self._miss_count)
            }
    
    def clear_cache(self) -> None:
        """Clear the module cache."""
        with self.lock:
            for module_name, module in self.cache.items():
                if hasattr(module, '__name__') and module.__name__ in sys.modules:
                    del sys.modules[module.__name__]
            self.cache.clear()


class DynamicModuleFactory:
    """Factory for creating and injecting code into modules at runtime."""
    
    @staticmethod
    def create_module(
        module_name: str, 
        module_code: str, 
        main_module_path: str = None,
        module_globals: Dict[str, Any] = None
    ) -> Optional[ModuleType]:
        """
        Dynamically create a module with specified name and inject code.
        
        Args:
            module_name: Name of the module to create
            module_code: Source code to inject into the module
            main_module_path: File path of the main module (for __file__ attribute)
            module_globals: Additional globals to inject into module namespace
            
        Returns:
            The dynamically created module, or None if creation fails
        """
        try:
            # Create the module
            dynamic_module = ModuleType(module_name)
            
            # Set standard module attributes
            dynamic_module.__file__ = main_module_path or "runtime_generated"
            dynamic_module.__package__ = module_name
            dynamic_module.__path__ = None
            dynamic_module.__doc__ = f"Dynamically generated module: {module_name}"
            
            # Inject additional globals if provided
            if module_globals:
                dynamic_module.__dict__.update(module_globals)
            
            # Execute the code in the module's namespace
            exec(module_code, dynamic_module.__dict__)
            
            # Register in sys.modules
            sys.modules[module_name] = dynamic_module
            
            return dynamic_module
            
        except Exception as e:
            print(f"Error creating module '{module_name}': {e}")
            traceback.print_exc()
            return None
    
    @staticmethod
    def create_from_template(
        module_name: str,
        template_code: str,
        replacements: Dict[str, str] = None,
        **kwargs
    ) -> Optional[ModuleType]:
        """Create a module from a template with string replacements."""
        if replacements:
            for placeholder, replacement in replacements.items():
                template_code = template_code.replace(f"{{{placeholder}}}", replacement)
        
        return DynamicModuleFactory.create_module(module_name, template_code, **kwargs)


# === Introspection System ===

class ModuleIntrospector:
    """Deep introspection utilities for modules and files."""
    
    def __init__(self, hash_algorithm: str = 'sha256'):
        self.hash_algorithm = hash_algorithm
        self._validate_hash_algorithm()
    
    def _validate_hash_algorithm(self) -> None:
        """Validate the hash algorithm is supported."""
        try:
            hashlib.new(self.hash_algorithm)
        except ValueError:
            raise ValueError(f"Unsupported hash algorithm: {self.hash_algorithm}")
    
    def _get_hasher(self):
        """Get a new hasher instance."""
        return hashlib.new(self.hash_algorithm)
    
    def get_file_metadata(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Collect comprehensive metadata about a file."""
        filepath = Path(filepath)
        
        try:
            stat = filepath.stat()
            content = filepath.read_bytes()
            
            hasher = self._get_hasher()
            hasher.update(content)
            
            return {
                "path": str(filepath),
                "filename": filepath.name,
                "stem": filepath.stem,
                "suffix": filepath.suffix,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
                "accessed": stat.st_atime,
                "hash": hasher.hexdigest(),
                "is_python": filepath.suffix == '.py',
                "is_text": self._is_text_file(filepath),
                "line_count": len(content.decode('utf-8', errors='ignore').splitlines()) if content else 0
            }
        except (FileNotFoundError, PermissionError, OSError) as e:
            return {
                "path": str(filepath),
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _is_text_file(self, filepath: Path) -> bool:
        """Check if a file is likely a text file."""
        text_extensions = {'.py', '.txt', '.md', '.rst', '.yaml', '.yml', '.json', '.xml', '.html', '.css', '.js'}
        return filepath.suffix.lower() in text_extensions
    
    def find_file_groups(
        self, 
        base_path: Union[str, Path], 
        max_depth: int = 2, 
        file_filter: Optional[Callable[[str], bool]] = None,
        include_singles: bool = False
    ) -> Dict[str, Set[str]]:
        """
        Group files by their content hash with configurable filtering.
        
        Args:
            base_path: Root directory to search
            max_depth: Maximum directory depth to traverse
            file_filter: Function to filter files (receives filename)
            include_singles: Whether to include groups with only one file
            
        Returns:
            Dictionary mapping content hashes to sets of file paths
        """
        base_path = Path(base_path)
        groups: Dict[str, Set[str]] = {}
        
        if file_filter is None:
            file_filter = lambda f: f.endswith(('.py', '.txt', '.md'))
        
        try:
            for root, dirs, files in os.walk(base_path):
                # Calculate and limit depth
                depth = len(Path(root).relative_to(base_path).parts)
                if depth > max_depth:
                    continue
                
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'venv'}]
                
                for file in files:
                    if not file_filter(file):
                        continue
                    
                    filepath = Path(root) / file
                    
                    try:
                        content = filepath.read_bytes()
                        hasher = self._get_hasher()
                        hasher.update(content)
                        hash_code = hasher.hexdigest()
                        
                        if hash_code not in groups:
                            groups[hash_code] = set()
                        groups[hash_code].add(str(filepath))
                        
                    except (PermissionError, IsADirectoryError, OSError) as e:
                        print(f"Could not process file {filepath}: {e}")
                        continue
        
        except Exception as e:
            print(f"Error walking directory {base_path}: {e}")
        
        # Filter out single-file groups if requested
        if not include_singles:
            groups = {h: files for h, files in groups.items() if len(files) > 1}
        
        return groups
    
    def inspect_module(self, module_name: str, deep_inspect: bool = True) -> Dict[str, Any]:
        """
        Comprehensively inspect a Python module.
        
        Args:
            module_name: Name of the module to inspect
            deep_inspect: Whether to perform deep introspection of members
            
        Returns:
            Dictionary containing module inspection results
        """
        try:
            module = importlib.import_module(module_name)
            
            module_info = {
                "name": getattr(module, '__name__', 'Unknown'),
                "file": getattr(module, '__file__', 'Unknown path'),
                "package": getattr(module, '__package__', None),
                "doc": getattr(module, '__doc__', 'No documentation'),
                "spec": str(getattr(module, '__spec__', None)),
                "attributes": {},
                "functions": {},
                "classes": {},
                "constants": {},
                "imports": []
            }
            
            # Get all members
            members = inspect.getmembers(module)
            
            for name, obj in members:
                if name.startswith('__') and name.endswith('__'):
                    continue  # Skip dunder attributes
                
                try:
                    if inspect.isfunction(obj):
                        func_info = {
                            "signature": str(inspect.signature(obj)),
                            "doc": getattr(obj, '__doc__', None),
                            "file": getattr(obj, '__code__', {}).co_filename if hasattr(obj, '__code__') else None,
                            "line": getattr(obj, '__code__', {}).co_firstlineno if hasattr(obj, '__code__') else None
                        }
                        
                        if deep_inspect:
                            try:
                                func_info["source"] = inspect.getsource(obj)
                            except (OSError, TypeError):
                                pass
                        
                        module_info['functions'][name] = func_info
                        
                    elif inspect.isclass(obj):
                        class_info = {
                            "doc": getattr(obj, '__doc__', None),
                            "bases": [base.__name__ for base in obj.__bases__],
                            "methods": [],
                            "properties": []
                        }
                        
                        if deep_inspect:
                            for method_name, method_obj in inspect.getmembers(obj):
                                if not method_name.startswith('_'):
                                    if inspect.ismethod(method_obj) or inspect.isfunction(method_obj):
                                        class_info["methods"].append(method_name)
                                    elif isinstance(method_obj, property):
                                        class_info["properties"].append(method_name)
                        
                        module_info['classes'][name] = class_info
                        
                    elif inspect.ismodule(obj):
                        module_info['imports'].append(name)
                        
                    else:
                        # Constants and other attributes
                        obj_type = type(obj).__name__
                        if obj_type in ('int', 'float', 'str', 'bool', 'list', 'dict', 'tuple'):
                            module_info['constants'][name] = {
                                "type": obj_type,
                                "value": str(obj)[:100]  # Truncate long values
                            }
                        else:
                            module_info['attributes'][name] = {
                                "type": obj_type,
                                "repr": str(obj)[:100]
                            }
                
                except Exception as member_error:
                    print(f"Error processing member {name}: {member_error}")
            
            return module_info
            
        except ImportError as e:
            return {
                "error": f"Could not import module '{module_name}': {e}",
                "error_type": "ImportError"
            }
        except Exception as e:
            return {
                "error": f"Unexpected error inspecting module '{module_name}': {e}",
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }


# === Main Runtime System ===

class ScalableReflectiveRuntime:
    """
    A comprehensive runtime system for dynamic module management,
    introspection, and code generation.
    """
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.module_index = ModuleIndex(config.max_cache_size)
        self.introspector = ModuleIntrospector(config.hash_algorithm)
        self.factory = DynamicModuleFactory()
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Cache and persistence
        self.module_cache_dir = self.config.base_dir / '.runtime_cache'
        self.index_path = self.module_cache_dir / 'module_index.pkl'
        
        # Ensure cache directory exists
        self.module_cache_dir.mkdir(exist_ok=True)
    
    def _load_content(self, path: Path) -> str:
        """Efficiently load file content with optional memory mapping."""
        if not self.config.use_mmap or path.stat().st_size < self.config.chunk_size:
            return path.read_text(encoding='utf-8', errors='replace')
        
        try:
            with open(path, 'r+b') as f:
                with mmap.mmap(f.fileno(), 0) as mm:
                    return mm.read().decode('utf-8', errors='replace')
        except (OSError, ValueError):
            # Fallback to regular reading
            return path.read_text(encoding='utf-8', errors='replace')
    
    def scan_directory(self, callback: Optional[Callable[[ModuleMetadata], None]] = None) -> None:
        """
        Scan the base directory to build the module index.
        
        Args:
            callback: Optional callback function called for each discovered module
        """
        print(f"Scanning directory: {self.config.base_dir}")
        
        for path in self._iter_python_files():
            try:
                metadata = ModuleMetadata.from_path(path)
                self.module_index.add(metadata.module_name, metadata)
                
                if callback:
                    callback(metadata)
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        print(f"Indexed {len(self.module_index.index)} modules")
    
    def _iter_python_files(self):
        """Iterate over Python files in the base directory."""
        for root, dirs, files in os.walk(self.config.base_dir):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if d not in self.config.excluded_dirs]
            
            # Check depth
            depth = len(Path(root).relative_to(self.config.base_dir).parts)
            if depth > self.config.max_scan_depth:
                continue
            
            for file in files:
                if file.endswith('.py'):
                    yield Path(root) / file
    
    def load_module(self, module_name: str) -> Optional[ModuleType]:
        """Load a module with caching support."""
        # Check cache first
        cached_module = self.module_index.get_cached_module(module_name)
        if cached_module:
            return cached_module
        
        # Get metadata
        metadata = self.module_index.get(module_name)
        if not metadata:
            print(f"Module '{module_name}' not found in index")
            return None
        
        # Load and create module
        try:
            content = self._load_content(metadata.original_path)
            module = self.factory.create_module(
                module_name, 
                content,
                str(metadata.original_path)
            )
            
            if module:
                self.module_index.cache_module(module_name, module)
            
            return module
            
        except Exception as e:
            print(f"Error loading module '{module_name}': {e}")
            return None
    
    def create_runtime_module(
        self,
        module_name: str,
        source_code: str,
        persist: bool = False
    ) -> Optional[ModuleType]:
        """
        Create a runtime module and optionally persist it as a .py file.
        
        Args:
            module_name: Name for the new module
            source_code: Python source code for the module
            persist: Whether to write the module to a .py file
            
        Returns:
            The created module or None if creation failed
        """
        module = self.factory.create_module(module_name, source_code)
        
        if module and persist:
            # Write to file for quine-like behavior
            module_file = self.config.base_dir / f"{module_name}.py"
            try:
                module_file.write_text(source_code, encoding='utf-8')
                print(f"Persisted module to {module_file}")
            except Exception as e:
                print(f"Warning: Could not persist module {module_name}: {e}")
        
        return module
    
    def find_duplicates(self) -> Dict[str, Set[str]]:
        """Find duplicate files in the base directory."""
        return self.introspector.find_file_groups(
            self.config.base_dir,
            max_depth=self.config.max_scan_depth,
            file_filter=lambda f: f.endswith(('.py', '.txt', '.md'))
        )
    
    def save_index(self) -> bool:
        """Persist the module index to disk."""
        try:
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.module_index.index, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Index saved to {self.index_path}")
            return True
        except Exception as e:
            print(f"Error saving index: {e}")
            return False
    
    def load_index(self) -> bool:
        """Load a previously saved module index."""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'rb') as f:
                    self.module_index.index = pickle.load(f)
                print(f"Index loaded from {self.index_path}")
                return True
        except Exception as e:
            print(f"Error loading index: {e}")
        return False
    
    def get_runtime_stats(self) -> Dict[str, Any]:
        """Get comprehensive runtime statistics."""
        stats = self.module_index.get_stats()
        stats.update({
            'base_directory': str(self.config.base_dir),
            'cache_directory': str(self.module_cache_dir),
            'config': {
                'max_cache_size': self.config.max_cache_size,
                'max_workers': self.config.max_workers,
                'chunk_size': self.config.chunk_size,
                'hash_algorithm': self.config.hash_algorithm
            }
        })
        return stats
    
    def shutdown(self) -> None:
        """Clean shutdown of the runtime system."""
        self.save_index()
        self.executor.shutdown(wait=True)
        self.module_index.clear_cache()


# === Utility Functions ===

def rpn_call(func: Callable, *args):
    """Execute a function in Reverse Polish Notation (args after function)."""
    @wraps(func)
    def wrapper(*positional_args):
        return func(*reversed(positional_args))
    return wrapper(*args)


def compose(*funcs):
    """Compose multiple functions, applying in reverse order."""
    def composed_func(arg):
        for func in reversed(funcs):
            arg = func(arg)
        return arg
    return composed_func


def identity(x):
    """Identity function for functional programming patterns."""
    return x


def create_quine_module(module_name: str, base_template: str = None, fallback_filename: str = None) -> str:
    """
    Generate a self-replicating module template.
    
    Args:
        module_name: Name for the quine module
        base_template: Optional base template, uses default if None
        fallback_filename: Filename to use in fallback file read
        
    Returns:
        Source code for a self-replicating module
    """
    if base_template is None:
        fallback_file = fallback_filename or f"{module_name}.py"
        base_template = f'''"""
Self-replicating module: {{module_name}}
Generated by Dynamic Runtime System
"""

def replicate():
    """Print the source code of this module.""" 
    import inspect
    import os
    frame = inspect.currentframe()
    module = inspect.getmodule(frame)
    if module is None:
        # fallback: read this file directly
        with open(os.path.abspath("{fallback_file}"), 'r', encoding='utf-8') as f:
            print(f.read())
    else:
        print(inspect.getsource(module))

def get_self():
    """Return the source code of this module as a string."""
    import inspect
    import os
    frame = inspect.currentframe()
    module = inspect.getmodule(frame)
    if module is None:
        # fallback: read this file directly
        with open(os.path.abspath("{fallback_file}"), 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return inspect.getsource(module)

if __name__ == "__main__":
    replicate()
'''
    
    return base_template.format(module_name=module_name)


# === Demo and Testing ===

def demo_runtime_system():
    """Demonstrate the runtime system capabilities."""
    print("=== Dynamic Runtime System Demo ===\n")
    
    # Setup
    config = RuntimeConfig(
        base_dir=Path.cwd(),
        max_cache_size=100,
        max_workers=2
    )
    
    runtime = ScalableReflectiveRuntime(config)
    
    # Try to load existing index
    if not runtime.load_index():
        print("Building new module index...")
        runtime.scan_directory()
        runtime.save_index()
    
    # Show stats
    print("Runtime Statistics:")
    stats = runtime.get_runtime_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # Create a demo quine module
    print("\n=== Creating Demo Quine Module ===")
    quine_code = create_quine_module("demo_quine")
    demo_module = runtime.create_runtime_module("demo_quine", quine_code, persist=True)
    
    if demo_module and hasattr(demo_module, 'get_self'):
        print("Quine module created successfully!")
        print("Self-replication test:")
        print(demo_module.get_self()[:200] + "...")
    
    # Find duplicates
    print("\n=== Finding Duplicate Files ===")
    duplicates = runtime.find_duplicates()
    if duplicates:
        print(f"Found {len(duplicates)} groups of duplicate files:")
        for i, (hash_code, files) in enumerate(list(duplicates.items())[:3], 1):
            print(f"  Group {i} (hash: {hash_code[:10]}...):")
            for file in files:
                print(f"    - {file}")
    else:
        print("No duplicate files found.")
    
    # Module introspection example
    print("\n=== Module Introspection ===")
    try:
        json_info = runtime.introspector.inspect_module('json', deep_inspect=False)
        if 'error' not in json_info:
            print(f"Inspected 'json' module:")
            print(f"  Functions: {len(json_info.get('functions', {}))}")
            print(f"  Classes: {len(json_info.get('classes', {}))}")
            print(f"  Constants: {len(json_info.get('constants', {}))}")
        else:
            print(f"Error inspecting 'json' module: {json_info['error']}")
    except Exception as e:
        print(f"Introspection failed: {e}")
    
    # Cleanup
    runtime.shutdown()
    print("\n=== Demo Complete ===")


def main():
    """Main entry point."""
    demo_runtime_system()


if __name__ == "__main__":
    main()