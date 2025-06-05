# Imports
import ast
import datetime
import functools
import hashlib
import inspect
import json
import marshal
import sqlite3
import threading
import time
import types
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Type,
                    TypeVar, Union, Set)

# Type Variables
T = TypeVar('T')
R = TypeVar('R')

# Sentinel for missing values in transducers
_MISSING = object()


# --- Group 1: Logical MRO Analysis ---

class LogicalMRO:
    """
    Analyzes class hierarchies, MRO, and super() calls to build a
    logical representation of method resolution.
    """
    def __init__(self):
        self.mro_data: Dict[str, Any] = {}

    def _analyze_super_calls(self, method: Callable) -> List[Dict]:
        super_calls = []
        if not callable(method) or inspect.isbuiltin(method):
            return super_calls
        try:
            source = inspect.getsource(method)
            tree = ast.parse(source)

            class SuperVisitor(ast.NodeVisitor):
                def visit_Call(self, node: ast.Call):
                    # Covers super().method_name(...)
                    if isinstance(node.func, ast.Attribute) and \
                       isinstance(node.func.value, ast.Call) and \
                       isinstance(node.func.value.func, ast.Name) and \
                       node.func.value.func.id == 'super':
                        super_calls.append({
                            "line": node.lineno,
                            "method": node.func.attr,
                            "type": "explicit_args" if node.func.value.args else "implicit_no_args"
                        })
                    # Covers super(CurrentClass, self).method_name(...)
                    # This is a simplified check; full analysis of super() args is complex.
                    elif isinstance(node.func, ast.Attribute) and \
                         isinstance(node.func.value, ast.Name) and \
                         node.func.value.id == 'super':
                         # This is a basic super() call like `super()`, not `super().method()`
                         # This part might need refinement if it's intended for `super.method()`
                         # The AST for `super()` itself (without method call) is just `ast.Call(func=ast.Name(id='super'))`
                         pass # Already handled by the next elif for general super() calls

                    # Covers `super()` if it's the direct function being called (less common for method calls)
                    elif isinstance(node.func, ast.Name) and node.func.id == 'super':
                         # This typically means `s = super()` then `s.method()`.
                         # The method call itself would be `ast.Call(func=ast.Attribute(value=ast.Name(id='s')))`
                         # For direct `super().foo()` this is handled by the first case.
                         # If it's just `super()`, it might not have `attr`.
                        super_calls.append({
                            "line": node.lineno,
                            "method": "N/A (super() assignment or direct call)",
                            "type": "implicit_no_args" # or "explicit_args" if node.args
                        })
                    self.generic_visit(node)

            SuperVisitor().visit(tree)
        except (TypeError, OSError, SyntaxError): # inspect.getsource can raise TypeError/OSError
            # Could log this error
            pass # Keep super_calls empty if source is unavailable/unparsable
        return super_calls

    def _encode_class_info(self, cls: Type) -> Dict:
        methods = {}
        for name, member in inspect.getmembers(cls):
            # We only want methods defined directly in this class for its own "methods" dict
            if callable(member) and name in cls.__dict__:
                methods[name] = {
                    "defined_in": cls.__name__,
                    "super_calls": self._analyze_super_calls(member)
                }
        return {
            "name": cls.__name__,
            "mro": [c.__name__ for c in cls.__mro__],
            "methods": methods
        }

    def analyze_classes(self, *classes: Type) -> Dict:
        """
        Analyzes the given classes and populates `self.mro_data`.
        Returns the generated MRO data.
        """
        analysis_result = {
            "classes": {},
            "method_dispatch": {} # Simplified: actual dispatch depends on MRO path
        }

        all_analyzed_classes = set()
        for cls in classes:
            # Analyze cls and all its bases to get a full picture
            for mro_cls in cls.__mro__:
                if mro_cls not in all_analyzed_classes and mro_cls is not object:
                    class_info = self._encode_class_info(mro_cls)
                    analysis_result["classes"][mro_cls.__name__] = class_info
                    all_analyzed_classes.add(mro_cls)

                    for method_name, method_info in class_info["methods"].items():
                        # Construct a simplified resolution path for methods defined in this class
                        path_key = f"{mro_cls.__name__}.{method_name}"
                        analysis_result["method_dispatch"][path_key] = {
                            "initial_definition": mro_cls.__name__,
                            "resolution_path_from_here": [
                                base.__name__ for base in mro_cls.__mro__
                                if hasattr(base, method_name) and method_name in base.__dict__
                            ],
                            "super_calls_in_this_impl": method_info["super_calls"]
                        }
        self.mro_data = analysis_result
        return analysis_result

    def __repr__(self) -> str:
        if not self.mro_data or "classes" not in self.mro_data:
            return "LogicalMRO (No MRO data analyzed. Call analyze_classes(your_class).)"

        s_expressions = []
        for cls_name, cls_info in self.mro_data["classes"].items():
            methods_s_expr = []
            if "methods" in cls_info: # Ensure methods key exists
                for name, info in cls_info["methods"].items():
                    super_calls_s_expr = ' '.join([
                        f"(super_call_to: {call['method']} type: {call['type']})"
                        for call in info.get('super_calls', [])
                    ])
                    methods_s_expr.append(f"(method {name} {super_calls_s_expr})")
            
            mro_list = cls_info.get("mro", [])
            s_expressions.append(
                f"(class {cls_name} (mro {' '.join(mro_list)}) {' '.join(methods_s_expr)})"
            )
        return "\n".join(s_expressions)


# --- Group 2: Temporal Code & Execution Framework ---

@dataclass
class TemporalCode:
    """Represents code that can be serialized, stored, and potentially re-executed."""
    source: bytes  # Marshalled code object
    ttl: int
    emanation_time: datetime.datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_function(cls, func: Callable, ttl: int, metadata: Optional[Dict[str, Any]] = None):
        """Create TemporalCode from a function."""
        if not hasattr(func, '__code__'):
            raise TypeError("Input 'func' must be a function with a '__code__' attribute.")

        code_obj = func.__code__
        marshalled_code = marshal.dumps(code_obj)
        
        func_metadata = {'name': func.__name__}
        # Store number of free variables; crucial for recreating functions with closures
        if code_obj.co_freevars > 0:
            func_metadata['closure_vars_count'] = code_obj.co_freevars
        
        if metadata:
            func_metadata.update(metadata)

        return cls(
            source=marshalled_code,
            ttl=ttl,
            emanation_time=datetime.datetime.now(),
            metadata=func_metadata
        )

    def instantiate(self) -> Callable:
        """Turn serialized code back into a callable function."""
        code_obj = marshal.loads(self.source)
        
        closure = None
        # If the original function was a closure, create dummy cell objects.
        # The actual cell *values* are not part of the marshalled code object.
        # This allows the function to be reconstructed if its logic depends on
        # the structure of a closure, but not necessarily pre-filled values from original context.
        if 'closure_vars_count' in self.metadata:
            closure = tuple(types.CellType(None) for _ in range(self.metadata['closure_vars_count']))

        # globals() here refers to the globals of *this current module* where instantiate is called.
        # This is a common default; specific applications might need a different global context.
        func = types.FunctionType(code_obj, globals(), 
                                  self.metadata.get('name', 'temporal_func'), 
                                  None,  # Default args
                                  closure) # Closure tuple
        return func

class TemporalJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

class TemporalStore:
    """Handles storage and retrieval of TemporalCode instances in an SQLite database."""
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.db = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
        self._lock = threading.Lock() # For thread-safe operations on the DB connection
        
        # Register adapters for datetime (bi-directional)
        sqlite3.register_adapter(datetime.datetime, lambda dt_obj: dt_obj.isoformat())
        sqlite3.register_converter("TIMESTAMP", lambda ts_bytes: datetime.datetime.fromisoformat(ts_bytes.decode()))
        
        self._setup_database()

    def _setup_database(self):
        with self._lock, self.db:
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS temporal_code (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_hash TEXT UNIQUE NOT NULL, -- Ensures code is stored once
                source BLOB NOT NULL,
                ttl INTEGER NOT NULL,
                emanation_time TIMESTAMP NOT NULL,
                metadata TEXT, -- JSON string
                mro_path TEXT -- Optional, for MRO-related temporal logic
            )""")
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS emanation_graph (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hash TEXT NOT NULL, -- Hash of the source code
                target_hash TEXT NOT NULL, -- Hash of the target (emanated) code
                emanation_time TIMESTAMP NOT NULL
                -- Optional: FOREIGN KEYs to temporal_code(code_hash) if needed,
                -- but direct hash storage is also viable.
            )""")
            # Consider adding indexes on code_hash and emanation_time for performance

    def store_code(self, temporal_code: TemporalCode, mro_path: Optional[str] = None) -> Optional[int]:
        """
        Stores TemporalCode in the database. Returns the ID of the inserted row or None on failure.
        Uses code_hash to prevent duplicate storage of the same code bytes.
        """
        code_hash = hashlib.sha256(temporal_code.source).hexdigest()
        metadata_json = json.dumps(temporal_code.metadata, cls=TemporalJSONEncoder)
        
        with self._lock, self.db:
            try:
                cursor = self.db.execute("""
                INSERT OR IGNORE INTO temporal_code 
                       (code_hash, source, ttl, emanation_time, metadata, mro_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    code_hash,
                    temporal_code.source,
                    temporal_code.ttl,
                    temporal_code.emanation_time,
                    metadata_json,
                    mro_path or temporal_code.metadata.get('mro_path') # Prioritize arg then metadata
                ))
                if cursor.rowcount > 0:
                    return cursor.lastrowid
                else: # Row was ignored (already exists), fetch its ID
                    cursor = self.db.execute("SELECT id FROM temporal_code WHERE code_hash = ?", (code_hash,))
                    row = cursor.fetchone()
                    return row[0] if row else None
            except sqlite3.Error as e:
                print(f"Error storing temporal code: {e}")
                return None

    def store_emanation(self, source_code: TemporalCode, target_code: TemporalCode) -> Optional[int]:
        """Stores an emanation relationship between two TemporalCode instances."""
        source_hash = hashlib.sha256(source_code.source).hexdigest()
        target_hash = hashlib.sha256(target_code.source).hexdigest()
        
        with self._lock, self.db:
            try:
                cursor = self.db.execute("""
                INSERT INTO emanation_graph (source_hash, target_hash, emanation_time)
                VALUES (?, ?, ?)
                """, (source_hash, target_hash, target_code.emanation_time))
                return cursor.lastrowid
            except sqlite3.Error as e:
                print(f"Error storing emanation: {e}")
                return None

    def get_code_by_hash(self, code_hash: str) -> Optional[TemporalCode]:
        """Retrieves TemporalCode by its hash."""
        with self._lock, self.db:
            cursor = self.db.execute(
                "SELECT source, ttl, emanation_time, metadata FROM temporal_code WHERE code_hash = ?",
                (code_hash,)
            )
            row = cursor.fetchone()
        if row:
            source, ttl, emanation_time, metadata_json = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            return TemporalCode(source, ttl, emanation_time, metadata)
        return None

    def close(self):
        with self._lock:
            self.db.close()

class TemporalMRO:
    """
    Manages temporal aspects of method execution, including code emanation.
    This is distinct from LogicalMRO (static analysis) and profile_class_methods (performance).
    """
    def __init__(self, store: TemporalStore):
        self.store = store
        self._emanation_lock = threading.Lock() # For operations involving emanation logic

    def _get_calling_function(self, frame_depth: int = 2) -> Optional[Callable]:
        """
        Tries to get the original function that was decorated by temporal_method.
        `frame_depth=1` is `_get_calling_function` itself.
        `frame_depth=2` is `temporal_context`.
        `frame_depth=3` is the `wrapper` inside `temporal_method`.
        The original function `func` is in the closure of `wrapper`.
        """
        try:
            # frame of temporal_context -> frame of wrapper -> original func in wrapper's closure
            frame = inspect.currentframe()
            for _ in range(frame_depth): # Go up to the wrapper's frame
                if frame.f_back:
                    frame = frame.f_back
                else:
                    return None # Not enough frames
            
            # 'func' is the name used in the temporal_method decorator for the original function
            original_func = frame.f_locals.get('func') 
            if callable(original_func):
                return original_func
        except Exception as e:
            print(f"Error introspecting calling function: {e}")
        return None

    def temporal_context(self, ttl: int, metadata: Optional[Dict[str,Any]] = None) -> Optional[TemporalCode]:
        """
        Captures the current executing function (expected to be decorated) as TemporalCode.
        """
        with self._emanation_lock:
            # The calling function is the 'wrapper' from 'temporal_method'.
            # We need the original 'func' that 'wrapper' closes over.
            original_func = self._get_calling_function(frame_depth=3)
            
            if original_func:
                temporal_code = TemporalCode.from_function(original_func, ttl, metadata)
                self.store.store_code(temporal_code) # Store its initial state
                return temporal_code
            else:
                print("Warning: Could not identify original function for temporal_context.")
                return None

    def emanate(self, code: TemporalCode, new_metadata_update: Optional[Dict[str, Any]] = None) -> Optional[TemporalCode]:
        """
        Propagates (emanates) a TemporalCode instance, typically with decremented TTL.
        Returns the new TemporalCode instance if emanation occurs.
        """
        if code.ttl <= 0:
            return None # TTL exhausted

        with self._emanation_lock:
            new_meta = code.metadata.copy()
            if new_metadata_update:
                new_meta.update(new_metadata_update)
            
            # Create new instance with decremented TTL
            new_temporal_code = TemporalCode(
                source=code.source,
                ttl=code.ttl - 1,
                emanation_time=datetime.datetime.now(),
                metadata=new_meta
            )
            
            # Store the new code version and the emanation relationship
            self.store.store_code(new_temporal_code)
            self.store.store_emanation(code, new_temporal_code)
            
            return new_temporal_code

def temporal_method(ttl: int, initial_metadata: Optional[Dict[str,Any]] = None):
    """
    Decorator for methods to make them operate within a TemporalMRO context.
    Assumes the instance (`self`) has a `_temporal_mro` attribute of type TemporalMRO.
    """
    def decorator(func: Callable): # func is the original method
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, '_temporal_mro') or not isinstance(self._temporal_mro, TemporalMRO):
                raise AttributeError(f"Instance {self} using @temporal_method lacks a '_temporal_mro' attribute of type TemporalMRO.")
            
            mro_ctx: TemporalMRO = self._temporal_mro
            
            # Capture current execution as temporal code
            # Metadata for this specific call context
            call_metadata = (initial_metadata or {}).copy()
            call_metadata.update({
                'call_args_repr': repr(args), # repr for safety
                'call_kwargs_repr': repr(kwargs)
            })
            temporal_code_instance = mro_ctx.temporal_context(ttl, metadata=call_metadata)
            
            # Execute original method logic
            result = func(self, *args, **kwargs)
            
            if temporal_code_instance:
                # Update metadata of the stored code with execution results (optional)
                temporal_code_instance.metadata['execution_result_repr'] = repr(result)
                # Re-store to update metadata (store_code handles INSERT OR IGNORE logic,
                # but for metadata update, explicit UPDATE might be better or versioning)
                # For simplicity, we assume metadata updates are less frequent or handled by re-storing.
                # A more robust system might version metadata or have specific update paths.
                # For now, we rely on the fact that the hash is the same, so store_code might ignore it.
                # If metadata needs frequent updates for an existing hash, TemporalStore needs an update method.
                # Let's assume for now the initial store is primary.
                pass # Metadata updated on local obj; re-store if needed based on policy.

            return result
        return wrapper
    return decorator

class EventDrivenTemporalStore(TemporalStore):
    """Extends TemporalStore with an event queue for processing TemporalCode."""
    def __init__(self, db_path: str = ":memory:"):
        super().__init__(db_path)
        self.event_queue: List[Any] = [] # Can store TemporalCode or other event types
        self._event_lock = threading.Lock() # For thread-safe access to event_queue

    def trigger_event(self, event: Any):
        with self._event_lock:
            self.event_queue.append(event)
            print(f"Event triggered: {type(event).__name__}, Queue size: {len(self.event_queue)}")

    def process_events(self, max_events: int = -1):
        """Processes events from the queue. max_events=-1 means process all."""
        processed_count = 0
        while True:
            with self._event_lock:
                if not self.event_queue or (max_events != -1 and processed_count >= max_events):
                    break
                event = self.event_queue.pop(0)
            
            print(f"Processing event: {type(event).__name__}...")
            self.execute_event(event)
            processed_count += 1
        print(f"Finished processing. Total events processed in this run: {processed_count}")

    def execute_event(self, event: Any):
        """Handles execution of an event, specifically for TemporalCode events."""
        if isinstance(event, TemporalCode):
            print(f"Executing TemporalCode: {event.metadata.get('name', 'N/A')}, TTL: {event.ttl}")
            if event.ttl <= 0:
                print("TemporalCode TTL exhausted, not executing.")
                return

            func_to_execute = event.instantiate()
            
            # For temporal methods, they expect 'self' with '_temporal_mro'.
            # We create a dynamic instance to satisfy this for event-driven execution.
            # This is a simplified context. Real applications might need more complex state.
            
            # Construct a 'self' context for the function if its name implies it's a method
            # This is heuristic. A more robust way would be to store if it's a method in metadata.
            is_method_like = 'self' in inspect.signature(func_to_execute).parameters

            if is_method_like:
                # Create a minimal 'self' context for the method
                # The TemporalMRO instance here uses 'self' (the EventDrivenTemporalStore) as its store.
                # This allows emanated methods to interact with the same event system.
                temporal_mro_for_event = TemporalMRO(store=self)

                # The 'TemporalInstance' will be the 'self' for the executed method
                # It needs the _temporal_mro attribute that temporal_method expects
                # It also needs an attribute corresponding to the method name itself
                method_name = event.metadata.get('name', 'unknown_method')

                # Prepare attributes for the dynamic instance
                instance_attrs = {
                    '_temporal_mro': temporal_mro_for_event,
                    'event_metadata': event.metadata, # Pass along event metadata
                    # Bind the function as a method to this instance
                    method_name: types.MethodType(func_to_execute, None) # Unbound initially
                }
                
                # Create the dynamic type and instance
                DynamicInstanceType = type('TemporalEventInstance', (object,), instance_attrs)
                instance = DynamicInstanceType()
                
                # Now correctly bind the method to the instance
                bound_method = types.MethodType(func_to_execute, instance)
                setattr(instance, method_name, bound_method)

                try:
                    print(f"  Invoking method '{method_name}' on dynamic instance...")
                    # Args/kwargs might need to be reconstructed from metadata if required by func
                    # For simplicity, calling without args here.
                    # Original args/kwargs could be stored in event.metadata if needed.
                    bound_method() 
                except Exception as e:
                    print(f"Error executing event method {method_name}: {e}")
            else: # Simple function, not a method
                try:
                    print(f"  Invoking function '{func_to_execute.__name__}'...")
                    func_to_execute()
                except Exception as e:
                    print(f"Error executing event function {func_to_execute.__name__}: {e}")
        else:
            print(f"Skipping non-TemporalCode event: {type(event)}")


# --- Group 3: Transducer Implementation ---

class Reduced:
    """Sentinel to signal early termination in a reduction."""
    def __init__(self, value: Any):
        self.value = value

def ensure_reduced(x: Any) -> Union[Any, Reduced]:
    """Wraps x in Reduced if not already."""
    return x if isinstance(x, Reduced) else Reduced(x)

def unreduced(x: Any) -> Any:
    """Unwraps a Reduced value; returns x if not Reduced."""
    return x.value if isinstance(x, Reduced) else x

def custom_reduce(
    reducing_fn: Callable[[Any, T], Any],
    collection: Iterable[T],
    initializer: Any = _MISSING
) -> Any:
    """
    Custom reduce implementation supporting early termination with Reduced.
    The reducing_fn should handle 3 arities:
    - reducing_fn() -> initial accumulator (if initializer is _MISSING)
    - reducing_fn(accumulator) -> final result transformation (completion step)
    - reducing_fn(accumulator, item) -> new accumulator (reduction step)
    """
    it = iter(collection)
    if initializer is _MISSING:
        try:
            # Arity 0 call for initial value if not provided
            accumulator = reducing_fn() 
        except TypeError: # If reducing_fn doesn't support 0-arity for init
            try:
                accumulator = next(it) # Fallback to first item
            except StopIteration:
                raise TypeError("custom_reduce() of empty sequence with no initial value and no 0-arity reducing_fn")
    else:
        accumulator = initializer

    for item in it:
        accumulator = reducing_fn(accumulator, item)
        if isinstance(accumulator, Reduced):
            accumulator = accumulator.value # unwrap for final result
            break
    
    # Arity 1 call for completion/finalization
    return reducing_fn(accumulator)


class Transducer:
    """Base class for transducers."""
    def __init__(self, transform_wiring_fn: Callable[[Callable], Callable]):
        """
        Args:
            transform_wiring_fn: A function that takes a 'downstream' reducing function
                                 and returns a 'new' reducing function that incorporates
                                 this transducer's logic.
                                 E.g., for Map(f), this is _map_wiring_fn.
        """
        self.transform_wiring_fn = transform_wiring_fn

    def __call__(self, downstream_reducing_fn: Callable) -> Callable:
        """
        Applies this transducer's transformation to a downstream reducing function.
        Returns a new reducing function.
        """
        return self.transform_wiring_fn(downstream_reducing_fn)

class Map(Transducer):
    """Transducer for applying a function to each element."""
    def __init__(self, func: Callable[[T], R]):
        self.func = func
        def _map_wiring_fn(downstream_rf: Callable) -> Callable:
            # downstream_rf is the next step (e.g., Filter's step, or final append)
            @functools.wraps(downstream_rf) # Preserve metadata if downstream_rf is wrapped
            def new_reducing_fn(acc=_MISSING, item=_MISSING):
                if acc is _MISSING: # Arity 0 (init)
                    return downstream_rf()
                if item is _MISSING: # Arity 1 (completion)
                    return downstream_rf(acc)
                # Arity 2 (step)
                return downstream_rf(acc, self.func(item))
            return new_reducing_fn
        super().__init__(_map_wiring_fn)

class Filter(Transducer):
    """Transducer for filtering elements based on a predicate."""
    def __init__(self, predicate: Callable[[T], bool]):
        self.predicate = predicate
        def _filter_wiring_fn(downstream_rf: Callable) -> Callable:
            @functools.wraps(downstream_rf)
            def new_reducing_fn(acc=_MISSING, item=_MISSING):
                if acc is _MISSING: # Arity 0
                    return downstream_rf()
                if item is _MISSING: # Arity 1
                    return downstream_rf(acc)
                # Arity 2
                return downstream_rf(acc, item) if self.predicate(item) else acc
            return new_reducing_fn
        super().__init__(_filter_wiring_fn)

class Cat(Transducer):
    """
    Transducer for concatenating/flattening collections produced by a previous step.
    Often used with Map in `mapcat`.
    """
    def __init__(self):
        def _cat_wiring_fn(downstream_rf: Callable) -> Callable:
            @functools.wraps(downstream_rf)
            def new_reducing_fn(acc=_MISSING, sub_collection=_MISSING):
                if acc is _MISSING: # Arity 0
                    return downstream_rf()
                if sub_collection is _MISSING: # Arity 1
                    return downstream_rf(acc)
                # Arity 2: sub_collection is an iterable (e.g., from Map(lambda x: [x, x+1]))
                # Reduce this sub_collection using the downstream_rf
                current_acc = acc
                if isinstance(sub_collection, str) or not isinstance(sub_collection, Iterable):
                    # If it's a string or not iterable, pass it as a single item
                    # This behavior might need adjustment based on desired mapcat outcome
                    # For typical mapcat, sub_collection is expected to be an inner collection
                     current_acc = downstream_rf(current_acc, sub_collection)
                else:
                    for item_in_sub in sub_collection:
                        current_acc = downstream_rf(current_acc, item_in_sub)
                        if isinstance(current_acc, Reduced): # Propagate early termination
                            break
                return current_acc
            return new_reducing_fn
        super().__init__(_cat_wiring_fn)

def compose(*functions: Callable) -> Callable:
    """Composes functions from right to left (f(g(h(x))))."""
    if not functions:
        return lambda x: x # Identity function
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)

def transduce(
    xform: Transducer, # The transducer (or composed transducer chain)
    reducing_fn: Callable, # The final reduction logic (e.g., append_to_list)
    initial_value: Any, # Initial accumulator for the reduction
    collection: Iterable[T]
) -> Any:
    """
    Applies a transducer `xform` to a `collection`, using `reducing_fn`
    for accumulation, starting with `initial_value`.
    """
    # The transducer decorates the reducing_fn
    transformed_reducing_fn = xform(reducing_fn)
    return custom_reduce(transformed_reducing_fn, collection, initial_value)

def append_to_list(acc: List = _MISSING, item: Any = _MISSING) -> List:
    """A standard reducing function for building lists."""
    if acc is _MISSING: # Arity 0 (init)
        return []
    if item is _MISSING: # Arity 1 (completion)
        return acc 
    # Arity 2 (step)
    acc.append(item)
    return acc

def into(target_container_factory: Callable[[], Union[List, Set, Dict]], # e.g. list, set
         xform: Transducer, 
         collection: Iterable[T]) -> Union[List, Set, Dict]:
    """
    Transduces a collection and pours results into a new target container.
    Uses a reducing function appropriate for the target type.
    """
    # For simplicity, only list appending is fully shown here.
    # Set would use `acc.add(item)`, dict `acc.update(item)` or `acc[k]=v`.
    if target_container_factory is list:
        rf = append_to_list
        initial_val = rf() # Get initial empty list []
    elif target_container_factory is set:
        def append_to_set(acc=_MISSING, item=_MISSING):
            if acc is _MISSING: return set()
            if item is _MISSING: return acc
            acc.add(item)
            return acc
        rf = append_to_set
        initial_val = rf()
    else:
        raise TypeError(f"Unsupported target container factory: {target_container_factory}")
    
    return transduce(xform, rf, initial_val, collection)

def mapcat(func_producing_iterable: Callable[[T], Iterable[R]]) -> Transducer:
    """A transducer that maps `f` over items and concatenates the resulting iterables."""
    # mapcat(f) is equivalent to compose(Map(f), Cat())
    # Cat flattens the iterables produced by Map(f)
    return compose(Map(func_producing_iterable), Cat())


# --- Group 4: Method Profiling Utilities ---

def profile_class_methods(cls: Type) -> Type:
    """
    Class decorator to profile method execution times.
    Stores profiling data in `cls._profiling_data_`. Modifies class in-place.
    """
    # Ensure the storage attribute exists on the class
    if not hasattr(cls, '_profiling_data_'):
        cls._profiling_data_: Dict[str, List[Dict[str, Any]]] = {}

    for attr_name, attr_value in list(cls.__dict__.items()): # Iterate copy for modification
        if callable(attr_value) and not attr_name.startswith("__"):
            original_method = attr_value

            # Handle staticmethod, classmethod, or regular method
            if isinstance(original_method, staticmethod):
                unwrapped_method = original_method.__func__
                @functools.wraps(unwrapped_method)
                def static_wrapper(*args, __orig_method=unwrapped_method, **kwargs):
                    start_time = time.perf_counter_ns()
                    result = __orig_method(*args, **kwargs)
                    end_time = time.perf_counter_ns()
                    duration_ns = end_time - start_time
                    cls._profiling_data_.setdefault(attr_name, []).append({
                        'type': 'staticmethod', 'duration_ns': duration_ns,
                        'args_repr': repr(args), 'kwargs_repr': repr(kwargs)
                    })
                    return result
                # Wrap the wrapper as staticmethod to avoid passing self implicitly
                setattr(cls, attr_name, staticmethod(static_wrapper))

            elif isinstance(original_method, classmethod):
                unwrapped_method = original_method.__func__
                @functools.wraps(unwrapped_method)
                def class_wrapper(c, *args, __orig_method=unwrapped_method, **kwargs): # c is the class
                    start_time = time.perf_counter_ns()
                    result = __orig_method(c, *args, **kwargs)
                    end_time = time.perf_counter_ns()
                    duration_ns = end_time - start_time
                    cls._profiling_data_.setdefault(attr_name, []).append({
                        'type': 'classmethod', 'duration_ns': duration_ns,
                        'args_repr': repr(args), 'kwargs_repr': repr(kwargs)
                    })
                    return result
                setattr(cls, attr_name, classmethod(class_wrapper))
            
            else: # Regular instance method (or function assigned to class)
                @functools.wraps(original_method)
                def instance_wrapper(self_or_cls, *args, __orig_method=original_method, **kwargs): # self_or_cls is the instance
                    start_time = time.perf_counter_ns()
                    result = __orig_method(self_or_cls, *args, **kwargs)
                    end_time = time.perf_counter_ns()
                    duration_ns = end_time - start_time
                    cls._profiling_data_.setdefault(attr_name, []).append({
                        'type': 'instancemethod', 'duration_ns': duration_ns,
                        'args_repr': repr(args), 'kwargs_repr': repr(kwargs)
                    })
                    return result
                setattr(cls, attr_name, instance_wrapper)
    return cls


# --- Group 5: Demonstrative main() function ---

def main():
    print("=" * 50)
    print("### 1. LogicalMRO Demonstration ###")
    print("=" * 50)

    class A:
        def m(self): print("A.m"); super().m()
        def n(self): print("A.n")

    class B(A):
        def m(self): print("B.m"); super().m()

    class C(A):
        def m(self): print("C.m"); super().m() # Will call A.m
        def n(self): print("C.n"); super().n() # Will call A.n

    class D(B, C):
        def m(self): print("D.m"); super().m() # Will call B.m
        def o(self): print("D.o")

    # D.m() -> B.m() -> C.m() -> A.m() -> object.m (if it exists)
    # print("MRO for D:", [cls.__name__ for cls in D.__mro__])
    # D().m() # Test call

    mro_analyzer = LogicalMRO()
    mro_analyzer.analyze_classes(D, B, C, A) # Analyze D and its hierarchy
    print(mro_analyzer) # Uses the __repr__
    # You can also access mro_analyzer.mro_data directly for the raw dictionary

    print("\n" + "=" * 50)
    print("### 2. Temporal Code & Execution Demonstration ###")
    print("=" * 50)

    # Using EventDrivenTemporalStore for this demo
    # db_file = "temporal_demo.sqlite"
    # if os.path.exists(db_file): os.remove(db_file) # Clean start for demo
    # event_store = EventDrivenTemporalStore(db_path=db_file)
    event_store = EventDrivenTemporalStore(db_path=":memory:") # In-memory for simplicity
    
    temporal_manager = TemporalMRO(store=event_store)

    class MyTemporalAgent:
        def __init__(self, name: str, tmro: TemporalMRO):
            self.name = name
            self._temporal_mro = tmro # Required by @temporal_method
            self.counter = 0

        @temporal_method(ttl=3, initial_metadata={'agent_name': 'Agent007'})
        def perform_action(self, task_id: str):
            self.counter += 1
            print(f"Agent '{self.name}' (Counter: {self.counter}) performing action for Task '{task_id}'. TTL of this context might be decreasing.")
            
            # Current temporal code instance (captured by @temporal_method)
            current_code_context: Optional[TemporalCode] = self._temporal_mro.temporal_context(ttl=0) # Get current context without re-storing

            if current_code_context and current_code_context.ttl > 0:
                print(f"  Action has TTL {current_code_context.ttl}. Will try to emanate for next step.")
                # Emanate: create a new version of this code (perform_action) with reduced TTL
                new_code = self._temporal_mro.emanate(current_code_context, {'emanated_from_task': task_id})
                if new_code:
                    print(f"  Emanated new TemporalCode (TTL: {new_code.ttl}) for future execution.")
                    # Trigger an event for the new code to be processed by the event store
                    self._temporal_mro.store.trigger_event(new_code) # store is EventDrivenTemporalStore
            else:
                print(f"  Action at TTL {current_code_context.ttl if current_code_context else 'N/A'}. No further emanation.")
            
            if self.counter > 3:
                 raise ValueError("Simulated error after 3 calls in one agent instance.")
            return f"Task {task_id} processed by {self.name}, counter {self.counter}"

    agent = MyTemporalAgent("Alice", temporal_manager)
    
    print("\n--- Initial call to agent.perform_action('task1') ---")
    try:
        result = agent.perform_action("task1")
        print(f"Initial call result: {result}")
    except Exception as e:
        print(f"Error in initial call: {e}")

    print("\n--- Processing events from EventDrivenTemporalStore ---")
    # The first call to perform_action might have triggered events (emanated code)
    # Process these events. Each event execution might trigger more events until TTL runs out.
    event_store.process_events(max_events=5) # Limit to prevent infinite loops in buggy code

    print("\n--- Second direct call to agent.perform_action('task2') ---")
    # This will be a new temporal context capture for 'task2'
    try:
        result = agent.perform_action("task2") # Agent's counter continues from its instance state
        print(f"Second call result: {result}")
    except Exception as e:
        print(f"Error in second call: {e}")


    print("\n--- Processing any new events ---")
    event_store.process_events(max_events=5)

    # Check database content (simplified)
    with event_store.db:
        codes_count = event_store.db.execute("SELECT COUNT(*) FROM temporal_code").fetchone()[0]
        emanations_count = event_store.db.execute("SELECT COUNT(*) FROM emanation_graph").fetchone()[0]
        print(f"\nDatabase state: {codes_count} stored codes, {emanations_count} emanation records.")

    event_store.close()


    print("\n" + "=" * 50)
    print("### 3. Transducers Demonstration ###")
    print("=" * 50)
    
    numbers = range(1, 11) # 1 to 10

    # Transducer 1: map(x*2), filter(x > 5)
    xform1 = compose(
        Map(lambda x: x * 2),      # e.g., [1..10] -> [2,4,6,8,10,12,14,16,18,20]
        Filter(lambda x: x > 10)   # e.g., [2,4,...,20] -> [12,14,16,18,20]
    )
    result1 = transduce(xform1, append_to_list, [], numbers)
    print(f"Transduce with map(x*2) then filter(x > 10): {result1}") # Expected: [12, 14, 16, 18, 20]

    # Transducer 2: Using mapcat
    # mapcat item -> [item, item*10], then filter out items < 20
    xform2 = compose(
        mapcat(lambda x: [x, x * 10]), # 1->[1,10], 2->[2,20], 3->[3,30] ...
                                       # Results in: [1,10, 2,20, 3,30, 4,40, 5,50] for input [1..5]
        Filter(lambda x: x >= 20)      # Filter this flattened list
    )
    result2 = into(list, xform2, range(1, 6)) # Use 'into' helper
    print(f"Transduce with mapcat(x->[x,x*10]) then filter(x >= 20) on [1-5]: {result2}") # Expected: [20, 30, 40, 50]

    # Transducer 3: Early termination with Reduced
    def sum_until_limit_rf(acc=_MISSING, item=_MISSING, limit=10):
        if acc is _MISSING: return 0
        if item is _MISSING: return acc
        
        new_sum = acc + item
        if new_sum > limit:
            return Reduced(new_sum) # Terminate early
        return new_sum
    
    # Need to use functools.partial to pass the 'limit' to the reducing function
    # when using with transduce, as transduce expects rf(acc,item) signature.
    # Or, make the RF a class/closure. For simplicity here, let's assume rf is fixed for the demo.
    # The custom_reduce itself can handle this directly if sum_until_limit_rf is passed to it.
    # For transduce, the reducing_fn is typically generic like append_to_list.
    # Let's demo with custom_reduce directly for this.
    
    print(f"Custom reduce with early termination (sum numbers, stop if sum > 10):")
    limited_sum_rf = functools.partial(sum_until_limit_rf, limit=10)
    # Need to adjust custom_reduce or rf for initializer if using 0-arity for initial value
    # For this RF, initial value is 0.
    result3 = custom_reduce(limited_sum_rf, numbers, 0) 
    print(f"  Data: {list(numbers)}. Result: {result3}") # 1+2+3+4+5 = 15 (stops when 1+2+3+4=10, adds 5, 15 > 10, reduced)
                                                     # Actual: 1+2+3+4=10. Next item 5. 10+5=15. Reduced(15).
                                                     # If limit 7: 1+2+3=6. Next 4. 6+4=10. Reduced(10).

    print("\n" + "=" * 50)
    print("### 4. Method Profiling Demonstration ###")
    print("=" * 50)

    @profile_class_methods
    class ProfiledCalculator:
        version = "1.0"

        def __init__(self, name="Calc"):
            self.name = name
            time.sleep(0.01) # Simulate some init work

        def add(self, x, y):
            time.sleep(0.02) # Simulate work
            return x + y

        @staticmethod
        def multiply(x, y):
            time.sleep(0.03) # Simulate work
            return x * y
        
        @classmethod
        def get_info(cls):
            time.sleep(0.005)
            return f"Calculator class, version {cls.version}"

    calc_instance = ProfiledCalculator("MyCalc")
    calc_instance.add(10, 20)
    calc_instance.add(5, 3)
    ProfiledCalculator.multiply(4, 5) # Call static method
    ProfiledCalculator.get_info()     # Call class method

    print("Profiling Data (_profiling_data_ attribute on the class):")
    for method_name, records in ProfiledCalculator._profiling_data_.items():
        print(f"  Method: {method_name}")
        for record in records:
            # Duration is in nanoseconds, convert to ms for readability
            duration_ms = record['duration_ns'] / 1_000_000
            print(f"    - Type: {record['type']}, Duration: {duration_ms:.3f} ms, "
                  f"Args: {record['args_repr']}, Kwargs: {record['kwargs_repr']}")
    
    print("\n" + "=" * 50)
    print("Main demonstration finished.")
    print("=" * 50)

if __name__ == "__main__":
    main()