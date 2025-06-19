from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Standard Library Imports - 3.13 std libs **ONLY**
#------------------------------------------------------------------------------
import os
import io
import re
import sys
import ast
import dis
import mmap
import json
import uuid
import site
import time
import cmath
import errno
import shlex
import signal
import random
import pickle
import ctypes
import socket
import struct
import pstats
import shutil
import tomllib
import decimal
import pathlib
import logging
import inspect
import asyncio
import hashlib
import argparse
import cProfile
import platform
import tempfile
import mimetypes
import functools
import linecache
import traceback
import threading
import importlib
import subprocess
import tracemalloc
import http.server
from math import sqrt
from io import StringIO
from array import array
from queue import Queue, Empty
from abc import ABC, abstractmethod
from enum import Enum, auto, StrEnum
from collections import namedtuple, OrderedDict, defaultdict, deque
from operator import mul
from typing import (
    Any, Dict, List, Optional, Union, Callable, TypeVar,
    Tuple, Generic, Set, Coroutine, Type, NamedTuple,
    ClassVar, Protocol, runtime_checkable, AsyncIterator, Iterator
)
from types import (
    SimpleNamespace, ModuleType, MethodType,
    FunctionType, CodeType, TracebackType, FrameType
)
from dataclasses import dataclass, field
from functools import reduce, lru_cache, partial, wraps
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path, PureWindowsPath
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from importlib.util import spec_from_file_location, module_from_spec
libc = None
OLLAMA_DEFAULT_HOST = "127.0.0.1"
OLLAMA_DEFAULT_PORT = 11434
IS_WINDOWS = os.name == 'nt'
IS_POSIX = os.name == 'posix'
if IS_WINDOWS:
    try:
        from ctypes import windll
        from ctypes import wintypes
        from ctypes.wintypes import HANDLE, DWORD, LPWSTR, LPVOID, BOOL
        from pathlib import PureWindowsPath
        def set_process_priority(priority: int):
            windll.kernel32.SetPriorityClass(wintypes.HANDLE(-1), priority)
        libc = ctypes.windll.msvcrt
        set_process_priority(1)
    except ImportError:
        print(f"{__file__} failed to import ctypes on platform: {os.name}")
elif IS_POSIX:
    try:
        import resource
        libc = ctypes.CDLL("libc.so.6")
    except ImportError:
        print(f"{__file__} failed to import ctypes on platform: {os.name}")
class AsyncOllamaClient:
    """
    An asynchronous client for Ollama API using only Python's standard asyncio library.
    """
    def __init__(self, host: str = OLLAMA_DEFAULT_HOST, port: int = OLLAMA_DEFAULT_PORT):
        self.host = host
        self.port = port
        self.logger = logging.getLogger(self.__class__.__name__)
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    async def _send_http_request(self, method: str, api_path: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        body_bytes = b""
        if payload:
            try:
                body_str = json.dumps(payload)
                body_bytes = body_str.encode('utf-8')
            except TypeError as e:
                self.logger.error(f"Payload serialization error: {e}")
                return {"error": "Payload serialization error", "details": str(e)}
        request_headers_list = [
            f"{method.upper()} {api_path} HTTP/1.1",
            f"Host: {self.host}:{self.port}",
            "Connection: close", # Simplifies by not handling persistent connections
            "Accept: application/json",
            "User-Agent: PythonAsyncioStdlibClient/1.0"
        ]
        if payload:
            request_headers_list.append("Content-Type: application/json; charset=utf-8")
            request_headers_list.append(f"Content-Length: {len(body_bytes)}")
        http_request_str = "\r\n".join(request_headers_list) + "\r\n\r\n"
        http_request_bytes = http_request_str.encode('utf-8') + body_bytes
        reader, writer = None, None
        try:
            self.logger.debug(f"Connecting to Ollama: {self.host}:{self.port}")
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=10.0 # Connection timeout
            )
            self.logger.debug(f"Sending request: {method.upper()} {api_path}")
            writer.write(http_request_bytes)
            await writer.drain()
            status_line_bytes = await asyncio.wait_for(reader.readline(), timeout=10.0) # Response timeout
            status_line = status_line_bytes.decode('utf-8').strip()
            self.logger.debug(f"Ollama status: {status_line}")
            version, status_code_str, *reason_parts = status_line.split(' ', 2)
            status_code = int(status_code_str)
            reason = reason_parts[0] if reason_parts else ""
            response_headers = {}
            content_length = 0
            while True:
                header_line_bytes = await asyncio.wait_for(reader.readline(), timeout=5.0)
                header_line = header_line_bytes.decode('utf-8').strip()
                if not header_line: break # End of headers
                name, value = header_line.split(':', 1)
                response_headers[name.strip().lower()] = value.strip()
                if name.strip().lower() == 'content-length':
                    content_length = int(value.strip())
            self.logger.debug(f"Ollama response headers: {response_headers}")
            response_body_bytes = b""
            if content_length > 0:
                response_body_bytes = await asyncio.wait_for(reader.readexactly(content_length), timeout=30.0) # Body read timeout
            # Note: This doesn't handle chunked transfer encoding, Ollama usually sends Content-Length.
            response_body_str = response_body_bytes.decode('utf-8')
            self.logger.debug(f"Ollama response body (first 200 chars): {response_body_str[:200]}")
            if not (200 <= status_code < 300):
                self.logger.error(f"Ollama API error {status_code} {reason}: {response_body_str}")
                return {"error": f"Ollama API Error {status_code}", "reason": reason, "details": response_body_str}
            if not response_body_str.strip(): # Handle empty success bodies if any API does that
                 return {"status_code": status_code, "message": "Success with empty body"}
            try:
                return json.loads(response_body_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Ollama response JSON decode error: {e}. Body: {response_body_str}")
                return {"error": "JSON decode error", "details": str(e), "body": response_body_str}
        except asyncio.TimeoutError:
            self.logger.error(f"Ollama request to {api_path} timed out.")
            return {"error": "Request timed out"}
        except ConnectionRefusedError:
            self.logger.error(f"Ollama connection refused at {self.host}:{self.port}. Is Ollama running?")
            return {"error": "Connection refused"}
        except Exception as e:
            self.logger.error(f"Ollama HTTP request to {api_path} failed: {e}", exc_info=True)
            return {"error": "HTTP request failed", "details": str(e)}
        finally:
            if writer and not writer.is_closing():
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception as e_close:
                    self.logger.error(f"Error closing Ollama connection: {e_close}")
    async def generate_embedding(self, text: str, model: str = "nomic-embed-text") -> Optional[List[float]]:
        self.logger.info(f"Requesting embedding from Ollama (model: {model}) for text: \"{text[:30]}...\"")
        payload = {"model": model, "prompt": text}
        response = await self._send_http_request("POST", "/api/embeddings", payload)
        if response and "embedding" in response:
            self.logger.info("Embedding received successfully.")
            return response["embedding"]
        self.logger.warning(f"Failed to get embedding from Ollama. Response: {response}")
        return None
    async def generate_response(self, prompt: str, model: str = "gemma2:latest", stream: bool = False) -> Optional[str]:
        self.logger.info(f"Requesting response from Ollama (model: {model}, stream: {stream}) for prompt: \"{prompt[:30]}...\"")
        # This stdlib client currently does not support true streaming for responses.
        # It will make a non-streaming request if stream=True.
        # Proper streaming would require iterative reading and yielding of JSON lines.
        if stream:
             self.logger.warning("Streaming is set to True, but this basic client will make a non-streaming request.")
        payload = {"model": model, "prompt": prompt, "stream": False} # Force non-streaming for this client
        response = await self._send_http_request("POST", "/api/generate", payload)
        if response and "response" in response: # For non-streaming
            self.logger.info("Response received successfully.")
            return response["response"]
        # If Ollama were to return an error structure like {"error": "message"}
        if response and "error" in response and isinstance(response["error"], str):
            self.logger.error(f"Ollama generation error: {response['error']}")
            return f"Error from Ollama: {response['error']}"

        self.logger.warning(f"Failed to get a valid response string from Ollama. Response: {response}")
        return None
# Helper to convert AST expression nodes (like type annotations or bases) to string
def _get_ast_node_name(node: ast.expr) -> str:
    """
    Converts an AST expression node to its string representation.
    Handles simple names, attributes (e.g., module.Class), and uses ast.unparse if available.
    """
    if hasattr(ast, 'unparse'): # ast.unparse is available in Python 3.9+
        try:
            return ast.unparse(node).strip()
        except Exception: # Fallback if unparse fails for some specific nodes
            pass
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parts = []
        curr = node
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
            return ".".join(reversed(parts))
        else: # e.g. func().attr
            return f"ComplexAttribute:{type(curr).__name__}.{node.attr}"
    elif isinstance(node, ast.Subscript): # e.g. list[int], Dict[str, int]
        value_str = _get_ast_node_name(node.value)
        slice_str = _get_ast_node_name(node.slice)
        return f"{value_str}[{slice_str}]"
    elif isinstance(node, ast.Constant): # Python 3.8+ for simple constants like None, True, "string"
        return str(node.value)
    # Add more specific handlers if needed for other node types like ast.Tuple for Tuple[int, str]
    return f"UnsupportedNodeType:{type(node).__name__}"
class ClassInfo(NamedTuple):
    name: str
    docstring: Optional[str]
    bases: List[str]
    methods: List[str] # List of method names, could be expanded to FunctionInfo for methods
class FunctionInfo(NamedTuple):
    name: str
    docstring: Optional[str]
    return_type: Optional[str]
    args: List[str] # List of argument names
    rpn_callable: bool = False  # Domain-specific; implies more than just 'rpn'; it "works(last time we checked, or; according to the docs)": bool
@dataclass
class FileMetadata:
    path: Path
    mime_type: str
    size: int
    created: float
    modified: float
    hash: str
    symlinks: List[Path] = field(default_factory=list)
    content: Optional[str] = None
    python_classes: Optional[List[ClassInfo]] = field(default=None, repr=False)
    python_functions: Optional[List[FunctionInfo]] = field(default=None, repr=False)
    parse_error: Optional[str] = field(default=None, repr=False)
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
def hash_state(value: Any) -> int:
    """Hash a state value in a deterministic way"""
    if isinstance(value, int):
        return value * 2654435761 % 2**32  # Knuth's multiplicative hash
    elif isinstance(value, str):
        return sum(ord(c) * (31 ** i) for i, c in enumerate(value)) % 2**32
    else:
        return hash(str(value)) % 2**32
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
class ContentRegistry:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.metadata: Dict[str, FileMetadata] = {}
        self.modules: Dict[str, Any] = {}
        self._init_mimetypes()
    def _init_mimetypes(self):
        mimetypes.add_type('text/markdown', '.md')
        mimetypes.add_type('text/plain', '.txt')
        # mimetypes.add_type('application/data', '.json')
    def _compute_hash(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    def _load_text_content(self, path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return None
    def _parse_python_content(self, content: str, file_path: Path) -> Tuple[Optional[List[ClassInfo]], Optional[List[FunctionInfo]], Optional[str]]:
        """
        Parses Python code content and extracts class and function information.
        Returns (class_infos, function_infos, error_message).
        """
        classes_found = []
        functions_found = []
        try:
            tree = ast.parse(content, filename=str(file_path))
            for node in tree.body: # Iterate over top-level nodes
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node, clean=False)
                    arg_names = [arg.arg for arg in node.args.args]
                    return_annotation_str = _get_ast_node_name(node.returns) if node.returns else None
                    functions_found.append(FunctionInfo(
                        name=node.name,
                        docstring=docstring,
                        return_type=return_annotation_str,
                        args=arg_names,
                        rpn_callable=False # Default, needs specific logic if rpn_callable is to be determined
                    ))
                elif isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node, clean=False)
                    base_names = [_get_ast_node_name(base_node) for base_node in node.bases]
                    method_names = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
                    classes_found.append(ClassInfo(
                        name=node.name,
                        docstring=docstring,
                        bases=base_names,
                        methods=method_names
                    ))
            return classes_found if classes_found else None, functions_found if functions_found else None, None
        except SyntaxError as e:
            return None, None, f"SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset} in {file_path.name}"
        except Exception as e:
            return None, None, f"ASTParsingError: {str(e)} in {file_path.name}"
    def register_file(self, path: Path) -> Optional[FileMetadata]:
        if not path.is_file():
            return None
        stat = path.stat()
        mime_type = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        symlinks_list = [p for p in path.parent.glob(f'*{path.name}*') if p.is_symlink()]
        text_content = None
        is_python_file = path.suffix == '.py' or \
                         (mime_type and ('python' in mime_type or 'x-python' in mime_type))
        if is_python_file or 'text' in mime_type:
             text_content = self._load_text_content(path)
        metadata = FileMetadata(
            path=path, mime_type=mime_type, size=stat.st_size,
            created=stat.st_ctime, modified=stat.st_mtime,
            hash=self._compute_hash(path), symlinks=symlinks_list,
            content=text_content
        )
        if is_python_file and metadata.content:
            classes, functions, error_msg = self._parse_python_content(metadata.content, path)
            metadata.python_classes = classes
            metadata.python_functions = functions
            metadata.parse_error = error_msg
        # module_name = f"content_{rel_path.stem.replace('-', '_').replace('.', '_')}" # Sanitize module name
        # Remove or comment out the dynamic module execution block:
        # spec = importlib.util.spec_from_file_location(module_name, str(path))
        # if spec and spec.loader:
        #     try:
        #         module = importlib.util.module_from_spec(spec)
        #         # spec.loader.exec_module(module) # <-- DO NOT EXECUTE
        #         # self.modules[module_name] = module 
        #     except Exception as e:
        #         print(f"Error creating module object from {path}: {e}")
        rel_path = path.relative_to(self.root_dir)
        self.metadata[str(rel_path)] = metadata
        return metadata
        rel_path = path.relative_to(self.root_dir)
        self.metadata[str(rel_path)] = metadata
        return metadata
    def scan_directory(self):
        for path in self.root_dir.rglob('*'):
            if path.is_file():
                self.register_file(path)
    def export_metadata(self, output_path: Path):
        metadata_dict = {}
        for k, v in self.metadata.items():
            item_data = {
                'path': str(v.path),
                'mime_type': v.mime_type,
                'size': v.size,
                'created': datetime.fromtimestamp(v.created).isoformat(),
                'modified': datetime.fromtimestamp(v.modified).isoformat(),
                'hash': v.hash,
                'symlinks': [str(s) for s in v.symlinks],
                'has_content': v.content is not None,
                'parse_error': v.parse_error
            }
            # NamedTuples need to be converted to dicts for JSON serialization
            if v.python_classes:
                item_data['python_classes'] = [c._asdict() for c in v.python_classes]
            else:
                item_data['python_classes'] = None
            if v.python_functions:
                item_data['python_functions'] = [f._asdict() for f in v.python_functions]
            else:
                item_data['python_functions'] = None
            metadata_dict[str(k)] = item_data
        output_path.write_text(json.dumps(metadata_dict, indent=2, ensure_ascii=False))

    def _parse_python_content(self, content: str, file_path: Path) -> Tuple[Optional[List[ClassInfo]], Optional[List[FunctionInfo]], Optional[str]]:
        """
        Parses Python code content and extracts class and function information.
        Returns (class_infos, function_infos, error_message).
        """
        classes = []
        functions = []
        try:
            tree = ast.parse(content, filename=str(file_path))
            for node in tree.body: # Iterate over top-level nodes
                if isinstance(node, ast.FunctionDef):
                    functions.append(FunctionInfo(name=node.name))
                elif isinstance(node, ast.ClassDef):
                    base_names = [_get_ast_node_name(base_node) for base_node in node.bases]
                    method_names = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_names.append(item.name)
                    classes.append(ClassInfo(name=node.name, bases=base_names, methods=method_names))
            return classes, functions, None
        except SyntaxError as e:
            return None, None, f"SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset}"
        except Exception as e:
            return None, None, f"ASTParsingError: {str(e)}"

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
class Morphology(enum.Enum):
    """
    Represents the floor morphic state of a BYTE_WORD.
    
    C = 0: Floor morphic state (stable, low-energy)
    C = 1: Dynamic or high-energy state
    
    The control bit (C) indicates whether other holoicons can point to this holoicon:
    - DYNAMIC (1): Other holoicons CAN point to this holoicon
    - MORPHIC (0): Other holoicons CANNOT point to this holoicon
    
    This ontology maps to thermodynamic character: intensive & extensive.
    A 'quine' (self-instantiated runtime) is a low-energy, intensive system,
    while a dynamic holoicon is a high-energy, extensive system inherently
    tied to its environment.
    """
    MORPHIC = 0      # Stable, low-energy state
    DYNAMIC = 1      # High-energy, potentially transformative state
    
    # Fundamental computational orientation and symmetry
    MARKOVIAN = -1    # Forward-evolving, irreversible
    NON_MARKOVIAN = math.e  # Reversible, with memory


class QuantumState(enum.Enum):
    """Represents a computational state that tracks its quantum-like properties."""
    SUPERPOSITION = 1   # Known by handle only
    ENTANGLED = 2       # Referenced but not loaded
    COLLAPSED = 4       # Fully materialized
    DECOHERENT = 8      # Garbage collected
    def superposition(self, other: 'QuantumState', coeff1: MorphicComplex, coeff2: MorphicComplex) -> 'QuantumState':
        """
        Create a superposition of two quantum states.
        |ψ⟩ = a|ψ₁⟩ + b|ψ₂⟩
        """
        if self.dimension != other.dimension:
            raise ValueError("Quantum states must belong to same Hilbert space")
            
        new_amplitudes = []
        for i in range(len(self.amplitudes)):
            new_amp = (self.amplitudes[i] * coeff1) + (other.amplitudes[i] * coeff2)
            new_amplitudes.append(new_amp)
            
        return QuantumState(new_amplitudes, self.dimension)
    
    def entangle(self, other: 'QuantumState') -> 'QuantumState':
        """
        Create an entangled state from two quantum states.
        |ψ⟩ = (|ψ₁⟩|0⟩ + |ψ₂⟩|1⟩)/√2
        This is a simplified version of entanglement for demonstration.
        """
        # For simplicity, we'll just return a superposition
        coeff = MorphicComplex(1/math.sqrt(2), 0)
        return self.superposition(other, coeff, coeff)
class WordSize(enum.IntEnum):
    """Standardized computational word sizes"""
    BYTE = 1     # 8-bit
    SHORT = 2    # 16-bit
    INT = 4      # 32-bit
    LONG = 8     # 64-bit

class BYTE_WORD:
    """Basic 8-bit word representation."""
    def __init__(self, value: int = 0):
        if not 0 <= value <= 255:
            raise ValueError("BYTE_WORD value must be between 0 and 255")
        self.value = value

    def __repr__(self) -> str:
        return f"BYTE_WORD(value={self.value:08b})"


class ByteWord:
    """
    Enhanced representation of an 8-bit BYTE_WORD with a comprehensive interpretation of its structure.
    
    Bit Decomposition:
    - T (4 bits): State or data field
    - V (3 bits): Morphism selector or transformation rule
    - C (1 bit): Floor morphic state (pointability)
    """
    def __init__(self, raw: int):
        """
        Initialize a ByteWord from its raw 8-bit representation.
        
        Args:
            raw (int): 8-bit integer representing the BYTE_WORD
        """
        if not 0 <= raw <= 255:
            raise ValueError("ByteWord must be an 8-bit integer (0-255)")
            
        self.raw = raw
        self.value = raw & 0xFF  # Ensure 8-bit resolution
        
        # Decompose the raw value
        self.state_data = (raw >> 4) & 0x0F       # High nibble (4 bits)
        self.morphism = (raw >> 1) & 0x07         # Middle 3 bits
        self.floor_morphic = Morphology(raw & 0x01)  # Least significant bit
        
        self._refcount = 1
        self._state = QuantumState.SUPERPOSITION

    @property
    def pointable(self) -> bool:
        """
        Determine if other holoicons can point to this holoicon.
        
        Returns:
            bool: True if the holoicon is in a dynamic (pointable) state
        """
        return self.floor_morphic == Morphology.DYNAMIC

    def __repr__(self) -> str:
        return f"ByteWord(T={self.state_data:04b}, V={self.morphism:03b}, C={self.floor_morphic.value})"

    @staticmethod
    def xnor(a: int, b: int, width: int = 4) -> int:
        """Perform a bitwise XNOR operation."""
        return ~(a ^ b) & ((1 << width) - 1)  # Mask to width-bit output

    @staticmethod
    def abelian_transform(t: int, v: int, c: int) -> int:
        """
        Perform the XNOR-based Abelian transformation.
        
        Args:
            t: State/data field
            v: Morphism selector
            c: Floor morphic state
            
        Returns:
            Transformed state value
        """
        if c == 1:
            return ByteWord.xnor(t, v)  # Apply XNOR transformation
        return t  # Identity morphism when c = 0

    @staticmethod
    def extract_lsb(state: Union[str, int, bytes], word_size: int) -> Any:
        """
        Extract least significant bit/byte based on word size.
        
        Args:
            state: Input value to extract from
            word_size: Size of word to determine extraction method
            
        Returns:
            Extracted least significant bit/byte
        """
        if word_size == 1:
            return state[-1] if isinstance(state, str) else str(state)[-1]
        elif word_size == 2:
            return (
                state & 0xFF if isinstance(state, int) else
                state[-1] if isinstance(state, bytes) else
                state.encode()[-1]
            )
        elif word_size >= 3:
            return hashlib.sha256(
                state.encode() if isinstance(state, str) else 
                state if isinstance(state, bytes) else
                str(state).encode()
            ).digest()[-1]


class MorphicComplex:
    """Represents a complex number with morphic properties.
        象迹 (xiàng jì) = 'Morpheme trace'
    """
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

@dataclass
class MorphologicalBasis(Generic[T, V, C]):
    """Defines a structured basis with symmetry evolution.
        形态导数  (xíngtài dǎoshù)
        形态 = morphology
        导数 = derivative
    """
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
    态律 (tài lǜ) = 'Morphological law'
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
    """Abstract morphism between type structures
    """
    
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
    态衍 (tài yǎn) = 'Morphological derivation'
        For composition rules
    旋化 (xuán huà) = 'Spiral transformation'
	    For non-associative evolution
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
# ============================================================================
# Python Object Abstraction
# ============================================================================

class PyObjABC(ABC):
    """Abstract Base Class for PyObject-like objects (including __Atom__)."""
    
    @abstractmethod
    def __getattribute__(self, name: str) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def __setattr__(self, name: str, value: Any) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def __class__(self) -> type:
        raise NotImplementedError
    
    @property
    def ob_refcnt(self) -> int:
        """Returns the object's reference count."""
        return self._refcount
    
    @ob_refcnt.setter
    def ob_refcnt(self, value: int) -> None:
        """Sets the object's reference count."""
        self._refcount = value
    
    @property
    def ob_ttl(self) -> Optional[int]:
        """Returns the object's time-to-live (in seconds or None)."""
        return self._ttl
    
    @ob_ttl.setter
    def ob_ttl(self, value: Optional[int]) -> None:
        """Sets the object's time-to-live."""
        self._ttl = value


@dataclass
class CPythonFrame:
    """
    Quantum-informed object representation.
    Maps conceptually to CPython's PyObject structure.
    """
    type_ptr: int             # Memory address of type object
    obj_type: Type[Any]           # Type information
    _value: Any                # Actual value (renamed to avoid conflict with property)
    refcount: int = field(default=1)
    ttl: Optional[int] = None
    _state: QuantumState = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize with timestamp and quantum properties"""
        self._birth_timestamp = time.time()
        self._refcount = self.refcount
        self._ttl = self.ttl 
        
        # Initialize _state to default SUPERPOSITION
        self._state = QuantumState.SUPERPOSITION
        
        if self.ttl is not None:
            self._ttl_expiration = self._birth_timestamp + self.ttl
        else:
            self._ttl_expiration = None

        # self._value is already set by dataclass init

        if self._state == QuantumState.SUPERPOSITION:
            self._superposition = [self._value]
            self._superposition_timestamp = time.time()
        else:
            self._superposition = None
            
        if self._state == QuantumState.ENTANGLED:
            self._entanglement = [self._value]
            self._entanglement_timestamp = time.time()
        else:
            self._entanglement = None
            
        if self.obj_type.__module__ == 'builtins':
            """All 'knowledge' aka data is treated as python modules and these are the flags for controlling what is canon."""
            self._is_primitive = True
            self._primitive_type = self.obj_type.__name__
            self._primitive_value = self._value
        else:
            self._is_primitive = False
    
    @classmethod
    def from_object(cls, obj: object) -> 'CPythonFrame':
        """Extract CPython frame data from any Python object"""
        return cls(
            type_ptr=id(type(obj)),
            obj_type=type(obj), # Use obj_type
            _value=obj,
            refcount=sys.getrefcount(obj) - 1 # Initial refcount estimate
            # Let ttl and state use their defaults unless specified
        )
    
    @property
    def value(self) -> Any:
        """Get the current value, potentially collapsing state."""
        if self._state == QuantumState.SUPERPOSITION:
            return random.choice(self._superposition)
        return self._value
    
    @property
    def state(self) -> QuantumState:
        """Current quantum-like state"""
        return self._state
    
    @state.setter
    def state(self, new_state: QuantumState) -> None:
        """Set the quantum state with appropriate side effects"""
        old_state = self._state
        self._state = new_state
        
        # Handle state transition side effects
        if new_state == QuantumState.COLLAPSED and old_state == QuantumState.SUPERPOSITION:
            if self._superposition:
                self._value = random.choice(self._superposition)
                self._superposition = None
    
    def collapse(self) -> Any:
        """Force state resolution"""
        if self._state != QuantumState.COLLAPSED:
            if self._state == QuantumState.SUPERPOSITION and self._superposition:
                self._value = random.choice(self._superposition) # Update internal value
                self._superposition = None # Clear superposition list
            elif self._state == QuantumState.ENTANGLED:
                 # Decide how entanglement collapses - maybe pick from list?
                 if self._entanglement:
                     self._value = random.choice(self._entanglement) # Example resolution
                 self._entanglement = None # Clear entanglement list
            self._state = QuantumState.COLLAPSED
        return self._value # Return the now-collapsed internal value

    
    def entangle_with(self, other: 'CPythonFrame') -> None:
        """Create quantum entanglement with another object.""" 
        if self._entanglement is None:
            self._entanglement = [self.value]
        if other._entanglement is None:
            other._entanglement = [other.value]
            
        self._entanglement.extend(other._entanglement)
        other._entanglement = self._entanglement
        self._state = other._state = QuantumState.ENTANGLED
    
    def check_ttl(self) -> bool:
        """Check if TTL expired and collapse state if necessary."""
        if self.ttl is not None and self._ttl_expiration is not None and time.time() >= self._ttl_expiration:
            self.collapse()
            return True
        return False
    
    def observe(self) -> Any:
        """Collapse state upon observation if necessary."""
        self.check_ttl()
        if self._state == QuantumState.SUPERPOSITION:
            self._state = QuantumState.COLLAPSED
            if self._superposition:
                self._value = random.choice(self._superposition)
        elif self._state == QuantumState.ENTANGLED:
            self._state = QuantumState.COLLAPSED
        return self.value


# ============================================================================
# Functional Programming Patterns - Transducers
# ============================================================================

class Missing:
    """Marker class to indicate a missing value."""
    pass


class Reduced:
    """Sentinel class to signal early termination during reduction."""
    def __init__(self, val: Any):
        self.val = val


def ensure_reduced(x: Any) -> Union[Any, Reduced]:
    """Ensure the value is wrapped in a Reduced sentinel."""
    return x if isinstance(x, Reduced) else Reduced(x)


def unreduced(x: Any) -> Any:
    """Unwrap a Reduced value or return the value itself."""
    return x.val if isinstance(x, Reduced) else x


def reduce(function: Callable[[Any, T], Any], iterable: Iterable[T], initializer: Any = Missing) -> Any:
    """A custom reduce implementation that supports early termination with Reduced."""
    accum_value = initializer if initializer is not Missing else function()
    for x in iterable:
        accum_value = function(accum_value, x)
        if isinstance(accum_value, Reduced):
            return accum_value.val
    return accum_value


class Transducer:
    """Base class for defining transducers."""
    def __init__(self, step: Callable[[Any, T], Any]):
        self.step = step

    def __call__(self, step: Callable[[Any, T], Any]) -> Callable[[Any, T], Any]:
        """The transducer's __call__ method allows it to be used as a decorator."""
        return self.step(step)


class Map(Transducer):
    """Transducer for mapping elements with a function."""
    def __init__(self, f: Callable[[T], R]):
        def _map_step(step):
            def new_step(r: Any = Missing, x: Optional[T] = Missing):
                if r is Missing:
                    return step()
                if x is Missing:
                    return step(r)
                return step(r, f(x))
            return new_step
        super().__init__(_map_step)


class Filter(Transducer):
    """Transducer for filtering elements based on a predicate."""
    def __init__(self, pred: Callable[[T], bool]):
        def _filter_step(step):
            def new_step(r: Any = Missing, x: Optional[T] = Missing):
                if r is Missing:
                    return step()
                if x is Missing:
                    return step(r)
                return step(r, x) if pred(x) else r
            return new_step
        super().__init__(_filter_step)


class Cat(Transducer):
    """Transducer for flattening nested collections."""
    def __init__(self):
        def _cat_step(step):
            def new_step(r: Any = Missing, x: Optional[Any] = Missing):
                if r is Missing:
                    return step()
                if x is Missing:
                    return step(r)
                    
                if not hasattr(x, '__iter__'):
                    raise TypeError(f"Expected iterable, got {type(x)}")
                    
                result = r
                for item in x:
                    result = step(result, item)
                    if isinstance(result, Reduced):
                        return result
                return result
            return new_step
        super().__init__(_cat_step)


def compose(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose functions in reverse order."""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), fns)


def transduce(xform: Transducer, f: Callable[[Any, T], Any], start: Any, coll: Iterable[T]) -> Any:
    """Apply a transducer to a collection with an initial value."""
    reducer = xform(f)
    return reduce(reducer, coll, start)


def mapcat(f: Callable[[T], Iterable[R]]) -> Transducer:
    """Map then flatten results into one collection."""
    return compose(Map(f), Cat())


def into(target: Union[list, set], xducer: Transducer, coll: Iterable[T]) -> Any:
    """Apply transducer and collect results into a target container."""
    def append(r: Any = Missing, x: Optional[Any] = Missing) -> Any:
        """Append to a collection."""
        if r is Missing:
            return []
        if hasattr(r, 'append'):
            r.append(x)
        elif hasattr(r, 'add'):
            r.add(x)
        return r
        
    return transduce(xducer, append, target, coll)


# ============================================================================
# Morphological Framework - Rule-based Transformations
# ============================================================================

class MorphologicalRule:
    """Rule that maps structural transformations in code morphologies."""
    
    def __init__(self, symmetry: str, conservation: str, lhs: str, rhs: List[Union[str, Morphology, ByteWord]]):
        self.symmetry = symmetry
        self.conservation = conservation
        self.lhs = lhs
        self.rhs = rhs
    
    def apply(self, input_seq: List[str]) -> List[str]:
        """Applies the morphological transformation to an input sequence."""
        if self.lhs in input_seq:
            idx = input_seq.index(self.lhs)
            return input_seq[:idx] + [str(elem) for elem in self.rhs] + input_seq[idx + 1:]
        return input_seq


@dataclass
class MorphologicPyOb:
    """
    The unification of Morphologic transformations and PyOb behavior.
    This is the grandparent class for all runtime polymorphs.
    It encapsulates stateful, structural, and computational potential.
    """
    symmetry: str
    conservation: str
    lhs: str
    rhs: List[Union[str, 'Morphology']]
    value: Any
    type_ptr: int = field(default_factory=lambda: id(object))
    ttl: Optional[int] = None
    state: QuantumState = field(default=QuantumState.SUPERPOSITION)
    
    def __post_init__(self):
        self._refcount = 1
        self._birth_timestamp = time.time()
        self._state = self.state
        
        if self.ttl is not None:
            self._ttl_expiration = self._birth_timestamp + self.ttl
        else:
            self._ttl_expiration = None
            
        if self.state == QuantumState.SUPERPOSITION:
            self._superposition = [self.value]
        else:
            self._superposition = None
            
        if self.state == QuantumState.ENTANGLED:
            self._entanglement = [self.value]
        else:
            self._entanglement = None
    
    def apply_transformation(self, input_seq: List[str]) -> List[str]:
        """
        Applies morphological transformation while preserving object state.
        """
        if self.lhs in input_seq:
            idx = input_seq.index(self.lhs)
            transformed = input_seq[:idx] + [str(elem) for elem in self.rhs] + input_seq[idx + 1:]
            self._state = QuantumState.ENTANGLED
            return transformed
        return input_seq
    
    def collapse(self) -> Any:
        """Collapse to resolved state."""
        if self._state != QuantumState.COLLAPSED:
            if self._state == QuantumState.SUPERPOSITION and self._superposition:
                self.value = random.choice(self._superposition)
            self._state = QuantumState.COLLAPSED
        return self.value
    
    def collapse_and_transform(self) -> Any:
        """Collapse to resolved state and apply morphological transformation to value."""
        collapsed_value = self.collapse()
        if isinstance(collapsed_value, list):
            return self.apply_transformation(collapsed_value)
        return collapsed_value
    
    def entangle_with(self, other: 'MorphologicPyOb') -> None:
        """Entangle with another MorphologicPyOb to preserve state & entanglement symmetry in Morphologic terms."""
        if self._entanglement is None:
            self._entanglement = [self.value]
        if other._entanglement is None:
            other._entanglement = [other.value]
            
        self._entanglement.extend(other._entanglement)
        other._entanglement = self._entanglement
        
        if self.lhs == other.lhs and self.conservation == other.conservation:
            self._state = QuantumState.ENTANGLED
            other._state = QuantumState.ENTANGLED




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

async def app_kernel():
    """
    Main asynchronous function for the application.
    Orchestrates file scanning, metadata processing, and Ollama interactions.
    """
    # Configure root logger for more comprehensive output during kernel execution
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    kernel_logger = logging.getLogger("AppKernel")
    kernel_logger.info("🚀 Starting Application Kernel...")

    # --- Content Registry Operations ---
    kernel_logger.info("Initializing Content Registry...")
    registry = ContentRegistry(Path.cwd()) # Scan current working directory
    kernel_logger.info(f"Scanning directory: {registry.root_dir.resolve()}")
    registry.scan_directory() # This is synchronous as per current ContentRegistry design
    kernel_logger.info(f"Content registry scan complete. Found {len(registry.metadata)} items.")

    output_metadata_path = registry.root_dir / "file_metadata_export.json"
    kernel_logger.info(f"Exporting metadata to: {output_metadata_path}")
    registry.export_metadata(output_metadata_path)
    kernel_logger.info("Metadata exported successfully.")

    # Example: Log AST info for the first Python file found
    first_py_meta: Optional[FileMetadata] = next(
        (meta for meta in registry.metadata.values() if meta.path.suffix == '.py' and (meta.python_classes or meta.python_functions)),
        None
    )
    if first_py_meta:
        kernel_logger.info(f"\n--- AST Details for: {first_py_meta.path.name} ---")
        if first_py_meta.parse_error:
            kernel_logger.warning(f"  Parse Error: {first_py_meta.parse_error}")
        if first_py_meta.python_classes:
            kernel_logger.info("  Classes:")
            for cls_info in first_py_meta.python_classes:
                bases = f"({', '.join(cls_info.bases)})" if cls_info.bases else ""
                kernel_logger.info(f"    - {cls_info.name}{bases}: {len(cls_info.methods)} methods. Doc: {cls_info.docstring[:20] if cls_info.docstring else 'N/A'}...")
        if first_py_meta.python_functions:
            kernel_logger.info("  Functions:")
            for func_info in first_py_meta.python_functions:
                kernel_logger.info(f"    - {func_info.name}({', '.join(func_info.args)}) -> {func_info.return_type or 'Any'}. Doc: {func_info.docstring[:20] if func_info.docstring else 'N/A'}...")
        kernel_logger.info("--- End AST Details ---")

    # --- Ollama Client Operations ---
    kernel_logger.info("\nInitializing Async Ollama Client...")
    ollama_client = AsyncOllamaClient()

    # Example: Generate an embedding
    text_to_embed = "Exploring Python's asyncio capabilities."
    kernel_logger.info(f"Requesting embedding for: \"{text_to_embed}\"")
    embedding_vector = await ollama_client.generate_embedding(text_to_embed)
    if embedding_vector:
        kernel_logger.info(f"Received embedding vector (first 3 elements): {embedding_vector[:3]}...")
    else:
        kernel_logger.warning("Failed to generate embedding. Ensure Ollama is running and 'nomic-embed-text' model is available.")

    # Example: Generate a response from Ollama
    prompt_for_llm = "What are the benefits of using static typing in Python?"
    kernel_logger.info(f"Requesting LLM response for: \"{prompt_for_llm}\"")
    llm_text_response = await ollama_client.generate_response(prompt_for_llm)
    if llm_text_response:
        kernel_logger.info(f"LLM Response:\n{'-'*30}\n{llm_text_response}\n{'-'*30}")
    else:
        kernel_logger.warning("Failed to generate response. Ensure Ollama is running and 'gemma2:latest' model is available.")
    if libc and hasattr(libc, 'printf'):
        libc.printf(b"Message from C library via FFI (app_kernel)\n")

    shell_command = "echo Hello from shell (app_kernel)"
    kernel_logger.info(f"Executing shell command: '{shell_command}'")
    try:
        # subprocess.run is blocking. For truly async, use asyncio.create_subprocess_shell
        # For simplicity here, keeping it blocking as it's a minor part.
        shell_result = subprocess.run(shell_command, shell=True, capture_output=True, text=True, check=False)
        kernel_logger.info(f"Shell command output: {shell_result.stdout.strip()}")
    except Exception as e_shell:
        kernel_logger.error(f"Shell command execution failed: {e_shell}")

    kernel_logger.info("🏁 Application Kernel finished its tasks.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    demo_runtime_system()

    # Global exception handler for asyncio tasks
    def asyncio_exception_handler(loop, context):
        logger = logging.getLogger("AsyncioGlobalHandler")
        msg = context.get("exception", context["message"])
        logger.error(f"Global unhandled asyncio exception: {msg}", exc_info=context.get("exception"))
        # loop.stop() # Example: stop the loop on critical unhandled errors

    loop = asyncio.new_event_loop() # Explicitly create a new event loop
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(asyncio_exception_handler)

    try:
        loop.run_until_complete(app_kernel())
    except KeyboardInterrupt:
        logging.info("\nApplication interrupted by user. Shutting down...")
    except Exception as e:
        logging.critical(f"Critical unhandled error in main execution: {e}", exc_info=True)
    finally:
        # Cleanup asyncio tasks if any are still pending (more advanced)
        # For simple scripts, just closing the loop might be enough.
        # Ensure all tasks are given a chance to close gracefully.
        all_tasks = asyncio.all_tasks(loop)
        if all_tasks:
            logging.info(f"Cancelling {len(all_tasks)} outstanding tasks...")
            for task in all_tasks:
                task.cancel()
            try:
                # Allow tasks to process cancellation
                loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
            except Exception as e_gather:
                logging.error(f"Error during task gathering on shutdown: {e_gather}")

        if hasattr(loop, "shutdown_asyncgens"): # Python 3.6+
             loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logging.info("Application shutdown complete.")
