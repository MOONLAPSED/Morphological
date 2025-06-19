#!/usr/bin/env python3
"""
Morphological Debugger/IDE - A self-compiled assembler and development environment
for ByteWord-based computation with Smalltalk browser paradigm.

This implements the core architecture for a "multiplayer REPL" development environment
that treats ByteWords as first-class semantic objects in Hilbert space.
"""

import asyncio
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable
import hashlib
import mimetypes
import ast
import importlib.util
import sys
from collections import defaultdict, deque

# Core ByteWord system (from your previous code)
class Morphology(Enum):
    MORPHIC = 0      # Stable, low-energy state (not pointable)
    DYNAMIC = 1      # High-energy state (pointable)

class QuantumState(Enum):
    SUPERPOSITION = 1   # Known by handle only
    ENTANGLED = 2       # Referenced but not loaded  
    COLLAPSED = 4       # Fully materialized
    DECOHERENT = 8      # Garbage collected

class ByteWord:
    """Enhanced 8-bit word with morphological properties"""
    def __init__(self, raw: int):
        if not 0 <= raw <= 255:
            raise ValueError("ByteWord must be 8-bit (0-255)")
        
        self.raw = raw
        self.state_data = (raw >> 4) & 0x0F      # T: High nibble (4 bits)
        self.morphism = (raw >> 1) & 0x07        # V: Middle 3 bits  
        self.floor_morphic = Morphology(raw & 0x01)  # C: LSB
        
        self._refcount = 1
        self._quantum_state = QuantumState.SUPERPOSITION
        self._observers: Set['MorphologicalObserver'] = set()
        
    @property
    def pointable(self) -> bool:
        return self.floor_morphic == Morphology.DYNAMIC
    
    def collapse(self) -> None:
        """Collapse from superposition to definite state"""
        if self._quantum_state == QuantumState.SUPERPOSITION:
            self._quantum_state = QuantumState.COLLAPSED
            for observer in self._observers:
                observer.on_collapse(self)
    
    def entangle(self, other: 'ByteWord') -> None:
        """Create quantum entanglement between ByteWords"""
        self._quantum_state = QuantumState.ENTANGLED
        other._quantum_state = QuantumState.ENTANGLED
        # Share observer sets for entangled states
        shared_observers = self._observers | other._observers
        self._observers = shared_observers
        other._observers = shared_observers
    
    def __repr__(self) -> str:
        return f"ByteWord(T={self.state_data:04b}, V={self.morphism:03b}, C={self.floor_morphic.value}, Q={self._quantum_state.name})"

class MorphologicalObserver(ABC):
    """Observer pattern for ByteWord state changes"""
    @abstractmethod
    def on_collapse(self, byteword: ByteWord) -> None:
        pass
    
    @abstractmethod
    def on_entanglement(self, byteword1: ByteWord, byteword2: ByteWord) -> None:
        pass

# Memory Management and Debugging
@dataclass
class MemoryRegion:
    """Represents a region of ByteWord memory"""
    start_addr: int
    size: int
    words: List[ByteWord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.words:
            self.words = [ByteWord(0) for _ in range(self.size)]
    
    def read_word(self, offset: int) -> ByteWord:
        if not 0 <= offset < self.size:
            raise IndexError(f"Memory offset {offset} out of range")
        return self.words[offset]
    
    def write_word(self, offset: int, word: ByteWord) -> None:
        if not 0 <= offset < self.size:
            raise IndexError(f"Memory offset {offset} out of range")
        self.words[offset] = word

class MorphologicalMemoryManager:
    """Memory manager for ByteWord-based computation"""
    
    def __init__(self):
        self.regions: Dict[str, MemoryRegion] = {}
        self.observers: List[MorphologicalObserver] = []
        self.breakpoints: Set[Tuple[str, int]] = set()  # (region_name, offset)
        
    def allocate_region(self, name: str, size: int) -> MemoryRegion:
        """Allocate a new memory region"""
        region = MemoryRegion(len(self.regions) * 256, size)
        self.regions[name] = region
        return region
    
    def set_breakpoint(self, region_name: str, offset: int) -> None:
        """Set a breakpoint at specific memory location"""
        self.breakpoints.add((region_name, offset))
    
    def read_word(self, region_name: str, offset: int) -> ByteWord:
        """Read word from memory with breakpoint checking"""
        if (region_name, offset) in self.breakpoints:
            self._trigger_breakpoint(region_name, offset, "READ")
        
        region = self.regions.get(region_name)
        if not region:
            raise KeyError(f"Memory region '{region_name}' not found")
        
        return region.read_word(offset)
    
    def write_word(self, region_name: str, offset: int, word: ByteWord) -> None:
        """Write word to memory with breakpoint checking"""
        if (region_name, offset) in self.breakpoints:
            self._trigger_breakpoint(region_name, offset, "WRITE")
        
        region = self.regions.get(region_name)
        if not region:
            raise KeyError(f"Memory region '{region_name}' not found")
        
        region.write_word(offset, word)
    
    def _trigger_breakpoint(self, region_name: str, offset: int, operation: str) -> None:
        """Handle breakpoint trigger"""
        print(f"🔴 BREAKPOINT: {operation} at {region_name}[{offset}]")
        # In a full implementation, this would pause execution and enter debugger

# Assembler for ByteWord operations
class ByteWordInstruction(Enum):
    """ByteWord assembly instructions"""
    LOAD = "LOAD"       # Load immediate value into register
    STORE = "STORE"     # Store register to memory
    MOVE = "MOVE"       # Move between registers
    XNOR = "XNOR"       # XNOR operation between two values
    MORPH = "MORPH"     # Change morphological state
    ENTANGLE = "ENTGL"  # Create quantum entanglement
    COLLAPSE = "COLLPS" # Force quantum collapse
    OBSERVE = "OBSERV"  # Register observer
    HALT = "HALT"       # Stop execution

@dataclass
class AssemblyInstruction:
    """Represents a single assembly instruction"""
    opcode: ByteWordInstruction
    operands: List[Union[str, int]]
    line_number: int
    source_line: str

class MorphologicalAssembler:
    """Self-compiled assembler for ByteWord programs"""
    
    def __init__(self, memory_manager: MorphologicalMemoryManager):
        self.memory_manager = memory_manager
        self.registers: Dict[str, ByteWord] = {
            f"R{i}": ByteWord(0) for i in range(8)
        }
        self.pc = 0  # Program counter
        self.program: List[AssemblyInstruction] = []
        self.labels: Dict[str, int] = {}
        
    def assemble(self, source_code: str) -> List[AssemblyInstruction]:
        """Assemble source code into ByteWord instructions"""
        lines = source_code.strip().split('\n')
        instructions = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith(';'):  # Skip empty lines and comments
                continue
                
            if line.endswith(':'):  # Label
                label = line[:-1]
                self.labels[label] = len(instructions)
                continue
            
            parts = line.split()
            if not parts:
                continue
                
            opcode_str = parts[0].upper()
            try:
                opcode = ByteWordInstruction(opcode_str)
            except ValueError:
                raise SyntaxError(f"Unknown instruction '{opcode_str}' at line {line_num}")
            
            operands = parts[1:] if len(parts) > 1 else []
            instruction = AssemblyInstruction(opcode, operands, line_num, line)
            instructions.append(instruction)
        
        self.program = instructions
        return instructions
    
    def execute_instruction(self, instruction: AssemblyInstruction) -> bool:
        """Execute a single instruction. Returns False if should halt."""
        opcode = instruction.opcode
        operands = instruction.operands
        
        if opcode == ByteWordInstruction.LOAD:
            # LOAD R0, 255
            reg_name = operands[0]
            value = int(operands[1])
            self.registers[reg_name] = ByteWord(value)
            
        elif opcode == ByteWordInstruction.STORE:
            # STORE R0, memory, 10
            reg_name = operands[0]
            region_name = operands[1]
            offset = int(operands[2])
            self.memory_manager.write_word(region_name, offset, self.registers[reg_name])
            
        elif opcode == ByteWordInstruction.MOVE:
            # MOVE R0, R1
            src_reg = operands[0]
            dst_reg = operands[1]
            self.registers[dst_reg] = self.registers[src_reg]
            
        elif opcode == ByteWordInstruction.ENTANGLE:
            # ENTANGLE R0, R1
            reg1 = operands[0]
            reg2 = operands[1]
            self.registers[reg1].entangle(self.registers[reg2])
            
        elif opcode == ByteWordInstruction.COLLAPSE:
            # COLLAPSE R0
            reg_name = operands[0]
            self.registers[reg_name].collapse()
            
        elif opcode == ByteWordInstruction.HALT:
            return False
            
        return True
    
    def run(self) -> None:
        """Execute the assembled program"""
        self.pc = 0
        while self.pc < len(self.program):
            instruction = self.program[self.pc]
            print(f"PC={self.pc:04d}: {instruction.source_line}")
            
            if not self.execute_instruction(instruction):
                break
                
            self.pc += 1
    
    def debug_step(self) -> Optional[AssemblyInstruction]:
        """Execute single instruction in debug mode"""
        if self.pc >= len(self.program):
            return None
            
        instruction = self.program[self.pc]
        print(f"🔍 DEBUG PC={self.pc:04d}: {instruction.source_line}")
        self.print_registers()
        
        if self.execute_instruction(instruction):
            self.pc += 1
            return instruction
        else:
            return None
    
    def print_registers(self) -> None:
        """Print current register state"""
        print("📊 REGISTERS:")
        for reg_name, word in self.registers.items():
            print(f"  {reg_name}: {word}")

# Smalltalk-style Object Browser
class ObjectBrowser:
    """Smalltalk-style browser for ByteWord objects and system inspection"""
    
    def __init__(self, memory_manager: MorphologicalMemoryManager):
        self.memory_manager = memory_manager
        self.current_object = None
        self.history: deque = deque(maxlen=50)
        
    def browse_memory_regions(self) -> Dict[str, Dict]:
        """Browse all memory regions like Smalltalk class browser"""
        regions_info = {}
        for name, region in self.memory_manager.regions.items():
            regions_info[name] = {
                'start_addr': region.start_addr,
                'size': region.size,
                'words_count': len(region.words),
                'quantum_states': self._analyze_quantum_states(region.words),
                'morphological_distribution': self._analyze_morphology(region.words)
            }
        return regions_info
    
    def _analyze_quantum_states(self, words: List[ByteWord]) -> Dict[str, int]:
        """Analyze quantum state distribution"""
        states = defaultdict(int)
        for word in words:
            states[word._quantum_state.name] += 1
        return dict(states)
    
    def _analyze_morphology(self, words: List[ByteWord]) -> Dict[str, int]:
        """Analyze morphological state distribution"""
        morphs = defaultdict(int)
        for word in words:
            morphs[word.floor_morphic.name] += 1
        return dict(morphs)
    
    def inspect_object(self, obj: Any) -> Dict[str, Any]:
        """Inspect any object Smalltalk-style"""
        self.current_object = obj
        self.history.append(obj)
        
        if isinstance(obj, ByteWord):
            return self._inspect_byteword(obj)
        elif isinstance(obj, MemoryRegion):
            return self._inspect_memory_region(obj)
        else:
            return self._inspect_generic(obj)
    
    def _inspect_byteword(self, word: ByteWord) -> Dict[str, Any]:
        """Detailed ByteWord inspection"""
        return {
            'type': 'ByteWord',
            'raw_value': word.raw,
            'binary': f"{word.raw:08b}",
            'hex': f"0x{word.raw:02X}",
            'state_data': {
                'value': word.state_data,
                'binary': f"{word.state_data:04b}"
            },
            'morphism': {
                'value': word.morphism,
                'binary': f"{word.morphism:03b}"
            },
            'floor_morphic': word.floor_morphic.name,
            'pointable': word.pointable,
            'quantum_state': word._quantum_state.name,
            'refcount': word._refcount,
            'observers_count': len(word._observers)
        }
    
    def _inspect_memory_region(self, region: MemoryRegion) -> Dict[str, Any]:
        """Detailed memory region inspection"""
        return {
            'type': 'MemoryRegion',
            'start_addr': region.start_addr,
            'size': region.size,
            'words_preview': [self._inspect_byteword(w) for w in region.words[:10]],
            'quantum_analysis': self._analyze_quantum_states(region.words),
            'morphological_analysis': self._analyze_morphology(region.words),
            'metadata': region.metadata
        }
    
    def _inspect_generic(self, obj: Any) -> Dict[str, Any]:
        """Generic object inspection"""
        return {
            'type': type(obj).__name__,
            'module': type(obj).__module__,
            'attributes': {k: str(v) for k, v in obj.__dict__.items() if not k.startswith('_')},
            'methods': [m for m in dir(obj) if callable(getattr(obj, m)) and not m.startswith('_')],
            'repr': repr(obj)
        }

# Main Development Environment
class MorphologicalIDE:
    """Main IDE combining debugger, assembler, and browser"""
    
    def __init__(self):
        self.memory_manager = MorphologicalMemoryManager()
        self.assembler = MorphologicalAssembler(self.memory_manager)
        self.browser = ObjectBrowser(self.memory_manager)
        self.running = False
        
    def run_repl(self) -> None:
        """Run the main REPL interface"""
        print("🔮 Morphological IDE - ByteWord Development Environment")
        print("Commands: assemble, run, debug, step, browse, inspect, memory, quit")
        
        while True:
            try:
                command = input("\n🧬 morpho> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "assemble":
                    self._handle_assemble()
                elif command == "run":
                    self._handle_run()
                elif command == "debug":
                    self._handle_debug()
                elif command == "step":
                    self._handle_step()
                elif command == "browse":
                    self._handle_browse()
                elif command == "memory":
                    self._handle_memory()
                elif command.startswith("inspect"):
                    self._handle_inspect(command)
                elif command == "help":
                    self._show_help()
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print("\n🛑 Interrupted")
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _handle_assemble(self) -> None:
        """Handle assembly command"""
        print("Enter ByteWord assembly code (end with '---'):")
        lines = []
        while True:
            line = input("    ")
            if line.strip() == "---":
                break
            lines.append(line)
        
        source_code = '\n'.join(lines)
        try:
            instructions = self.assembler.assemble(source_code)
            print(f"✅ Assembled {len(instructions)} instructions")
        except Exception as e:
            print(f"❌ Assembly error: {e}")
    
    def _handle_run(self) -> None:
        """Handle run command"""
        if not self.assembler.program:
            print("❌ No program assembled")
            return
        
        print("🚀 Running program...")
        self.assembler.run()
        print("✅ Program completed")
    
    def _handle_debug(self) -> None:
        """Handle debug command"""
        if not self.assembler.program:
            print("❌ No program assembled")
            return
        
        print("🔍 Debug mode - use 'step' to execute instructions")
        self.assembler.pc = 0
    
    def _handle_step(self) -> None:
        """Handle step command"""
        instruction = self.assembler.debug_step()
        if instruction:
            print(f"✅ Executed: {instruction.source_line}")
        else:
            print("🛑 Program halted or completed")
    
    def _handle_browse(self) -> None:
        """Handle browse command"""
        regions = self.browser.browse_memory_regions()
        if not regions:
            print("📭 No memory regions allocated")
            return
        
        print("🗂️  Memory Regions:")
        for name, info in regions.items():
            print(f"  {name}: {info['size']} words, {info['quantum_states']}")
    
    def _handle_memory(self) -> None:
        """Handle memory allocation"""
        name = input("Region name: ")
        size = int(input("Size (words): "))
        
        region = self.memory_manager.allocate_region(name, size)
        print(f"✅ Allocated region '{name}' with {size} words")
    
    def _handle_inspect(self, command: str) -> None:
        """Handle inspect command"""
        # Simple inspection of registers for now
        if "registers" in command:
            for reg_name, word in self.assembler.registers.items():
                info = self.browser.inspect_object(word)
                print(f"{reg_name}: {info}")
    
    def _show_help(self) -> None:
        """Show help information"""
        print("""
🔮 Morphological IDE Commands:
  assemble  - Enter assembly code for ByteWords
  run       - Execute assembled program
  debug     - Enter debug mode
  step      - Execute single instruction in debug mode
  browse    - Browse memory regions (Smalltalk-style)
  memory    - Allocate new memory region
  inspect   - Inspect objects (try 'inspect registers')
  quit/exit - Exit IDE
        """)

# Example usage and demo
def demo_morphological_ide():
    """Demonstrate the Morphological IDE"""
    ide = MorphologicalIDE()
    
    # Pre-allocate some memory for demo
    ide.memory_manager.allocate_region("main", 16)
    ide.memory_manager.allocate_region("stack", 8)
    
    # Demo assembly program
    demo_program = """
    ; Demo ByteWord program
    LOAD R0, 255        ; Load max value into R0
    LOAD R1, 85         ; Load 0b01010101 into R1
    STORE R0, main, 0   ; Store R0 to main memory
    STORE R1, main, 1   ; Store R1 to main memory
    ENTANGLE R0, R1     ; Create quantum entanglement
    COLLAPSE R0         ; Force collapse of R0
    HALT                ; Stop execution
    """
    
    print("🧪 Demo Program:")
    print(demo_program)
    
    # Assemble and run demo
    try:
        ide.assembler.assemble(demo_program)
        print("\n🚀 Running demo program...")
        ide.assembler.run()
        
        print("\n📊 Final register state:")
        ide.assembler.print_registers()
        
        print("\n🗂️  Memory regions:")
        regions = ide.browser.browse_memory_regions()
        for name, info in regions.items():
            print(f"  {name}: {info}")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    print("Starting Morphological IDE Demo...")
    demo_morphological_ide()
    
    print("\n" + "="*50)
    print("Starting Interactive REPL...")
    ide = MorphologicalIDE()
    ide.run_repl()