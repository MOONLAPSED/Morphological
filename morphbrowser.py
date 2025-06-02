#!/usr/bin/env python3
"""
ByteWord Self-Hosting Assembler/Debugger
A meta-development environment for ByteWord morphological computation

Combines:
- Self-compiled assembler for ByteWord instructions
- Memory debugger with semantic state visualization
- File browser with morphological awareness
- REPL with quantum state introspection
- Smalltalk-style browser for live object manipulation
"""

import sys
import os
import ast
import dis
import traceback
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum, IntEnum
import hashlib
import mimetypes
from collections import defaultdict

# Core ByteWord implementation from your code
class Morphology(Enum):
    MORPHIC = 0      # Stable, low-energy state (cannot be pointed to)
    DYNAMIC = 1      # High-energy state (can be pointed to)

class QuantumState(Enum):
    SUPERPOSITION = 1   # Known by handle only
    ENTANGLED = 2       # Referenced but not loaded  
    COLLAPSED = 4       # Fully materialized
    DECOHERENT = 8      # Garbage collected

class ByteWord:
    """Enhanced 8-bit BYTE_WORD with morphological properties"""
    def __init__(self, raw: int):
        if not 0 <= raw <= 255:
            raise ValueError("ByteWord must be an 8-bit integer (0-255)")
            
        self.raw = raw
        self.value = raw & 0xFF
        
        # Decompose: T(4) V(3) C(1)
        self.state_data = (raw >> 4) & 0x0F       # High nibble
        self.morphism = (raw >> 1) & 0x07         # Middle 3 bits  
        self.floor_morphic = Morphology(raw & 0x01)  # LSB
        
        self._refcount = 1
        self._quantum_state = QuantumState.SUPERPOSITION
        self._entangled_refs = set()

    @property 
    def pointable(self) -> bool:
        return self.floor_morphic == Morphology.DYNAMIC

    def entangle_with(self, other: 'ByteWord'):
        """Create quantum entanglement between ByteWords"""
        self._entangled_refs.add(id(other))
        other._entangled_refs.add(id(self))
        self._quantum_state = QuantumState.ENTANGLED
        other._quantum_state = QuantumState.ENTANGLED

    def collapse(self):
        """Collapse quantum state to definite value"""
        self._quantum_state = QuantumState.COLLAPSED
        return self.value

    def __repr__(self):
        return f"ByteWord(T={self.state_data:04b}, V={self.morphism:03b}, C={self.floor_morphic.value}, Q={self._quantum_state.name})"

@dataclass
class ByteWordInstruction:
    """Assembly instruction operating on ByteWords"""
    opcode: str
    operands: List[Union[ByteWord, int, str]]
    address: int
    source_line: Optional[str] = None
    
class MorphologicalMemory:
    """Memory model with semantic locality and quantum properties"""
    def __init__(self, size: int = 65536):
        self.size = size
        self.memory: List[Optional[ByteWord]] = [None] * size
        self.semantic_clusters: Dict[int, List[int]] = defaultdict(list)
        self.quantum_entanglements: Dict[int, List[int]] = defaultdict(list)
        
    def store(self, address: int, word: ByteWord):
        """Store ByteWord with semantic clustering"""
        if not 0 <= address < self.size:
            raise ValueError(f"Address {address} out of bounds")
            
        self.memory[address] = word
        
        # Cluster semantically similar words
        for addr, existing in enumerate(self.memory):
            if existing and addr != address:
                # Cluster based on morphism similarity
                if abs(word.morphism - existing.morphism) <= 1:
                    self.semantic_clusters[word.morphism].append(address)
                    self.semantic_clusters[existing.morphism].append(addr)
                    
        # Track quantum entanglements
        if word._entangled_refs:
            self.quantum_entanglements[address] = list(word._entangled_refs)
    
    def load(self, address: int) -> Optional[ByteWord]:
        """Load ByteWord and update quantum state"""
        if not 0 <= address < self.size:
            return None
            
        word = self.memory[address]
        if word:
            word.collapse()  # Observation collapses superposition
        return word
    
    def get_semantic_neighborhood(self, address: int, radius: int = 8) -> List[Tuple[int, ByteWord]]:
        """Get semantically related ByteWords in vicinity"""
        if not self.memory[address]:
            return []
            
        center_word = self.memory[address]
        neighborhood = []
        
        # Include address-local neighborhood
        for offset in range(-radius, radius + 1):
            addr = address + offset
            if 0 <= addr < self.size and self.memory[addr]:
                neighborhood.append((addr, self.memory[addr]))
        
        # Include semantic cluster members
        for cluster_addr in self.semantic_clusters[center_word.morphism]:
            if cluster_addr != address and self.memory[cluster_addr]:
                neighborhood.append((cluster_addr, self.memory[cluster_addr]))
                
        return neighborhood

class ByteWordAssembler:
    """Self-hosting assembler for ByteWord instructions"""
    
    OPCODES = {
        'LOAD': 0x01,   'STORE': 0x02,  'ADD': 0x03,    'SUB': 0x04,
        'AND': 0x05,    'OR': 0x06,     'XOR': 0x07,    'NOT': 0x08,
        'XNOR': 0x09,   'JMP': 0x0A,    'JZ': 0x0B,     'JNZ': 0x0C,
        'CALL': 0x0D,   'RET': 0x0E,    'PUSH': 0x0F,   'POP': 0x10,
        'QUINE': 0x11,  'ENTANGLE': 0x12, 'COLLAPSE': 0x13, 'EVOLVE': 0x14,
        'MORPH': 0x15,  'REFLECT': 0x16, 'HALT': 0xFF
    }
    
    def __init__(self):
        self.memory = MorphologicalMemory()
        self.program_counter = 0
        self.stack: List[ByteWord] = []
        self.registers: Dict[str, ByteWord] = {}
        self.labels: Dict[str, int] = {}
        self.instructions: List[ByteWordInstruction] = []
        
    def assemble(self, source: str) -> List[ByteWordInstruction]:
        """Assemble ByteWord assembly source"""
        instructions = []
        lines = source.strip().split('\n')
        address = 0
        
        # First pass: collect labels
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith(';'):
                continue
                
            if line.endswith(':'):
                label = line[:-1]
                self.labels[label] = address
            else:
                address += 1
        
        # Second pass: assemble instructions
        address = 0
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith(';') or line.endswith(':'):
                continue
                
            try:
                instruction = self._parse_instruction(line, address)
                instructions.append(instruction)
                address += 1
            except Exception as e:
                print(f"Assembly error on line {line_num + 1}: {e}")
                
        self.instructions = instructions
        return instructions
    
    def _parse_instruction(self, line: str, address: int) -> ByteWordInstruction:
        """Parse a single assembly instruction"""
        parts = line.replace(',', '').split()
        opcode = parts[0].upper()
        
        if opcode not in self.OPCODES:
            raise ValueError(f"Unknown opcode: {opcode}")
            
        operands = []
        for part in parts[1:]:
            if part.startswith('#'):  # Immediate value
                operands.append(int(part[1:], 0))
            elif part.startswith('@'):  # Memory address
                operands.append(int(part[1:], 0))
            elif part in self.labels:  # Label reference
                operands.append(self.labels[part])
            elif part.startswith('R'):  # Register
                operands.append(part)
            else:
                # Try to parse as ByteWord literal: T:V:C
                if ':' in part:
                    t, v, c = map(int, part.split(':'))
                    raw = (t << 4) | (v << 1) | c
                    operands.append(ByteWord(raw))
                else:
                    operands.append(int(part, 0))
                    
        return ByteWordInstruction(opcode, operands, address, line)

class ByteWordDebugger:
    """Interactive debugger with quantum state visualization"""
    
    def __init__(self, assembler: ByteWordAssembler):
        self.assembler = assembler
        self.breakpoints: set = set()
        self.watch_addresses: set = set()
        self.execution_trace: List[Tuple[int, str]] = []
        
    def set_breakpoint(self, address: int):
        """Set breakpoint at address"""
        self.breakpoints.add(address)
        print(f"Breakpoint set at address {address:04X}")
        
    def watch_memory(self, address: int):
        """Watch memory address for changes"""
        self.watch_addresses.add(address)
        print(f"Watching memory address {address:04X}")
        
    def step(self):
        """Execute single instruction"""
        if self.assembler.program_counter >= len(self.assembler.instructions):
            print("Program terminated")
            return False
            
        instruction = self.assembler.instructions[self.assembler.program_counter]
        self.execution_trace.append((self.assembler.program_counter, instruction.source_line))
        
        print(f"PC:{self.assembler.program_counter:04X} | {instruction.source_line}")
        self._execute_instruction(instruction)
        
        # Check watched memory
        for addr in self.watch_addresses:
            word = self.assembler.memory.load(addr)
            if word:
                print(f"WATCH {addr:04X}: {word}")
                
        return True
        
    def _execute_instruction(self, instruction: ByteWordInstruction):
        """Execute a single ByteWord instruction"""
        opcode = instruction.opcode
        operands = instruction.operands
        
        if opcode == 'LOAD':
            # LOAD R1, @addr or LOAD R1, #immediate
            reg, source = operands
            if isinstance(source, int) and source < self.assembler.memory.size:
                word = self.assembler.memory.load(source)
                if word:
                    self.assembler.registers[reg] = word
            elif isinstance(source, ByteWord):
                self.assembler.registers[reg] = source
                
        elif opcode == 'STORE':
            # STORE R1, @addr
            reg, addr = operands
            if reg in self.assembler.registers:
                self.assembler.memory.store(addr, self.assembler.registers[reg])
                
        elif opcode == 'QUINE':
            # QUINE R1 - create self-reproducing ByteWord
            reg = operands[0]
            if reg in self.assembler.registers:
                word = self.assembler.registers[reg]
                # Create morphogenic fixed point
                quine_word = ByteWord(word.raw)
                quine_word._quantum_state = QuantumState.COLLAPSED
                self.assembler.registers[f"{reg}_QUINE"] = quine_word
                
        elif opcode == 'ENTANGLE':
            # ENTANGLE R1, R2 - create quantum entanglement
            reg1, reg2 = operands
            if reg1 in self.assembler.registers and reg2 in self.assembler.registers:
                word1 = self.assembler.registers[reg1] 
                word2 = self.assembler.registers[reg2]
                word1.entangle_with(word2)
                
        elif opcode == 'COLLAPSE':
            # COLLAPSE R1 - collapse quantum superposition
            reg = operands[0]
            if reg in self.assembler.registers:
                self.assembler.registers[reg].collapse()
                
        elif opcode == 'HALT':
            self.assembler.program_counter = len(self.assembler.instructions)
            return
            
        self.assembler.program_counter += 1
        
    def visualize_memory(self, start: int = 0, count: int = 16):
        """Visualize memory with semantic clustering"""
        print(f"\nMemory Dump (0x{start:04X} - 0x{start+count-1:04X}):")
        print("Addr | Raw  | T:V:C | Morph | Quantum | Entangled")
        print("-" * 55)
        
        for i in range(count):
            addr = start + i
            word = self.assembler.memory.memory[addr]
            if word:
                entangled = "Yes" if word._entangled_refs else "No"
                print(f"{addr:04X} | {word.raw:02X}   | {word.state_data}:{word.morphism}:{word.floor_morphic.value} | "
                      f"{word.floor_morphic.name:7} | {word._quantum_state.name:11} | {entangled}")
            else:
                print(f"{addr:04X} | --   | -:-:- | ------- | ----------- | ---")
                
    def show_semantic_clusters(self):
        """Show semantic clustering of memory"""
        print("\nSemantic Clusters:")
        for morphism, addresses in self.assembler.memory.semantic_clusters.items():
            if addresses:
                print(f"Morphism {morphism}: {[f'0x{addr:04X}' for addr in addresses]}")

class ByteWordREPL:
    """Interactive REPL for ByteWord development"""
    
    def __init__(self):
        self.assembler = ByteWordAssembler()
        self.debugger = ByteWordDebugger(self.assembler)
        self.context = {'assembler': self.assembler, 'debugger': self.debugger}
        
    def run(self):
        """Start the interactive REPL"""
        print("ByteWord Morphological Development Environment")
        print("Type 'help' for commands, 'quit' to exit")
        
        while True:
            try:
                line = input("ByteWord> ").strip()
                if not line:
                    continue
                    
                if line == 'quit':
                    break
                elif line == 'help':
                    self._show_help()
                elif line.startswith('asm'):
                    self._handle_assembly(line)
                elif line.startswith('debug'):
                    self._handle_debug(line)
                elif line.startswith('mem'):
                    self._handle_memory(line)
                elif line.startswith('exec'):
                    self._handle_execution(line)
                else:
                    # Python evaluation in context
                    try:
                        result = eval(line, globals(), self.context)
                        if result is not None:
                            print(result)
                    except:
                        exec(line, globals(), self.context)
                        
            except KeyboardInterrupt:
                print("\nInterrupted")
            except Exception as e:
                print(f"Error: {e}")
                
    def _show_help(self):
        print("""
ByteWord Development Environment Commands:
  asm <code>         - Assemble ByteWord code
  debug step         - Single step execution  
  debug bp <addr>    - Set breakpoint
  debug watch <addr> - Watch memory address
  mem dump [start]   - Show memory dump
  mem clusters       - Show semantic clusters
  exec run           - Run assembled program
  quit               - Exit REPL
  
ByteWord Assembly Syntax:
  LOAD R1, #15:3:1   - Load ByteWord T=15,V=3,C=1 into R1
  STORE R1, @0x100   - Store R1 to memory address 0x100
  QUINE R1           - Create self-reproducing copy
  ENTANGLE R1, R2    - Create quantum entanglement
  COLLAPSE R1        - Collapse quantum state
  HALT               - Stop execution
        """)
        
    def _handle_assembly(self, line: str):
        code = line[3:].strip()
        if code:
            try:
                instructions = self.assembler.assemble(code)
                print(f"Assembled {len(instructions)} instructions")
                for i, instr in enumerate(instructions):
                    print(f"{i:04X}: {instr.source_line}")
            except Exception as e:
                print(f"Assembly error: {e}")
                
    def _handle_debug(self, line: str):
        parts = line.split()
        if len(parts) < 2:
            return
            
        cmd = parts[1]
        if cmd == 'step':
            self.debugger.step()
        elif cmd == 'bp' and len(parts) > 2:
            addr = int(parts[2], 0)
            self.debugger.set_breakpoint(addr)
        elif cmd == 'watch' and len(parts) > 2:
            addr = int(parts[2], 0)
            self.debugger.watch_memory(addr)
            
    def _handle_memory(self, line: str):
        parts = line.split()
        if len(parts) < 2:
            return
            
        cmd = parts[1]
        if cmd == 'dump':
            start = int(parts[2], 0) if len(parts) > 2 else 0
            self.debugger.visualize_memory(start)
        elif cmd == 'clusters':
            self.debugger.show_semantic_clusters()
            
    def _handle_execution(self, line: str):
        parts = line.split()
        if len(parts) < 2:
            return
            
        cmd = parts[1]
        if cmd == 'run':
            self.assembler.program_counter = 0
            while self.assembler.program_counter < len(self.assembler.instructions):
                if self.assembler.program_counter in self.debugger.breakpoints:
                    print(f"Breakpoint hit at {self.assembler.program_counter:04X}")
                    break
                if not self.debugger.step():
                    break

if __name__ == "__main__":
    repl = ByteWordREPL()
    repl.run()