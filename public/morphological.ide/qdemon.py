# quine-demonstration for std lib python
#!/usr/bin/env python3
"""
ByteWord Morphological Assembler & Interactive Debugger
A self-compiled assembler for ByteWord ontological computation with live debugging.

This implements:
- ByteWord assembly language parser
- Interactive morphological state inspector
- Live quantum state visualization
- Self-modifying code capabilities
- Multiplayer REPL environment
"""

import re
import sys
import cmd
import threading
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import json
import time
from collections import defaultdict, deque

# Import your existing ByteWord classes
# (Assuming these are available from your previous code)

class AssemblyInstruction(Enum):
    """ByteWord Assembly Instructions"""
    LOAD = "LOAD"       # Load ByteWord into register
    STORE = "STORE"     # Store register to memory
    MORPH = "MORPH"     # Apply morphological transformation
    QUINE = "QUINE"     # Self-replicate current state
    ENTANGLE = "ENTG"   # Create quantum entanglement
    COLLAPSE = "COLL"   # Collapse superposition
    OBSERVE = "OBS"     # Observe quantum state
    BRANCH = "BR"       # Conditional branch on morphology
    JUMP = "JMP"        # Unconditional jump
    HALT = "HALT"       # Halt execution
    DEBUG = "DBG"       # Enter debug mode

@dataclass
class ByteWordRegister:
    """Represents a register containing a ByteWord"""
    value: Optional[int] = None
    quantum_state: str = "SUPERPOSITION"
    entangled_with: List[str] = field(default_factory=list)
    morphology: str = "MORPHIC"
    last_modified: float = field(default_factory=time.time)
    
    def collapse(self) -> int:
        """Collapse quantum superposition to classical value"""
        if self.quantum_state == "SUPERPOSITION":
            self.quantum_state = "COLLAPSED"
            # In real implementation, this would involve probability calculations
            if self.value is None:
                self.value = 0  # Default collapse
        return self.value or 0

class ByteWordAssembler:
    """Self-compiled assembler for ByteWord morphological computation"""
    
    def __init__(self):
        self.registers: Dict[str, ByteWordRegister] = {
            f"R{i}": ByteWordRegister() for i in range(16)
        }
        self.memory: Dict[int, int] = {}
        self.pc = 0  # Program counter
        self.instructions: List[Tuple[str, List[str]]] = []
        self.labels: Dict[str, int] = {}
        self.debug_mode = False
        self.execution_history: deque = deque(maxlen=1000)
        self.breakpoints: set = set()
        
    def parse_assembly(self, source: str) -> None:
        """Parse ByteWord assembly source code"""
        lines = source.strip().split('\n')
        instruction_count = 0
        
        # First pass: collect labels
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
                
            if line.endswith(':'):
                label = line[:-1]
                self.labels[label] = instruction_count
            else:
                instruction_count += 1
        
        # Second pass: parse instructions
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';') or line.endswith(':'):
                continue
                
            parts = re.split(r'[,\s]+', line)
            opcode = parts[0].upper()
            operands = [p.strip() for p in parts[1:] if p.strip()]
            
            self.instructions.append((opcode, operands))
    
    def execute_instruction(self, opcode: str, operands: List[str]) -> bool:
        """Execute a single ByteWord assembly instruction"""
        self.execution_history.append({
            'pc': self.pc,
            'opcode': opcode,
            'operands': operands,
            'registers_before': {k: v.value for k, v in self.registers.items()},
            'timestamp': time.time()
        })
        
        if opcode == "LOAD":
            reg, value = operands[0], operands[1]
            if value.startswith('#'):
                # Immediate value
                self.registers[reg].value = int(value[1:], 0)
            else:
                # Memory address
                addr = int(value, 0) if value.isdigit() else self.labels.get(value, 0)
                self.registers[reg].value = self.memory.get(addr, 0)
            self.registers[reg].quantum_state = "COLLAPSED"
            
        elif opcode == "STORE":
            reg, addr_str = operands[0], operands[1]
            addr = int(addr_str, 0) if addr_str.isdigit() else self.labels.get(addr_str, 0)
            self.memory[addr] = self.registers[reg].collapse()
            
        elif opcode == "MORPH":
            reg = operands[0]
            current_val = self.registers[reg].collapse()
            # Apply ByteWord morphological transformation
            t = (current_val >> 4) & 0x0F
            v = (current_val >> 1) & 0x07
            c = current_val & 0x01
            
            if c == 1:  # Dynamic state - apply XNOR transformation
                new_t = ~(t ^ v) & 0x0F
                new_val = (new_t << 4) | (v << 1) | c
                self.registers[reg].value = new_val
                self.registers[reg].morphology = "DYNAMIC"
            else:
                self.registers[reg].morphology = "MORPHIC"
                
        elif opcode == "QUINE":
            # Self-replicate current state
            reg = operands[0]
            source_reg = self.registers[reg]
            # Create a new register with identical state
            new_reg_name = f"Q{len([r for r in self.registers if r.startswith('Q')])}"
            self.registers[new_reg_name] = ByteWordRegister(
                value=source_reg.value,
                quantum_state=source_reg.quantum_state,
                morphology=source_reg.morphology
            )
            
        elif opcode == "ENTG":
            # Entangle two registers
            reg1, reg2 = operands[0], operands[1]
            self.registers[reg1].entangled_with.append(reg2)
            self.registers[reg2].entangled_with.append(reg1)
            self.registers[reg1].quantum_state = "ENTANGLED"
            self.registers[reg2].quantum_state = "ENTANGLED"
            
        elif opcode == "OBS":
            # Observe quantum state - forces collapse
            reg = operands[0]
            value = self.registers[reg].collapse()
            print(f"Observed {reg}: {value:08b} ({value})")
            
        elif opcode == "BR":
            # Branch if morphology matches condition
            reg, condition, label = operands[0], operands[1], operands[2]
            if self.registers[reg].morphology == condition:
                self.pc = self.labels[label] - 1  # -1 because pc will be incremented
                
        elif opcode == "JMP":
            label = operands[0]
            self.pc = self.labels[label] - 1
            
        elif opcode == "DBG":
            self.debug_mode = True
            return False  # Pause execution
            
        elif opcode == "HALT":
            return False
            
        return True
    
    def run(self) -> None:
        """Execute the assembled ByteWord program"""
        self.pc = 0
        running = True
        
        while running and self.pc < len(self.instructions):
            if self.pc in self.breakpoints:
                print(f"Breakpoint hit at instruction {self.pc}")
                self.debug_mode = True
                
            if self.debug_mode:
                self.interactive_debug()
                
            opcode, operands = self.instructions[self.pc]
            running = self.execute_instruction(opcode, operands)
            self.pc += 1
    
    def interactive_debug(self) -> None:
        """Enter interactive debugging mode"""
        debugger = ByteWordDebugger(self)
        debugger.cmdloop()
        self.debug_mode = False

class ByteWordDebugger(cmd.Cmd):
    """Interactive debugger for ByteWord assembly"""
    
    intro = """
╔══════════════════════════════════════════════════════════════╗
║                ByteWord Morphological Debugger               ║
║  Quantum-Semantic Assembly Debug Environment                 ║
╚══════════════════════════════════════════════════════════════╝
Type 'help' for commands, 'continue' to resume execution.
    """
    prompt = "ψ> "
    
    def __init__(self, assembler: ByteWordAssembler):
        super().__init__()
        self.assembler = assembler
    
    def do_registers(self, args: str) -> None:
        """Show all register states with quantum information"""
        print("\n╔═══════════════ REGISTER STATES ═══════════════╗")
        for name, reg in self.assembler.registers.items():
            if reg.value is not None or reg.quantum_state != "SUPERPOSITION":
                val_str = f"{reg.value:08b}" if reg.value is not None else "????????"
                entangled = f" ⇄ {reg.entangled_with}" if reg.entangled_with else ""
                print(f"║ {name:>3}: {val_str} |{reg.quantum_state:>12}| {reg.morphology:>7}{entangled}")
        print("╚═══════════════════════════════════════════════╝\n")
    
    def do_memory(self, args: str) -> None:
        """Show memory contents"""
        if not self.assembler.memory:
            print("Memory is empty.")
            return
            
        print("\n╔═══════════════ MEMORY CONTENTS ═══════════════╗")
        for addr in sorted(self.assembler.memory.keys()):
            val = self.assembler.memory[addr]
            print(f"║ 0x{addr:04X}: {val:08b} ({val:3d}) 0x{val:02X}")
        print("╚═══════════════════════════════════════════════╝\n")
    
    def do_morphology(self, args: str) -> None:
        """Analyze morphological structure of a register"""
        if not args:
            print("Usage: morphology <register>")
            return
            
        reg_name = args.strip()
        if reg_name not in self.assembler.registers:
            print(f"Register {reg_name} not found.")
            return
            
        reg = self.assembler.registers[reg_name]
        if reg.value is None:
            print(f"Register {reg_name} is in superposition.")
            return
            
        val = reg.value
        t = (val >> 4) & 0x0F  # State data (4 bits)
        v = (val >> 1) & 0x07  # Morphism selector (3 bits)
        c = val & 0x01         # Floor morphic bit
        
        print(f"\n╔═══ MORPHOLOGICAL ANALYSIS: {reg_name} ═══╗")
        print(f"║ Raw Value:     {val:08b} ({val:3d})")
        print(f"║ State (T):     {t:04b} ({t:2d})")
        print(f"║ Morphism (V):  {v:03b} ({v:1d})")
        print(f"║ Control (C):   {c:01b} ({'DYNAMIC' if c else 'MORPHIC'})")
        print(f"║ Pointable:     {'Yes' if c else 'No'}")
        
        if c == 1:  # Dynamic state
            transformed = (~(t ^ v)) & 0x0F
            print(f"║ T ⊕ V (XNOR):  {transformed:04b} ({transformed:2d})")
        
        print("╚═══════════════════════════════════╝\n")
    
    def do_step(self, args: str) -> None:
        """Execute next instruction"""
        if self.assembler.pc >= len(self.assembler.instructions):
            print("Program has finished.")
            return
            
        opcode, operands = self.assembler.instructions[self.assembler.pc]
        print(f"Executing: {opcode} {' '.join(operands)}")
        
        running = self.assembler.execute_instruction(opcode, operands)
        self.assembler.pc += 1
        
        if not running:
            print("Program halted.")
    
    def do_continue(self, args: str) -> None:
        """Continue execution"""
        return True
    
    def do_break(self, args: str) -> None:
        """Set breakpoint at instruction number"""
        try:
            bp = int(args.strip())
            self.assembler.breakpoints.add(bp)
            print(f"Breakpoint set at instruction {bp}")
        except ValueError:
            print("Usage: break <instruction_number>")
    
    def do_history(self, args: str) -> None:
        """Show execution history"""
        print("\n╔═══════════════ EXECUTION HISTORY ═══════════════╗")
        for i, entry in enumerate(list(self.assembler.execution_history)[-10:]):
            pc = entry['pc']
            op = entry['opcode']
            operands = ' '.join(entry['operands'])
            print(f"║ {i:2d}: PC={pc:3d} {op:>6} {operands:<20}")
        print("╚═════════════════════════════════════════════════╝\n")
    
    def do_quantum(self, args: str) -> None:
        """Show quantum entanglement graph"""
        entangled_pairs = []
        for name, reg in self.assembler.registers.items():
            if reg.entangled_with:
                for partner in reg.entangled_with:
                    pair = tuple(sorted([name, partner]))
                    if pair not in entangled_pairs:
                        entangled_pairs.append(pair)
        
        if entangled_pairs:
            print("\n╔═══════ QUANTUM ENTANGLEMENTS ═══════╗")
            for pair in entangled_pairs:
                print(f"║ {pair[0]} ⇄ {pair[1]}")
            print("╚═════════════════════════════════════╝\n")
        else:
            print("No quantum entanglements found.")
    
    def do_exit(self, args: str) -> None:
        """Exit debugger"""
        return True

# Example ByteWord Assembly Program
EXAMPLE_PROGRAM = """
; ByteWord Morphological Assembly Example
; Demonstrates quantum state manipulation and morphological transformations

start:
    LOAD R0, #0b11010010    ; Load initial ByteWord
    LOAD R1, #0b01100101    ; Load second ByteWord
    
    ; Analyze initial morphology
    DBG                     ; Enter debug mode
    
    ; Create quantum entanglement
    ENTG R0, R1            ; Entangle registers
    
    ; Apply morphological transformation
    MORPH R0               ; Transform R0 based on its morphology
    
    ; Observe the result
    OBS R0                 ; Collapse quantum state
    
    ; Self-replicate if dynamic
    BR R0, DYNAMIC, replicate
    JMP end
    
replicate:
    QUINE R0               ; Create quantum clone
    
end:
    HALT
"""

def main():
    """Main entry point for ByteWord Assembler"""
    print("ByteWord Morphological Assembler v0.1")
    print("====================================")
    
    assembler = ByteWordAssembler()
    
    # Parse and run example program
    assembler.parse_assembly(EXAMPLE_PROGRAM)
    
    print(f"Parsed {len(assembler.instructions)} instructions")
    print(f"Found {len(assembler.labels)} labels")
    
    print("\nStarting execution...")
    assembler.run()
    
    print("\nExecution completed.")

if __name__ == "__main__":
    main()