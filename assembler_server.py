#!/usr/bin/env python3
"""
ByteWord Self-Compiled Assembler and Psi Orchestrator
A live debugging environment for morphological computation

Combines:
- Self-assembling ByteWord compilation
- Visual Studio-style memory debugging
- Smalltalk browser for live object inspection
- Multiplayer REPL capabilities
"""

import asyncio
import json
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from collections import defaultdict
import hashlib
import socket
import uuid


class Morphology(Enum):
    """ByteWord morphological states"""
    MORPHIC = 0      # Stable, low-energy, non-pointable
    DYNAMIC = 1      # High-energy, pointable, transformative


class QuantumState(Enum):
    """Computational quantum states"""
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
        self.state_data = (raw >> 4) & 0x0F       # T: High nibble (4 bits)
        self.morphism = (raw >> 1) & 0x07         # V: Middle 3 bits  
        self.floor_morphic = Morphology(raw & 0x01)  # C: LSB
        
        # Quantum properties
        self._quantum_state = QuantumState.SUPERPOSITION
        self._refcount = 0
        self._entangled_with: Set['ByteWord'] = set()
        self._creation_time = time.time()
        
    @property
    def pointable(self) -> bool:
        return self.floor_morphic == Morphology.DYNAMIC
        
    def observe(self) -> 'ByteWord':
        """Quantum observation collapses superposition"""
        if self._quantum_state == QuantumState.SUPERPOSITION:
            self._quantum_state = QuantumState.COLLAPSED
        self._refcount += 1
        return self
        
    def entangle(self, other: 'ByteWord') -> None:
        """Create quantum entanglement between ByteWords"""
        self._entangled_with.add(other)
        other._entangled_with.add(self)
        self._quantum_state = QuantumState.ENTANGLED
        other._quantum_state = QuantumState.ENTANGLED
        
    def decohere(self) -> None:
        """Quantum decoherence - prepare for garbage collection"""
        self._quantum_state = QuantumState.DECOHERENT
        for entangled in self._entangled_with:
            entangled._entangled_with.discard(self)
            
    def __repr__(self) -> str:
        return f"ByteWord(T={self.state_data:04b}, V={self.morphism:03b}, C={self.floor_morphic.value}, Q={self._quantum_state.name})"


@dataclass
class AssemblyInstruction:
    """Assembly instruction operating on ByteWords"""
    opcode: str
    operands: List[ByteWord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    address: Optional[int] = None
    
    def execute(self, context: 'ExecutionContext') -> Any:
        """Execute instruction in given context"""
        return context.execute_instruction(self)


class ExecutionContext:
    """Runtime execution context for ByteWord assembly"""
    
    def __init__(self, memory_size: int = 65536):
        self.memory = bytearray(memory_size)
        self.registers: Dict[str, ByteWord] = {}
        self.byteword_pool: Dict[int, ByteWord] = {}
        self.instruction_pointer = 0
        self.stack: List[ByteWord] = []
        self.quantum_registry: Dict[uuid.UUID, ByteWord] = {}
        
    def allocate_byteword(self, value: int) -> ByteWord:
        """Allocate and track a ByteWord"""
        bw = ByteWord(value)
        bw_id = uuid.uuid4()
        self.quantum_registry[bw_id] = bw
        self.byteword_pool[id(bw)] = bw
        return bw
        
    def execute_instruction(self, instr: AssemblyInstruction) -> Any:
        """Execute a single assembly instruction"""
        method_name = f"_op_{instr.opcode.lower()}"
        if hasattr(self, method_name):
            return getattr(self, method_name)(instr.operands)
        else:
            raise ValueError(f"Unknown opcode: {instr.opcode}")
            
    # ByteWord Assembly Operations
    def _op_load(self, operands: List[ByteWord]) -> None:
        """LOAD reg, value - Load ByteWord into register"""
        if len(operands) != 2:
            raise ValueError("LOAD requires 2 operands")
        reg_name = f"R{operands[0].raw}"
        self.registers[reg_name] = operands[1].observe()
        
    def _op_store(self, operands: List[ByteWord]) -> None:
        """STORE reg, addr - Store register to memory"""
        if len(operands) != 2:
            raise ValueError("STORE requires 2 operands")
        reg_name = f"R{operands[0].raw}"
        addr = operands[1].raw
        if reg_name in self.registers:
            self.memory[addr] = self.registers[reg_name].raw
            
    def _op_entangle(self, operands: List[ByteWord]) -> None:
        """ENTANGLE bw1, bw2 - Create quantum entanglement"""
        if len(operands) != 2:
            raise ValueError("ENTANGLE requires 2 operands")
        operands[0].entangle(operands[1])
        
    def _op_morph(self, operands: List[ByteWord]) -> ByteWord:
        """MORPH bw, target_state - Apply morphological transformation"""
        if len(operands) != 2:
            raise ValueError("MORPH requires 2 operands")
        source = operands[0]
        target_morph = Morphology(operands[1].raw & 1)
        
        # Create new ByteWord with transformed morphology
        new_raw = (source.raw & 0xFE) | target_morph.value
        result = self.allocate_byteword(new_raw)
        return result


class PsiOrchestrator:
    """Orchestrates ByteWord quantum states and morphological transitions"""
    
    def __init__(self):
        self.active_bytewords: Dict[uuid.UUID, ByteWord] = {}
        self.entanglement_graph: Dict[ByteWord, Set[ByteWord]] = defaultdict(set)
        self.morphological_history: List[Tuple[float, ByteWord, str]] = []
        self.observers: List[Callable] = []
        
    def register_byteword(self, bw: ByteWord) -> uuid.UUID:
        """Register ByteWord for orchestration"""
        bw_id = uuid.uuid4()
        self.active_bytewords[bw_id] = bw
        self._notify_observers("register", bw)
        return bw_id
        
    def track_morphology(self, bw: ByteWord, operation: str) -> None:
        """Track morphological state changes"""
        timestamp = time.time()
        self.morphological_history.append((timestamp, bw, operation))
        self._notify_observers("morph", bw, operation)
        
    def analyze_entanglement_topology(self) -> Dict[str, Any]:
        """Analyze quantum entanglement patterns"""
        topology = {
            "total_bytewords": len(self.active_bytewords),
            "entangled_pairs": 0,
            "isolated_bytewords": 0,
            "largest_entanglement_cluster": 0
        }
        
        # Count entangled pairs and clusters
        visited = set()
        clusters = []
        
        for bw in self.active_bytewords.values():
            if bw not in visited and bw._entangled_with:
                cluster = self._explore_entanglement_cluster(bw, visited)
                clusters.append(len(cluster))
                topology["entangled_pairs"] += len(cluster) * (len(cluster) - 1) // 2
            elif not bw._entangled_with:
                topology["isolated_bytewords"] += 1
                
        topology["largest_entanglement_cluster"] = max(clusters) if clusters else 0
        return topology
        
    def _explore_entanglement_cluster(self, start_bw: ByteWord, visited: Set[ByteWord]) -> Set[ByteWord]:
        """DFS to find all ByteWords in entanglement cluster"""
        cluster = set()
        stack = [start_bw]
        
        while stack:
            bw = stack.pop()
            if bw not in visited:
                visited.add(bw)
                cluster.add(bw)
                stack.extend(bw._entangled_with - visited)
                
        return cluster
        
    def add_observer(self, callback: Callable) -> None:
        """Add observer for ByteWord state changes"""
        self.observers.append(callback)
        
    def _notify_observers(self, event: str, *args) -> None:
        """Notify all observers of state changes"""
        for observer in self.observers:
            try:
                observer(event, *args)
            except Exception as e:
                print(f"Observer error: {e}")


class SmalltalkBrowser:
    """Smalltalk-style object browser for live ByteWord inspection"""
    
    def __init__(self, orchestrator: PsiOrchestrator):
        self.orchestrator = orchestrator
        self.current_selection: Optional[ByteWord] = None
        self.inspection_history: List[ByteWord] = []
        
    def browse_bytewords(self) -> Dict[str, Any]:
        """Browse all active ByteWords with hierarchical organization"""
        categories = {
            "morphic": [],
            "dynamic": [], 
            "superposition": [],
            "entangled": [],
            "collapsed": [],
            "decoherent": []
        }
        
        for bw in self.orchestrator.active_bytewords.values():
            # Categorize by morphology
            if bw.floor_morphic == Morphology.MORPHIC:
                categories["morphic"].append(self._byteword_info(bw))
            else:
                categories["dynamic"].append(self._byteword_info(bw))
                
            # Categorize by quantum state
            state_name = bw._quantum_state.name.lower()
            if state_name in categories:
                categories[state_name].append(self._byteword_info(bw))
                
        return categories
        
    def inspect_byteword(self, bw: ByteWord) -> Dict[str, Any]:
        """Deep inspection of a specific ByteWord"""
        self.current_selection = bw
        self.inspection_history.append(bw)
        
        return {
            "identity": {
                "id": id(bw),
                "raw_value": f"0x{bw.raw:02X}",
                "binary": f"{bw.raw:08b}",
                "creation_time": bw._creation_time
            },
            "morphology": {
                "state_data": f"{bw.state_data:04b}",
                "morphism": f"{bw.morphism:03b}",
                "floor_morphic": bw.floor_morphic.name,
                "pointable": bw.pointable
            },
            "quantum": {
                "state": bw._quantum_state.name,
                "refcount": bw._refcount,
                "entangled_count": len(bw._entangled_with),
                "entangled_ids": [id(e) for e in bw._entangled_with]
            },
            "operations": self._available_operations(bw)
        }
        
    def _byteword_info(self, bw: ByteWord) -> Dict[str, Any]:
        """Compact ByteWord information for browsing"""
        return {
            "id": id(bw),
            "value": f"0x{bw.raw:02X}",
            "morphology": bw.floor_morphic.name,
            "quantum_state": bw._quantum_state.name,
            "entangled": len(bw._entangled_with) > 0
        }
        
    def _available_operations(self, bw: ByteWord) -> List[str]:
        """List available operations for ByteWord"""
        ops = ["observe", "decohere"]
        if bw.pointable:
            ops.append("entangle")
        if bw._quantum_state == QuantumState.SUPERPOSITION:
            ops.append("collapse")
        return ops


class MultiplayerREPL:
    """Multiplayer REPL server for collaborative ByteWord development"""
    
    def __init__(self, port: int = 8888):
        self.port = port
        self.clients: Dict[str, Any] = {}
        self.shared_context = ExecutionContext()
        self.orchestrator = PsiOrchestrator()
        self.browser = SmalltalkBrowser(self.orchestrator)
        self.server = None
        
    async def start_server(self):
        """Start the multiplayer REPL server"""
        self.server = await asyncio.start_server(
            self.handle_client, 'localhost', self.port
        )
        print(f"ByteWord Multiplayer REPL started on port {self.port}")
        
    async def handle_client(self, reader, writer):
        """Handle individual client connections"""
        client_id = str(uuid.uuid4())[:8]
        addr = writer.get_extra_info('peername')
        print(f"Client {client_id} connected from {addr}")
        
        self.clients[client_id] = {
            'reader': reader,
            'writer': writer,
            'context': ExecutionContext()
        }
        
        try:
            await self.client_repl_loop(client_id)
        except asyncio.CancelledError:
            pass
        finally:
            writer.close()
            del self.clients[client_id]
            print(f"Client {client_id} disconnected")
            
    async def client_repl_loop(self, client_id: str):
        """Main REPL loop for client"""
        client = self.clients[client_id]
        writer = client['writer']
        reader = client['reader']
        
        # Send welcome message
        welcome = f"ByteWord Assembler REPL - Client {client_id}\n"
        welcome += "Commands: .browse, .inspect <id>, .morph <bw> <state>, .quit\n"
        welcome += "Assembly: LOAD, STORE, ENTANGLE, MORPH\n> "
        
        writer.write(welcome.encode())
        await writer.drain()
        
        while True:
            try:
                data = await reader.readline()
                if not data:
                    break
                    
                command = data.decode().strip()
                if command == '.quit':
                    break
                    
                response = await self.process_command(client_id, command)
                writer.write(f"{response}\n> ".encode())
                await writer.drain()
                
            except Exception as e:
                error_msg = f"Error: {e}\n> "
                writer.write(error_msg.encode())
                await writer.drain()
                
    async def process_command(self, client_id: str, command: str) -> str:
        """Process REPL command"""
        client = self.clients[client_id]
        
        if command.startswith('.browse'):
            categories = self.browser.browse_bytewords()
            return json.dumps(categories, indent=2)
            
        elif command.startswith('.inspect'):
            parts = command.split()
            if len(parts) == 2:
                bw_id = int(parts[1])
                # Find ByteWord by ID
                for bw in self.orchestrator.active_bytewords.values():
                    if id(bw) == bw_id:
                        info = self.browser.inspect_byteword(bw)
                        return json.dumps(info, indent=2)
                return f"ByteWord {bw_id} not found"
            return "Usage: .inspect <byteword_id>"
            
        elif command.startswith('.morph'):
            parts = command.split()
            if len(parts) == 3:
                bw_id = int(parts[1])
                new_state = int(parts[2])
                # Implementation for morphological transformation
                return f"Morphing ByteWord {bw_id} to state {new_state}"
            return "Usage: .morph <byteword_id> <new_state>"
            
        else:
            # Try to parse as assembly instruction
            try:
                return self.assemble_and_execute(client_id, command)
            except Exception as e:
                return f"Assembly error: {e}"
                
    def assemble_and_execute(self, client_id: str, assembly: str) -> str:
        """Assemble and execute ByteWord assembly"""
        client = self.clients[client_id]
        context = client['context']
        
        # Simple assembly parser
        parts = assembly.strip().split()
        if not parts:
            return "Empty instruction"
            
        opcode = parts[0].upper()
        operands = []
        
        # Parse operands as ByteWords
        for operand in parts[1:]:
            if operand.startswith('0x'):
                value = int(operand, 16)
            else:
                value = int(operand)
            operands.append(context.allocate_byteword(value))
            
        # Create and execute instruction
        instr = AssemblyInstruction(opcode, operands)
        result = context.execute_instruction(instr)
        
        # Register ByteWords with orchestrator
        for bw in operands:
            self.orchestrator.register_byteword(bw)
            
        return f"Executed: {assembly} -> {result}"


# Demo/Test Functions
def demo_byteword_system():
    """Demonstrate the ByteWord system"""
    print("=== ByteWord Self-Assembler Demo ===")
    
    # Create orchestrator and browser
    orchestrator = PsiOrchestrator()
    browser = SmalltalkBrowser(orchestrator)
    
    # Create some ByteWords
    bw1 = ByteWord(0b10110100)  # T=1011, V=010, C=0 (MORPHIC)
    bw2 = ByteWord(0b01001011)  # T=0100, V=101, C=1 (DYNAMIC)
    bw3 = ByteWord(0b11110001)  # T=1111, V=000, C=1 (DYNAMIC)
    
    # Register with orchestrator
    for bw in [bw1, bw2, bw3]:
        orchestrator.register_byteword(bw)
    
    # Demonstrate entanglement
    bw2.entangle(bw3)
    
    # Browse ByteWords
    print("\n=== Browsing ByteWords ===")
    categories = browser.browse_bytewords()
    for category, words in categories.items():
        if words:
            print(f"{category.upper()}: {len(words)} ByteWords")
    
    # Inspect specific ByteWord
    print(f"\n=== Inspecting ByteWord {id(bw2)} ===")
    inspection = browser.inspect_byteword(bw2)
    print(json.dumps(inspection, indent=2))
    
    # Analyze entanglement topology
    print("\n=== Entanglement Analysis ===")
    topology = orchestrator.analyze_entanglement_topology()
    print(json.dumps(topology, indent=2))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ByteWord Self-Assembler")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--server", action="store_true", help="Start multiplayer REPL server")
    parser.add_argument("--port", type=int, default=8888, help="Server port")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_byteword_system()
    
    if args.server:
        repl = MultiplayerREPL(args.port)
        asyncio.run(repl.start_server())
        asyncio.get_event_loop().run_forever()
    
    if not args.demo and not args.server:
        print("Use --demo for demonstration or --server to start multiplayer REPL")
    """
        # Run demo to see the system in action
    python byteword_assembler.py --demo

    # Start multiplayer REPL server
    python byteword_assembler.py --server --port 8888

    # Connect with telnet and try commands:
    # telnet localhost 8888
    # > LOAD 1 0xB4
    # > LOAD 2 0x4B  
    # > ENTANGLE 1 2
    # > .browse
    # > .inspect <id>
    """