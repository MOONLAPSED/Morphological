import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Code, Database, Zap, GitBranch, Eye, Cpu, MemoryStick, Play, Pause, StepForward, RotateCw } from 'lucide-react';

// ByteWord implementation matching your specification
class ByteWord {
  constructor(raw) {
    if (!Number.isInteger(raw) || raw < 0 || raw > 255) {
      throw new Error('ByteWord must be 8-bit integer (0-255)');
    }
    
    this.raw = raw;
    this.value = raw & 0xFF;
    this.stateData = (raw >> 4) & 0x0F;      // T: High nibble (4 bits)
    this.morphism = (raw >> 1) & 0x07;       // V: Middle 3 bits  
    this.floorMorphic = raw & 0x01;          // C: LSB (1 bit)
    this.refCount = 1;
    this.quantumState = 'SUPERPOSITION';
  }
  
  get pointable() {
    return this.floorMorphic === 1; // DYNAMIC state
  }
  
  get morphology() {
    return this.floorMorphic === 0 ? 'MORPHIC' : 'DYNAMIC';
  }
  
  toString() {
    return `T=${this.stateData.toString(2).padStart(4, '0')} V=${this.morphism.toString(2).padStart(3, '0')} C=${this.floorMorphic}`;
  }
  
  // XNOR-based Abelian transformation
  static abelianTransform(t, v, c) {
    if (c === 1) {
      return (~(t ^ v)) & 0x0F; // XNOR with 4-bit mask
    }
    return t; // Identity when c = 0
  }
  
  transform() {
    const newT = ByteWord.abelianTransform(this.stateData, this.morphism, this.floorMorphic);
    return new ByteWord((newT << 4) | (this.morphism << 1) | this.floorMorphic);
  }
}

// Memory model for ByteWord storage
class MorphologicalMemory {
  constructor(size = 256) {
    this.memory = new Array(size).fill(0).map((_, i) => new ByteWord(i % 256));
    this.size = size;
    this.watches = new Set();
  }
  
  read(address) {
    if (address < 0 || address >= this.size) {
      throw new Error(`Memory address ${address} out of bounds`);
    }
    return this.memory[address];
  }
  
  write(address, byteWord) {
    if (address < 0 || address >= this.size) {
      throw new Error(`Memory address ${address} out of bounds`);
    }
    this.memory[address] = byteWord;
    this.notifyWatches(address);
  }
  
  addWatch(address) {
    this.watches.add(address);
  }
  
  removeWatch(address) {
    this.watches.delete(address);
  }
  
  notifyWatches(address) {
    if (this.watches.has(address)) {
      console.log(`Watch triggered at address ${address}`);
    }
  }
}

// Assembly language parser for ByteWord operations
class MorphologicalAssembler {
  constructor() {
    this.labels = new Map();
    this.program = [];
  }
  
  parse(source) {
    const lines = source.split('\n').map(line => line.trim()).filter(line => line && !line.startsWith(';'));
    this.program = [];
    this.labels.clear();
    
    // First pass: collect labels
    lines.forEach((line, index) => {
      if (line.endsWith(':')) {
        this.labels.set(line.slice(0, -1), index);
      }
    });
    
    // Second pass: parse instructions
    lines.forEach(line => {
      if (!line.endsWith(':')) {
        this.program.push(this.parseInstruction(line));
      }
    });
    
    return this.program;
  }
  
  parseInstruction(line) {
    const parts = line.split(' ').filter(p => p);
    const opcode = parts[0].toUpperCase();
    const operands = parts.slice(1).map(op => op.replace(',', ''));
    
    return { opcode, operands, original: line };
  }
}

// Main Morphological IDE Component
export default function MorphologicalAssemblerIDE() {
  const [memory] = useState(() => new MorphologicalMemory(64));
  const [assembler] = useState(() => new MorphologicalAssembler());
  const [sourceCode, setSourceCode] = useState(`; Morphological Assembly Example
; T|V|C ByteWord operations

LOAD R0, #42    ; Load immediate value 42 into R0
MORPH R0        ; Apply morphological transformation
STORE R0, @0x10 ; Store result to memory address 0x10
WATCH @0x10     ; Set memory watch point

; Quantum state operations
ENTANGLE R1, R2 ; Create entangled ByteWord pair
COLLAPSE R1     ; Force quantum state collapse
MEASURE R1, R3  ; Measure and store result

; Control flow
LOOP:
  INC R0
  CMP R0, #100
  JNE LOOP

HALT`);
  
  const [program, setProgram] = useState([]);
  const [pc, setPc] = useState(0); // Program counter
  const [registers, setRegisters] = useState({
    R0: new ByteWord(0),
    R1: new ByteWord(0), 
    R2: new ByteWord(0),
    R3: new ByteWord(0)
  });
  const [isRunning, setIsRunning] = useState(false);
  const [selectedAddress, setSelectedAddress] = useState(0);
  const [watchPoints, setWatchPoints] = useState(new Set());
  const [consoleOutput, setConsoleOutput] = useState([]);
  
  const addConsoleOutput = (message, type = 'info') => {
    setConsoleOutput(prev => [...prev, { message, type, timestamp: Date.now() }]);
  };
  
  const assemble = () => {
    try {
      const assembled = assembler.parse(sourceCode);
      setProgram(assembled);
      setPc(0);
      addConsoleOutput('Assembly successful', 'success');
    } catch (error) {
      addConsoleOutput(`Assembly error: ${error.message}`, 'error');
    }
  };
  
  const step = () => {
    if (pc >= program.length) {
      addConsoleOutput('Program terminated', 'info');
      setIsRunning(false);
      return;
    }
    
    const instruction = program[pc];
    addConsoleOutput(`Executing: ${instruction.original}`, 'debug');
    
    // Simple instruction execution simulation
    switch (instruction.opcode) {
      case 'LOAD':
        if (instruction.operands[1].startsWith('#')) {
          const value = parseInt(instruction.operands[1].slice(1));
          setRegisters(prev => ({
            ...prev,
            [instruction.operands[0]]: new ByteWord(value & 0xFF)
          }));
        }
        break;
      case 'MORPH':
        setRegisters(prev => ({
          ...prev,
          [instruction.operands[0]]: prev[instruction.operands[0]].transform()
        }));
        break;
      case 'STORE':
        if (instruction.operands[1].startsWith('@')) {
          const addr = parseInt(instruction.operands[1].slice(1), 16);
          memory.write(addr, registers[instruction.operands[0]]);
        }
        break;
      case 'WATCH':
        if (instruction.operands[0].startsWith('@')) {
          const addr = parseInt(instruction.operands[0].slice(1), 16);
          memory.addWatch(addr);
          setWatchPoints(prev => new Set(prev).add(addr));
        }
        break;
      case 'HALT':
        addConsoleOutput('Program halted', 'success');
        setIsRunning(false);
        return;
    }
    
    setPc(prev => prev + 1);
  };
  
  const run = () => {
    setIsRunning(true);
    const interval = setInterval(() => {
      step();
      if (!isRunning) {
        clearInterval(interval);
      }
    }, 100);
  };
  
  const reset = () => {
    setIsRunning(false);
    setPc(0);
    setRegisters({
      R0: new ByteWord(0),
      R1: new ByteWord(0),
      R2: new ByteWord(0), 
      R3: new ByteWord(0)
    });
    setConsoleOutput([]);
  };
  
  // Auto-assemble on source change
  useEffect(() => {
    const timer = setTimeout(assemble, 500);
    return () => clearTimeout(timer);
  }, [sourceCode]);
  
  return (
    <div className="min-h-screen bg-gray-900 text-green-400 font-mono text-xs">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 p-2 flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h1 className="text-lg font-bold text-green-300">Morphological Assembler & Debugger</h1>
          <div className="flex items-center space-x-2 text-xs">
            <Cpu className="w-4 h-4" />
            <span>ByteWord Architecture</span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button 
            onClick={run} 
            disabled={isRunning}
            className="px-3 py-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded flex items-center space-x-1"
          >
            <Play className="w-3 h-3" />
            <span>Run</span>
          </button>
          <button 
            onClick={step}
            disabled={isRunning}
            className="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 rounded flex items-center space-x-1"
          >
            <StepForward className="w-3 h-3" />
            <span>Step</span>
          </button>
          <button 
            onClick={() => setIsRunning(false)}
            disabled={!isRunning}
            className="px-3 py-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded flex items-center space-x-1"
          >
            <Pause className="w-3 h-3" />
            <span>Pause</span>
          </button>
          <button 
            onClick={reset}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded flex items-center space-x-1"
          >
            <RotateCw className="w-3 h-3" />
            <span>Reset</span>
          </button>
        </div>
      </div>
      
      {/* Main Layout */}
      <div className="flex h-screen">
        {/* Left Panel - Source Code Editor */}
        <div className="w-1/2 border-r border-gray-700 flex flex-col">
          <div className="bg-gray-800 p-2 border-b border-gray-700 flex items-center space-x-2">
            <Code className="w-4 h-4" />
            <span>Morphological Assembly</span>
          </div>
          <textarea
            value={sourceCode}
            onChange={(e) => setSourceCode(e.target.value)}
            className="flex-1 bg-gray-900 text-green-400 p-4 resize-none outline-none"
            spellCheck={false}
          />
        </div>
        
        {/* Right Panel - Split between debugging tools */}
        <div className="w-1/2 flex flex-col">
          {/* Top Right - Memory & Registers */}
          <div className="h-1/2 border-b border-gray-700 flex">
            {/* Memory View */}
            <div className="w-1/2 border-r border-gray-700">
              <div className="bg-gray-800 p-2 border-b border-gray-700 flex items-center space-x-2">
                <MemoryStick className="w-4 h-4" />
                <span>Memory (ByteWords)</span>
              </div>
              <div className="p-2 h-full overflow-auto">
                {Array.from({ length: Math.min(memory.size, 32) }, (_, i) => {
                  const byteWord = memory.read(i);
                  const isWatched = watchPoints.has(i);
                  const isSelected = selectedAddress === i;
                  return (
                    <div 
                      key={i}
                      onClick={() => setSelectedAddress(i)}
                      className={`flex items-center space-x-2 p-1 rounded cursor-pointer hover:bg-gray-800 ${
                        isSelected ? 'bg-blue-800' : ''
                      } ${isWatched ? 'text-yellow-400' : ''}`}
                    >
                      <span className="w-8 text-gray-500">0x{i.toString(16).padStart(2, '0')}</span>
                      <span className="w-12">{byteWord.raw.toString(16).padStart(2, '0')}</span>
                      <span className="text-xs">{byteWord.toString()}</span>
                      <span className="text-xs text-cyan-400">{byteWord.morphology}</span>
                      {isWatched && <Eye className="w-3 h-3 text-yellow-400" />}
                    </div>
                  );
                })}
              </div>
            </div>
            
            {/* Registers */}
            <div className="w-1/2">
              <div className="bg-gray-800 p-2 border-b border-gray-700 flex items-center space-x-2">
                <Database className="w-4 h-4" />
                <span>Registers & State</span>
              </div>
              <div className="p-2">
                <div className="mb-4">
                  <div className="text-sm text-gray-400 mb-2">Program Counter: {pc}</div>
                  <div className="text-sm text-gray-400 mb-2">Status: {isRunning ? 'Running' : 'Stopped'}</div>
                </div>
                
                {Object.entries(registers).map(([name, byteWord]) => (
                  <div key={name} className="mb-2 p-2 bg-gray-800 rounded">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-cyan-400">{name}</span>
                      <span>0x{byteWord.raw.toString(16).padStart(2, '0')}</span>
                    </div>
                    <div className="text-xs text-gray-400">
                      {byteWord.toString()} | {byteWord.morphology}
                    </div>
                    <div className="text-xs text-purple-400">
                      State: {byteWord.quantumState}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* Bottom Right - Console & Program View */}
          <div className="h-1/2 flex">
            {/* Program Disassembly */}
            <div className="w-1/2 border-r border-gray-700">
              <div className="bg-gray-800 p-2 border-b border-gray-700 flex items-center space-x-2">
                <GitBranch className="w-4 h-4" />
                <span>Program</span>
              </div>
              <div className="p-2 h-full overflow-auto">
                {program.map((instruction, index) => (
                  <div 
                    key={index}
                    className={`p-1 rounded mb-1 ${
                      pc === index ? 'bg-yellow-900 text-yellow-200' : 'hover:bg-gray-800'
                    }`}
                  >
                    <span className="text-gray-500 w-8 inline-block">{index.toString().padStart(2, '0')}</span>
                    <span>{instruction.original}</span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Console Output */}
            <div className="w-1/2">
              <div className="bg-gray-800 p-2 border-b border-gray-700 flex items-center space-x-2">
                <Terminal className="w-4 h-4" />
                <span>Console</span>
              </div>
              <div className="p-2 h-full overflow-auto">
                {consoleOutput.map((output, index) => (
                  <div 
                    key={index}
                    className={`text-xs mb-1 ${
                      output.type === 'error' ? 'text-red-400' :
                      output.type === 'success' ? 'text-green-400' :
                      output.type === 'debug' ? 'text-blue-400' :
                      'text-gray-300'
                    }`}
                  >
                    {output.message}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}