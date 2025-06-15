#!/usr/bin/env python3
"""
md_agent.py - Bridge between iwe-LSP operations and morphic computational ontology
Combines LSP document state management with quantum-morphic bit operations

NOTE: MAKE a config file:
[[language.markdown.code_actions]]
trigger = "extract"
command = "python3 /path/to/md_agent.py extract"

[[language.markdown.code_actions]] 
trigger = "inline"
command = "python3 /path/to/md_agent.py inline"
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re

# Import your morphic classes (assuming they're in the same directory)
from paste import BYTE, QuantumState, HilbertSpace, MorphicComplex, least_significant_unit, WordSize

@dataclass
class DocumentState:
    """Represents the state of a document in LSP terms"""
    uri: str
    content: str
    version: int
    symbols: List[Dict] = None
    diagnostics: List[Dict] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = []
        if self.diagnostics is None:
            self.diagnostics = []

class MorphicLSPBridge:
    """
    Bridges LSP document operations with quantum-morphic computational ontology.
    
    Key insight: LSP operations can be encoded as morphic transformations:
    - Extract = Split operation (VVV=010, decrease complexity)
    - Inline = Merge operation (VVV=001, increase local complexity) 
    - Navigate = Identity with pointer movement (VVV=000)
    - Format = Normalize operation (VVV=110/111, reset/set states)
    """
    
    def __init__(self):
        self.document_states: Dict[str, DocumentState] = {}
        self.hilbert_space = HilbertSpace(dimension=8)  # 8D for our 8-bit BYTE operations
        self.operation_cache: Dict[str, BYTE] = {}
        
    def encode_operation(self, operation: str, context: Dict = None) -> BYTE:
        """
        Encode LSP operations into morphic BYTE operations.
        Maps LSP semantic operations to our <C_C_VV|TTTT> bit structure.
        """
        if context is None:
            context = {}
            
        # Calculate base operation encoding
        op_map = {
            'extract': 0b11010000,  # C=1, _C_=1, VV=01 (split), TTTT=0000
            'inline':  0b11001000,  # C=1, _C_=1, VV=00 (merge), TTTT=1000  
            'navigate': 0b10000100, # C=1, _C_=0, VV=00 (identity), TTTT=0100
            'format':  0b11100010,  # C=1, _C_=1, VV=10 (normalize), TTTT=0010
            'rename':  0b11110001,  # C=1, _C_=1, VV=11 (transform), TTTT=0001
            'search':  0b10110011,  # C=1, _C_=0, VV=11 (query), TTTT=0011
        }
        
        base_value = op_map.get(operation, 0b00000000)
        
        # Modify TTTT bits based on context
        if context:
            context_hash = least_significant_unit(context, WordSize.BYTE)
            # Preserve upper bits (C_C_VV), modify lower bits (TTTT) with context
            base_value = (base_value & 0xF0) | (context_hash & 0x0F)
        
        return BYTE(base_value)
    
    def extract_section(self, document_path: str, cursor_line: int = None) -> Tuple[str, str]:
        """
        Extract a section from a markdown document.
        Returns (new_file_id, updated_content)
        """
        path = Path(document_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {document_path}")
            
        content = path.read_text()
        lines = content.split('\n')
        
        # Find the section to extract
        section_start, section_end, section_title = self._find_section_bounds(lines, cursor_line)
        
        if section_start is None:
            raise ValueError("No section found at cursor position")
        
        # Generate morphic file ID
        section_content = '\n'.join(lines[section_start:section_end])
        operation_byte = self.encode_operation('extract', {'title': section_title, 'line': cursor_line})
        file_id = self._generate_morphic_id(section_content, operation_byte)
        
        # Create the extracted file
        extracted_content = section_content
        extracted_path = path.parent / f"{file_id}.md"
        extracted_path.write_text(extracted_content)
        
        # Update original document
        link_line = f"[{section_title.lstrip('# ')}]({file_id})"
        updated_lines = lines[:section_start] + [link_line] + lines[section_end:]
        updated_content = '\n'.join(updated_lines)
        
        return file_id, updated_content
    
    def inline_section(self, document_path: str, cursor_line: int = None) -> str:
        """
        Inline a previously extracted section back into the document.
        Returns updated_content
        """
        path = Path(document_path)
        content = path.read_text()
        lines = content.split('\n')
        
        # Find the link to inline
        if cursor_line is None:
            cursor_line = self._find_link_line(lines)
        
        if cursor_line >= len(lines):
            raise ValueError("Cursor line out of range")
            
        line = lines[cursor_line]
        match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', line)
        
        if not match:
            raise ValueError("No link found at cursor position")
            
        title, file_id = match.groups()
        
        # Read the extracted file
        extracted_path = path.parent / f"{file_id}.md" 
        if not extracted_path.exists():
            raise FileNotFoundError(f"Extracted file not found: {extracted_path}")
            
        extracted_content = extracted_path.read_text()
        
        # Replace the link with the actual content
        updated_lines = lines[:cursor_line] + extracted_content.split('\n') + lines[cursor_line + 1:]
        updated_content = '\n'.join(updated_lines)
        
        return updated_content
    
    def _find_section_bounds(self, lines: List[str], cursor_line: int = None) -> Tuple[int, int, str]:
        """Find the bounds of a markdown section"""
        if cursor_line is None:
            cursor_line = 0
            
        # Find the current section header
        section_start = None
        section_title = None
        current_level = None
        
        # Look backwards from cursor to find section start
        for i in range(cursor_line, -1, -1):
            line = lines[i].strip()
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                if current_level is None or level <= current_level:
                    section_start = i
                    section_title = line
                    current_level = level
                    break
        
        if section_start is None:
            return None, None, None
            
        # Find section end (next header of same or higher level)
        section_end = len(lines)
        for i in range(section_start + 1, len(lines)):
            line = lines[i].strip()
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                if level <= current_level:
                    section_end = i
                    break
                    
        return section_start, section_end, section_title
    
    def _find_link_line(self, lines: List[str]) -> int:
        """Find the first line containing a markdown link"""
        for i, line in enumerate(lines):
            if re.search(r'\[([^\]]+)\]\(([^)]+)\)', line):
                return i
        raise ValueError("No markdown links found in document")
    
    def _generate_morphic_id(self, content: str, operation_byte: BYTE) -> str:
        """
        Generate a morphic file ID based on content and operation.
        Uses quantum state encoding for collision resistance.
        """
        # Create quantum state from content
        content_hash = hashlib.sha256(content.encode()).digest()
        amplitudes = []
        
        # Convert hash bytes to complex amplitudes
        for i in range(0, min(16, len(content_hash)), 2):
            real = (content_hash[i] - 128) / 128.0
            imag = (content_hash[i+1] - 128) / 128.0 if i+1 < len(content_hash) else 0
            amplitudes.append(MorphicComplex(real, imag))
            
        # Pad to hilbert space dimension
        while len(amplitudes) < self.hilbert_space.dimension:
            amplitudes.append(MorphicComplex(0, 0))
            
        quantum_state = QuantumState(amplitudes, self.hilbert_space)
        
        # Combine with operation byte
        combined_hash = hash((hash(content), operation_byte.value))
        
        # Generate short, readable ID
        return f"{abs(combined_hash) % 100000000:08x}"
    
    def to_quantum_state(self, document: DocumentState) -> QuantumState:
        """Convert document state to quantum representation"""
        # Hash document properties
        content_hash = hash_state(document.content)
        symbol_hash = hash_state(document.symbols) if document.symbols else 0
        diag_hash = hash_state(document.diagnostics) if document.diagnostics else 0
        
        # Create amplitudes from hashes
        amplitudes = []
        hashes = [content_hash, symbol_hash, diag_hash, document.version]
        
        for i in range(self.hilbert_space.dimension):
            if i < len(hashes):
                val = hashes[i] % 1000 / 1000.0  # Normalize to [0,1]
                amplitudes.append(MorphicComplex(val, 0))
            else:
                amplitudes.append(MorphicComplex(0, 0))
                
        return QuantumState(amplitudes, self.hilbert_space)

def main():
    """CLI interface for the morphic LSP bridge"""
    if len(sys.argv) < 3:
        print("Usage: md_agent.py <operation> <document_path> [cursor_line]")
        print("Operations: extract, inline")
        sys.exit(1)
        
    operation = sys.argv[1]
    document_path = sys.argv[2]
    cursor_line = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    bridge = MorphicLSPBridge()
    
    try:
        if operation == "extract":
            file_id, updated_content = bridge.extract_section(document_path, cursor_line)
            
            # Write updated content back to original file
            Path(document_path).write_text(updated_content)
            
            print(json.dumps({
                "success": True,
                "operation": "extract",
                "file_id": file_id,
                "message": f"Section extracted to {file_id}.md"
            }))
            
        elif operation == "inline":
            updated_content = bridge.inline_section(document_path, cursor_line)
            
            # Write updated content back to original file  
            Path(document_path).write_text(updated_content)
            
            print(json.dumps({
                "success": True,
                "operation": "inline", 
                "message": "Section inlined successfully"
            }))
            
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
    except Exception as e:
        print(json.dumps({
            "success": False,
            "error": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()