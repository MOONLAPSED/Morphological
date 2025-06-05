import uuid, random, math
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────
# 1. ChurchWinding: wraps an integer into 8‐bit Church numeral
# ──────────────────────────────────────────────────────────────
class ChurchWinding:
    def __init__(self, raw: int):
        self.raw               = raw  # original signed integer
        self.count             = abs(raw) % 256  # 0..255
        self.direction         = 1 if raw >= 0 else -1

    def apply(self, f: callable, x):
        result = x
        for _ in range(self.count):
            if self.direction > 0:
                result = f(result)
            else:
                if not hasattr(f, 'inverse'):
                    raise ValueError("Function lacks .inverse() for negative winding")
                result = f.inverse(result)
        return result

    def compose(self, other: 'ChurchWinding') -> 'ChurchWinding':
        new_raw_count = (self.count + other.count) % 256
        # Non‐associative rule:
        if self.count > 127:
            new_signed = new_raw_count * self.direction
        else:
            new_signed = new_raw_count * other.direction
        return ChurchWinding(new_signed)


# ──────────────────────────────────────────────────────────────
# 2. TorusOrientation (2 bits)
# ──────────────────────────────────────────────────────────────
class TorusOrientation(Enum):
    TYPE_MAJOR   = 0b00
    VALUE_MINOR  = 0b01
    COMPUTE_TWIST= 0b10
    HOLE_VACUUM  = 0b11


# ──────────────────────────────────────────────────────────────
# 3. ToroidalByteWord: “Quantum of semantic energy”
# ──────────────────────────────────────────────────────────────
class ToroidalByteWord:
    def __init__(self, byte_value: int):
        self.raw_bits = byte_value & 0xFF

        # Decompose into T (2 bits), V (3 bits), C (3 bits)
        t_bits = (byte_value >> 6) & 0b11
        v_bits = (byte_value >> 3) & 0b111
        c_bits = byte_value & 0b111

        self.type_orientation = TorusOrientation(t_bits)
        self.value_winding    = ChurchWinding(v_bits)
        self.compute_twist    = ChurchWinding(c_bits)

        self.energy_level     = self._calc_orbital_energy()
        self.gravitational_neighbors: List['ToroidalByteWord'] = []
        self.pointer_chain: Optional['ToroidalByteWord'] = None

    def _calc_orbital_energy(self) -> float:
        V_norm = self.value_winding.count / 256.0
        C_norm = self.compute_twist.count / 256.0
        return math.sqrt(V_norm*V_norm + C_norm*C_norm)

    def torus_position(self) -> Tuple[float,float,float]:
        theta = 2*math.pi*(self.value_winding.count / 256.0)
        phi   = 2*math.pi*(self.compute_twist.count / 256.0)
        R, r = 2.0, 1.0
        x = (R + r*math.cos(phi))*math.cos(theta)
        y = (R + r*math.cos(phi))*math.sin(theta)
        z = r*math.sin(phi)
        return (x, y, z)

    def compose(self, other: 'ToroidalByteWord') -> 'ToroidalByteWord':
        # 1. Winding interaction
        new_v = self.value_winding.compose(other.value_winding)
        new_c = self.compute_twist.compose(other.compute_twist)

        # 2. Orientation interaction
        new_orient_val = (self.type_orientation.value + other.type_orientation.value) % 4
        new_orientation = TorusOrientation(new_orient_val)

        # 3. Repack into byte
        new_byte = ((new_orientation.value << 6)
                    | ((new_v.count & 0b111) << 3)
                    | (new_c.count & 0b111))
        result = ToroidalByteWord(new_byte)

        # 4. Orbital coupling
        self._establish_orbital_coupling(other, result)
        return result

    def _establish_orbital_coupling(self, other: 'ToroidalByteWord', result: 'ToroidalByteWord'):
        trio = sorted([self, other, result], key=lambda x: x.energy_level, reverse=True)
        center = trio[0]
        for body in trio[1:]:
            center.gravitational_neighbors.append(body)
            body.pointer_chain = center

    def propagate_orbital(self, steps: int = 1) -> List['ToroidalByteWord']:
        history = [self]
        current = self
        seen = set()

        for step in range(steps):
            if current.raw_bits in seen:
                break  # Prevent runaway loops
            seen.add(current.raw_bits)

            bit_count = bin(current.raw_bits).count('1')
            if bit_count >= 4:
                if current.gravitational_neighbors:
                    if len(current.gravitational_neighbors) == 1:
                        n = current.gravitational_neighbors[0]
                        current = n.compose(current)
                    else:
                        # Distribute coupling among all neighbors sequentially
                        for n in current.gravitational_neighbors:
                            current = current.compose(n)
                else:
                    current = current.compose(current)
            else:
                # “Low‐energy” random walk
                theta_shift = (step * 37) % 256
                shifted = (current.raw_bits + theta_shift) % 256
                current = ToroidalByteWord(shifted)

            history.append(current)
        return history

    def measure_topology(self) -> dict:
        pos = self.torus_position()
        return {
            'winding_number_major': self.value_winding.count,
            'winding_number_minor': self.compute_twist.count,
            'spinor_orientation': self.type_orientation.name,
            'orbital_energy': self.energy_level,
            'torus_position': pos,
            'gravitational_mass': len(self.gravitational_neighbors),
            'church_encoding': f"λf.f^{self.value_winding.count}(λx.f^{self.compute_twist.count}(x))"
        }

    def to_float(self) -> float:
        x, y, z = self.torus_position()
        return math.atan2(y, x) / math.pi  # normalized angle [-1..1]

    def says(self) -> str:
        topo = self.measure_topology()
        return (f"I am {topo['church_encoding']} wound "
                f"{topo['winding_number_major']}×{topo['winding_number_minor']} "
                f"around the {topo['spinor_orientation']} axis")


# ──────────────────────────────────────────────────────────────
# 4. QuantumAtom: “Agentic Quine” model
# ──────────────────────────────────────────────────────────────
class QuantumState(Enum):
    SUPERPOSITION = auto()
    ENTANGLED     = auto()
    COLLAPSED     = auto()

# Keep a global registry for lookup:
_ATOM_REGISTRY: Dict[uuid.UUID, 'QuantumAtom'] = {}

def lookup_atom_by_qid(qid: uuid.UUID) -> 'QuantumAtom':
    return _ATOM_REGISTRY[qid]

class QuantumAtom:
    def __init__(self):
        self.qid              = uuid.uuid4()
        self.state            = QuantumState.SUPERPOSITION
        self.gen              = 0
        self.entanglements: Dict[uuid.UUID, List[uuid.UUID]] = {}
        self.child: Optional['QuantumAtom'] = None
        _ATOM_REGISTRY[self.qid] = self

    def entangle_with(self, other: 'QuantumAtom') -> uuid.UUID:
        eid = uuid.uuid4()
        self.entanglements[eid] = [other.qid]
        other.entanglements[eid] = [self.qid]
        self.state = QuantumState.ENTANGLED
        other.state = QuantumState.ENTANGLED
        return eid

    def measure(self, observable: str) -> float:
        # "Non‐collapsing" read
        return random.random()

    def collapse(self, observable: str) -> float:
        val = self.measure(observable)
        self.state = QuantumState.COLLAPSED
        for eid, partners in self.entanglements.items():
            for pqid in partners:
                partner = lookup_atom_by_qid(pqid)
                partner._receive_collapse(eid, observable, val)
        return val

    def _receive_collapse(self, eid: uuid.UUID, observable: str, val: float):
        if self.state == QuantumState.ENTANGLED:
            self.state = QuantumState.COLLAPSED
            # (Callback / logging can happen here.)

    def quine_self(self) -> 'QuantumAtom':
        child = QuantumAtom()
        child.gen = self.gen + 1
        eid = self.entangle_with(child)
        self.child = child
        return child

    def execute(self) -> Optional['QuantumAtom']:
        if self.state == QuantumState.ENTANGLED:
            if random.random() < 0.5:
                return self.quine_self()
            else:
                self.collapse('temporal_position')
                return None
        elif self.state == QuantumState.SUPERPOSITION:
            # Force initial entanglement with a “vacuum” or dummy atom
            dummy = QuantumAtom()
            self.entangle_with(dummy)
            return None
        # If already COLLAPSED, no action
        return None


# ──────────────────────────────────────────────────────────────
# 5. MorphologicalTorusField: “Multi‐Level Universe of ByteWords”
# ──────────────────────────────────────────────────────────────
class MorphologicalTorusField:
    def __init__(self, max_levels: int = 8):
        self.max_levels = max_levels
        self.torus_levels: List[List[ToroidalByteWord]] = []
        for lvl in range(max_levels):
            level_list = []
            capacity = 2 ** lvl
            for pos in range(capacity):
                byte_val = ((lvl & 0b111) << 5) | (pos & 0b11111)
                # Ensures we remain within 8 bits
                tbw = ToroidalByteWord(byte_val)
                level_list.append(tbw)
            self.torus_levels.append(level_list)

    def field_composition(self, lvl1: int, pos1: int, lvl2: int, pos2: int) -> ToroidalByteWord:
        a = self.torus_levels[lvl1][pos1]
        b = self.torus_levels[lvl2][pos2]
        result = a.compose(b)
        target = min(lvl1 + lvl2, self.max_levels - 1)
        self.torus_levels[target].append(result)
        return result

    def field_evolution(self, steps: int = 10) -> dict:
        evolution_log = []
        for step in range(steps):
            interactions = []
            for lvl_idx, level in enumerate(self.torus_levels):
                for idx, tbw in enumerate(level):
                    history = tbw.propagate_orbital(1)
                    if len(history) > 1:
                        interactions.append({
                            'level': lvl_idx,
                            'position': idx,
                            'initial': history[0].measure_topology(),
                            'final': history[-1].measure_topology()
                        })
            evolution_log.append({'step': step, 'interactions': interactions})

        return {
            'evolution_log': evolution_log,
            'final_field_state': self._measure_field_state()
        }

    def _measure_field_state(self) -> dict:
        total_energy = 0.0
        total_windings = {'major': 0, 'minor': 0}
        for level in self.torus_levels:
            for tbw in level:
                total_energy += tbw.energy_level
                topo = tbw.measure_topology()
                total_windings['major'] += topo['winding_number_major']
                total_windings['minor'] += topo['winding_number_minor']
        return {
            'total_field_energy': total_energy,
            'total_windings': total_windings,
            'field_levels': len(self.torus_levels),
            'active_tori': sum(len(lvl) for lvl in self.torus_levels)
        }
