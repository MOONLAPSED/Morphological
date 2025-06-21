import math
import random
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from functools import reduce
from collections import defaultdict

# Physical constants
K_B = 1.380649e-23    # Boltzmann constant (J/K)
T_ENV = 300.0         # Environmental temperature (K)
LN2 = math.log(2.0)   # Natural log of 2

def landauer_cost(bits: int) -> float:
    """Calculate Landauer erasure cost in Joules"""
    return bits * K_B * T_ENV * LN2

class ByteWordNode:
    """
    A ByteWord node with internal density distribution over microstates.
    Represents a morphological unit in the computational groupoid.
    """
    
    def __init__(self, raw: int, internal_states: int = 256):
        self.raw = raw & 0xFF  # Keep 8-bit
        self.internal_states = internal_states
        
        # Initialize uniform density distribution
        self.density = [1.0 / internal_states] * internal_states
        self.neighbors = []
        self.live_mask = [True] * internal_states  # Which microstates are alive
        
        # Morphological properties
        self._entropy = None
        self._coherence = None
        
    def add_neighbor(self, other: 'ByteWordNode'):
        """Add bidirectional neighbor relationship"""
        if other not in self.neighbors:
            self.neighbors.append(other)
            other.neighbors.append(self)
    
    def normalize_density(self):
        """Normalize density to sum to 1.0"""
        total = sum(self.density)
        if total > 0:
            self.density = [d / total for d in self.density]
    
    def prune_dead_states(self, threshold: float = 1e-3):
        """Remove microstates below threshold - ontic death"""
        new_density = []
        new_live_mask = []
        
        for i, (d, alive) in enumerate(zip(self.density, self.live_mask)):
            if alive and d >= threshold:
                new_density.append(d)
                new_live_mask.append(True)
            elif alive:
                # State dies
                new_live_mask.append(False)
        
        # Update only live states
        live_density = [d for d, alive in zip(self.density, self.live_mask) if alive]
        dead_count = len([d for d, alive in zip(self.density, self.live_mask) if not alive])
        
        # Redistribute dead mass to living states
        if live_density and dead_count > 0:
            dead_mass = sum(d for d, alive in zip(self.density, self.live_mask) if not alive)
            redistribution = dead_mass / len(live_density)
            live_density = [d + redistribution for d in live_density]
        
        # Rebuild full arrays
        self.density = []
        for i, alive in enumerate(self.live_mask):
            if alive and live_density:
                self.density.append(live_density.pop(0))
            else:
                self.density.append(0.0)
                self.live_mask[i] = False
        
        self.normalize_density()
    
    def morphological_entropy(self) -> float:
        """Calculate Shannon entropy of density distribution"""
        if self._entropy is not None:
            return self._entropy
            
        entropy = 0.0
        for d in self.density:
            if d > 0:
                entropy -= d * math.log2(d)
        
        self._entropy = entropy
        return entropy
    
    def semantic_coherence(self) -> float:
        """Measure coherence as inverse of entropy normalized"""
        max_entropy = math.log2(len([d for d in self.density if d > 0]))
        if max_entropy == 0:
            return 1.0
        return 1.0 - (self.morphological_entropy() / max_entropy)
    
    def laplacian_step(self, dt: float = 0.1):
        """Single Laplacian diffusion step with neighbors"""
        if not self.neighbors:
            return
        
        # Calculate neighbor average for live states only
        neighbor_densities = []
        for neighbor in self.neighbors:
            # Only consider live microstates
            live_neighbor_density = [
                d for d, alive in zip(neighbor.density, neighbor.live_mask) if alive
            ]
            if live_neighbor_density:
                neighbor_densities.append(live_neighbor_density)
        
        if not neighbor_densities:
            return
        
        # Average neighbor densities (pad/truncate to match)
        max_len = max(len(nd) for nd in neighbor_densities)
        padded_neighbors = []
        for nd in neighbor_densities:
            padded = nd + [0.0] * (max_len - len(nd))
            padded_neighbors.append(padded)
        
        # Calculate average
        avg_neighbor = [
            sum(nd[i] for nd in padded_neighbors) / len(padded_neighbors)
            for i in range(max_len)
        ]
        
        # Get our live density
        our_live_density = [
            d for d, alive in zip(self.density, self.live_mask) if alive
        ]
        
        # Pad to match
        if len(our_live_density) < len(avg_neighbor):
            our_live_density.extend([0.0] * (len(avg_neighbor) - len(our_live_density)))
        elif len(avg_neighbor) < len(our_live_density):
            avg_neighbor.extend([0.0] * (len(our_live_density) - len(avg_neighbor)))
        
        # Laplacian update: neighbor_avg - self
        delta = [avg - self_d for avg, self_d in zip(avg_neighbor, our_live_density)]
        new_live_density = [
            max(0.0, self_d + dt * d) 
            for self_d, d in zip(our_live_density, delta)
        ]
        
        # Normalize
        total = sum(new_live_density)
        if total > 0:
            new_live_density = [d / total for d in new_live_density]
        
        # Update our density array
        live_idx = 0
        for i, alive in enumerate(self.live_mask):
            if alive and live_idx < len(new_live_density):
                self.density[i] = new_live_density[live_idx]
                live_idx += 1
            elif not alive:
                self.density[i] = 0.0
        
        # Clear cached values
        self._entropy = None
        self._coherence = None
    
    def __repr__(self):
        return f"ByteWordNode(0x{self.raw:02X}, entropy={self.morphological_entropy():.3f})"

@dataclass(frozen=True)
class ToroidalByteWord:
    """
    ByteWord on a toroidal manifold with winding numbers and orientation.
    Represents a point in the morphological phase space.
    """
    winding: Tuple[int, int]  # (w1, w2) mod N
    orientation: int          # 0..3 (2 bits of orientation)
    
    @classmethod
    def random(cls, N: int = 256) -> 'ToroidalByteWord':
        """Generate random toroidal ByteWord"""
        return cls(
            winding=(random.randrange(N), random.randrange(N)),
            orientation=random.randrange(4)
        )
    
    def collapse(self, choose: Optional[Tuple[int, int]] = None, N: int = 256) -> Tuple['ToroidalByteWord', float]:
        """
        Collapse superposition to definite state.
        Returns (collapsed_state, landauer_cost_in_joules)
        """
        # Calculate entropy cost - log2 of possible states
        total_states = N * N * 4  # winding pairs × orientations
        bits_erased = math.log2(total_states)
        cost = landauer_cost(int(bits_erased))
        
        if choose is None:
            w1, w2 = random.randrange(N), random.randrange(N)
        else:
            w1, w2 = choose
            
        new_orientation = random.randrange(4)
        collapsed = ToroidalByteWord((w1, w2), new_orientation)
        
        return collapsed, cost
    
    def compose(self, other: 'ToroidalByteWord') -> 'ToroidalByteWord':
        """
        Non-associative composition requiring winding resonance.
        Only succeeds if windings match exactly (resonance condition).
        """
        if self.winding != other.winding:
            raise ValueError(f"Winding mismatch: {self.winding} ≠ {other.winding}")
        
        # XOR orientations (Abelian group operation)
        new_orientation = self.orientation ^ other.orientation
        
        return ToroidalByteWord(self.winding, new_orientation)
    
    def distance(self, other: 'ToroidalByteWord', N: int = 256) -> float:
        """
        Toroidal distance considering winding and orientation
        """
        # Toroidal distance in winding space
        w1_dist = min(abs(self.winding[0] - other.winding[0]), 
                     N - abs(self.winding[0] - other.winding[0]))
        w2_dist = min(abs(self.winding[1] - other.winding[1]),
                     N - abs(self.winding[1] - other.winding[1]))
        
        winding_dist = math.sqrt(w1_dist**2 + w2_dist**2)
        
        # Orientation distance (on circle)
        ori_dist = min(abs(self.orientation - other.orientation),
                      4 - abs(self.orientation - other.orientation))
        
        return winding_dist + ori_dist

class MorphologicalNetwork:
    """
    Network of ByteWordNodes forming a computational groupoid.
    Implements morphodynamics through Laplacian evolution.
    """
    
    def __init__(self):
        self.nodes: List[ByteWordNode] = []
        self.time_step = 0
        self.total_entropy = 0.0
        self.landauer_debt = 0.0  # Accumulated thermodynamic cost
    
    def add_node(self, raw_value: int) -> ByteWordNode:
        """Add new node to network"""
        node = ByteWordNode(raw_value)
        self.nodes.append(node)
        return node
    
    def connect_nodes(self, idx1: int, idx2: int):
        """Connect two nodes by index"""
        if 0 <= idx1 < len(self.nodes) and 0 <= idx2 < len(self.nodes):
            self.nodes[idx1].add_neighbor(self.nodes[idx2])
    
    def step(self, dt: float = 0.1, prune_threshold: float = 1e-3):
        """Single evolution step of the morphodynamic system"""
        # Laplacian diffusion step for all nodes
        for node in self.nodes:
            node.laplacian_step(dt)
        
        # Prune dead states (pays Landauer cost)
        pruning_cost = 0.0
        for node in self.nodes:
            old_live_count = sum(node.live_mask)
            node.prune_dead_states(prune_threshold)
            new_live_count = sum(node.live_mask)
            
            if new_live_count < old_live_count:
                bits_erased = old_live_count - new_live_count
                pruning_cost += landauer_cost(bits_erased)
        
        self.landauer_debt += pruning_cost
        self.time_step += 1
        
        # Update total entropy
        self.total_entropy = sum(node.morphological_entropy() for node in self.nodes)
    
    def evolve(self, steps: int, dt: float = 0.1) -> List[Dict]:
        """Evolve system for multiple steps, returning trajectory"""
        trajectory = []
        
        for _ in range(steps):
            # Record state before step
            state = {
                'time': self.time_step,
                'total_entropy': self.total_entropy,
                'landauer_debt': self.landauer_debt,
                'node_entropies': [node.morphological_entropy() for node in self.nodes],
                'node_coherences': [node.semantic_coherence() for node in self.nodes]
            }
            trajectory.append(state)
            
            # Take evolution step
            self.step(dt)
        
        return trajectory
    
    def semantic_graph(self) -> Dict[int, List[int]]:
        """Return adjacency representation of semantic relationships"""
        graph = defaultdict(list)
        for i, node in enumerate(self.nodes):
            for neighbor in node.neighbors:
                j = self.nodes.index(neighbor)
                graph[i].append(j)
        return dict(graph)
    
    def __repr__(self):
        return f"MorphologicalNetwork({len(self.nodes)} nodes, t={self.time_step}, entropy={self.total_entropy:.3f})"

def demo_morphological_system():
    """Demonstrate the morphological ByteWord system"""
    print("=== Morphological ByteWord System Demo ===\n")
    
    # Create network
    network = MorphologicalNetwork()
    
    # Add some nodes
    node_values = [0b10101010, 0b11001100, 0b11110000, 0b01010101]
    for val in node_values:
        network.add_node(val)
    
    # Connect in a ring topology
    for i in range(len(network.nodes)):
        network.connect_nodes(i, (i + 1) % len(network.nodes))
    
    print(f"Created network: {network}")
    print(f"Semantic graph: {network.semantic_graph()}")
    
    # Show initial node states
    print("\nInitial node states:")
    for i, node in enumerate(network.nodes):
        print(f"  Node {i}: {node}")
    
    # Evolve system
    print(f"\nEvolving system for 10 steps...")
    trajectory = network.evolve(10, dt=0.05)
    
    print(f"Final network: {network}")
    print(f"Total Landauer debt: {network.landauer_debt:.2e} J")
    
    # Show final node states  
    print("\nFinal node states:")
    for i, node in enumerate(network.nodes):
        print(f"  Node {i}: {node}")
    
    # Demonstrate toroidal ByteWords
    print("\n=== Toroidal ByteWord Demo ===")
    
    # Create random toroidal words
    tword1 = ToroidalByteWord.random()
    tword2 = ToroidalByteWord.random()
    
    print(f"Random toroidal word 1: {tword1}")
    print(f"Random toroidal word 2: {tword2}")
    
    # Collapse first word
    collapsed, cost = tword1.collapse()
    print(f"\nCollapsed word 1: {collapsed}")
    print(f"Landauer cost: {cost:.2e} J")
    
    # Try composition (might fail due to winding mismatch)
    try:
        composed = collapsed.compose(collapsed)  # Self-composition
        print(f"Self-composition: {composed}")
    except ValueError as e:
        print(f"Composition failed: {e}")
    
    # Distance calculation
    distance = tword1.distance(tword2)
    print(f"Distance between toroidal words: {distance:.3f}")

if __name__ == "__main__":
    demo_morphological_system()