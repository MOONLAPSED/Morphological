## Requirements

![Win11](/public/Win11%20High%20Contrast%20(Black).png)
- Windows 11
- Windows-search: `Turn Windows features on or off`: Enable:
  - "Containers"
  - "Virtual Machine Platform"
  - "Windows Hypervisor Platform"
  - "Windows Sandbox"
  - "Windows Subsystem for Linux"
- Must be run as Administrator

![this:](/public/Screenshot1.png)


## Usage
1. Open the `/platform/` folder.
2. Double-click `sandbox_config.wsb` (the custom icon shows if Windows Sandbox is enabled).
3. Wait for the terminal to finish auto-setup.
4. When prompted, enter:
   ```bat
   .\invoke_setup.bat
   ```

---
## Temporary; on morphhemes

```python
class Morpheme:
    def __init__(self, source: str, 炁: float):
        """
        # Hanzi for Morphological Computing

        ## Overview

        As we move deeper into morphological source code and computational ontology, traditional symbols like Ψ (psi), ∇ (nabla), and ε (epsilon) no longer serve us.

        Instead, we adopt **ideograms** — characters that **carry meaning in their form**, not just in their function.

        ---

        ## Key Characters

        ### 炁 (qì) – Activation Field / Will / Ψ

        - **Original meaning**: Esoteric term for cosmic energy
        - **In our system**: The energetic cost of staying mutable
        - **Usage**: `self.炁`  
        - **Translation**: Activation field, thermodynamic drive, computational will

        > Example: A high-energy ByteWord has high 炁. When 炁 fades, it collapses to low energy.

        ---

        ### 旋 (xuán) – Spiral / Rotation / Non-Associativity

        - **Original meaning**: Swirl, rotation, vortex
        - **In our system**: Represents the **path dependence** of composition
        - **Usage**: `word.compose(other, mode='旋')`
        - **Translation**: Spiral logic, rotational semantics

        > Example: `(a * b) * c ≠ a * (b * c)` because 旋 changes the result.

        ---

        ### 象 (xiàng) – Morpheme / Symbol / Representation

        - **Original meaning**: Elephant (symbolic representation)
        - **In our system**: The smallest unit of meaningful computation
        - **Usage**: `象 = ByteWord(...)`
        - **Translation**: Morpheme, symbolic value, ByteWord

        > Example: Each 象 carries its own phase, type, and value.

        ---

        ### 衍 (yǎn) – Derivation / Evolution / Propagation

        - **Original meaning**: Derive, evolve, propagate
        - **In our system**: Used for `.propagate()` and `.compose()`
        - **Usage**: `象.衍(steps=10)`
        - **Translation**: Morphological derivation, evolutionary step

        > Example: `象.衍()` evolves the morpheme through time and structure.

        ---

        ### 态 (tài) – State / Phase / Morphology

        - **Original meaning**: State, condition, appearance
        - **In our system**: Tracks phase (high/low energy)
        - **Usage**: `象.tai` or `象.态`
        - **Translation**: Morphological phase, computational state

        > Example: High energy = 态=1; Low energy = 态=0

        ---

        ### 镜 (jìng) – Mirror / Reflexivity / Observation

        - **Original meaning**: Mirror, reflective surface
        - **In our system**: Used for **observation**, **reflection**, and **entanglement**
        - **Usage**: `象.镜(other)` → observe interaction
        - **Translation**: Reflexive operation, entanglement check

        > Example: Two morphemes mirror each other → they annihilate via 炁 cancellation.

        ---

        ## Phrasebook of Morphological Computing

        | Concept | Hanzi | Pinyin | Meaning |
        |--------|-------|--------|---------|
        | Activation Field | 炁 | qì | Will, potential, decay |
        | Morpheme | 象 | xiàng | Smallest unit of morphological meaning |
        | Morphological Derivative | 态衍 | tài yǎn | Evolution over time and structure |
        | Phase Transition | 态转 | tài zhuǎn | Switch between high and low energy |
        | Self-Cancellation | 镜消 | jìng xiāo | Annihilation via opposite 炁 |
        | Composition Rule | 旋构 | xuán gòu | Path-dependent combination |
        | Statistical Coherence | 统协 | tǒng xié | Distributed agreement across runtimes |
        """
        self.source = source
        self.炁 = 炁  # Activation field — the price of will
```

<a href="https://github.com/MOONLAPSED/Morphological">Morphological Source Code</a> © 2025 by <a href="https://github.com/MOONLAPSED">Moonlapsed</a> is licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a><img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" style="max-width: 1em;max-height:1em;margin-left: .2em;">
