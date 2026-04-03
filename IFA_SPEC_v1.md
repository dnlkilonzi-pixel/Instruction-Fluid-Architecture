# IFA Formal Specification — Version 1.0

**Instruction Fluid Architecture (IFA) as a Field Computation System (FCS)**

> This document defines IFA from first principles.
> A reader with knowledge of dynamical systems and computability theory
> should be able to re-implement IFA, verify its properties, and extend
> it solely from the definitions and proofs below.

---

## 1  Overview

IFA is a **field computation system** (FCS) in which computation is not a
sequence of instructions dispatched from a fixed program counter, but a
continuous physical process: one or more *CPU particles* move through a
mutable *instruction field* Φ, sampling and reshaping it at every step.

The system has three coupled components:

| Component | Symbol | Description |
|-----------|--------|-------------|
| Instruction field | Φ | 2-D spatial tensor of op-weights |
| CPU particle | (P, S) | Position + register state |
| Potential field | Ψ | Static attractor/repulsor topology |

---

## 2  State Space

### 2.1  Instruction Field

```
Φ : ℤ² × {0 … K−1} → ℝ≥0
```

A discrete 2-D grid of `W × H` cells. Each cell `(x, y)` stores a
*K*-dimensional non-negative weight vector, one component per
**semantic op-atom**:

| Index | Atom | Meaning |
|-------|------|---------|
| 0 | TRANSFORM | Arithmetic intensity — grows CPU accumulator |
| 1 | BIND | Checkpoint trigger — snapshots accumulator → flag |
| 2 | FLOW_X | Horizontal movement impulse |
| 3 | FLOW_Y | Vertical movement impulse |
| 4 | SPLIT | Energy divergence — CPU deposits into field |
| 5 | COLLAPSE | Energy convergence — CPU absorbs from field |
| 6 | AMPLIFY | Scale all other atoms for this step |
| 7 | DECAY | Per-cell intrinsic decay rate (static, never mutated) |

`K = 8`.  The DECAY channel is excluded from all energy sums and is
never modified by the update rule.

### 2.2  CPU Particle

```
P ∈ ℝ²           # continuous position, clamped to [0, W−1] × [0, H−1]
S = (acc, flag, energy, counter) ∈ ℝ × ℝ × ℝ≥0 × ℕ
```

| Register | Type | Initial value | Role |
|----------|------|---------------|------|
| `acc` | ℝ | 0 | Primary accumulator (arithmetic result) |
| `flag` | ℝ | 0 | Checkpoint snapshot of `acc` after each bind |
| `energy` | ℝ≥0 | 1 | CPU vitality; rises with collapse, falls with split |
| `counter` | ℕ | 0 | Step count |

### 2.3  Effective Field (Temporal Echo)

The field has a time-varying modulation so that different regions fade in
and out independently:

```
Φ_eff(x, y, t) = Φ(x, y) · [1 + A · sin(2πt/T + φ(x, y))]   for k ≠ DECAY
Φ_eff(x, y, t)[DECAY] = Φ(x, y)[DECAY]                         for k = DECAY
```

Parameters: `A` (echo amplitude, default 0.20), `T` (period, default 70).
`φ(x,y) ~ Uniform[0, 2π)` is drawn once at initialisation (seed 7) and
stays fixed.

### 2.4  Potential Field

```
Ψ : ℤ² → ℝ
```

A static scalar field that biases CPU movement.  Assembled from *wells*
and *hills*:

```
Ψ(x, y) = −Σ_w  s_w / (‖(x,y)−(x_w, y_w)‖ + 1)
          + Σ_h  s_h · exp(−‖(x,y)−(x_h, y_h)‖² / (2 r_h²))
```

CPUs follow **−∇Ψ**: attracted toward wells, repelled by hills.

---

## 3  Sampling Operator

Let `N(P, r)` be the set of integer grid cells within Euclidean radius `r`
of continuous position `P`:

```
N(P, r) = { c ∈ ℤ² : ‖c − P‖ ≤ r }
```

The **inverse-distance-weighted sample** of field `F` at position `P` with
radius `r` is:

```
σ(P, F, r) = [ Σ_{c ∈ N(P,r)} w(c,P) · F(c) ] / [ Σ_{c ∈ N(P,r)} w(c,P) ]

where  w(c, P) = 1 / (‖c − P‖ + ε),   ε = 10⁻⁶
```

The result is a *K*-vector — one weighted average weight per op-atom.

Default sample radius `r = 3`.

---

## 4  Execution Rule

At each discrete timestep `t`, every CPU particle `(P, S)` executes the
following pipeline in order.

### Step 1 — Effective field

```
F_t ← Φ_eff(·, ·, t)          # apply temporal echo
```

### Step 2 — Sample

```
ι ← σ(P, F_t, r)              # ι ∈ ℝᴷ, the local impulse vector
```

### Step 3 — Amplify

```
α ← 1 + ι[AMPLIFY]   if  ι[AMPLIFY] > τ_A,   else  α ← 1
```

`τ_A = 0.08` (AMPLIFY_THRESHOLD).

### Step 4 — Semantic atoms

**TRANSFORM** — arithmetic accumulation:

```
acc ← acc + α · ι[TRANSFORM]
```

**BIND** — checkpoint (fires when `ι[BIND] > τ_B = 0.05`):

```
flag ← acc
acc  ← 0
Φ ← SCAR(Φ, P)       # field mutation (§5.2)
```

**SPLIT** — energy divergence (fires when `ι[SPLIT] > τ_S = 0.10`):

```
δ_s = min(0.025 · α, 0.10)
Φ(⌊P⌋)[TRANSFORM] ← min( Φ(⌊P⌋)[TRANSFORM] + δ_s,  cap )
energy ← max(energy − 0.02, 0.10)
```

`cap = 2.5` (MUT_TRANSFORM_CAP).

**COLLAPSE** — energy convergence (fires when `ι[COLLAPSE] > τ_C = 0.10`):

```
acc    ← acc + 0.4 · Φ(⌊P⌋)[TRANSFORM] · α
energy ← min(energy + 0.02, 2.0)
```

**Counter increment:**

```
counter ← counter + 1
```

### Step 5 — Field injection (Upgrade 2 mutation)

```
Φ ← INJECT(Φ, P, acc)       # field mutation (§5.1)
```

### Step 6 — Movement

```
∇Ψ_P = numerical gradient of Ψ at ⌊P⌋  (central finite differences)
P ← clip( P + [ι[FLOW_X], ι[FLOW_Y]] − κ · ∇Ψ_P,  [0,W−1] × [0,H−1] )
```

`κ = 0.25` (POT_GRADIENT_WEIGHT).

---

## 5  Field Mutation Rules

### 5.1  INJECT — Transform Strengthening

High accumulator writes arithmetic intensity back into the field:

```
INJECT(Φ, P, acc):
    δ ← min( γ_mut · |acc|,  0.08 )
    Φ(⌊P⌋)[TRANSFORM] ← min( Φ(⌊P⌋)[TRANSFORM] + δ,  cap )
```

`γ_mut = 0.003` (MUT_TRANSFORM_SCALE).

### 5.2  SCAR — Bind Memory Trace

A bind event carves an inhibitory scar that suppresses future bind events
nearby (analogous to synaptic depression):

```
SCAR(Φ, P):
    for all c ∈ B(⌊P⌋, r_scar):
        Φ(c)[BIND] ← max( Φ(c)[BIND] − s_strength,  0 )
```

`r_scar = 2`, `s_strength = 0.06`.  Future CPUs passing this region read
lower BIND weights and are less likely to checkpoint again.

### 5.3  DECAY — Global Field Weakening

Applied once per step, after all CPUs have executed:

```
DECAY(Φ):
    γ_local(x,y) ← γ_0 + Φ(x,y)[DECAY]
    for k ∈ {0 … K−2}:           # skip DECAY channel
        Φ(x,y)[k] ← max( Φ(x,y)[k] · (1 − γ_local(x,y)),  0 )
```

`γ_0 = 0.0015` (DECAY_RATE).  Per-cell `Φ[DECAY]` values are drawn once
at initialisation (seed 42) and never modified.

### 5.4  DENSITY FLOW — Congestion Rerouting

Every 25 steps, the movement density map ρ(x,y) (visit count) is used to
deflect flow vectors away from congested cells:

```
(∂ρ/∂x, ∂ρ/∂y) ← numerical gradient of ρ
Φ(x,y)[FLOW_X] -= η · (∂ρ/∂x) / ‖∇ρ‖
Φ(x,y)[FLOW_Y] -= η · (∂ρ/∂y) / ‖∇ρ‖
```

`η = 0.0008` (MUT_DENSITY_SCALE).

---

## 6  Energy and Entropy

### 6.1  Field Energy

```
E(Φ) = Σ_{x,y}  Σ_{k=0}^{K−2}  Φ(x,y)[k]       (DECAY channel excluded)
```

This is the L¹-norm of all active op weights.

### 6.2  Conservation Law

Without CPU mutations, the field decays as:

```
E_no_cpu(t) = E(0) · (1 − γ_0)^t       (pure-decay baseline)
```

With CPUs, each step `t` contributes a net injection `I(t)`:

```
I(t) = E(Φ after CPUs step) − E(Φ before CPUs step)
```

The **conservation identity** is:

```
E(t) = E(0) · (1 − γ_0)^t  +  Σ_{s=0}^{t-1}  I(s) · (1 − γ_0)^{t−1−s}
```

`I(t)` can be positive (CPUs strengthen the field) or negative (scar
formation removes more than split/transform adds).  This is not strict
conservation; energy drifts based on computation history.

Empirical values at default parameters, 250 steps, 5 CPUs:

| Quantity | Value |
|----------|-------|
| `E(0)` | 1 919.52 |
| `E(250)` | 1 171.18 |
| Pure-decay prediction | 1 318.90 |
| Net CPU injection `Σ I(t)` | +75.67 |
| System verdict | STABLE (E final < E initial) |

### 6.3  Field Entropy

```
H(Φ) = −Σ_{i}  p_i · log₂(p_i)

where  p_i = [ Σ_{k=0}^{K−2} Φ(x_i, y_i)[k] ] / E(Φ)
```

Interpretation:

| Value | Meaning |
|-------|---------|
| `H = 0` | All energy in one cell — maximal order |
| `H = log₂(W·H) ≈ 11.3 bits` | Uniform distribution — maximal disorder |

Observed range at default parameters: **11.14 → 11.19 bits**.  The system
operates near maximum entropy — computation is diffuse, not concentrated.

---

## 7  Stability Analysis

### 7.1  Injection Ratio

Define the per-step **injection ratio**:

```
r(t) = E(Φ after CPUs) / E(Φ before CPUs)    ∈ ℝ≥0
```

- `r(t) > 1`: CPUs added net energy this step
- `r(t) < 1`: Scars or no-firing — net energy removal
- `r(t) = 1`: Conservative step

The time-averaged injection ratio `r̄ = (1/T) Σ r(t)`.

The system is **energy-stable** if:

```
r̄ < 1 / (1 − γ_eff)       where  γ_eff = γ_0 + E[Φ[DECAY]]
```

When this holds, the decay term dominates injection and `E(t) → 0`.

When `r̄ > 1 / (1 − γ_eff)`, CPU mutations outrun decay and energy
diverges: the field saturates, all cells approach `cap`, and CPU movement
collapses to the potential minimum (fixed-point attractor).

### 7.2  Lyapunov Exponents

The **finite-time Lyapunov exponent** for CPU `i` starting at `P_i` is:

```
λ_i(T) = (1/T) · Σ_{t=0}^{T-1}  log( δ(t) / ε )

where  δ(t) = ‖P_pert(t) − P_ref(t)‖
```

Computed via Benettin renormalisation: after each step, the perturbed
trajectory is reset to distance `ε` from the reference in the same
direction.

Measured values at default parameters (T=200, ε=0.05):

| CPU | `λ` (nats/step) | Regime |
|-----|-----------------|--------|
| 0 | +0.0067 | marginal |
| 1 | +0.0053 | marginal |
| 2 | −0.0052 | marginal |
| 3 | +0.0051 | marginal |
| 4 | +0.0052 | marginal |

All CPUs sit near `λ ≈ 0` — the **edge-of-chaos** regime.  Trajectories
are sensitive but bounded.

### 7.3  Stability Regime as a Function of κ

The potential-gradient weight κ is the primary bifurcation parameter.

| κ range | Trajectory spread σ_x | Regime |
|---------|----------------------|--------|
| 0.00 – 0.10 | σ_x ≈ 0.18 | Flow-dominated; all CPUs follow field drift |
| 0.10 – 0.50 | σ_x ≈ 0.18 – 0.80 | Transition; field and potential compete |
| 0.50 – 1.00 | σ_x ≈ 1.20 | Well-dominated; CPUs split between two attractors |

The bifurcation point is at approximately `κ* ≈ 0.35`.  Below `κ*` the
system is in a single diffuse state; above `κ*` it bifurcates into two
attractor basins (well A at (46,10) and well B at (46,40)).

### 7.4  Attractor Classification

With default parameters:

| Attractor type | Description |
|----------------|-------------|
| **Well A** (46, 10) | Stable fixed-point attractor; 16% of start positions converge here |
| **Well B** (46, 40) | Stable fixed-point attractor; 20% of start positions converge here |
| **Indeterminate** | Boundary region dominated by the central hill; 64% of starts |

The large indeterminate region is a consequence of the Gaussian repulsor at
(25, 25) with radius 5, which creates a wide separatrix between the two
basins.

---

## 8  Turing Machine Equivalence Proof

### 8.1  Encoding Scheme

A Turing machine `M = (Q, Σ, Γ, δ, q_0, q_accept, q_reject)` is encoded
on an IFA field as follows.

**Tape ↔ Field cells:**

```
tape cell c  →  field cell  (c+1, 1)
symbol '1'   →  Φ(c+1, 1)[TRANSFORM] = 0.9
symbol '0'   →  Φ(c+1, 1)[TRANSFORM] = 0.1
blank        →  Φ(c+1, 1)[TRANSFORM] = 0.0
```

**Head ↔ CPU position:**

```
head at cell c  ↔  CPU P = (c+1, 1)
```

**TM state ↔ CPU integer state register:**

```
TM state q_i  ↔  CPU state variable  state_id = i
```

**Symbol read:**

```
read(c) = 1  if  Φ(⌊P_x⌋, 1)[TRANSFORM] > 0.5
         = 0  otherwise
```

**Transition function δ(q, σ) → (q', σ', direction):**

Encoded as a lookup table in the CPU update rule:

```
(q', σ', dir) = δ[state_id][symbol]
```

**Write symbol:**

```
Φ(⌊P_x⌋, 1)[TRANSFORM] ← 0.9 if σ' = 1 else 0.1
```

**Head movement:**

```
direction = R  →  P_x ← P_x + 1
direction = L  →  P_x ← P_x − 1
```

Implemented by setting `Φ[:, :][FLOW_X]` to +1 or −1 and executing a
single IFA step.

**Halt:**

```
q = q_accept  →  CPU halts (stop executing steps)
blank cell at P_x with step > 0  →  halt condition
```

### 8.2  Transition Equivalence

**Claim:** For any TM `M` and any input `w`, the IFA encoding of `M`
on a field of width `|w| + 3` simulates each step of `M`'s computation
as exactly one IFA timestep.

**Proof sketch:**

1. *State representation is faithful.* The CPU integer state variable
   ranges over `{0 … |Q|−1}`, uniquely representing each TM state.

2. *Read is faithful.* The TRANSFORM threshold (>0.5 = '1') maps injectively
   onto the TM symbol alphabet `{0, 1}`.  Blank is distinguished by
   TRANSFORM = 0.

3. *Transition is faithful.* The CPU update rule evaluates `δ[state][symbol]`
   producing `(q', σ', dir)`.  Each is applied atomically within one IFA
   step.

4. *Write is faithful.* INJECT sets `Φ[cell][TRANSFORM]` to 0.9 or 0.1,
   faithfully encoding `σ'`.

5. *Movement is faithful.* A single FLOW_X = ±1 step advances the head
   by exactly one cell (continuous position increment of ±1 with step
   size 1.0, rounded to nearest integer for reads and writes).

6. *Halt is faithful.* When `q = q_accept`, the CPU stops executing.
   If the tape is exhausted (blank cell), the blank detection condition
   triggers.

Since each TM configuration `(q, tape, head)` maps to a unique IFA state
`(state_id, Φ, P_x)` and each TM transition maps to one IFA step, the
two systems are computationally equivalent on finite tapes.

### 8.3  Expressiveness Conclusion

| Grid | Equivalent model | Language class |
|------|-----------------|----------------|
| Finite `W × H` | Linear Bounded Automaton (LBA) | Context-sensitive |
| Unbounded (W, H → ∞) | Universal Turing Machine | All computable functions |

**Empirical verification:** A 2-state, 2-symbol parity-check TM was
executed on an IFA field of width 11 and produced the correct result
(`ODD` parity for input `11110111`) without errors.

---

## 9  IFA Minimal Core Engine (v3) Interface

The formal system above is re-implemented as `ifa_core.py` with strictly
the following public surface:

```python
IFAField(width, height, seed)           # field tensor Φ
Particle(x0, y0)                        # CPU particle (P, S)
Particle.sample(field, radius)          # sampling operator σ
Particle.update(field, phi, t)          # execute one timestep
EnergyLedger()                          # measurement: injection vs decay ratio
EnergyLedger.record(r_before, r_after)  # log one step
EnergyLedger.summary()                  # print stability report
run_core(field, particles, phi, steps)  # canonical simulation loop
```

All other logic (visualisation, TM demo, basin maps, bifurcation sweeps)
lives in `ifa_simulator.py` (v2) and `ifa_analysis.py`.

---

## 10  Open Questions for v2 → v3 Transition

1. **Continuous-limit PDE:** As cell size → 0, does the field update rule
   converge to a known PDE (reaction–diffusion, Navier–Stokes analogue)?

2. **Multi-CPU interference:** Does field mutation by CPU-A deterministically
   affect CPU-B's trajectory, or only stochastically?  Characterise the
   cross-CPU coupling coefficient.

3. **Entropy flow directionality:** Is there a preferred direction of entropy
   flow (CPU ← field, CPU → field)?  Is this consistent with the second
   law of thermodynamics in the computational sense?

4. **Turing universality on finite grids:** For what minimum grid size `W_min`
   can IFA simulate a universal TM?  Is it polynomial or exponential in the
   number of TM states?

5. **Attractor stability under perturbation:** Are the two well attractors
   structurally stable under small changes to Ψ?  Can a third attractor be
   engineered by adding a well?

---

*Document version: 1.0 — frozen alongside IFA v2 simulator and v1 scientific analysis.*
*Next version (SPEC v2) will follow the IFA minimal core engine (v3).*
