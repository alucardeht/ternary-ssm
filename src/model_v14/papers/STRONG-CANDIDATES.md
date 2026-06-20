# Strong Paper-Worthy Candidates

This registry tracks verified flagship candidates that meet the project's strict academic and empirical standards.

## Training Dynamics & Speed (R507)

### 1. Krylov-Subspace Recurrent Descent
- **Mechanism**: Condenses $\mathcal{O}(T)$ temporal backpropagation into an $\mathcal{O}(K)$ matrix-free subspace inversion, bypassing the need to unroll the SSM state sequentially.
- **Originality Vector**: Applies Krylov subspace methods directly to the ternary recurrent objective, decoupling sequence length from backward pass latency.
- **Workspace Path**: `.workspace/research/r507-training-speed/`
- **Closeout Evidence Needed**: Live VRAM/Throughput delta proving $\mathcal{O}(K)$ convergence.

### 2. Causal Entropy-Gradient Sheaf
- **Mechanism**: Replaces sequential BPTT with a topological sheaf, allowing exact global gradient recovery with parallelized stalks.
- **Originality Vector**: Mathematical paradigm shift framing causal credit assignment as a sheaf cohomology problem, massively parallelizing the backward pass across time.
- **Workspace Path**: `.workspace/research/r507-training-speed/`
- **Closeout Evidence Needed**: GPU profiling confirming parallel stalk execution without sequence-locking.

### 3. Sparse Sign-Flip Cotangent (SSFC)
- **Mechanism**: Utilizes the Dirac delta nature of discrete boundaries in ternary quantization to compute cotangents via extremely sparse sign-flips instead of dense floats.
- **Originality Vector**: Native SpMV speedup derived fundamentally from the discrete $-1, 0, 1$ state space.
- **Workspace Path**: `.workspace/research/r507-training-speed/`
- **Closeout Evidence Needed**: Empirical verification of sparse vs dense backward latency.

### 4. Discrete-Phase Area Integrator
- **Mechanism**: Reframes backpropagation as a discrete topological boundary flux via Stoke's theorem, avoiding continuous gradient materialization.
- **Originality Vector**: Projects a continuous loss manifold onto a discrete area integral.
- **Workspace Path**: `.workspace/research/r507-training-speed/`
- **Closeout Evidence Needed**: Convergence proof under varying sequence lengths.

### 5. Ternary Latent-Energy Hamiltonian
- **Mechanism**: Establishes a strict Hamiltonian invariant for the forward-backward pass, allowing energy-conserving updates without local approximations.
- **Originality Vector**: Provides an exact mathematical falsifier ($\Delta \text{CE} > 0$) for ternary optimization.
- **Workspace Path**: `.workspace/research/r507-training-speed/`
- **Closeout Evidence Needed**: Loss curve validation ensuring the Hamiltonian invariant holds.

### 6. Ternary Commitment Escrow (Dynamic Relaxation)
- **Mechanism**: Avoids expensive false-discharge repair cycles by tying strict quantization to variance-certified momentum.
- **Originality Vector**: Structural shift in the optimizer step, delaying ternary collapse until momentum clears the variance escrow.
- **Workspace Path**: `.workspace/research/r507-training-speed/`
- **Closeout Evidence Needed**: Step-time measurement verifying reduction in repair cycles.
