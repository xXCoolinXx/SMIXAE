# E Synthetic Experiment Details

## Zoo of Manifolds

(1) Draw from a sparse mixture of manifolds and sum:
$X=\sum Z_{i}V_{i}+b_{i}$

(2) Train SAE

(3) Evaluate Manifold Capture

**Figure 16:** Synthetic Evaluation Pipeline. We construct a controlled benchmark for manifold recovery by sparse autoencoders. (1) We define a zoo of manifolds (spheres, tori, Möbius strips, etc.) and generate data points by sampling from a sparse mixture: each observation is formed as $X=\sum_{i}Z_{i}U_{i}$ where $Z_{i}$ are local coordinates on the i-th manifold and $U_{i}$ are ambient basis matrices embedding each manifold into high-dimensional space. (2) An SAE is trained on the resulting superposed activations. (3) We evaluate whether the SAE recovers the individual manifolds from the mixture, assessing both subspace capture (for direction-based SAEs) and simplicial capture (for point-based SAEs) as defined in Sections 3 and 5.

**Figure 17:** Ising coupling matrix $J_{ij}$ recovers latent manifold structure across sparsity regimes. Red = positive coupling; blue = mutual exclusion $(J<0)$. At low K (tiling), atoms are shared across manifolds and block structure is weak. At intermediate $K(Kpprox8-16)$, clean block-diagonal structure emerges. At high K (dilution), atoms over-tile individual manifolds, fragmenting blocks.

**Manifold zoo.** Table 4 summarizes the eight manifold types used in the synthetic benchmark. For each type, we list the intrinsic dimension $d_{i}$ (the number of free parameters), the embedding dimension $k_{i}$ (the dimension of the ambient subspace containing the manifold, which determines the number of atoms needed for subspace capture), the parametric embedding, and the parameter ranges used across variants.

**Table 4:** Manifold zoo used in the synthetic benchmark.

| Type | $d_{i}$ | $k_{i}$ | Embedding $\gamma_{i}(	heta)\in\mathbb{R}^{k_{i}}$ | Variant parameters |
| :--- | :--- | :--- | :--- | :--- |
| Circle | 1 | 2 | $(r\cos	heta, r\sin	heta)$ | $r\in\{0.5,0.75,1.0,1.5,2.0,3.0\}$ |
| Sphere | 2 | 3 | $(r\sin\phi\cos	heta, r\sin\phi\sin	heta, r\cos\phi)$ | $r\in\{0.5,0.75,1.0,1.5,2.0,3.0\}$ |
| Torus | 2 | 4 | Clifford: $((R+r\cos\phi)\cos	heta, ..., r\sin\phi)$ | $(R,r)\in\{(2,0.5),(2,1),(3,1),...\}$ |
| Möbius | 2 | 3 | $((1+t\cosrac{\phi}{2})\cos\phi, ..., t\sinrac{\phi}{2})$ | $w\in\{0.2,0.3,0.5,0.7,1.0,1.5\}$ |
| Swiss roll | 2 | 3 | $(	heta\cos	heta, h^{2}	heta\sin	heta)$ | $(	heta_{max},h_{max})\in\{(2\pi,1.5),...,(4.5\pi,6)\}$ |
| Helix | 1 | 3 | $(r\cos	heta, r\sin	heta, lpha	heta)$ | $lpha\in\{0.1,0.2,0.3,0.4,0.5,0.6\}$ turns |
| Flat disk | 2 | 2 | $(r\cos	heta, r\sin	heta)$ with $r\sim\sqrt{\mathcal{U}(0,1)}$ | $R\in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0\}$ |
| Segment | 1 | 1 | $(t)$ | $	ext{length} \in \{0.5, 0.75, 1.0, 1.5, 2.0, 3.0\}$ |

The distinction between $d_{i}$ and $k_{i}$ is important throughout the paper. The intrinsic dimension $d_{i}$ governs the manifold's degrees of freedom and determines the expected number of localized detectors in the tiling regime. The embedding dimension $k_{i}$ determines the number of atoms required for subspace capture (Definition 3): a circle is parameterized by a single angle $(d_{i}=1)$ but its embedding $(\cos	heta, \sin	heta)$ lives in a 2-dimensional subspace $(k_{i}=2)$, so two atoms are needed to span it. Similarly, the torus is intrinsically 2-dimensional but requires a 4-dimensional Clifford embedding to faithfully represent its topology.

**Normalization.** A critical design choice is ensuring that all manifold instances contribute equally to the reconstruction loss. Without normalization, manifold types with large embeddings (e.g., the Swiss roll, whose coordinates scale as $	heta\sim3\pi$ to $4.5\pi$) would dominate the SAE's capacity, while small-norm manifolds (e.g., a circle with $r=0.5$) would be treated as noise. We address this by centering and isotropically rescaling each instance at construction time. Concretely, for each manifold instance $i$ with parameters $ho_{i}$, we draw a calibration sample of 50,000 points from the raw embedding $V_{i}$, compute the sample mean $\mu_{i}$ and the RMS norm of the centered samples $\sigma_{i}=\sqrt{\mathbb{E}[||\gamma_{i}(	heta)-\mu_{i}||^{2}]}$, and define the normalized embedding as

$$	ilde{\gamma}_{i}(	heta)=rac{\gamma_{i}(	heta)-\mu_{i}}{\sigma_{i}} \quad (12)$$

This transformation is an isotropic rescaling composed with a translation: it preserves all angles, relative distances, curvature ratios, and topological structure. After normalization, every instance has RMS norm exactly 1 in local coordinates, regardless of manifold type or variant parameters.

**Ambient embedding.** For each of the 48 manifold instances (8 types $	imes$ 6 variants), we draw a random orthonormal matrix $V_{i}\in\mathbb{R}^{k_{i}	imes d}$ by sampling a $d	imes k_{i}$ Gaussian matrix and taking the Q factor of its QR decomposition (transposed to obtain orthonormal rows). This ensures that $||zV_{i}||_{2}=||z||_{2}$ for all $z$: the ambient embedding is norm-preserving. The bias vectors $b_{i}$ are set to zero throughout (i.e., $\sigma_{bias}=0$).

**Sparse mixture sampling.** Observations follow the generative model

$$x=\sum_{i\in S}	ilde{\gamma}_{i}(	heta_{i})V_{i}+\epsilon, \quad |S|=L_{0}, \quad (13)$$

where the active set $S$ is drawn uniformly at random (without replacement) from the 48 instances, intrinsic coordinates $	heta_{i}$ are sampled uniformly on each manifold, and $\epsilon\sim\mathcal{N}(0,\sigma_{\epsilon}^{2}I_{d})$ with $\sigma_{\epsilon}=10^{-5}$. The noise level is deliberately kept small so that reconstruction quality reflects the SAE's geometric organization rather than denoising ability. We generate $N=2,000,000$ training samples. The evaluation set consists of 1,000,000 samples generated at $L_{0}=4$ with a separate random seed, along with the corresponding per-manifold contributions $m_{i}=	ilde{\gamma}_{i}(	heta_{i})V_{i}$ and active masks. This separation ensures that evaluation measures capture on in-distribution superposition with known ground truth.

## SAE Training

We use TopK sparse autoencoders throughout the synthetic experiments. The encoder is a linear map $W_{enc}\in\mathbb{R}^{c	imes d}$ followed by TopK selection (retaining only the $k$ largest activations and zeroing the rest). The decoder is a linear map $W_{dec}\in\mathbb{R}^{d	imes c}$ with unit-norm columns, applied to the sparse code to produce the reconstruction $\hat{x}=W_{dec}	ext{TopK}(W_{enc}x)$. The dictionary size is $c=512$ throughout, yielding an expansion factor of $c/d=4$ relative to the ambient dimension $d=128$.

We train separate SAEs for each sparsity budget $k\in\{3,4,6,8,10,14,16,20,25\}$. This range is chosen to span all three theoretical regimes. All SAEs are trained with Adam (learning rate $3	imes10^{-3}$, no weight decay) for 10 epochs with batch size 1,024. The loss function combines $l_{1}$ reconstruction error with a dead-neuron reanimation term. An atom is considered dead if it has zero activation for every sample in the current batch. The reanimation term encourages dead atoms to develop nonzero pre-activations, preventing capacity waste.

**Restricted $R^{2}$ (subspace capture score).** The primary metric tests Definition 3 directly. For a given SAE trained at sparsity $k$, we proceed as follows:

1. Encode the full evaluation set $\{x^{(j)}\}$ through the SAE to obtain codes $\{z^{(j)}\}$
2. For each manifold instance $i$, select the rows where $i$ is active (using the ground-truth active masks) to obtain the manifold-specific codes $Z_{i}\in\mathbb{R}^{n_{1}	imes c}$ and the corresponding true contributions $M_{i}\in\mathbb{R}^{n_{i}	imes d}$
3. Greedily select $n$ atoms by iteratively choosing the decoder direction $d_{j}$ that explains the most residual variance of $M_{i}$. At each step, the selected atom's projection is removed from the residual before selecting the next.
4. Mask the codes to retain only the $n$ selected atoms: $Z_{i}^{(n)}=Z_{i}\odot e_{selected}$, where $e_{selected}$ is a binary mask.
5. Decode: $\hat{M}_{i}^{(n)}=Z_{i}^{(n)}W_{dec}^{	op}$
6. Compute the restricted $R^{2}$:

$$R^{2}(i,k,n)=1-rac{\sum_{j}||m_{i}^{(j)}-\hat{m}_{i}^{(j,n)}||^{2}}{\sum_{j}||m_{i}^{(j)}-\overline{m}_{i}||^{2}}. \quad (14)$$

where $\overline{m_{i}}$ is the mean of the true contributions. An $R^{2}$ near 1 at $n=k_{i}$ indicates compact subspace capture. We report $R^{2}$ for $n$ ranging from $\max(1,k_{i}-2)$ to $k_{i}+2$ to visualize how capture improves around the embedding dimension. Note that the greedy selection operates on the decoder directions of the trained SAE, not on the codes. This is important: we are asking whether $n$ decoder directions span the manifold's ambient subspace, using the codes the SAE actually produces on in-distribution (superposed) inputs.

**Support size.** For each manifold instance $i$ and SAE sparsity $k$, the support size $|S_{\mathcal{M}}|$ counts the number of unique dictionary atoms that fire on at least 10% of the manifold's evaluation points. To avoid counting near-zero activations (e.g., from ReLU tails or numerical noise), we apply a per-atom magnitude threshold: for each atom $j$, we compute the 10th percentile of its nonzero activations and discard activations below this threshold. Atoms must additionally fire on at least 30 points (an absolute floor) to be counted. This filtering ensures that the support size reflects genuinely active atoms rather than numerical artifacts.

**Receptive field spread.** For each atom $j$ in the support of manifold $i$, we gather all manifold points where $j$ fires (after the robustness filtering described above) and compute the mean pairwise Euclidean distance among those points in ambient space. This quantity measures how broadly the atom's receptive field covers the manifold. We then take the median across all atoms in the support and normalize by the manifold's own mean pairwise distance (computed from a subsample of up to 2,000 points), yielding a dimensionless quantity between 0 (maximally localized: each atom fires on a tight cluster) and 1 (maximally global: each atom fires uniformly across the entire manifold).

**Ising coupling inference.** To recover manifold structure from SAE codes without supervision, we binarize the codes $(s_{j}=	ext{sign}(z_{j}))$ and fit a pairwise Ising model

$$p(s)\propto \exp\left(\sum_{i<j}J_{ij}s_{i}s_{j}+\sum_{i}h_{i}s_{i}ight) \quad (15)$$

using pseudo-likelihood maximization (PLM) with L-BFGS optimization. We enforce symmetry by setting $J=(W+W^{	op})/2$ during optimization rather than as a post-hoc correction. Regularization strength is selected via the extended Bayesian information criterion (EBIC) with $\gamma=0.5$, following the IsingFit procedure of Van Borkulo et al. (2014). The fields $h_{i}$ absorb marginal firing rates, so universally active atoms have large $|h_{i}|$ but small $|J_{ij}|$, and indirectly correlated atoms are factored out by construction. We then apply Louvain community detection to $|J|$ to partition atoms into candidate manifold groups, and validate each group by checking for a sharp PCA spectral gap in its code vectors (indicating low-dimensional structure consistent with a manifold).
