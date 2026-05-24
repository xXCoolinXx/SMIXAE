# Background

Both Francel 2026 (this repository) and Bhalla et. al. 2026 (Do SAEs Capture Concept Manifolds?) currently use the same toy model design. The format of the toy model is as follows

1. Construct a list of manifolds (line segment, sphere, torus, etc.)
2. Sample a random matrix from $\mathbb{R}^{n \times n}$
3. Utilize QR decomposition to get a random orthonormal basis in the space <- THIS STEP IS PROBLEMATIC
4. Generate random points on each manifold, embed the manifold using its own basis (from above), and sum these points together. This is subject to some sparsity condition, so only a few manifolds will be present at once
5. Add a very small noise term and add apply RMSNorm to make it a realistic proxy for an LLM activation. 

# Problems

This approach is subtley wrong, for two key reasons.

## Implicit Reliance on Single Dimensional Compressed Sensing Results

In step 3 of the above toy model recipe, you may note that it samples a random orthonormal basis using QR decomposition. This step is highly problematic, for it does not control for mutual coherence (\mu) of the entire set of manifold vectors. This is not a consideration for single dimensional vectors, for one may pack in exponentially many single dimensional vectors in a single activation with minimal effects on recovery.

However, considering that the \bar{k} value, or average dimensionality of each manifold, is around 2, we must include FAR FEWER manifolds if we wish to ensure recovery. 

## Lack of translation

This model implicitly assumes that the activation space is centered at the origin. While not necessarily a problematic assumption, SMIXAE routes based on group activation norm. Without an affine translation, SMIXAE naturally must center all manifolds at the origin, leading to poor recovery. Considering that the mean of activations in real LLMs is non-zero due to RMSNorm, it seems reasonable that the toy model should also simulate this, until SMIXAE is updated to route based on something other than group activation norm. 

## Lack of torus

Torus needs to be added, with 6 different configurations. 

# Solutions

## Control mutual coherence

The mutual coherence of the manifolds needs to be controlled to ensure optimal recovery and test both SMIXAE and SAEs in a fairer (and, arguably, more realistic) setting. It is conjectured tha the true number of features in the model is closer to $O(d_model/\bar{k})$, so the model should be able to shift the corresponding manifold subspaces until they have low mutual coherence. As such, our toy model should incorporate this concept. 

The specific mathematical tool is the Grassmannian. The actual subspaces we want to obtain for our manifolds implicitly exist on the Grassmannian manifold. The Grassmannian is smooth and compact, and hence we can perform optimization on it to find a set of 3-dimensional subspaces which have low mutual coherence, and then embed each manifold (which lives in $\mathbb{R}^3$) into these subspaces.

This task should use `pymanopt`. 

## Proper Inclusion of Affine Shifts

Currently, the code applies per-manifold random affine shifts. This actually doesn't make any sense, and was based on flawed assumptions about the structure of the activation manifold. 

## Torus

Torus needs to be added, with 6 different configurations. Torus should be embedded in 3 dim space, even though it in principle can have 4 coordinates. 

# Code Quality

The generation of the output plots should be decoupled from the training process itself, so that the generation can be modified on its own without re-training. 
