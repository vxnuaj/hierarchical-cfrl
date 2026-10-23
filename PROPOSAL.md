# Hierarchical Continuous Flow Language Models for Reinforcement Learning

## Hypothesis

Hierarchical continuous representations improve the effectiveness of pathwise reinforcement learning in continuous flow language models compared with flat continuous representations.

## Idea

Continuous flow language models generate text by evolving continuous-valued representations that are eventually decoded into discrete tokens.

Pathwise reinforcement learning allows reward signals from a completed generation to influence the model through its continuous generation trajectory.

We propose introducing **multi-scale hierarchical continuous states** into flow language models and investigating whether this hierarchy makes reinforcement learning more effective.

The intuition is that coarse continuous states may capture global semantic or reasoning structure, while fine continuous states capture token-level details. This separation may improve reward prediction, credit assignment, and learning efficiency.

## Proposed Experiment

Compare a flat continuous flow language model against a hierarchical continuous flow language model, training both with the same pathwise reinforcement learning method.

Measure:
- Learning efficiency, measured by reward achieved per training sample
- Final reasoning accuracy and verifier reward
- Reward prediction accuracy and gradient quality
- Training stability

Control for model size, computational budget, and base language modeling performance to determine whether hierarchy provides a specific benefit for reinforcement learning.

## Research Question

**Does introducing hierarchical continuous representations make pathwise reinforcement learning more effective in continuous flow language models?**

## Related Papers

1. **Backpropagating Verifier Rewards Through Continuous Flow Language Models** (2026). Introduces a critic-based method for propagating verifier rewards through intermediate continuous states in flow language models.
   - https://arxiv.org/abs/2610.19080

2. **Denoising Hierarchical Representations: Joint Continuous Diffusion for Language Modeling** (2026). Explores hierarchical continuous representations by jointly denoising token representations and coarser clusters, including applications to flow language models.
   - https://arxiv.org/abs/2610.08738
