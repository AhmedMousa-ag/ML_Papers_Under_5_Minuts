# Dual PatchNorm

Authors: Manoj Kumar, Mostafa Dehghani, Neil Houlsby.

Authors propose Dual PatchNorm which is a two-layer normalization layer before and after the patch embedding layer in vision transformers.

## Introduction

Authors try to improve ViT models. first, they tried different orders of LayerNorm, but they didn't succeed as they found that pre-LN strategy in ViT is close to optimal.

But they observed that placing additional LayerNorms before and after the standard ViT-projection layer, which they call Dual PatchNorm (DPN), can improve significantly over well-tuned ViT baselines.

They did experiment with three different datasets and it demonstrate the efficacy of DPN. They also observed that the LayerNorm scale parameters upweight the pixels at the center and corners of each patch.

```python
hp, wp = patch_size[0], patch_size[1]
x = einops.rearrange(
	x, "b (ht hp) (wt wp) c -> b (ht wt) (hp wp c)", hp=hp, wp=wp)
x = nn.LayerNorm(name="ln0")(x)
x = nn.Dense(output_features, name="dense")(x)
x = nn.LayerNorm(name="ln1")(x)
```

## Background

### Patch Embedding Layer in Vision Transformer

Vision Transformer consists of a patch embedding layer (PE) followed by a stack of Transformer blocks. The PE layer first rearranges the image $x ∈ R^{H×W ×3}$  into a sequence of patches $x_p ∈ R^{\frac{HW}{P^2}×P^2}$ where $P$ denotes the patch size. It then projects each patch independently with a dense projection to constitute a sequence of "visual tokens" $x_t ∈ R^{\frac{HW}{P^2}×D}$, $P controls the trade-off between the granularity of the visual tokens and the computational cost in the subsequent Transformer layers.

### Layer Normalization

Given a sequence of $N$ patches $x ∈ R^{N ×D}$, LayerNorm as applied in ViTs consist of two operations:

**(1)**   $ x = \frac {x-µ(x)}{σ(x)}$

**(2)**   $y = γx + β$

where $µ(x) ∈ R^N , σ(x) ∈ R^N , γ ∈ R^D , β ∈ R^D$.

The first Equation normalizes each patch $x_i ∈ R^D$ of the sequence to have zero mean and unit standard deviation. Then Second Equation applies learnable shifts and scales $β$ and $γ$ which are shared across all patches.

## Methods

### Alternate LayerNorm Placements:

ViTs incorporate LayerNorm before every self-attention and MLP layer, commonly known as the pre-LN strategy. For each of the self-attention and MLP layers, authors evaluated 3 strategies: 1- Place LayerNorm before (pre-LN), 2- after (post-LN), and before and after (pre+post-LN) leading to nine different combinations.

### Dual PatchNorm

Instead of adding LayerNorms to the Transformer block, authors also propose to apply LayerNorms in the stem alone, both before and after the patch embedding layer. In particular, they replace $x=PE(x)$ with $x = LN(PE(LN(x)))$ and keep the rest of the architecture fixed. they call this Dual PatchNorm (DPN).

## Experiments

![image](https://user-images.githubusercontent.com/59775002/218676964-ba556ba2-fde1-4318-b436-7cb96b66e510.png)

### Comparison ToViT


### Avlations and Analysis

Authors assess three alternate strategies: **Pre, Post** and **Post Pos. Pre** applies LayerNorm only to the inputs, **Post** only to the outputs, and **Post PosEmb** to the outputs after being summed with positional embeddings.

Nex table displays the accuracy gains with two alternate strategies: **Pre** is unstable on B/32 leading to a significant drop in accuracy. Additionally, **Pre** obtains minor drops in accuracy on S/32 and Ti/16.

**Post** and **Post PosEmb** achieve worse performance on smaller outputs of the embedding layer is necessary to obtain consistent improvements in accuracy across all ViT variants.

![image](https://user-images.githubusercontent.com/59775002/218677207-43c9f641-efee-4033-b39d-22389af2ab27.png)
