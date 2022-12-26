# Scalable Diffusion Models with Transformers

Authors: William Peebles and, Saining Xie.

Authors train latent diffusion models of images replacing the commonly used U-Net backbone with a transformer that operates on latent patches.

## Introductions

Authors focus on a new class of diffusion models based on transformers. They call them Diffusion Transformers, or DiTs for short. DiTs adhere to the best practices of Vision Transformers ViTs.

More specifically, they study the scaling behavior of transformers with respect to network complexity vs sample quality. The successfully replaced the U-Net backbone with a transformer. They further show that DiTs are scalable architectures for diffusion models: there is a strong correlation between the network complexity vs sample quality.

## Diffusion Transformers

### Diffusion Transformer Design Space

$ FIgure 3

DiT is based on the Vision Transformer (ViT) architecture which operates on sequences of patches.

**Patchify:** The input to DiT is a spatial representation $z$ (for $256 × 256 ×  3$ images, $z$ has shape $32  × 32  × 4$). The first layer of DiT is "patchify", which converts the spatial input into a sequence of $T$ tokens, each of dimension $d$, by linearly embedding each patch in the input. Following patchify, authors apply standard ViT fequency-based positional embeddings (the sine-cosine version) to all input tokens. The number of tokens $T$ created by patchify is determined by the patch size hyperparameter $p$. As shown in next Figure, halving $p$ will quadruple $T$, and this *at least* quadruple total transformer Gflops. Although it has a significant impact on Gflops, note that changing $p$ has no meaningful impact on downstream parameter counst.

$ Figure 4

**DiT block design.** Following patchify, the input tokens are processed by a sequence of transofrmer blocks. In addition to noised image inputs, diffusion models sometimes process additional conditional information such as noice timesetps $t$, class labels $c$, natural language, etc. Authors explore four variant of transformer blocks that process conditional inputs differently.

$Figure 5

1- *In-context conditioning*. Simply append the vector embeddings of $t$ and $c$ as two additional tokens in the inputs sequence, treating them no differently from the image tokens. This is similar to $cls$ tokens in ViTs, and it allows us to use standard ViTs blocks without modification. After the final block, the conditioning tokens removed from the sequence. This approach introduces negligible new Gflops to the model.

2- *Cross-attention block*. Concatenate the embedding of $t$ and $c$ into a lenght-two sequence, separate from the image token sequence. The transformer block is modified to include an additional multi-head self-attention block, similar to original design from Vaswani *et al*, and also similar to the oned used by LDM for conditioning on class labels. Cross-attention adds the most Gflops to the model, roughly a *15% overhead.*

3- *Adaptive layer norm (adaLN) block.* Following the widespread usage of adaptive normalization layers in GANs and diffusion models with U-Net backbones, authors explore replacing standard layer norm layers in transformer blocks with adaptive layer norm (adaLN), Rather than directly learn dimension-wise scale and shift parameters $γ$ and $β$, they regress them from the sum of the embedding vectors of $t$ and $c$.

4- *adaLN-Zero block*. Prior work on ResNets has found that initalizing each residual block as the identity function is beneficial. For example, Goyal *et al*. found that zero-initializing the final batch norm scale factor $γ$ in each block accelerates large-scale training in the supervised learning setting. Diffusion U-Net models use a similar initalization strategy, zero-initalizing the final convolutional layer in each block prior to any residual connections. Authors explore a modification of the adaLN DiT block which does the same. In addition to regressing $γ$ and $β$, they also regress dimension-wise scaling parameters $α$ that are applied immediately prior to any residual connections within the DiT block.

**Model Size.** Authors apply a sequence of $N$ DiT blocks, each operating at the hidden dimension size $d$. Following ViT, they use standard transformer configs that jointly scale $N$, $d$ and attentions heads. Specifically, they use four configs: DiT-S, DiT-B, DiT-L and DiT-XL. They cover a wide range of model sizes and flop allocations, from 0.3 to 118.6 Gflops, allowing us to guage scaling performance.

$ Table 1

**Transformer decoder.** After the final DiT block, we need to decode our sequence of image tokens into an output noise prediction and an output diagonal covariance prediction. Both of these outputs have shape equal to the original spatial input. They use a standard linear decoder to do this; they apply the final layer norm (adaptive if using adaLN) and linearly decode each token into a $p×p×2C$ tensor, where $C$ is the number of channels in the spatial input ot DiT. Finally, rearrange the decoded tokens into their original spatial layout to get the predicted noise and covariance.

*The complete DiT design space we explore is patch size,
transformer block architecture and model size.*

## Experiments

**DiT block design.** Authors train four of our highest Gflop DiT-XL/2 models, each using a different block design in-context (119.4 Gflops), cross-attention (137.6 Gflops),adaptive layer norm (adaLN, 118.6 Gflops) or adaLN-zero (118.6 Gflops).

$ Figure 7

## Benchmark

$ Table 2

$ Table 1
