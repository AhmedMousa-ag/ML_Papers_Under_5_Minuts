# Scalable Diffusion Models with Transformers

Authors: William Peebles and, Saining Xie.

Authors train latent diffusion models of images replacing the commonly used U-Net backbone with a transformer that operates on latent patches.

## Introductions

Authors focus on a new class of diffusion models based on transformers. They call them Diffusion Transformers or DiTs for short. DiTs adhere to the best practices of Vision Transformers ViTs.

More specifically, they study the scaling behavior of transformers with respect to network complexity vs sample quality. They successfully replaced the U-Net backbone with a transformer. They further show that DiTs are scalable architectures for diffusion models: there is a strong correlation between the network complexity vs sample quality.

## Diffusion Transformers

### Diffusion Transformer Design Space

![image](https://user-images.githubusercontent.com/59775002/209646034-9e9b2e61-47b9-4f83-a71e-df9bc6956d3c.png)

DiT is based on the Vision Transformer (ViT) architecture which operates on sequences of patches.

**Patchify:** The input to DiT is a spatial representation $z$ (for $256 × 256 ×  3$ images, $z$ has shape $32  × 32  × 4$). The first layer of DiT is "patchify", which converts the spatial input into a sequence of $T$ tokens, each of dimension $d$, by linearly embedding each patch in the input. Following patchify, authors apply standard ViT frequency-based positional embeddings (the sine-cosine version) to all input tokens. The number of tokens $T$ created by patchify is determined by the patch size hyperparameter $p$. As shown in the next Figure, halving $p$ will quadruple $T$, and this *at least* quadruple total transformer Gflops. Although it has a significant impact on Gflops, note that changing $p$ has no meaningful impact on downstream parameter counts.

![image](https://user-images.githubusercontent.com/59775002/209646194-ba48fc8b-16b4-4a5a-894d-8c3dd1eaaf64.png)

**DiT block design.** Following patchify, the input tokens are processed by a sequence of transformer blocks. In addition to noised image inputs, diffusion models sometimes process additional conditional information such as noise timesetps $t$, class labels $c$, natural language, etc. Authors explore four variants of transformer blocks that process conditional inputs differently.

![image](https://user-images.githubusercontent.com/59775002/209646247-a63c77b5-91a8-4a1f-8606-528a1bc412e5.png)

1- *In-context conditioning*. Simply append the vector embeddings of $t$ and $c$ as two additional tokens in the input sequence, treating them no differently from the image tokens. This is similar to $cls$ tokens in ViTs, and it allows us to use standard ViTs blocks without modification. After the final block, the conditioning tokens removed from the sequence. This approach introduces negligible new Gflops to the model.

2- *Cross-attention block*. Concatenate the embedding of $t$ and $c$ into a length-two sequence, separate from the image token sequence. The transformer block is modified to include an additional multi-head self-attention block, similar to the original design from Vaswani *et al*, and also similar to the one used by LDM for conditioning on class labels. Cross-attention adds the most Gflops to the model, roughly a *15% overhead.*

3- *Adaptive layer norm (adaLN) block.* Following the widespread usage of adaptive normalization layers in GANs and diffusion models with U-Net backbones, authors explore replacing standard layer norm layers in transformer blocks with adaptive layer norm (adaLN), Rather than directly learn dimension-wise scale and shift parameters $γ$ and $β$, they regress them from the sum of the embedding vectors of $t$ and $c$.

4- *adaLN-Zero block*. Prior work on ResNets has found that initializing each residual block as the identity function is beneficial. For example, Goyal *et al*. found that zero-initializing the final batch norm scale factor $γ$ in each block accelerates large-scale training in the supervised learning setting. Diffusion U-Net models use a similar initialization strategy, zero-initializing the final convolutional layer in each block prior to any residual connections. Authors explore a modification of the adaLN DiT block which does the same. In addition to regressing $γ$ and $β$, they also regress dimension-wise scaling parameters $α$ that are applied immediately prior to any residual connections within the DiT block.

**Model Size.** Authors apply a sequence of $N$ DiT blocks, each operating at the hidden dimension size $d$. Following ViT, they use standard transformer configs that jointly scale $N$, $d$ and attentions heads. Specifically, they use four configs: DiT-S, DiT-B, DiT-L and DiT-XL. They cover a wide range of model sizes and flop allocations, from 0.3 to 118.6 Gflops, allowing us to gauge scaling performance.

![image](https://user-images.githubusercontent.com/59775002/209646764-d039eeca-cb45-43bc-aab0-83a783fbcefa.png)

**Transformer decoder.** After the final DiT block, we need to decode our sequence of image tokens into an output noise prediction and an output diagonal covariance prediction. Both of these outputs have shape equal to the original spatial input. They use a standard linear decoder to do this; they apply the final layer norm (adaptive if using adaLN) and linearly decode each token into a $p×p×2C$ tensor, where $C$ is the number of channels in the spatial input to DiT. Finally, rearrange the decoded tokens into their original spatial layout to get the predicted noise and covariance.

*The complete DiT design space we explore is patch size,
transformer block architecture, and model size.*

## Experiments

**DiT block design.** Authors train four of our highest Gflop DiT-XL/2 models, each using a different block design in-context (119.4 Gflops), cross-attention (137.6 Gflops), adaptive layer norm (adaLN, 118.6 Gflops) or adaLN-zero (118.6 Gflops).

![image](https://user-images.githubusercontent.com/59775002/209647643-c47f3157-8993-4ff9-8fb9-24915813ea72.png)


![image](https://user-images.githubusercontent.com/59775002/209647708-ce8f0389-ef72-49b1-9bd8-94dd78353511.png)



## Benchmark

![image](https://user-images.githubusercontent.com/59775002/209647917-c45678e8-6a4d-45e7-a083-613658ae0cbb.png)


![image](https://user-images.githubusercontent.com/59775002/209647978-8e1970fe-404d-47eb-8450-47128ed89a44.png)
