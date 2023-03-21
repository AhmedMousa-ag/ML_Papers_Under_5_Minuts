# FateZero: Fusing Attentions for Zero-shot Text-based Video Editing

Authors: Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan and Qifeng Chen.

Authors propose a zero-shot text-based editing method (*FateZero*) on real-world videos without per-prompt training or use-specific mask.

## Introduction

Previous or concurrent diffusion-based editing methods majorly work on images, and to edit real images, their methods utilize deterministic DDIM for the image-to-noise inversion, then the inverted noise gradually generates the edited images under the condition of the target prompt. based on this pipeline, several methods have been proposed in terms of cross-attention guidance.

Authors proposed *FateZero* which is a zero-shot video editing that doesn't need to be trained for each target prompt individually and has no user-specific mask.
And to keep the temporal consistency of the edited video, they used two novel designs. Firstly, instead of solely relying on inversion and generation, they stored all the self and cross-attention maps at every step of the inversion process. This enabled subsequently replacing them during the denoising steps of the DDIM pipeline.

Author's contribution can be summarized as follows:

* They present the first framework for temporal-consistent zero-shot text-based video editing using pre-trained text-to-image model.
* They propose to fuse the attention maps in the inversion process and generation process to preserve the motion and structure consistency during editing.
* Their Attention Blending Block utilizes the source prompt's cross-attention map during attention fusion to prevent source semantic leakage and improve the shape-editing capability.
* They show extensive applications of their method in video style editing, video local editing, video object replacement *, etc*.

## Methods

![image](https://user-images.githubusercontent.com/59775002/226563043-a47becd7-dd8e-41ca-b685-9bc9809cd0e3.png)

### FateZero Video Editing

Authors use pre-trained text-to-image model as a base model, which contains a UNet for $T$-timestep denoising. Instead of straight-forwardly exploiting the regular pipeline of latent editing guided by reconstruction attention, they made several critical modifications for video editing as follows.

**Inversion Attenion Fusion:** Direct editing using the inverted noise results in frame inconsistency, which may be attributed to two factors. **First**, the invertible property of DDIM only holds in the limit of small steps. Nevertheless, the present requirements of 50 DDIM denoising steps lead to an accumulation of errors with each subsequent step. Second, using large classifier-free guidance $s_{cfg}>>1$ can increase the edit ability in denoising, but the large editing freedom leads to inconsistent neighboring frames. Therefore, previous methods require optimization of text-embedding or other regularization.
To alleviate these issues, authors utilize the attention maps during inversion steps, which is available because the source prompt $p_{src}$ and initial latent $z_0$ are provided to the UNet during inversion. Formally, during inversion, they store the intermediate self-attention maps $[s^{src}_t]t^T_{=1}$, cross-attention maps $[c^{src}_t]t^T_{=1}$ at each timestep $t$ and the final latent feature maps $z_T$ as

$z_T,[c^{src}_t]t^T=1,[s^{src}_]t^T_{=1}$ $= DDIM-INV(z_0,P_{src})$

Where $DDIM-INV$ stands for the DDIM inversion pipeline.
During the editing stage, we can obtain the noise to remove by fusing the attention from inversion:

$\^e_t = ATT-FUSION(ε_0,z_t,t,p_{edit},c^{src}_t,s^{src}_t)$

where $p_{edit}$ represents the modified prompt. In function $ATT-FUSION$, we inject the cross-attention maps of the unchanged part of the prompt similar to Prompt-to-Prompt. We also replace self-attention maps to preserve the original structure and motion during the style and attribute editing.

**Attention Map Blending:** Inversion-time attention fusion might be insufficient in local attrition editing, as shown in the next Figure. In the third column, replacing self-attention $s^{edit}∈ R^{hw \space × \space hw}$ with $s^{src}$ brings unnecessary structure leakage and generated image has unpleasant blending artifacts in the visualization. On the other hand, if we keep $s^{edit}$ during the DDIM denoising pipeline, the structure of the background and watermelon has unwanted changes, and the pose of the original rabbit is also lost. Inspired by the fact that the cross-attention map provides the semantic layout of the image, as visualized in the second row of the next figure, we obtain a binary mask $M_t$ by thresholding the cross-attention map of the edited words during inversion by a constant $τ$. Then the self-attention maps of editing stage $s^{edit}_t$ and inversion stage $s^{src}_t$ are blended with the binary mask $M_t$. Formally the attention map fusion is implemented as

$M_t = HEAVISIDESTEP(c^{src}_t,τ),$

$s^{fused}_t = M_t ⊙ s^{edit}_t + (1-M_t) ⊙ s^{src}_t.$

![image](https://user-images.githubusercontent.com/59775002/226563179-25268a2d-7fca-4b20-a1c3-775ee31b7ec3.png)

**Spatial-Temporal Self-Attention:** Denoising each frame individually produces inconsistent video. Inspired by the casual self-attention and recent one-shot video generation method. authors reshape the original self-attention to Spatial-Temporal Self-Attention without changing pre-trained weights. Specifically, they implement $ATTENTION(Q,K,V)$ for feature $z^i$ at temporal index $i ∈ [1,n]$ as

$Q=W^QZ^i,K=W^K[Z^i;Z^W],V=W^V[Z^i;Z^W]$

where $[·]$ denotes the concatentation operation and $W^Q,W^K,W^V$ are the projection matrices from pretrained model. Empirically, they find it enough to warp the middle frame $z^w=z^{Round[{{n}\over{2}}]}$ for attribute and style editing. Thus, the spatial-temporal self-attention map is represented as $s^{src}_t ∈ R^{hw\space × \space fhw}$, where $f=2$ is the number of frames used as key and value. It captures both the structure of a single frame and the temporal correspondence with the warped frames.

### Shpe-Aware Video Editing

Different from appearance editing, reforming the shape of a specific object in the video is much more challenging. To this end, a pre-trained video diffusion model is needed. Since there is no publicly-available generic video diffusion model, we perform the editing on the one-shot video diffusion model instead. In this case, authors compare their editing method with simple DDIM inversion which achieves better performance in terms of editing ability, motion consistency and temporal consistency.

## Comparison

![image](https://user-images.githubusercontent.com/59775002/226563310-42f75d4b-e947-43c7-aec8-dcf63cf66e61.png)

![image](https://user-images.githubusercontent.com/59775002/226563386-b933d4c7-76eb-4bbd-baee-99293185891b.png)

![image](https://user-images.githubusercontent.com/59775002/226563467-a1750e70-6374-4553-9038-ad417b2cb0ae.png)

![image](https://user-images.githubusercontent.com/59775002/226563635-1b0c41c4-def2-4582-800a-0b028b548ff6.png)
