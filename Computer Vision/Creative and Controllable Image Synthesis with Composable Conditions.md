# Composer: Creative and Controllable Image Synthesis with Composable Conditions

Authors: Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao And Jingren Zhou.

Authors use *Compositionality* as the core idea to control the output of diffusion model image. such as spatial layout and palette.

They decompose an image into representative factors and then train a diffusion model with all these factors as the conditions to recompose the input.

## Introduction

![image](https://user-images.githubusercontent.com/59775002/221852744-2f2ebc81-921d-4c7b-9303-5240584b0c06.png)


Generative models often struggle to accurately produce images with specifications for semantics, shape, style, and color all at once.
Authors argue that the key to controllable image generation relies not only on conditioning but even more significantly on **Compositionality** (Lake et al.,2017) where we can exponentially expand the control space by introducing an enormous number of potential combinations. Similar concepts are explored in the field of language and scene understanding, where the compositionality is termed *compositional generalization,* the skill of recognizing or generating a potentially infinite number of novel combinations from a limited number of known components.

Authors in this work build upon the above idea and present Composer, a realization of *compositional generative modes*. which refer to generative models that are capable of seamlessly recombining visual components to produce new images.
Authors specifically implement Composer as a multi-conditional diffusion model with a UNet backbone. At every training iteration of Composer, there are two phases: in the **1)** **decomposition phase**, we break down images in a batch into individual representations using computer vision algorithms or pre-trained models: whereas in the **2)** ***composition phase***, we optimize Composer so that it can reconstruct these images from their representation subsets.

While conceptually simple and easy to implement, Composer is surprisingly powerful, enabling encouraging performance on both traditional and previously unexplored image generation and manipulation tasks.

## Method

![image](https://user-images.githubusercontent.com/59775002/221853059-dd2aad6b-7039-4759-ac7b-327d485f1fe5.png)

### Diffusion Models

**Guidance directions:** Composer is a diffusion model accepting multiple conditions, which enables various directions with classifier-free guidance:

$\^e_θ(X_t,c) = ωe_θ(x_t,c_2)+(1-ω)e_θ(x_t,c_1) $

where $c_1$ and $c_2$ are two sets of conditions. Different choices of $c_1$ and $c_2$ represent different emphases on conditions.

Conditions within $(c_2,c_1)$ are emphasized with a guidance weight of $ω$, those withing $(c_1,c_2)$ are suppressed with a guidance weight of $(1-ω)$, and conditions within $c_1 ∩ c_2$ are given a guidance weight of $1.0$.

*Bidirectional guidance:* By reversing an image $x_0$ to its latent $x_T$ using condition $c_1$, and then sampling from $x_T$ using another condition $c_1$, will be able to manipulate the image in a disentangled manner using Composer, where the manipulation direction is defined by the difference between $c_2$ and $c_1$.

### Decomposition

Authors decompose an image into decoupled representations that capture various aspects of it. they extract eight representations in this work on-the-fly during training.

***Caption:* **Authors directly use title or description information in image-text training data as image captions.

***Semantics and style:*** Authors use image embedding extracted by the pre-trained CLIP ViT-L/14@336px (Radford et al.,2021) model to represent the semantics and style of an image, similar to unCLIP(Ramesh et al., 2022).

***Color:*** Authors represent color statistics of an image using the smoothed CIELab histogram(Sergeyk, 2016). They quantize the CIELab color space to 11 hue values, 5 saturation values, and 5 light values, and they use a smoothing sigma of 10.

***Sketch:*** Authors apply instance segmentation on an image using the pre-trained YOLOv5 model to extract its instance masks. Instance segmentation masks reflect the category and shape information of visual objects.

***Depthmap:*** Authors use a pre-trained monocular depth estimation model (Ranftl et al.,2022) to extract the depth map of an image, which roughly captures the image's layout.

***Intensity:*** Authors introduce raw grayscale images as representation to force the model to learn a disentangled degree of freedom for manipulating colors. To introduce randomness, they uniformly sample from a set of predefined RGB channel weights to create grayscale images.

***Masking:*** Authors introduce image masks to enable Composer to restrict image generation or manipulation to an editable region. They use a 4-channel representation, where the first 3 channels correspond to the masked RGB image, while the last channel corresponds to the binary mask.

### Composition

Authors use diffusion models to recompose images from a set of representations. They leverage the GLIDE (Nichol et al., 2021) architecture and modify its conditioning modules.
They explore two different mechanisms to condition the model on their representations:

***1) Global conditioning:*** For global representations including CLIP sentence embeddings, image embeddings and color palettes, authors project and add them to the timestep embedding.
In addition, the authors project image embedding and color palettes into eight extra tokens and concatenate them with CLIP word embeddings, which are then used as the context for cross-attention in GLIDE, similar to unCLIP(Ramesh et al., 2022).

![image](https://user-images.githubusercontent.com/59775002/221853540-f1af221e-8f82-45f9-a697-fe7f1cb75697.png)

***2) Localized conditioning:*** For localized representations including sketches, segmentation masks, depth maps, intensity images, and masked images, authors project them into uniform-dimensional embeddings with the same spatial size as the noisy latent $x_t$ using stacked convolutional layers. Then compute the sum of these embeddings and concatenate the result to $x_t$ before feeding it into the UNet.

![image](https://user-images.githubusercontent.com/59775002/221853754-016aca0d-5fe9-4122-92ef-675566e00759.png)

***Joint training strategy:*** It is essential to devise a joint training strategy that enables the model to learn to decode images from a variety of combinations of conditions. The authors used simple and effective configurations where they used an independent dropout probability of 0.5 for each condition, a probability of 0.1 for dropping all conditions, and a probability of 0.1 for retaining all conditions. They used a special dropout probability of 0.7 for intensity images because they contain the vast majority of information about the images and may underweight other conditions during training.
