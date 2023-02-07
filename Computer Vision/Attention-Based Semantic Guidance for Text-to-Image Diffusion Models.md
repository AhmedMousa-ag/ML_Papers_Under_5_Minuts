# Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models

Authors: Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or.

Authors introduce the concept of Generative Semantic Nursing (GSN), where they seek to intervene in the generative process on the fly during inference time to improve the faithfulness of the generated images.

## Introduction

![image](https://user-images.githubusercontent.com/59775002/217196629-dcbeab92-de1b-4797-9bc6-2af2747e0e6b.png)

Produced images by text-based image generation models do not always faithfully reflect the semantic meaning of the target prompt. thereby authors observe two key semantic issues: **1)** "catastrophic neglect", where one or more of the subjects of the prompt are not generated. **2)** "incorrect attribute binding", where the model binds attributes to the wrong subjects or fails to bind them entirely.

Thereby authors introduce the concept of "Generative Semantic Nursing" (GSN). In the GSN process, one slightly shifts the latent code at each timestep of the denoising process such that the latent is encouraged to better consider the semantic information passed from the input text prompt.

Authors propose a form of GSN dubbed *Attend-and-Excite*, which leverages the powerful cross-attention maps of a pre-trained diffusion model. The attention maps define a probability distribution over the text tokens for each image patch, which determines the dominant tokens in the patch. Although each path can attend freely to all text tokens, there is no mechanism to ensure that all tokens are attended to by some patch in the image.

Attend-and-Excite embodies this intuition by demanding that each subject token is dominant in some path in the image. authors carefully guide the latent at each denoising timestep and encourage the model to attend to all subject tokens and strengthen their activations. Importantly, the author's approach is applied on the fly during inference time and requires no additional training or fine-tuning.

## Attend-and-Excite

![image](https://user-images.githubusercontent.com/59775002/217196810-2a06ff98-cc6c-491f-9033-370d1ff33765.png)

The idea of *generative semantic nursing* is authors gradually shift the noised latent code at each timestep $t$ toward a more semantically-faithful generation. At each denoising step $t$, authors consider the attention maps of the subject tokens in the prompt $P$. Intuitively, for a subject to be present in the synthesized image, it should have a high influence on some patch in the image. As such authors define a loss objective that attempts to maximize the attention values for each subject token. then update the noised latent at time $t$ according to the gradient of the computed loss. This encourages the latent at the next timestep to better incorporate all subject tokens in its representation. This manipulation occurs on the fly during inference.

### Extracting the Cross-Attention Maps

![image](https://user-images.githubusercontent.com/59775002/217196923-66ef792f-b32b-4f04-a160-eee3decd5f80.png)

Given the input text prompt $P$, authors consider the set of all subject tokens (e.g., nouns) $S=\{s_1,....,s_k\}$ present in $P$. the objective is to extract a spatial attention map for each token $s∈ S$, indicating the influence of the token $s$ on each image patch.
Given the noised latent $z_t$ at the current timestep, we perform a forward pass through the pre-trained UNet network using $z_t$ and $P$. then considering the resulting cross-attention map obtained after averaging all $16×16$ attention layers and heads. The resulting aggregated map $A_t$ contains $N$ spatial attention maps, one for each of the tokens of $P,i.e.A_t ∈ R ^{16×16×N}$.

The pre-trained CLIP text encoder prepends a specialized token $sot$ to $P$ indicating the start of the text. During the text encoding process. this token receives global information about the prompt. This leads to $sot$ obtaining a high probability in the token distribution defined in $A_t$. Since we are interested in enhancing the actual prompt tokens, we re-weigh the attention values by ignoring the attention of $sot$ and performing a Softmax operation on the remaining tokens. After the Sotmax operation, the $(i,j)$-th entry of the resulting matrix $A_t$ indicates the probability of each of the textual tokens being present in the corresponding image patch. We then extract the $16 × 16$ normalized attention map for each subject token $s$.

### Obtaining Smooth Attention Maps

Observe that the attention values $A^s_t$ calculated above may not fully reflect whether an object is generated in the resulting image. Specifically, a single patch with a high attention value could stem from partial information being passed from the token $s$. This may occur when the model does not generate the full subject, but rather a patch that resembles some part of the subject, *e.g.*, a silhouette that resembles an animal's body part.
To avoid such adversarial solutions, we apply a Gaussian filer over $A^s_t$ in Step 5 of the algorithm. This ensures that the attention value of the maximally-activated patch is dependent on its neighboring patches since, after this operation, each patch is a linear combination of its neighboring patches in the original map.

### Performing On the Fly Optimization

For each subject token in $S$, the author's optimization encourages the existence of at least one patch of $A^s_t$ with a high activation value. Therefore, they define the loss quantifying this desired behavior as

![image](https://user-images.githubusercontent.com/59775002/217197028-00de225f-6c13-40fc-b867-f2c987d23b92.png)

That is, the loss attempts to strengthen the activations of the most neglected subject token at the current timestep $t$. It should be noted that different timesteps may strengthen different tokens, encouraging all neglected subject tokens to be strengthened at some timestep.
Having computed loss $L$, we shift the current latent $z_t$ by

$z^{'}_t  ← z_t − α_t · ∇_{zt} L,$

where $α_t$ is a scalar defining the step size of the gradient update. Finally, we perform another forward pass through $SD$ using $z^{'}_t$ to calculate $z_{t-1}$ for the next denoising step (Step 16 of the Algorithm). The above update process is repeated for a subset of the timesteps $t=T,T-1,....,t_{end}$ where we set $T=50$, following Stable Diffusion, and $t_{end}=25$. This is based on the observation that the final timesteps do not alter the spatial locations of objects in the generated image.

### Iterative Latent Refinement

If the attention values of a token do not reach a certain value in the early denoising stages, the corresponding object will not be generated. Therefore, we iteratively update $z_t$ until a pre-defined minimum attention value is achieved for all subject tokens. Yet, many updates of $z_t$ may lead to the latent becoming out-of-distribution, resulting in incoherent images. As such, this refinement is performed gradually across a small subset of timesteps.

Specifically, we demand that each subject token reaches a maximum attention value of at least $0.8$. To do so gradually, we perform the iterative updates at various denoising steps. We set the iterations to $t_1=0,t_2=10$ and $t_3=20$ with minimum required attention values of $T_1 = 0.05,\space T_2=0.5$ and $T_3=0.8$. This gradual refinement prevents $z_t$ from becoming out-of-distribution while encouraging more faithful generations.

## Comparisons

### Qualitative Comparisons

![image](https://user-images.githubusercontent.com/59775002/217197345-b4ad7141-260a-48e1-9ef6-6b906f3fadce.png)

### Quantitative Analysis

![image](https://user-images.githubusercontent.com/59775002/217197497-51a69b66-3b1a-4ff9-9b27-d478fc66ff07.png)
