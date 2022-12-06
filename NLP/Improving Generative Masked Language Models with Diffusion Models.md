# DiffusionBERT: Improving Generative Masked Language Models with Diffusion Models

Authors: Zhengfu He, Tianxiang Sun, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu.

Authors leverage pre-trained NLP models (e.g BERT) as a good initialization for faster converigence in diffusion method, by propusing a new noise schedule for forward diffusion process. And investigating several designs of incorporating the time step into BERT.

## Introduction

Applying diffusion models to text data is still challenging and under-explored due to the discrete nature of the text.

Prior works can be divided into two methods, 1- extending diffusion models to discrete state spaces, 2- performing the diffusion process and its reverse in the continuous domain and bridging the continuous and discrete domain through embedding and rounding.

![image](https://user-images.githubusercontent.com/59775002/205863060-67373178-d8ef-4268-a804-a7ef4c567c30.png)

However, none of these works leveraged pre-trained language models "PLM". So authors suggest that in the forward process of diffusion model to add small amount of noise gradually to the data. Then a neural network ($p θ$) is employed to learn the reverse process step by step. Such a denoising neural network is naturally related to a wide class of "PLMs" that are pre-trained with denoising objectives such as **BERT**, hence pre-trained denoising language models can serve as a good starting point to learn the reverse diffusion process.

In the discrete domain, the forward diffusion process can be implemented by a chain of transition matrices that gradually corrupt the clean text. As shown in Figure 1, the clean text "Hello world !" is gradually corrupted into " [MASK]  [MASK]  [MASK]" during the diffusion process. In this work, authors explore using pre-trained denoising language model to learn the reverse diffusion process and demonstrate their advantages in accelerating convergence and improving generation quality. Further, they propose a new noise schedule of the forward process based on the principle of distributing the corrupted information uniformly across the forward process. The noise schedule, called *spindle schedule, generates noise for $Xt$ conditioned not only on $x t−1$ but also on $x0$, making the forward process non-Markovian without changing the original training objective, Note that the denoising model takes as input $xt$ and time step $t$ to predict $x_t-_1$, where $t$ is unseen during the pre-training of language models so they investigate several ways of incorporating the time step into PLMs. As a result, they found that the best result is achieved by throwing away the time information, which they call *time-agnostic decoding* (TAD).

## DiffusionBERT

In contrast to recently proposed diffusion models for text, e.g.,Diffusion-LM and DIffuSeq which are based on *continuous* diffusion models, authors instead explore *discrete* diffusion models to integrate PLMs as the backbone. We first introduce a specific instance of discrete diffusion models, which considers a transition matrix with an absorbing state for the sake of using PLMs. Secondly, they introduce a new noise schedule of the forward diffusion process, called spindle schedule, which is based on the principle of distributing the corrupted information uniformly across the forward process, Then authors investigate several alternatives of incorporating the time step into PLMs for predicting $x_t-_1$ given $xt$ and $t$.

### Diffusion Models with a Discrete Abosrbing State

To be combined with pre-trained denoising language models, we incorporate an absorbing state, e.g., [MASK] for BERT, in the Markov process. In particular, each token in the sequence either stays the same or transitions to [MASK] with some probability. Formally, each entry of the transition matrix at step $t$ is as follows,

![image](https://user-images.githubusercontent.com/59775002/205863187-8963adb3-7df0-4aa1-9140-3c3c092e593a.png)

Where [M] is the abbreviation of [MASK]. Such a Markov process converges to a stationary distribution $q(x T )$, which places all probability mass on a sequence with all [MASK] tokens.

The $t$-step marginal $ q(x^i_t |x^i_0 )$ can be easily obtained in a closed form,

![image](https://user-images.githubusercontent.com/59775002/205863257-dcb8d028-0482-4cff-a8e2-b0cb4f2e8d6d.png)

By the end we can derive a training objective to optimize $p θ (x _t −_1 |x_t , t)$ and generate a sample by performing the reverse diffusion process:

![image](https://user-images.githubusercontent.com/59775002/205863362-2e1f5278-30d3-47dd-b6ef-7e5019a1433b.png)

### Spindle Noise Schedule

The noise schedule in the continuous domain, such as the linear schedule and the cosine schedule, has shown to be important to the performance of diffusion models.

In contrast to the continuous domain where the noise can be easily controlled by the variance of the Gaussian, (1) *it's less obvious how to control the degree of noise added at each step in the discrete domain*. For the discrete domain, the noise schedule $β t = (T − t + 1)^-1$ has been explored for the case of the uniform transition matrix and the absorbing-state transition matrix. However, (2) *such a schedule assumes all tokens carry the same amount of information and does not consider the linguistic difference among the tokens in a sequence. Besides*, (3) *it violates the easy-first-generation nature of denoising language models.* That is, the model tends to generate tokens that are most frequently appearing (and is least surprising) in the training corpus to achieve a higher likelihood. As the context becomes richer, more details come up in the sequence,

To address the above issues, authors consider a noise schedule that (1) measures the added noise at each step by corrupted information and encourages the corrupted information to be uniformly distributed across the diffusion steps. Once the information is measured independently for each token, (2) different tokens in a sequence are assigned different probabilities of transition to the [MASK] token. Moreover, inspired by the easy-first-generation phenomenon, (3) they put the tokens in a sequence in descending order of their information and divide them into$T$ buckets. Each bucket is ensured to contain the same amount of information. That is, they mask the most informative tokens at the start of the forward process and mask the least informative tokens at the end of the forward process such that the learnable reverse process follows an easy-first generative behavior.

In particular, distributing corrupted information uniformly across the forward steps can be formally described by:

![image](https://user-images.githubusercontent.com/59775002/205863524-9e2560b8-28aa-4bfe-a34e-857500560963.png)

Where $H$ denotes the entropy, which measures the amount of information of a random variable, $x^i$ denoted $i$-th token in the sequence and $n$ denotes the length of the sequence. According to Eq. (7), denotes the probability that the $i$-th token remains the same at step $t$, i.e., $x^i_t = x^i_0$. They expect that $α^i_t > α^j_t if H(x ^i_t ) < H(x^j_t )$ such that easy (low information) tokens emerges earlier than hard (high information) tokens during the reverse process.

Considering these aforementioned properties, they construct as $α^i_t$ follow:

![image](https://user-images.githubusercontent.com/59775002/205863637-9e1f7054-45ff-4132-ba58-e6d2d04a4a27.png)

Where $S(t)$ is introduced to control the effect of the informativeness at time step $t$. It's designed to be sinusoidal to ensure $S(0) = S(T) = 0$ such that $x_t$ can retain all (zero) information when $t = 0 (t=T)$. The effect of $S(t)$ is controlled by a hyperparameter $λ$. When $λ = 0$, the noise schedule is degraded to $β t = (T −t+1)^−1$. In the proposed schedule, the transition probability at time step $t$ depends not only on the current state but also on the original text, making the forward diffusion process non-Markovian.

![image](https://user-images.githubusercontent.com/59775002/205863740-58ff36ef-2af2-4c24-9ca0-ee84e94d46a2.png)

### The Design Space of the Feeding Time Steps

Typically, a diffusion model takes as input a noised sample and the time step to predict the denoised sample during the reverse process, i.e., $p θ (x t−1 |x t , t)$. However, t is an additional variable that is unseen during the pre-training of language models and therefore it is less trivial how to feed the time information into the PLMs. Here we explore three design choices of feeding time steps.

**Layer-wise Time Embedding** A straightforward choice is to include the time step as the same way as positional encoding, i.e., using the Transformer sinusoidal embedding or a learnable MLP in each Transformer layer. Note that this way is commonly adopted in previous work.

**Prefix Time Embedding** Prompting language models by prepending trainable soft tokens to the input sequence has shown promising results recently (Lester et al., 2021; Sun et al., 2022). Hence, we also explore including a time step token embedding $v(t)$ as a prefix of the input token embeddings $v(x^1_t), v(x^2_t ), · · · , v(x^n_t )$. In particular, the time step token is inserted in between the [CLS] token and the input sequence. These added time step token embeddings are trained along with the PLM.

**Time-Agnostic Decoding** Another alternative is not to explicitly incorporate the time step $t$ because it can be implied by the noised sample $x_t$ . In contrast to the image data, it is easier to implicitly infer the diffusion time step by counting the number of corrupted tokens (i.e., [MASK]) in the noised sequence. In this way, the PLM has to perform iterative decoding while being ignorant of the current time step, i.e., $p θ (x_t−_1 |x_t )$.
