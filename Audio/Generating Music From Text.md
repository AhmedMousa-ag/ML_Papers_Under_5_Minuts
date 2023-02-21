# MusicLM: Generating Music From Text

Authors: Andrea, Timo, Denk, Zalán, Jesse, Mauro,  Antoine, Qingqing, Aren, Adam, Marco, Matt, Neil, and Christian.

Authors introduce MusicLM which is a model for generating high-fidelity music from text description and it generates music at 24 kHz that remains consistent over several minutes.

## Introduction

Authors introduce **MusicLM**, a model for generating high-fidelity music from text descriptions. they leverage **AudioLM**'s multi-stage autoregressive modeling as the generative component while extending it to incorporate text conditioning.

And to address the main challenge of paired data scarcity, they rely on **MuLan** which is a joint music-text model that is trained to project music and its corresponding text description to representations close to each other in an embedding space.

When authors trained **MusicLM** on a large dataset of *unlabeled* music, the model learns to generate long and coherent music at 24 kHz, for text descriptions of significant complexity, such as *“enchanting jazz song with a memorable saxophone solo and a solo singer”* or *“Berlin 90s techno with a low bass and strong kick”.*

And to address the lack of evaluation data for this task, authors introduce **MusicCaps**, a new high-quality music caption dataset with 5.5k examples prepared by expert musicians, which authors publicly release to support future research.

## Method

![image](https://user-images.githubusercontent.com/59775002/220276983-4bf762e5-c25d-4947-ac22-cf66558529b1.png)

### Representation and Tokenization of Audio and Text

Authors use three models for extracting audio representations that will serve for conditional autoregressive music generation. In particular, by following the approach of **AudioLM**, they use the self-supervised audio representations of **SoundStream** as acoustic tokens to enable high-fidelity synthesis, and w2v-BERT as semantic tokens to facilitate long-term coherent generation.
To represent the conditioning authors rely on **MuLan** text embedding at inference time. All three of these models are trained independently and then frozen, such that they provide the discrete audio and text representations for the sequence-to-sequence modeling.

**SoundSteam:** Authors use SoundStream model for 24 kHz monophonic audio with a striding factor of 480, resulting in 50 Hz embeddings. The quantization of these embeddings is learned during training by an RVQ with 12 quantizers, each with a vocabulary size of 1024. This results in a bitrate of 6 kbps, where one second of audio is represented by 600 tokens. and authors refere to these as *acoustic toekns,* denoted by $A$.

**w2v-BERT:** Similar to AudioLM, authors use an intermediate layer of masked-language-modeling (MLM) module of a w2v-BERT model with 600M parameters. after pretraining and freezing the model. they extract embeddings from the 7th layer and quantize them using the centroids of a learned k-means over the embeddings. they use 1024 clusters and a sampling rate of 25 Hz, resulting in 25 semantic tokens for every second of audio, denoted by $S$.

**MuLan:** Since MuLan operates on 10-second audio inputs and we need to process longer audio sequences, we calculate the audio embeddings on 10-second windows with 1-second stride and average the resulting embeddings. then we discretize the resulting embedding by applying an RVQ with 12 vector quantizers, each with a vocabulary size of 1024.
This process yields 12 MuLan audio tokens $M_A$ for an audio sequence. During inference, we use conditioning the MuLan text embedding extracted from the text prompt. and quantize it with the same RVQ as the one used for the audio embeddings, to obtain 12 tokens $M_T$.

### Hierarchical Modeling of Audio Representations

![image](https://user-images.githubusercontent.com/59775002/220277112-78fd02f8-3445-4638-b56c-8ee124f2300b.png)

Authors propose a hierarchical sequence-to-sequence modeling task, where each stage is modeled autoregressively by a separate decoder-only Transformer.

The first stage is the *** semantic modeling*** stage, which learns the mapping from **MuLan** audio tokens to the semantic tokens $S$, by modeling the distribution $p(S_t|S_{<t},M_A)$, where $t$ is the position in the sequence corresponding to a time step.

The second stage is ***acoustic modeling*** stage, where acoustic tokens $A_q$ are predicted conditioned on both the MuLan audio tokens and the semantic tokens, modeling the distribution $p(A_t|A_{<t},S,M_A)$.

## Comparison

![image](https://user-images.githubusercontent.com/59775002/220277226-236873f9-6cc6-4d61-ac1c-0c2652d7815b.png)
