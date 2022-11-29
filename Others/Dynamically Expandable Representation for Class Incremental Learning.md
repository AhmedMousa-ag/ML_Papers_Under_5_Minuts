# Dynamically Expandable Representation for Class Incremental Learning

Authors: Shipeng Yan, Jiangwei Xie and Xuming He.

Authors address the problem of class incremental learning, in particular they consider the task setting of incremental learning with limited memory and aim to achieve better stability-plasticity trade-off.

## Introduction

Humans can easily accumulate visual knowledge from past experiences and incrementally learn novel concepts. Inspired by this, the problem of class incremental learning aims to design algorithms that can learn novel concepts in a sequential manner and eventually perform well on all observed classes. "real world application such as intelligent robot."

There has been much effort attempting to address in literature, among them, perhaps the most effective strategy is to keep a memory buffer that stores part of observed data for rehearsal in future. However due to limited memory size, it still faces several typical challenges. such as sacrificing the model plasticity for stability or being susceptible to forgetting due to greater degradation of old concepts.

In this work, authors aim to achieve better stability-plasticity trade-off  by adopting a two-stage learning strategy: 1-decoupling the adaptation of feature representation and, 2- final classifier head (*or classifier for short*) of a deep network.

Within this framework, they propose a novel data representation, referred to as *super-feature*, capable of increasing its dimensionality to accommodate new classes. Their main idea is to freeze the previously learned representation and augment it with additional feature dimensions from a new learnable extractor in each incremental step. This enables to retain the existing knowledge and provides enough flexibility to learn novel concepts. Moreover this super-feature is expanded dynamically based on the complexity of novel concepts to maintain a compact representation.

## Methods

![image](https://user-images.githubusercontent.com/59775002/204486336-5cf1e05f-c988-4f1f-bd2f-48b655e0f725.png)

### Method Overview

Their method adopts the rehearsal strategy, which saves a part of the data as the memory $Mt$ for future training. For the learning of step $t$, they decoupled the learning process into two sequential stages as follows:

1- *Representation Learning Stage*. To achieve better trade-off between stability and plasticity, they fix the previous feature representation and expand it with a new feature extractor trained on the incoming and memory data. They design an auxiliary loss on the novel extractor to promote it to learn diverse and discriminative features. To improve the model efficiency, they dynamically expand the representation according to the complexity of new classes via introducing a channel-level mask-based pruning method.

2- *Classifier Learning Stage*. After the learning of representation, they retrain the classifier with currently available data $D̃ t = D t ∪ M t$ at step $t$ to deal with the class imbalance problem via adopting the balanced finetuning method.

### Expandable Representation Learning

At step $t$, the model is composed of a super-feature extractor $Φ t$ and the classifier $H t$. The super-feature extractor $Φ t$ is built by expanding the feature extractor $Φ t−1$ with a newly created feature extractor $ft$. Specifically, given an image $x ∈ D̃ t$, the feature $u$ extracted by $Φ t$ is obtained by concatenation as follows: $u = Φ t (x) = [Φ t−1 (x), F t (x)]$

Here they reuse the previous $F1,....,Ft-1$ and encourage the new extractor $Ft$ to learn only novel aspects of new classes. The feature $u$ is then fed into the classifier $Ht$ to make prediction as follows: $p H t (y|x) = Softmax(H t (u))$ 

Then the prediction $ŷ = arg max p H t (y|x), ŷ ∈ Ỹ t$ . The classifier is designed to match its new input and output dimensions for step $t$. The parameters of $Ht-1$ to retain old knowledge and its newly added parameters are randomly initialized.

To reduce catastrophic forgetting, we freeze the learned function $Φ t−1$ at step $t$, as it captures the intrinsic structure of previous data. In detail, the parameters of last step super-feature extractor $θ Φ t−1$ and the statistics of Batch Normalization are not updated. Besides, they instantiate $Ft$ with $Ft-1$ as initialization to reuse previous knowledge for fast adaption and forward transfer.

Instead of assuming the prior distribution for $t$-th step which is unimodal, they expand the model with new parameters by creating a separate feature extractor $Ft$ for incoming data and take a uniform distribution as the prior distribution $p(θ F t |D 1:t−1 )$ which provides enough flexibility for the model to adapt to novel concepts. Meanwhile, for simplicity, they approximate the prior distribution $p(θ Φ t−1 |D 1:t−1 )$ on the old parameters $θ Φ t−1$ as the Dirac distribution, which maintains the information learned on $D 1:t−1 .$ By integrating two prior distribution assumptions on $p(θ Φ t−1 |D 1:t−1 )$ and $p(θ Φ t−1 |D 1:t−1 )$.

#### Training Loss

The model learns with cross-entropy loss on memory and incoming data as follows: 

![image](https://user-images.githubusercontent.com/59775002/204486482-c2bd68f5-b01b-47de-a945-4d0eec66f6a4.png)

Where $xi$ is image and $yi$ is the corresponding label. To enforce the network to learn the diverse and discriminative features for novel concepts, we further develop an auxiliary loss operating on the novel feature $Ft(x)$. Specifically, we introduce an auxiliary classifier, which predicts the probability $p H at (y|x) = Softmax(H t a (F t (x))$ to encourage the network to learn features to discriminate between old and new concepts, the label space of  $H t a is |Y t |+1$ including the new category set $Yt$ and other class by treating all old concepts as one category. Thusly, they introduce the auxiliary loss and obtain the expandable representation loss as follows: $L ER = L H t + λ a L H at$ where $λ a$ is the hyper-parameter to control the effect of the auxiliary classifier. It is worth noting that $λ a =0$ for first step $t$ =1.

### Dynamical Expansion

To remove the model redundancy and maintain a compact representation, the model dynamically expands the super-feature according to the complexity of novel concepts. Specifically, we adopt a differentiable channel-level mask-based method to prune filters of the extractor $Ft$, in which the masks are learned with the representation jointly. After the learning of the mask, we binarize the mask and prune the feature extractor $Ft$ to obtain the pruned network $FPt$.

#### Channel-level Masks

The pruning method is based on differentiable channel-level masks. which is adapted from HAT. For the novel feature extractor $Ft$, the input feature map of convolutional layer $l$ for a given image $x$ is denoted as $fl$. we introduce the channel mask $∈ R c l$ to control the size of layer $l$ where $m il ∈ [0, 1]$ and $c l$ is the number of channels of layer $l$. $f1$ is modulated with mask as follows: $Eqution 5

To make the value of $m1$ fall into the interval [0,1], the gating function is adopted as follows: $m l = σ(se l )$ where $e l$ means learnable mask parameters, the gating function $σ(·)$ uses the sigmoid function in this work and $s$ is the scaling factor to control the sharpness of the function.

During training $φ t (x)$ is $F t (x)$ with the softmasks. For inference, we assign $s$ a large value to binarize masks and obtain the pruned network.

#### Mask Learning

During epoch, a linear annealing schedule is applied for $s$ as follows: 

![image](https://user-images.githubusercontent.com/59775002/204486675-fa171692-579a-4595-aaa7-b373cc9eaf1a.png)

where $b$ is the batch index, $s max >>1 $ is the hyper-parameter to control the schedule, $B$ is the number of batches in one epoch. The training epoch starts with all channels activated in a uniform way. Then the mask is progressively binarized with the increasing of batch index within epoch.

#### Sparsity Loss

At every step, we encourage the model to maximally reduce the number of parameters with a minimal performance drop. Motivated by this, we add a sparsity loss based on the ratio of used weights in all available weights:

![image](https://user-images.githubusercontent.com/59775002/204486800-92a9facd-bb51-4eee-b6ee-33a85001814f.png)

where $L$ is the number of layers, $Kt$ is the kernel size of convolution layer $l$, layer $l=0$ refers to input image, and $||mo||1=3$ after adding the sparsity loss, the final loss function is: 

![image](https://user-images.githubusercontent.com/59775002/204486926-1b6bc65e-9927-4c47-9b72-05aae36994a4.png)

where $λ s$ is the hyper-parameter to control the model size.

### Classifier Learning

At the representation learning stage, we re-train the classifier head in order to reduce the bias in the classifier weight introduced by imbalanced training. Specifically, we first re-initialize the classifier with random weights and then sample a class-balanced subset from currently available data $D̃ t$. We train the classifier head only using the cross-entropy loss with a temperature $δ$ in the softmax. The temperature controls the smoothness of the Softmax function to improve the margins between classes.

## Comparison between other methods

![image](https://user-images.githubusercontent.com/59775002/204487039-b938406e-ac9a-4230-848b-a9dc12cf778c.png)
