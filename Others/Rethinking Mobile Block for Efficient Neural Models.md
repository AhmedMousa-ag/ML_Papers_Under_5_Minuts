# Rethinking Mobile Block for Efficient Neural Models

Authors: Jiangning Zhang, Xiangtai Li, Jian Li, Liang Liu, Zhucun Xue, Boshen Zhang, Zhengkai Jiang, Tianxin Huang, Yabiao Wang, Chengjie Wang.

Authors designed efficient models with low parameters and FLOPs for dense predictions by introducing **I**nverted **R**esidual **M**obile **B**lock (iRMB) for mobile applications.

## Introduction

$Figure 1

*"Can we combine the most naive Conv and MHSA to further develop a simple yet efficient block for mobile application, just like proven effective IRB?"*

Authors try to answer previous question after being inspired by the fact that there's no Transformer-based or hybrid efficient block is popular like CNN-based IRB at the moment.

Authors rethink Inverted Residual Block in MobileNetv2 and MHSA/FFN modules in Transformer, which are proven to be effective widely, from a structural point of view.

As shown in previous Figure, 1- **Left**, authors inductively abstract a general **Meta Mobile Block** that takes arguments expansion ratio $λ$ and efficient operator $F$ to instantiate different modules, *i.e.,* IRB, MHSA, and Feed-Forward Network (FFN). These findings means that *the framework of the meta block provides the basic ability to the model, while the differences of model performances essentially come from different structural instantiations.* Therefore authors only use Conv and MHSA modules to deduce detailed instantiation, designing a simple yet effective *Inverted Residual Mobile Block* (iRMB) for mobile applications. As well as building a ResNet-like 4-phase Efficient MOdel (EMO) based on a series of IRMBs for dense applications.

## Related Work

**1- Efficient CNN Models**

**2- Vision Transformer**

## Methodology

$Figure 2

### Meta Mobile Block

Authors abstracted a general Meta Mobile ($M^2$) Block from Inverted Residual Block in MobileNetv2 with core MHSA and FFN modules in Transformers. Which takes arguments *expansion ratio $λ$ and efficient operator $F$* to instantiate different modules, they argue that the $M^2$ block can reveal the consistent essence expression of the above three modules, and its different instantiations are very important to model performance.

EMO contains one deduced IRMB absorbing advantage of lightweight CNN and Transformer. As an example take image input $X(∈ R^{C×H×W})$, $M^2$ block firstly use a expansion $MLP_e$ with output/input ratio equaling $λ$ to expand channel dimension:

 $X_e = MLP_e (X)(∈ R^{λC×H×W})$.

Then intermediate operator $F$ enhances image features further, *e.g.,* identity operator, static convolution, dynamic MHSA, *etc*. Considering that $M^2$ block is suitable for efficient network design, authors present $F$ as the concept of *efficient operator*, formulated as:

$X_f = F (X_e )(∈ R^{λC×H×W})$.

Finally, a shrinkage $MLP_s$ with inverted input/output ratio equaling $λ$ to shrink channel dimension:

$X s = MLP_s (X_f )(∈ R^{C×H×W})$

Where a residual connection is used to get the final output $Y = X + X_s (∈ R^{C×H×W} )$. Notice that normalization and activation functions are omitted for clarity.

### Inverted Residual Mobile Block

Based on the inductive $M^2$ block, authors instantiated an effective yet efficient modern *Inverted Residual Mobile Block* (iRMB) for mobile applications, which absorbs CNN-like efficiency to model local features and Transformer-like dynamic modeling capability to learn long-distance interactions. Specifically, $F$ in iRMB as modeled as cascaded *MHSA* and *Convolution* operations, formulated as $F$(·) = Conv(MHSA(·)).

However, naive implementation can lead to unaffordable expenses for two main reasons: **1** $λ$ is generally greater than one that the intermediate dimension would be multiple to input dimension, causing a quadratic increase of parameters and computations. **2** FLOPs of MHSA is proportional to quadratic of total image pixels. Therefore, employing more efficient Window-MHSA (W-MHSA) and depth-wise convolution (DW-Conv) with a skip connection to trade-off cost and accuracy, view in last figure **LEFT** notice that attention matrix in W-MHSA is calculated by **$X$** for efficiency, *i.e*, $Q=K=X$$(∈ R^{C×H×W})$, while the expanded value $V =X_e (∈ R^{λC×H×W})$. Therefore, this improvement is termed as *Expanded Window MHSA* (EW-MHSA) that is more applicative, formulated as:

$F (·) = (DW-Conv, Skip)(EW-MHSA(·))$.

Also, this cascading manner can increase the expansion speed of the receptive field and reduce the MPL of the model to $O(2W/(k − 1 + 2w))$.

**Efficient Equivalent Implementation.** MHSA is usually used in channel consistent projection ($λ=1$), meaning that the FLOPs of multiplying attention matrix times expended $X_e(λ>1)$ will increase by $λ-1$. Fortunately, the information flow from $X$ to expended $V(X_e)$ involves only linear operations, *i.e.,* $MLP_e$$(·)$, so can derive an equivalent proposition: *"When the groups of MLP_e equals to the head number of W-MHSA, the multiplication result of exchanging order remains unchanged*". In this paper called matrix multiplication before $MLP_e$ *pre-attn* that is used by default.

**Bosting Existing Models.** To assess iRMB performance, we set $λ$ to 4 and replace standard Transformer structure in columnar DeiT and pyramid-like PVT As shown in next table, we surprisingly found that iRMB can improve performance with fewer parameters and computations in the same training setting, especially for columnar structure.

$Table 2

### EMO for Dense Prediction

When designing efficient visual models for mobile applications, authors define the following criteria as an efficient model:

**1- Usability:** Simple implementation that does not use complex operators and is easy to optimize for application.

**2- Uniformity:** As few core modules as possible to reduce model complexity.

**3- Effectiveness:** Good performance for classification and dense prediction.

**4- Efficiency:** Fewer parameters and calculations with accuracy trade-off.

Based on the above criteria, authors designed a ResNet-like 4-phase Efficient MOdel (EMO) based on a series of iRMBs for dense applications.

**1)** For the overall framework, EMO consists of only iRMBs without diversified modules Á , which is a departure from recent efficient methods in terms of designing idea.
**2)** For the specific module, iRMB consists of only standard convolution and multi-head self-attention without other complex operators. Also, benefitted by DW-Conv, iRMB can adapt to down-sampling operation through the stride and does not require any position embeddings for introducing inductive bias to MHSA.
3) For variant settings, authors employ gradually increasing expansion rates and channel numbers.

## Comparison

$Table 6

$Table 7
