# Dropout Reduces Underfitting

Authors: Zhuang Liu, Zhiqiu Xu, Joseph Jin, Zhiqiang Shen, And Trevor Darrell.

Authors in this paper demonstrate that dropout can also mitigate underfitting when used at the start of training.

## Introduction

![image](https://user-images.githubusercontent.com/59775002/223388072-2a74380f-6b4b-41db-8b14-08ee395fe4e5.png)

Dropout randomly deactivates each neuron with probability $p$, preventing different features from co-adapting with each other. After applying dropout, training loss typically increases, while test error decreases, narrowing the model's generalization gap.

Authors in this paper demonstrate an alternative use of dropout for tackling underfitting. They begin their investigation into dropout training dynamics by making an intriguing observation on gradient norms, which then leads to key empirical findings: during the initial stages of training, dropout reduces gradient variance across mini-batches and allows the model to update in more consistent directions. These directions are also more aligned with the entire dataset's gradient direction. Consequently, the model can optimize the training loss more effectively with respect to the whole training set, rather than being swayed by individual mini-batches.
In other words, dropout counteracts SGD and prevents excessive regularization due to randomness in sampling mini-batches during early training.

Based on this, authors introduce *early dropout* which is **only used during early training** to help underfitting models fit better.

Early dropout **lowers** the final training loss compared to no dropout and standard dropout. Authors also propose for models that already use standard dropout to remove dropout during earlier training epochs to mitigate overfitting and they refer to this approach as ***late dropout*** and demonstrate that it improves generalization accuracy for large models.

![image](https://user-images.githubusercontent.com/59775002/223388156-3eb005ed-f071-4156-823c-104ade36ed07.png)

## How Dropout Can Reduce Underfitting

In this study authors compare two ViT-T/16 training processes on ImageNet: one without dropout as a baseline and the other with a 0.1 dropout rate.

**Gradient Norm.** Authors investigated the impact of dropout on the strength of gradients $g$, measured by their $L_2$ norm $||g||_2$. For the dropout model, they measure the entire model's gradient, even though a subset of weights may have been deactivated due to dropout.

![image](https://user-images.githubusercontent.com/59775002/223388259-575e12b6-4042-4482-b94f-de023216fc9f.png)

**Model Distance.** Since the gradient steps are smaller, the dropout model is expected to travel a smaller distance from its initial point than the baseline model.
To measure the distance between the two models, authors use the $L_2$ norm, represented by $||W_1 - W_2||_2$, where $W_i$ denotes the parameters of each model.

**Gradient Direction Variance.** Authors hypothesize the same for the two models: the dropout model is producing more consistent gradient directions across mini-batches. To test this, authors collected a set of mini-batch gradients $G$ by training a model checkpoint on randomly selected batches. Then measure the gradient direction variance (GDV) by computing the average pairwise cosine distance:

$Eqution GDV

![image](https://user-images.githubusercontent.com/59775002/223388346-272d4fa1-7795-4f48-a9e4-bc1d842d8e62.png)

**Gradient Direction Error.** However, what should be the correct direction to take? To fit the training data, the underlying objective is to minimize the loss on the entire training set, not just on any single mini-batch. Thereby authors compute the gradient for a given model on the whole training set, where dropout is set to inference mode to capture the full model's gradient. Then evaluate how far the actual mini-batch gradient $g_{step}$ is from this whole dataset "ground-truth" gradient $\^g$. The average cosine distance defined from all $g_{step} ∈ G$ to $\^g$ as the gradient direction "error" (GDE):

![image](https://user-images.githubusercontent.com/59775002/223388839-07da578f-e900-4dee-98ae-ae9ef03adb6f.png)

![image](https://user-images.githubusercontent.com/59775002/223388687-3ec5af76-1d41-4571-8985-e242fa63fa46.png)

**Bias-variance Tradeoff.** This analysis at early training can be viewed through the lens of the bias-variance tradeoff.
For no-dropout models, an SGD mini-batch provides an unbiased estimate of the whole-dataset gradient is equal to the whole-dataset gradient. However, with dropout, the estimate becomes more or less biased, as the mini-batch gradient are generated by different sub-networks, whose expected gradient may not match the full network's gradient. Nevertheless, the gradient variance is significantly reduced, leading to a reduction in gradient error. Intuitively, this reduction in variance and error helps prevent the model from overfitting to specific batches, especially during the early stages of training when the model is undergoing significant changes.

## Approach

Based on previous analysis, authors know that using dropout early can potentially improve the model's ability to fit the training data.

### Underfitting and overfitting regimes.

Whether it is desirable to fit the training data better depends on whether the model is in an underfitting or overfitting regime, which can be difficult to define precisely. Authors considered if a model generalizes better with standard dropout then it's in an overfitting regime, and if the model performs better without dropout, then consider it to be in an underfitting regime.

### Early dropout

In their default settings, models at underfitting regimes do not use dropout. To improve their ability to fit the training data, authors proposed early dropout, **using dropout before a certain iteration, and then disabling it for the rest of training.** which resulted in reducing final training loss and improves accuracy.

### Late dropout.

Overfitting models already have standard dropout included in their training settings. During the early stages of training, dropout may cause overfitting unintentionally, which is not desirable. To reduce overfitting, authors propose late dropout, **not using dropout before a certain iteration, and then using it for the rest of training.** This is a symmetric approach to early dropout.

### Hyper-parameters.

**1) number of epochs to wait before turning dropout on or off:** results show that this choice can be robust enough to vary from 1% to 50% of the total epochs.

**2) Drop rate $p$:** Which is similar to the standard dropout rate and is also moderately robust.

## Comparison

![image](https://user-images.githubusercontent.com/59775002/223389230-25f9287b-017d-4774-b68c-49f8016b1139.png)

![image](https://user-images.githubusercontent.com/59775002/223389318-28f98feb-4c17-4f63-a305-0324f8d89988.png)