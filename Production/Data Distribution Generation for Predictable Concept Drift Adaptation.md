
# DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation

Authors: Wendi Li, Xiao Yang, Weiqing Liu, Yingce Xia, Jiang Bian.

Authors propose a novel method DDG-DA that can effectively forecast the evolution of data distribution and improve the performance of models. Specifically, they first train a predictor to estimate the future data distribution, then leverage it to generate training samples, and finally train models on the generated data.

## Introduction

![image](https://user-images.githubusercontent.com/59775002/207271208-534ea5c5-8149-4adf-bc7e-de9e7f587bd6.png)

To handle concept drift, previous studies usually leverage a two-step approach. Particularly, the first step is detecting the occurrence of concept drift in the streaming data, followed by the second step, if concept drift does occur, which adapts the model with new coming data by training a new model purely with the latest data or making an ensemble between the new model with the current one, or fine-tuning the current model with latest data.

Most of these studies share the same assumption that the latest data contains more useful information, and thus the detection of concept drift and following model adaptation is mainly applied to the latest data.

![image](https://user-images.githubusercontent.com/59775002/207271652-0b673817-0426-40ce-876e-23f69fe76333.png)

Most of the existing studies paid less attention to the scenarios that concept drift evolves in a gradual nonrandom way, which are in fact more common in streaming data.

In this paper, authors propose a novel method DDG-DA to predict the data distribution of the next time-step sequentially, such that the model of the downstream learning task can be trained on the data sample from the predicted distribution instead of catching up with the latest concept drift only. In practice, DDG-DA is designed as a dynamic data generator that can create sample data from previously observed data by following predicted future data distribution. In other words, DDG-DA generates the resampling probability of each historical data sample to construct the future data distribution in estimation. However, it's challenging, in reality, to train this data generator to maximize the similarity between the predicted data distribution and the ground truth future data distribution.

To address this challenge, the authors propose to first represent a data distribution by learning a model under this data distribution and then create a differentiable distribution distance to train the data generator. To verify the effectiveness of this approach, they also conduct a thorough theoreticalness of this approach, they also conduct a thorough theoretical analysis to prove the equivalence between traditional distribution distance, e.g. KL-divergence, and the proposed differentiable distribution distance.

## Method Design

### Overall Design

In streaming data, forecasting models are trained and adapted on historical data (training data $D^t_{train}$) and make predictions on unseen data (test data $D^{t}_{test}$) in the future. Training data and test data change over time. for each timestamp, $t$, the target of $task^{(t)} : = (D^{(t)}_{train}$$,D^{(t)}_{test})$ is to learn a new model or adapt an existing model on historical data $D^{(t)}_{train}$ and minimize the loss on $D^{(t)}_{test}$. the data come continuously and might never end. Therefore models can leverage a limited size of $D^{(t)}_{train}$ at timestamp $t$ due to storage limitation. $D^{(t)}_{train}$ is a dataset sampled from training data distribution $p^{(t)}_{train}(x,y)$ and $D^{(t)}_{test}$ from test data distribution $p^{(t)}_{test}(x,y)$. The two distributions may be different. This distribution gap is harmful to the forecasting acurracy on $D^{(t)}_{test}$ when learning forecasting models on $D^{(t)}_{train}$ with a distribution different from $D^{(t)}_{test}$.

#### DDG-DA Learning

![image](https://user-images.githubusercontent.com/59775002/207271848-25a7b307-c5a3-4910-9b67-6fe9bc1177f1.png)

To bridge this gap, DDG-DA (annotated as $M Θ$ ) tries to model the concept drift and predict the test data distribution $p^{(t)}_{test}(x,y)$. The framework of DDG-DA is demonstrated in prev figure. DDG-DA will act like a weighted data sampler to resample on $D^{(t)}_{train}$ and create a new training dataset $D^{(t)}_{resame}(Θ)$ whose data distribution is $p^{(t)}_{resame}(x,y;Θ)$ (the distribution of the resampled dataset serves as the prediction of test distribution). DDG-DA tries to minimize difference between the $p^{(t)}_{resam}(x,y;Θ)$ (the predicted data distribution) and the test data distribution $p^{(t)}_{test}(x,y)$ (the ground truth data distribution).

During training process of DDG-DA, $MΘ$ tries to learn patterns on $task^{(t)} ∈ Task_{train}$ by minimizing the data distribution distance between $p^{(t)}_{resam}(x,y;Θ)$ and $p^{(t)}_{test}(x,y)$. the knowlege learned by $MΘ$ from $Task_{train}$ is expected to transfer to new tasks from $Task_{test}$.

#### DDG-DA Forecast

For a given task $task^{(t)} ∈ Task_{test}$ ,the forecasting model is trained on dataset $D^{(t)}_{resam}(Θ)$ under distribution $p^{(t)}_{resam}(x,y;Θ)$ and forecasts on test data $D^(t)_{test}$.$p^{(t)}_{resam}(x,y;Θ)$ is the distribution of resampled dataset $D^(t)_{resam}(Θ)$ is more similar to $D^{(t)}_{test}$ than $D6{(t)}_{train}$.

### Model Design and Learning Process

![image](https://user-images.githubusercontent.com/59775002/207271979-dc04adec-3203-440b-8af2-df3fa58550ce.png)

#### Feature Design

DDG-DA is expected to guide the model learning process in each $task^{(t)}$ by forecasting test data distribution. Historical data distribution information is useful to predict the target distribution of $D^{(t)}_{test}$ and is input into DDG-DA. DDG-DA will learn concept drift patterns from training tasks and help to adapt models in test tasks.

DDG-DA could be formalized as $q^{(t)}_{train} = M_Θ (g(D^{(t)}_{train})).g$ is a feature extractor. It takes $D^{(t)}_{train}$ as input and outputs historical data distribution information. $MΘ$ leverages the extracted information and outputs the resampling probabilities for samples in $D^(t)_{train}$.

#### Objective Function

$MΘ$ accepts the extracted feature and outputs the resampling probability on $D^{(t)}_{train}$. The resampled dataset's joint distribution $p^{(t)}_{resam}(x,y;Θ)$ serves as the distribution prediction. The learning target of DDG-DA is to minimize the difference between $p^{(t)}_{resam}(x,y;Θ)$ and $p6{(t)}_{test}(x,y)$. they focus on the most important concept assume the difference between $p^{(t)}_{test}(x)$ and $p^{(t)}_{resam}(x;Θ)$ are minor. The loss of DDG-DA could be reformulated as:

![image](https://user-images.githubusercontent.com/59775002/207272123-3930d403-5158-40a9-928c-edb06742f48a.png)

Where $D_{KL}$ represents the Kullback-Leibler divergence.

Normal distribution assumption is reasonable for unknown variables and often used in maximum likelihood estimation. Under this assumption, $p^{(t)}_{test}(y|x) = N(y^{(t)}_{test}(x),σ)$ where $σ$ is constant.

Summarizing losses of all training tasks, the optimization target of DDG-DA could be formalized as follows:

![image](https://user-images.githubusercontent.com/59775002/207272323-c2e635d4-57bc-48a6-ac86-8cdf80c523cb.png)

DDG-DA learns knowledge from $Task_{train}$ and transfers it to unseen test tasks. In each task, DDG-DA forecasts the future data distribution and generates dataset $D^(t)_{resam}(Θ)$ Learning the forecasting models on $D^{(t)}_{resam}(Θ)$, it adapts to upcoming streaming data better.

#### Optimization

DDG-DA adopts a model with a closed-form optimization solution as $y^{(t)}_{proxy}(x;φ^(t))$. there're many choices, such as logistic regression, kernel-based nonlinear model, and differentiable closed-form solvers. Authors choose a linear model for simplicity $h(x; φ_{(t)} ) = xφ^{(t)} for y_{proxy} (x; φ^{(t)})$.

The resampling probability $q^{(t)}_{train}$ outputted by $M_Θ$ could be regarded as sample weights when learning forecasting models. The loss function can be formulated as:

![image](https://user-images.githubusercontent.com/59775002/207272424-65c576fa-008f-48e5-9b9b-470cb508e7bc.png)

Where $X^{(t)},y^{(t)}$ and $Q^{(t)}$ represent the concatenated features, labels and resampling probability in $D^{(t)}_{train}$.

## Comparison

![image](https://user-images.githubusercontent.com/59775002/207272520-235ed290-3885-4886-aff8-d7edd714874a.png)
