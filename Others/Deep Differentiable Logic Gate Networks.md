#  Deep Differentiable Logic Gate Networks

Ahutors: Felix Petersen, Christian Borgelt, Hilde Kuehne, Oliver Deussen.

Authors explore logic gate network for machine learning tasks by learning combinations of logic gates, and they propose differentiable logic network which is an architecture that combines real-valued logics and a continuously parameterized relaxation of the network.

## Introduction

The problem in training networks of discrete components like logic gates, is that they are non-differentiable and therefore, conventionally, cannot be optimized via standard methods such as gradient descent . One approach for this would be gradient-free optimization methods such as evolutionary training, which works for small models, but becomes infeasible for larger ones.

In this work authors propose an approach for gradient-based training of logic gate networks (aka. arithmetic / algebric circuits). Logic networks are based on binary logic gates, such as "and", and "xor". So authors relax them continuosly during training to differentiable logic gate network, which allows efficiently training them with gradient descent. and for this authors used real-valued logic and learn which logic gate to use at each neuron.

Logic gate networks allow for very fast classification, with speeds beyond a million images per second on a single CPU core (for MNIST at > 97.5% accuracy). The computational cost of a layer with $n$ neurons is $Θ(n)$ with very small constants (as only logic gates of Booleans are required), while, in comparison, a fully connected layer (with m input neurons) requires $Θ(n · m)$ computations with significantly larger constants (as it requires floating-point arithmetic). While the training can be more expensive than for regular neural networks (however, just by a constant and asymptotically less expensive), to our knowledge, the proposed method is the fastest available architecture at inference time.

## Logic Gate Network

![image](https://user-images.githubusercontent.com/59775002/212854366-c389b284-57a3-4258-9ee5-210cbb58b524.png)


Logic gate networks are networks similar to neural networks where each neuron is represented by a binary logic gate like ‘and’, ‘nand’, and ‘nor’ and accordingly has only two inputs (instead of all neurons in the previous layer as it is the case in fully-connected neural networks). Given a binary vector as input, pairs of Boolean values are selected, binary logic gates are applied to them, and their output is then used as input for layers further downstream. Logic gate networks do not use weights. Instead, they are parameterized via the choice of logic gate at each neuron. In contrast to fully connected neural networks, binary logic gate networks are sparse because each neuron has only 2 instead of $n$ inputs, where $n$ is the number of neurons per layer. In logic gate networks, we do not need activation functions as they are intrinsically non-linear.

Previous figure illustrates a small logic gate network. In the illustration, each node corresponds to a single logic operator. Note that the distribution over operators (red) is part of the differentiable relaxation discussed in the next section.
As logic gate networks build on bit-wise logic operations only, their execution is very efficient.

## Differentiable Logic Gate Networks

![image](https://user-images.githubusercontent.com/59775002/212854506-581d4efc-ae84-4510-8870-a310d01095d3.png)

**Differentiable Logics** To make binary logic networks differentiable, we leverage the following relaxation. First, instead of hard binary activations / values $a ∈ {0, 1}$, authors relax all values to probabilistic activations $a ∈ [0, 1]$. Second, we replace the logic gates by computing the expected value of the activation given probabilities of independent inputs $a_1$ and $a_2$ . For example, the probability that two independent events with probabilities $a_1$ and $a_2$ both occur is $a_1 · a_2$ . These operators correspond to the probabilistic T-norm and T-conorm; a report of the full set of relaxations corresponding to the probabilistic interpretation in Table 1.

Accordingly, authors define the activation of a neuron with the $i$th operator as:

$a = f i (a 1 , a 2 ) ,$

where $f_i$ is the $i$th real-valued operator corresponding to previous table and $a_1$ , $a_2$ are the inputs to the neuron.

**Differentiable Choice of Operator** While real-valued logics allow differentiation, they do not allow training as the operators are not continuously parameterized and thus (under hard binary inputs) the activations in the network will always be $a ∈ {0, 1}$. Thus, we propose to represent the choice of which logic gate is present at each neuron by a categorical probability distribution. For this, we parameterize each neuron with 16 floats (i.e., $w ∈ R^{16} $), which, by softmax, map to the probability simplex (i.e., a categorical probability distribution
such that all entries sum up to 1 and P it has only non-negative values). That is, $p_i = e^{wi} /( \sum_j e^{wj}$ ), and thus $p$ lies in the probability simplex $p ∈ ∆^{15}$ . During training, we evaluate for each neuron all 16 relaxed binary logic gates and use the categorical probability distribution to compute their weighted average. Thus, authors define the activation $a$ of a differentiable logic gate neuron as

![image](https://user-images.githubusercontent.com/59775002/212854662-4814e77b-ec60-48fa-b077-09a4eebb6aad.png)

**Aggregation of Output Neurons** Now, we may have $n$ output neurons $a_1$ , $a_2$ , ..., $a_n ∈ [0, 1]$, but we may want the logic gate network to only predict $k < n$ values of a larger range than $[0, 1]$. Further, we may want to be able to produce graded outputs. Thus, we can aggregate the outputs as

![image](https://user-images.githubusercontent.com/59775002/212854736-da4b3cd3-2e2d-4f11-8912-cfe2967acc06.png)

where $τ$ is a normalization temperature and $β$ is an optional offset.

#### Training Considerations

**Training** For learning, authors randomly initialize the connections and the parameterization of each neuron. For the initial parameterization of each neuron, they draw elements of $w$ independently from a standard normal distribution. In all reported experiments, they used the same number of neurons in each layer (except for the input) and between 4 and 8 layers, which they call straight network. they trained all models with the Adam optimizer at a constant learning rate of 0.01.

**Discretization** After training, during inference, authors discretize the probability distributions by only taking their mode (i.e., their most likely value), and thus the network can be computed with Boolean values, which makes inference very fast. In practice, they observe that most neurons converge to one logic gate operation; therefore, the discretization step introduces only a small error, e.g., for MNIST, the gap is smaller than 0.1%. We note that all reported results are accuracies after discretization.

**Classification** n the application of a classification learning setting with $k$ classes (e.g., 10) and $n$ output neurons (e.g., $1 00$), we group the output into $k$ groups of size $n/k$ (e.g., 100). Then, we count the number of $1$s which corresponds to the classification score such that the predicted class can be retrieved via the arg max of the class scores. During differentiable training, we sum up the probabilities of the outputs in each group instead of counting the 1s, and we can train the model using a softmax cross-entropy classification loss.

**Regression** For regression learning, let us assume that we need to predict a $k$-dimensional output vector. Here, $τ$ and $β$ play the role of an affine transformation to transform the range of possible predictions from $0$ to $n/k$ to an application specific and more suitable range. Here, the optional bias $β$ is important, e.g., if we want to predict values outside the range of $[0, n/k/τ ]$. In some cases, it is desirable to cover the entire range of real numbers, which may be achieved using a logit transform $logit(x) = σ^{−1} (x) = log{^x_{1−x}}$ in combination with $τ = n/k, β = 0$. During differentiable training, we sum up the probabilities of the outputs in each group instead of counting the 1s, and we can train the model, e.g., using an MSE loss.

## Current Limitations and Opportunities

**Expensive Training** A limitation of differentiable logic gate networks is their relatively higher training cost compared to (performance-wise) comparable conventional neural networks. The higher training cost is because multiple differentiable operators need to be evaluated for each neuron, and in their real valued differentiable form, most of these operators require floating-point value multiplications. However, the practical computational cost can be reduced through improved implementations. authors note that, asymptotically, differentiable logic gate networks are cheaper to train compared to conventional neural networks due to their sparsity.

**Convolutions and Other Architectures** Convolutional logic gate networks and other architectural components such as residual connections are interesting and important directions for future research.

**Edge Computing and Embedded Machine Learning** authors would like to emphasize that the current limitations to rather small architectures (compared to large deep learning architectures) does not need to be a limitation: For example, in edge computing and embedded machine learning, models are already limited to tiny architectures because they run, e.g., on mobile CPUs, microcontrollers, or IoT devices. In these cases, training cost is not a concern because it is done before deployment.
Authors also note that there are many other applications in industry where the training cost is negligible in comparison to the inference cost.

## Experiments results

![image](https://user-images.githubusercontent.com/59775002/212854870-010fb832-265a-4e4e-8519-3b2ade241046.png)
