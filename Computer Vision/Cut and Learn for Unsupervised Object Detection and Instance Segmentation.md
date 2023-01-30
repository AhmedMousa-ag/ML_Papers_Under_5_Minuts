# Cut and Learn for Unsupervised Object Detection and Instance Segmentation

Authors: Xudong Wang, Rohit Girdhar, Stella X. Yu, and Ishan Misra.

Authors propose **Cut**-and-**Le**a**R**n (*CutLER*), which is a simple approach for training unsupervised object detection and segmentation models.

## Introduction

Author's method **Cut**-and-**Le**a**R**n (CutLER) consists of three simple architecture and data-agnostic mechanisms. CutLER is trained exclusively on unlabeled ImageNet data without the need of additional training data.

Authors first propose MaskCut which can automatically produce multiple initial coarse masks for each image using the pre-trained self-supervised features.

Secondly, they propose a simple loss-dropping strategy to train detectors using the coarse masks while being robust to objects missed by MaskCut.

**Features of CutLER:**

***1) Simplicity:*** CutLER is simple to train and agnostic to the choice of detection and backbone architectures.

***2) Zero-shot detector:*** CutLEER trained solely on ImageNet shows strong zero-shot performance on 11 different bench-marks where it outperforms prior work trained with additional in-domain data.

***3) Robustness:*** CutER exhibits strong robustness against domain shifts when tested on images from different domains such as video frames, sketches, paintings, clip arts, etc.

***4) Pretraining for supervised detection:*** CutLER can serve as a pre-trained model for training fully supervised object detection and instance segmentation models and improves performance on COCO, including on few-shot object detection benchmarks.

## Method

$ Figure 2

As illustrated in previous figure, authors propose MaskCut (See next figure) that generates multiple binary masks per image using self-supervised features from DINO. Second, they show a dynamic loss-dropping strategy, called DropLoss that can learn a detector from MaskCut's initial masks while encouraging the model to explore objects missed by MaskCut. Third, they further improve the performance of their method through multiple rounds of self-training

Preliminaries

**Normalized Cuts** (Ncut) treats the image segmentation problem as a graph partitioning task. They construct a fully connected undirected graph via representing each image as a node. Each pair of nodes is connected by edges with weights $W_{ij}$ that measure the similarity of connected nodes. NCut minimizes the cost of partitioning the graph into two sub-graphs, a bipartition, by solving a generalized eigenvalue system for finding the eigenvector $x$ that corresponds to the second smaller eigenvalue $λ$, where $D$ is a $N ×N$ diagonal matrix $P$ with $d(i) = ∑_j W{_ij}$ and $W$ is a $N×N$ symmetrical matrix.

$Equation 1

**DINO and TokenCut** DINO finds that the self-supervised ViT can automatically learn a certain degree of perceptual grouping of image patches.

TokenCut leverages the DINO features for NCut and obtaining foreground/background segments in an image.

### MaskCut for Discovering Multiple Objects

$ Figure 3

Vanilla NCut is limited to discovering a single object in an image. thereby authors propose MaskCut that extends NCut to discover multiple objects per image by iteratively applying NCut to a masked similarity matrix. After getting the bipartition $x^t$ from NCut at stage $t$, we get two disjoint groups of patches and construct a binary mask $M^t$, where:

$ Eqution 2

To determine which group corresponds to the foreground, they make use of two criteria:

1) intuitively, the foreground patches should be more prominent than background patches. Therefore, the foreground mask should contain the patch corresponding to the maximum absolute value in the second smallest eigenvector $M^t$
2) Authors incorporate a simple but empirically effective object-centric prior, the foreground set should contain less than two of the four corners. They reverse the partitioning of the foreground and background, *i.e.,* $M^t_{ij} = 1- M^t_{ij}$, if the criteria 1 is not satisfied while the current foreground set contains two corners or the criteria 2 is not satisfied. In practice, authors also set all $W_{ij}< τ^{ncut}$ to $1e^{-5}$ and $W_{ij} ≥ τ^{ncut}$ to $1$.

To get a mask for the $(t+1)^{th}$ object, authors update the node similarity $W^{t+1}_{ij}$ via masking out these nodes corresponding to the foreground in previous stages:

$ Eqution 3

Where $M̂_{ij}= 1 − M_{ij}$. Using the updated $W^{t+1}_{ij}$, repeat Equation 1, and 2 to get a mask $M^{t+1}$. repeat this process $t$ times and set $t=3$ by default.

### DropLoss for Exploring Image Regions

A standard detection loss penalized predicted regions $r_i$ that do not overlap with the "ground-truth". Since the "ground-truth" masks given by MaskCut may miss instances, the standard loss does not enable the detector to discover new instances not labeled in the "groun-truth". Therefore authors propose to ignore the loss of predicted regions $r_i$ that have a small overlap with the "ground-truth". More specifically, during training they drop the loss for each predicted region $r_i$ that has a maximum overlap of $τ^{IoU}$ with any of the "ground-truth" instances:

$L_{drop} (r_i) = 1 (IoU^{max}>τ^{IoU})L_{vanilla} (r_i)$

Where $IoU^{max}_i$ denotes the maximum IoU with all "ground-truth" for $r_i$ and $L_{vanilla}$ refers to the vanilla loss function of detectors. $L_{drop}$ does not penalize the model for detecting objects missed in the "ground-truth" and thus encourages the exploration of different image regions. In practice, authors use a low threshold $τ_{IoU} = 0.01$.

### Multi-Round Self-Training

Empirically, authors found that despite learning from the coarse masks obtained by MaskCut, detection models "clean" the ground truth and produce masks (and boxes) that are better than the initial coarse masks used for training. The detectors refine mask quality, and DropLoss strategy encourages them to discover new objects masks. Thus authors leverage this property and use multiple rounds of self-training to improve the detector's performance.

Authors use the predicted masks and proposals with a confidence score over $0.75-0.5t$ from the $t^{th}$-round as the additional pseudo annotations for the $(t+1)^{th}$-round of self-training. To de-duplicate the predictions and the ground truth from round $t$, authors filter out ground-truth masks with an $IoU > 0.5$ with the predicted masks. Authors found that three rounds of self-training are sufficient to obtain good performance. Each round steadily increases the number of "ground-truth" samples used to train the model.

## Comparisons

$ Table 3

$ Table 4

$ Table 5
