# Spotlight

Authors spot light on the cruciality of provisioning ML models, where they discuess key areas including *model performance*, *data monitoring*, *detecting outlier drifts*, and *model explain-ability* as well as open source solutions.
## Challenges

* Design systems that monitor live deployments and take action or raise alerts when events impacts model performance.

* Inability to detect model performance degradation can lead to stale models and increased technical debt.
* Measuring model performance implies having timely access to labels for live data which is mostly not available due to operational and financial constraints.
* Monitoring systems requires functionality to determine when significant changes to data and predictive distributions happen.

* Building trust in ML systems and make decision process transparent as models often are *black boxes*.

# Monitoring

## Performance and metrics

For live data the number of frequency of collected labels depends on the application, for example time-series prediction or interned ad serving labels are automatic, but for many other applications labels are expensive to produce such as medical diagnostic system based on image recognition which requires domain knowledge and time consuming.

A fundamental draw back of uni-variate metrics is that correlations between features are not captured. and multivariate metrics such as covariance matrices and multivariate histograms remain difficult to implement due to increased computational cost and curse of dimensionality as well as the need for online update rule.

## Outlier Detection

ML models often fail to generalize outside of the training data distribution and models are typically not well calibrated which can lead to overconfident predictions on out of distribution instances. there for **outlier detection** is a key to flag anomalies whose model predictions can't be trusted.

The type of outlier detector depends on the modality, dimensionality of the data, availability of labeled normal and outlier data and whether the detector is pre-trained (offline) or updated online.

Pre-trained detector can be deployed as a separate static ML model while the online detector is deployed as a stateful application.

It is important to note that the problem of unsupervised anomaly detection for real-world data is far from solved "e.g. natural images or noisy times series". though different studies on image data also illustrate that generative density models can assign higher likelihood values to out-of-distribution instances compared to inlier data.

## Drift Detection

While outliers refer to individual instances, drift/shift detection checks whether two samples are drawn from the same underlying distribution or not via a statistical hypothesis test.
We can distinguish co-variate shift from label shift of the model predictions.

1- Co-variate shift: the input distribution *p(x)* changes while the conditional label distribution *p(y|x)* remains unchanged.

2- Label shift happens when *p(y)* changes but the conditional *p(x|y)* does not.

In practice for high dimensional data to work such as images, a dimensionality reduction must be applied before applying the hypothesis test. other observations shows that randomly initialized encoders and black box dimensionality reduction are promising pre-processing methods, followed by a two-sample test such as maximum mean discrepancy for multivariate case in combination with a permutation test to obtain p-values.

## Deploying Model Monitoring

The techniques discussed before need to be deployed alongside the running models but in a manner which does not affect their core performance.

KFServing and Seldon Core which run on Kubernetes solve this by utilizing the eventing based project by allowing serverless components to be connected to event streams.

# Explainability

## The need for model explanations

1- ML model explanations allow users to build trust in model predictions and improve transparency

2- User can verify which factors contribute to certain predictions, which introduce a layer of accountability,

3- The data used for pre-trained models isn't curated by user and often leads to biased training sets and discrimination, and that goes hand to hand when drift detection flags outlier, an explanation methods can help determine whether the model prediction can be trusted or not.

Explanation algorithms can be grouped into:

### White-Box

Which assumes access to model internals such as being able to take dradients with respect to the input.

### Black-Box 

Which doesn't assume anything beyond being able to access the prediction API endpoint. and the only way to interact with the model is by requesting predictions.

## Deploying Black-Box Explanations

Black-box explanation algorithms work by taking an input instance whose prediction is to be explained and by repeatedly querying the model with modified versions of the input to approximate it's predictive behavior. The actual query strategy, definition of modified instances and explanation output is specific to each algorithm. In production this translates to having two deployments, the original model and the explainer, exposing a prediction endpoint and an explanation endpoint respectively. When the prediction endpoint is called with a data point. a prediction is returned as usual, but when the explanation endpoint is called with the same data point, this triggers the black-box explanation algorithm to internally query the model and produce an explanation.

This pattern has the advantage of using the underlying infrastructure to auto-scale the model deployment if a high volume of explanations is requested.

Alternatively, this pattern can be implemented on a carbon copy of the model so as to separate production prediction requests from introspective explanation requests.
