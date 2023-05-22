# Data Science and Machine Learning Reading List: From 101 to Advanced Topics


## Part I: Foundations
### 1 Probability Theory and Statistical Inference
#### Probability Space and Measure
* [Probability Space](https://en.wikipedia.org/wiki/Probability_space) `wiki`

### Common Probability Distributions
* [Univariate Distribution Relationships](http://www.math.wm.edu/~leemis/2008amstat.pdf) `paper`

### Convergence of Random Variables
* [Law of Large Numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers) `wiki`
* [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) `wiki`

## Statistical Inference

### Estimation
* [Method of Moments Estimation](https://en.wikipedia.org/wiki/Method_of_moments_(statistics)) `wiki`
* [Maximum Likelihood Estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) `wiki`
* [Confidence Interval](https://en.wikipedia.org/wiki/Confidence_interval) `wiki`
* [Bootstrap: A Statistical Method](https://www.stat.rutgers.edu/home/mxie/rcpapers/bootstrap.pdf) `paper`

## Hypothesis Testing
* [Tests of Hypotheses Using Statistics](https://web.williams.edu/Mathematics/sjmiller/public_html/BrownClasses/162/Handouts/StatsTests04.pdf) `lecture notes`

### Concepts
* [Type I and Type II Errors](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors) `wiki`
* [$P$-Value](https://en.wikipedia.org/wiki/P-value) `wiki`
* [American Statistical Association Releases Statement on Statistical Significance and $P$-Values](https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf) `other`

### Parametric Tests
* [$Z$-Test](https://en.wikipedia.org/wiki/Z-test) `wiki`
* [Test of Proportion](https://online.stat.psu.edu/statprogram/reviews/statistical-concepts/proportions) `wiki`
* [One Sample $T$-Test](https://en.wikipedia.org/wiki/Student%27s_t-test#One-sample_t-test) `wiki`
* [Two Sample $T$-Test with Equal Variance](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_similar_variances_(1/2_%3C_sX1/sX2_%3C_2%29)) `wiki`
* [Two Sample $T$-Test with Unequal Variance](https://en.wikipedia.org/wiki/Student%27s_t-test#Equal_or_unequal_sample_sizes,_unequal_variances_(sX1_%3E_2sX2_or_sX2_%3E_2sX1%29)) `wiki`
* [Paired $T$-Test](https://en.wikipedia.org/wiki/Student%27s_t-test#Dependent_t-test_for_paired_samples) `wiki`
* [Pearson's $\chi^2$ Test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test) `wiki`

### Nonparametric Tests
* [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) `wiki`
* [Wilcoxon Rank-Sum Test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test) `wiki`
* [Wilcoxon Signed-Rank Test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) `wiki`

## Bayesian Statistical Inference
* [Formal Description of Bayesian Inference](https://en.wikipedia.org/wiki/Bayesian_inference#Formal_description_of_Bayesian_inference) `wiki`
* [What's the Difference between a Confidence Interval and a Credible Interval?](https://stats.stackexchange.com/questions/2272/whats-the-difference-between-a-confidence-interval-and-a-credible-interval) `forum`


# 2 Linear Regression

## Classical Linear Regression

### Ordinary Least Squares
* Section 3.2, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

### Gauss-Markov Theorem
* Section 3.2.2, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

### Inference
* Section 3.2, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`
### R Squared
* [Coefficient of Determination](https://en.wikipedia.org/wiki/Coefficient_of_determination) `wiki`

### Violations of Linear Regression Assumptions and Remedies
* [Violations of Classical Linear Regression Assumptions](https://www.bauer.uh.edu/jhess/documents/ViolationsofClassicalLinearRegressionAssumptions.doc) `lecture notes`
* [Generalized Least Squares](https://en.wikipedia.org/wiki/Generalized_least_squares) `wiki`
* [Newey–West Estimator](https://en.wikipedia.org/wiki/Newey%E2%80%93West_estimator) `wiki`
* [Cochrane–Orcutt Estimation](https://en.wikipedia.org/wiki/Cochrane%E2%80%93Orcutt_estimation) `wiki`

## Shrinkage and High Dimensional Methods

### Subset Selection
* Section 3.3.1, 3.3.2, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

### Principle Component Regression
* Section 3.5.1, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

### Partial Least Squares
* Section 3.5.2, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

### Ridge Regression
* Section 3.4.1, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

### Lasso and Its Variants
* Section 3.4.2, 3.4.3, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`
* [Regression Shrinkage and Selection via the Lasso](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.35.7574&rep=rep1&type=pdf) `paper`
* [Least Angle Regression](https://statweb.stanford.edu/~tibs/ftp/lars.pdf) `paper`
* [Regularization and Variable Selection via the Elastic Net](https://www.jstor.org/stable/3647580?seq=1) `paper`
* [Model Selection and Estimation in Regression with Grouped Variables](http://www.columbia.edu/~my2550/papers/glasso.final.pdf) `paper`
* [The Adaptive Lasso and Its Oracle Properties](http://users.stat.umn.edu/~zouxx019/Papers/adalasso.pdf) `paper`
* [Exact Post-Selection Inference, with Application to the Lasso](https://www.stat.cmu.edu/~ryantibs/statml/lectures/Lee-Sun-Sun-Taylor.pdf) `paper`

### Other Advanced Methods
* [Variable Selection via Nonconcave Penalized Likelihood and Its Oracle Properties](https://fan.princeton.edu/papers/01/penlike.pdf) `paper`
* [Nearly Unbiased Variable Selection under Minimax Concave Penalty](https://arxiv.org/pdf/1002.4734.pdf) `paper`
* [Sure Independence Screening for Ultrahigh Dimensional Feature Space](https://fan.princeton.edu/papers/06/SIS.pdf) `paper`

## Robust Methods
### Least Trimmed Squares
* Section 8.4.2, [Linear Models with R, 2nd](https://www.routledge.com/Linear-Models-with-R/Faraway/p/book/9781439887332) `book chapter`

### Quantile Regression
* [Quantile Regression](https://en.wikipedia.org/wiki/Quantile_regression) `wiki`


# 3 Time Series Analysis

## Linear Time Series Models
### Stationarity 
* Section 2.1, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

### Serial Correlation
* Section 2.2, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

### White Noise
* Section 2.3, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

### ARMA
* Section 2.6, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

### Unit-Root Nonstationarity
* Section 2.7, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

## Nonlinear Linear Time Series Models

### Threshold Autoregressive Model
* Section 4.1.2, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

### Markov Switching Model
* Section 4.1.4, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

## Volatility Models

### ARCH
* Section 3.4, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

### GARCH and Its Extensions
* Section 3.5, 3.6, 3.8, 3.9, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`


## Multivariate Time Series Models

### Vector Autoregressive Model
* Section 8.2, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

### Cointegration and Error-Correction Model
* [Cointegration](https://en.wikipedia.org/wiki/Cointegration) `wiki`
* [Error Correction Model](https://en.wikipedia.org/wiki/Error_correction_model) `wiki`
* [Cointegration: The Engle and Granger Approach](https://warwick.ac.uk/fac/soc/economics/staff/gboero/personal/hand2_cointeg.pdf) `lecture notes`

## State-Space Model, Structural Time Series, and Kalman Filter
* Chapter 5, [A Very Short Course on Time Series Analysis](https://bookdown.org/rdpeng/timeseriesbook/state-space-models-and-the-kalman-filter.html) `book chapter`
* [Forecasting at Scale](https://peerj.com/preprints/3190.pdf) `paper`
* [Predicting the Present with Bayesian Structural Time Series](https://people.ischool.berkeley.edu/~hal/Papers/2013/pred-present-with-bsts.pdf) `paper`


# 4 Supervised Learning

## Logistic Regression
* Section 4.4, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`


## Naive Bayes Classifier
* Section 6.6.3, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

### Relationship to Logistic Regression
* Section 4, [Generative and Discriminative Classifiers: Naive Bayes and Logistic Regression](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf) `lecture notes`

## Linear Discriminant Analysis
* Secion 4.4, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

## K-Nearest Neighbor
* Secion 13.3, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

## Support Vector Machine
* Secion 12.1-12.3, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

## Classification and Regression Trees
* Secion 9.2, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

## Bagging
* Section 8.7, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

## Random Forests
* Section 15.1-15.3, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

## Boosting
* [Boosting](https://web.stanford.edu/~hastie/TALKS/boost.pdf)  `lecture notes`
* [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/latest/tutorials/model.html) `documentation`
* [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/pdf/1603.02754.pdf) `paper`
* [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://www.audentia-gestion.fr/MICROSOFT/lightgbm.pdf) `paper`
* [CatBoost: Gradient Boosting with Categorical Features Support](http://learningsys.org/nips17/assets/papers/paper_11.pdf) `paper`

## Stacking
* Section 8.8, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`
* [Super Learner in Prediction](https://core.ac.uk/download/pdf/61320298.pdf) `paper`

# 5 Unsupervised Learning

## Clustering

### K-Means Clustering
* Section 14.3.6, 14.3.11, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

#### Relation to Gaussian Mixture Models and EM Algorithm
* Section 8.5.1, 14.3.7, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

### Hierarchical Clustering
* Section 14.3.12, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

### Density-Based Clustering
* [A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) `paper`
* [OPTICS: Ordering Points to Identify the Clustering Structure](https://www.dbs.ifi.lmu.de/Publikationen/Papers/OPTICS.pdf) `paper`

### Clustering Time Series Data
* [Clustering of Time Series Data - A Survey](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.115.6594&rep=rep1&type=pdf) `paper`

## Dimension Reduction
### Principal Component Analysis
* [Principal Component Analysis](https://www.comp.nus.edu.sg/~cs5240/lecture/pca.pdf) `lecture notes`

### Statistical Factor Analysis
* Secion 9.5, [Analysis of Financial Time Series, 3rd](https://www.wiley.com/en-us/Analysis+of+Financial+Time+Series%2C+3rd+Edition-p-9780470414354) `book chapter`

## Graphical Lasso
* Section 17.1, 17.3, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`
* [Sparse Inverse Covariance Estimation with the Graphical Lasso](http://statweb.stanford.edu/~tibs/ftp/graph.pdf) `paper`
* [Review of Sparse Inverse Covariance Matrix with the Graphical Lasso](https://github.com/CamDavidsonPilon/Graphical-Lasso-in-Finance/blob/master/report.pdf) `paper`
* [The `huge` Package for High-Dimensional Undirected Graph Estimation in R](https://cran.r-project.org/web/packages/huge/vignettes/vignette.pdf) `paper`


# 6 Reinforcement Learning

## Overview
* [Introduction to Reinforcement Learning with David Silver](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver) `lecture notes`
* [Deep Reinforcement Learning: An Overview](https://arxiv.org/pdf/1701.07274.pdf) `paper`
* [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) `other`

## Model-Free Methods

### Value-Based Methods
#### Temporal Difference Methods
* Section 6.1, [Reinforcement Learning: An Introduction, 2nd](https://mitpress.mit.edu/books/reinforcement-learning-second-edition) `book chapter`

#### SARSA
* Section 6.4, [Reinforcement Learning: An Introduction, 2nd](https://mitpress.mit.edu/books/reinforcement-learning-second-edition) `book chapter`

#### Q-Learning
* Section 6.5, [Reinforcement Learning: An Introduction, 2nd](https://mitpress.mit.edu/books/reinforcement-learning-second-edition) `book chapter`

#### Deep Q Network
* [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf) `paper`
* [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf) `paper`
* [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf) `paper`

### Policy-Based Methods
#### REINFORCE
* Section 13.3, [Reinforcement Learning: An Introduction, 2nd](https://mitpress.mit.edu/books/reinforcement-learning-second-edition) `book chapter`

#### Actor-Critic Methods
* Section 13.5, [Reinforcement Learning: An Introduction, 2nd](https://mitpress.mit.edu/books/reinforcement-learning-second-edition) `book chapter`

## Model-Based Methids
* [CS294-112 Lecture 9: Model-Based Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_9_model_based_rl.pdf) `lecture notes`


# 7 Deep Learning

## Overview
* [Deep Learning](https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf) `paper`
* [Deep Learning](https://www.deeplearningbook.org/) `book`

## Building Blocks and Feedforward Neural Networks
### Backpropagation
* [CS231n Lecture 4: Backpropagation and Neural Networks](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture4.pdf) `lecture notes`
* [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) `lecture notes`

### Vanishing and Exploding Gradient 
* [CSC321 Lecture 15: Exploding and Vanishing Gradients](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/readings/L15%20Exploding%20and%20Vanishing%20Gradients.pdf) `lecture notes`
* [On the Difficulty of Training Recurrent Neural Networks](https://arxiv.org/pdf/1211.5063.pdf) `paper`

### Weight Initialization
* [Weight Initialization Schemes - Xavier (Glorot) and He](https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init) `paper`
* [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
](https://arxiv.org/pdf/1502.01852.pdf) `paper`
* [Understanding the Difficulty of Training Deep Feedforward Neural Networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) `paper`

### Activation Functions
* [Activation Functions: Comparison of Trends in Practice and Research for Deep Learning](https://arxiv.org/pdf/1811.03378.pdf) `paper`
* [Swish: A Self-Gated Activation Function](https://arxiv.org/pdf/1710.05941v1.pdf) `paper`

### Batch Normalization
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf) `paper`

### Dropout
* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) `paper`
* [Introduction of Dropout and Ensemble Model in the History of Deep Learning](https://medium.com/unpackai/introduction-of-dropout-and-ensemble-model-in-the-history-of-deep-learning-a4c2a512dcca) `forum`

### The Mathe Behind Deep Learning
* Section 5-7, [A Selective Overview of Deep Learning](https://arxiv.org/pdf/1904.05526.pdf) `paper` 

## Convolutional Neural Networks
* [CS231n Lecture 5: Convolutional Neural Networks](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture5.pdf) `lecture notes`
* [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/) `lecture notes`

## Recurrent Neural Networks
* [CS231n Lecture 10: Recurrent Neural Networks](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf) `lecture notes`
* [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) `blog`
* [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf) `paper`

## Generative Learning
* [CS231n Lecture 13: Generative Models](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture13.pdf) `lecture notes`
* [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf) `paper`



### Deep Learning for Computer Vision
* [CS231n Lecture 9: CNN Architectures](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf) `lecture notes`
* [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet) `paper`
* LeNet-5: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) `paper`
* AlexNet: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) `paper`
* GoogleNet: [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf) `paper`
* VGGNet: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) `paper`
* ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) `paper`
* Xception: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf) `paper`
* ResNeXt: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
* SENet: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf) `paper`


### Deep Learning for Natural Language Processing

#### Word2Vec
* [The Illustrated Word2Vec](http://jalammar.github.io/illustrated-word2vec/) `blog`
* [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) `blog`
* [Word2Vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/) `blog`
* [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) `other`

#### Attention and Transformer
* [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) `paper`
* [Visualizing a Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) `blog`
* [Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) `blog`
* [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) `blog`

#### ELMo
* ELMo: [Deep Contextualized Word Representations](https://arxiv.org/pdf/1802.05365.pdf) `paper`
* [Deep Contextualized Word Representations with ELMo](https://www.mihaileric.com/posts/deep-contextualized-word-representations-elmo/) `blog`

#### GPT Series
* GPT2: [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) `paper`
* [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) `blog`
* GPT3: [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) `paper`

#### BERT and Its Variants
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) `paper`
* [The Illustrated BERT, ELMo, and Co.](http://jalammar.github.io/illustrated-bert/) `blog`
* [A Visual Guide to Using BERT for the First Time](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/) `blog`
* [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) `paper`
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf) `paper`
* [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/pdf/1909.11942.pdf) `paper`

# 8 Data Preparation

## Feature Engineering

### Feature Scaling
* [Standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) `documentation`
* [Normalization](https://scikit-learn.org/stable/modules/preprocessing.html#normalization) `documentation`
* [Scaling Features to a Range](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) `documentation`
* [Mapping to a Uniform Distribution](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html#sklearn.preprocessing.QuantileTransformer) `documentation`

### Feature Encoding
* [OneHot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder) `documentation`
* [Label Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder) `documentation`
* [Ordinal Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder) `documentation`

### Automated FEature Engineering
* [Featuretools: An Open Source Python Framework for Automated Feature Engineering](https://www.featuretools.com/) `documentation`

## Data Augmentation 
* [A survey on Image Data Augmentation for Deep Learning](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0) `paper`
* [The Effectiveness of Data Augmentation in Image Classification using Deep Learning](https://arxiv.org/pdf/1712.04621.pdf) `paper`

## Missing Data
* [The Prevention and Handling of the Missing Data](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3668100/) `paper`
* [Statistical Analysis with Missing Data, 3rd](https://www.wiley.com/en-us/Statistical+Analysis+with+Missing+Data%2C+3rd+Edition-p-9780470526798) `book`

## Imbalanced Labels
### Sampling
* [SMOTE: Synthetic Minority Over-Sampling Technique](https://arxiv.org/pdf/1106.1813.pdf) `paper`
* [Imbalanced-learn: A Python Toolbox to Tackle the Curse of
Imbalanced Datasets in Machine Learning](https://www.jmlr.org/papers/volume18/16-365/16-365.pdf) `paper`

### Recalibration
* Section 6.3, [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf) `paper`


# 9 Model Interpretation

## Global Model-Agnostic Methods

### Partial Dependence Plot
* Section 8.1, [Intepretable Machine Learning](https://christophm.github.io/interpretable-ml-book/pdp.html) `book chapter`
* [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/download/pdf_1/euclid.aos/1013203451) `paper`

## Permutation Feature Importance
* Section 8.5, [Intepretable Machine Learning](https://christophm.github.io/interpretable-ml-book/feature-importance.html) `book chapter`
* [All Models are Wrong, but Many are Useful: Learning a Variable’s Importance by Studying an Entire Class of Prediction Models Simultaneously](https://arxiv.org/pdf/1801.01489.pdf) `paper`

### Global Surrogate
* Section 8.6, [Intepretable Machine Learning](https://christophm.github.io/interpretable-ml-book/global.html) `book chapter`

## Local Model-Agnostic Methods

### Individual Conditional Expectation
* Section 9.1, [Intepretable Machine Learning](https://christophm.github.io/interpretable-ml-book/ice.html) `book chapter`
* [Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation](https://arxiv.org/pdf/1309.6392.pdf) `paper`

### Local Surrogate
* Section 9.2, [Intepretable Machine Learning](https://christophm.github.io/interpretable-ml-book/lime.html) `book chapter`
* ["Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf) `paper`

### SHAP Values
* Section 9.5, [Intepretable Machine Learning](https://christophm.github.io/interpretable-ml-book/shapley.html) `book chapter`
* Section 9.6, [Intepretable Machine Learning](https://christophm.github.io/interpretable-ml-book/shap.html) `book chapter`
* [A unified Approach to Interpreting Model Predictions](https://arxiv.org/pdf/1705.07874.pdf) `paper`

## Method Comparison
* [Interpretability Methods in Machine Learning: A Brief Survey](https://www.twosigma.com/articles/interpretability-methods-in-machine-learning-a-brief-survey/)


# 10 Model Evaluation

## Bias-Variance Tradeoff
* [Bias-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) `wiki`
* [Overfitting](https://en.wikipedia.org/wiki/Overfitting) `wiki`

## Cross Validation
* Section 7.10, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`
* [A Study of CrossValidation and Bootstrap for Accuracy Estimation and Model Selectio](https://ai.stanford.edu/~ronnyk/accEst.pdf) `paper`

## Hyperparameter Optimization

### Grid Search
* [Exhaustive Grid Search](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) `documentation`

### Random Search
* [Randomized Parameter Search](https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-search) `documentation`

### Bayesian Optimization
* [CSC321 lecture 21: Bayesian Hyperparameter Optimization](https://netman.aiops.org/~peidan/ANM2019/2.MachineLearningBasics/LectureCoverage/27.BayesianOptimization.pdf) `lecture notes`
* [Practical Bayesian Optimization of Machine Learning Algorithms](https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf) `paper`

## Model Evaluation Metrics
### Classification metrics
* [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall) `wiki`
* [F1-Score](https://en.wikipedia.org/wiki/F-score) `wiki`
* [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix) `wiki`
* [Area Under the ROC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) `wiki`

### Regression Metrics
* [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) `wiki`
* [Mean Absolute Deviation](https://en.wikipedia.org/wiki/Average_absolute_deviation) `wiki`
* [Mean Absolute Percentage Error](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) `wiki`

### Ranking Metrics
* [Recall and Precision at $K$ for Recommender Systems](https://medium.com/@m_n_malaeb/recall-and-precision-at-k-for-recommender-systems-618483226c54) `blog`
* [Normalized Discounted Cumulative Gain](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG) `wiki`
* [Ranking Measures and Loss Functions in Learning to Rank](https://papers.nips.cc/paper/2009/file/2f55707d4193dc27118a0f19a1985716-Paper.pdf) `paper`

### Clustering Metrics
* [Silhouette Coefficient](https://en.wikipedia.org/wiki/Silhouette_(clustering)) `wiki`
* [Dunn Index](https://en.wikipedia.org/wiki/Dunn_index) `wiki`

### Information Criteria
#### Akaike Information Criterion
* Section 7.5, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

#### Bayes Information Criterion
* Section 7.7, [Elements of Statistical Learning, 2nd](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf) `book chapter`

# 11 Causal Inference


## Overview
* [A Survey on Causal Inference](https://arxiv.org/pdf/2002.02770.pdf) `paper`
* [Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction](https://www.amazon.com/Causal-Inference-Statistics-Biomedical-Sciences/dp/0521885884) `book`


## A/B Testing
* [Trustworthy Online Controlled Experiments](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/D97B26382EB0EB2DC2019A7A7B518F59) `book`


## Matching Methods and Re-weighting Methods
* [An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/) `paper`


## Difference in Differences
* [Difference-in-Differences Estimation](https://personal.utdallas.edu/~d.sul/Econo2/lect_10_diffindiffs.pdf) `lecture notes`


## Uplift Trees
* [Modeling Uplift Directly: Uplift Decision Tree with KL Divergence and Euclidean Distance as Splitting Criteria](https://tech.wayfair.com/data-science/2019/10/modeling-uplift-directly-uplift-decision-tree-with-kl-divergence-and-euclidean-distance-as-splitting-criteria/) `blog`
* [Real-World Uplift Modelling with Significance-Based Uplift Trees](https://stochasticsolutions.com/pdf/sig-based-up-trees.pdf) `paper`
* [Recursive Partitioning for Heterogeneous Causal Effects](https://www.pnas.org/content/113/27/7353) `paper`
* [Uplift Modeling with Multiple Treatments and General Response Types](https://arxiv.org/pdf/1705.08492.pdf) `paper`

## Meta Learners
* [Meta-Learners for Eestimating Heterogeneous Treatment Effects Using Machine Learning](https://arxiv.org/pdf/1706.03461.pdf) `paper`
* [Quasi-Oracle Estimation of Heterogeneous Treatment Effects](https://arxiv.org/pdf/1712.04912.pdf) `paper`


## Causal Inference in Industry
* [A Comparison of Approaches to Advertising Measurement: Evidence from Big Field Experiments at Facebook](https://www.kellogg.northwestern.edu/faculty/gordon_b/files/fb_comparison.pdf) `paper`
* [CausalML: Python Package for Causal Machine Learning](https://arxiv.org/pdf/2002.11631.pdf) `paper`
* [Evaluating Online Ad Campaigns in a Pipeline: Causal Models at Scale](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36552.pdf) `paper`
* [From Infrastructure to Culture: A/B Testing Challenges in Large Scale Social Networks](https://content.linkedin.com/content/dam/engineering/site-assets/pdfs/ABTestingSocialNetwork_share.pdf) `paper`
* [Improve User Retention with Causal Learning](http://proceedings.mlr.press/v104/du19a/du19a.pdf) `paper`
* [Inferring Causal Impact Using Bayesian Structural Time-Series Models](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/41854.pdf) `paper`
* [Measuring Ad Effectiveness Using Geo Experiments](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/38355.pdf) `paper`
* [Online Controlled Experiments at Large Scale](http://chbrown.github.io/kdd-2013-usb/kdd/p1168.pdf) `paper`
* [Online Experimentation at Microsoft](https://ai.stanford.edu/~ronnyk/ExPThinkWeek2009Public.pdf)
* [Overlapping Experiment Infrastructure: More, Better, Faster Experimentation](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36500.pdf) `paper`
* [The Online Display Ad Effectiveness Funnel & Carryover: Lessons from 432 Field Experiments](https://poseidon01.ssrn.com/delivery.php?ID=780025027093084078006098081104117031007048068055025069102114082123091069028115004024007063049014102035101119123093091021030066000033062052083122074127067004086065090005062075024076120065104092095069115119026114011001072031111124008007103123098066117098&EXT=pdf&INDEX=TRUE) `paper`
* [Twitter Experimentation: Technical Overview](https://blog.twitter.com/engineering/en_us/a/2015/twitter-experimentation-technical-overview.html) `blog`
* [Under the Hood of Uber’s Experimentation Platform](https://eng.uber.com/xp/) `blog`
* [Using Causal Inference to Improve the Uber User Experience](https://eng.uber.com/causal-inference-at-uber/) `blog`


# 12 Information Technology

## Recommender System
### Content-Based Filtering 
* [Content-based Filtering](https://developers.google.com/machine-learning/recommendation/content-based/basics) `other`

### Collaborative Filtering
#### Neighborhood Methods
* [Amazon.com recommendations: Item-to-item collaborative filtering](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11673444) `paper`
* [Collaborative Filtering](https://developers.google.com/machine-learning/recommendation/collaborative/basics) `other`

#### Matrix Factorization
* [Matrix Factorization for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) `paper`
* [Matrix Factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) `other`

### Factorization Machines
* [Factorization Machines](https://ieeexplore.ieee.org/document/5694074) `paper`
* [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) `paper`

### Neural Recommender Systems
* [A Large-Scale Deep Architecture for Personalized Grocery Basket Recommendations](https://arxiv.org/pdf/1910.12757.pdf) `paper`
* [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1905.06874.pdf) `paper`
* [Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1803.02349.pdf) `paper`
* [COLD: Towards the Next Generation of Pre-Ranking System](https://arxiv.org/pdf/2007.16122.pdf) `paper` 
* [Collaborative Multi-modal Deep Learning for the Personalized Product Retrieval in Facebook Marketplace](https://arxiv.org/pdf/1805.12312.pdf) `paper`
* [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/pdf/1708.05123.pdf) `paper`
* [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf) `paper`
* [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091.pdf) `paper`
* [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) `paper`
* [Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09248.pdf)
* [Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction](https://arxiv.org/pdf/2006.05639.pdf) `paper`
* [The Architectural Implications of Facebook's DNN-based Personalized Recommendation](https://arxiv.org/pdf/1906.03109.pdf) `paper`
* [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) `paper`

### More Industrial Examples 
* [Powered by AI: Instagram’s Explore Recommender System](https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system/) `blog`
* [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.fb.com/wp-content/uploads/2016/11/practical-lessons-from-predicting-clicks-on-ads-at-facebook.pdf) `paper`
* [Talent Search and Recommendation Systems at LinkedIn](https://arxiv.org/pdf/1809.06481.pdf) `paper`
* [The Netflix Recommender System: Algorithms, Business Value, and Innovation](https://dl.acm.org/doi/pdf/10.1145/2843948) `paper`
* [Two Decades of Recommender Systems at Amazon.com](https://assets.amazon.science/76/9e/7eac89c14a838746e91dde0a5e9f/two-decades-of-recommender-systems-at-amazon.pdf) `paper`
* [Using Deep Learning at Scale in Twitter’s Timelines](https://blog.twitter.com/engineering/en_us/topics/insights/2017/using-deep-learning-at-scale-in-twitters-timelines.html) `blog`


## Search Ranking
* [Amazon Search: The Joy of Ranking Products](https://assets.amazon.science/89/cd/34289f1f4d25b5857d776bdf04d5/amazon-search-the-joy-of-ranking-products.pdf) `paper`
* [Applying Deep Learning To Airbnb Search](https://arxiv.org/pdf/1810.09591.pdf) `paper`
* [Bringing Personalized Search to Etsy](https://codeascraft.com/2020/10/29/bringing-personalized-search-to-etsy/) `blog`
* [Embedding-based Retrieval in Facebook Search](https://arxiv.org/pdf/2006.11632.pdf) `paper`
* [Food Discovery with Uber Eats: Building a Query Understanding Engine](https://eng.uber.com/uber-eats-query-understanding/) `blog`
* [Improving Deep Learning for Airbnb Search](https://arxiv.org/pdf/2002.05515.pdf)) `paper`
* [In-session Personalization for Talent Search](https://arxiv.org/pdf/1809.06488.pdf) `paper`
* [Learning to Rank Personalized Search Results in Professional Networks](https://arxiv.org/pdf/1605.04624.pdf) `paper`
* [Machine Learning-Powered Search Ranking of Airbnb Experiences](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) `blog`
* [MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search](http://research.baidu.com/Public/uploads/5d12eca098d40.pdf) `paper`
* [Ranking Relevance in Yahoo Search](https://www.kdd.org/kdd2016/papers/files/adf0361-yinA.pdf) `paper`
* [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://dl.acm.org/doi/pdf/10.1145/3219819.3219885) `paper`
* [Reinforcement Learning to Rank in E-Commerce Search Engine]([https://arxiv.org/pdf/1803.00710.pdf) `paper`
* [Shop The Look: Building a Large Scale Visual Shopping System at Pinterest](https://dl.acm.org/doi/pdf/10.1145/3394486.3403372)`paper`
* [Talent Search and Recommendation Systems at LinkedIn: Practical Challenges and Lessons Learned](https://arxiv.org/pdf/1809.06481.pdf) `paper`
* [Towards Personalized and Semantic Retrieval for E-commerce Search via Embedding Learning](https://arxiv.org/pdf/2006.02282.pdf) `paper`


## Ads Bidding
* [Ads Allocation in Feed via Constrained Optimization](https://dl.acm.org/doi/pdf/10.1145/3394486.3403391) `paper`
* [Bid Optimizing and Inventory Scoring in Targeted Online Advertising](https://dl.acm.org/doi/abs/10.1145/2339530.2339655?casa_token=anTqe_x1G6QAAAAA:-wbGohnu46dO9P8Tc1VaBHrNPd0zCJo87Hnoq4kR81-DpOli_R_qEtcgGlbUw2VRgxOB-PPYIFyjWQ) `paper`
* [How We Built A Context-Specific Bidding System for Etsy Ads](https://codeascraft.com/2021/03/23/how-we-built-a-context-specific-bidding-system-for-etsy-ads/) `blog`


## Supply and Demand Forecasting
* [Bayesian Intermittent Demand Forecasting for Large Inventories](https://papers.nips.cc/paper/2016/file/03255088ed63354a54e0e5ed957e9008-Paper.pdf) `paper`
* [Forecasting at Uber: An Introduction](https://eng.uber.com/forecasting-introduction/) `blog`
* [Managing Supply and Demand Balance Through Machine Learning](https://doordash.engineering/2021/06/29/managing-supply-and-demand-balance-through-machine-learning/) `blog`
* [Probabilistic Demand Forecasting at Scale](http://www.vldb.org/pvldb/vol10/p1694-schelter.pdf)


## Marketplace Pricing and Matching
* [Dynamic Pricing and Matching in Ride-Hailing Platforms](https://poseidon01.ssrn.com/delivery.php?ID=713094097085071091029127000069078069000039039014031001095005103028030082091006095071118026012125037127020069115070101124006101023054032039051015086126101078099071026064046062001103094092006127098093097120100081000087068122092095097080102028127105082082&EXT=pdf&INDEX=TRUE) `paper`
* [Driver Surge Pricing](https://arxiv.org/pdf/1905.07544.pdf) `paper`
* [Dynamic Pricing and Matching in Ride-Sharing](https://www.naefrontiers.org/184199/Abstract) `other`
* [Predicting Real-Time Surge Pricing of Ride-Sourcing Companies](https://www.sciencedirect.com/science/article/abs/pii/S0968090X19301627) `paper`


## Customer Relation Management
* [Customer Acquisition via Display Advertising Using Multi-Armed Bandit Experiments](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1602&context=statistics_papers) `paper`
* [A Hidden Markov Model of Customer Relationship Dynamics](https://pubsonline.informs.org/doi/abs/10.1287/mksc.1070.0294) `paper`
* [Customer Purchase Intent Prediction Under Online Multi-Channel Promotion: A Feature-Combined Deep Learning Framework](https://ieeexplore.ieee.org/abstract/document/8795449) `paper`
* [Deep Learning for Understanding Consumer Histories](https://engineering.zalando.com/posts/2016/10/deep-learning-for-understanding-consumer-histories.html) `blog`
* [Large Scale Cross Category Analysis of Consumer Review Content on Sales Conversion Leveraging Deep Learning](https://journals.sagepub.com/doi/abs/10.1177/0022243719866690?journalCode=mrja) `paper`
* [Predicting Online Shopping Behaviour from Clickstream Data Using Deep Learning](https://www.sciencedirect.com/science/article/pii/S0957417420301676?casa_token=lU1gjhvQdPwAAAAA:CFOx2EPduvRpzem39yFhI039nUpDIxPgUvv-AtV1KlgazMO7QmPQBBiZ-H736Pjqupd8iPr9o9g) `paper`
* [Sentence-Based Text Analysis for Customer Reviews](https://pubsonline.informs.org/doi/abs/10.1287/mksc.2016.0993?journalCode=mksc) `paper`
* [Targeting Prospective Customers: Robustness of Machine-Learning Methods to Typical Data Challenges](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2019.3308?casa_token=6kX8O8O7Qj4AAAAA:q_pI9SZZjo1pCMddcNGhyIXjqsd7mhgEnf0eKQ1N_KJ55RykgJPxQC7wvAk5Pgp4T72YNAmKOFfa) `paper`
* [Uncovering the Path to Purchase Using Topic Models](https://journals.sagepub.com/doi/abs/10.1177/0022243720954376) `paper`


### Churn Prediction
* [A Comparison of Machine Learning Techniques for Customer Churn Prediction](https://www.acadpubl.eu/jsi/2018-119-10/articles/10b/2.pdf) `paper`
* [Customer Attrition Analysis for Financial Services Using Proportional Hazard Models](https://www.sciencedirect.com/science/article/pii/S0377221703000699?casa_token=cRIe3c9zSyMAAAAA:lSWvbBABJVJUYeMJk46W9URXfoYaWsc4uajNxQJM9pSGxT7Tek1TjDTFAIXcVvGZG43o82UnSWc) `paper`
* [Customer Churn Prediction in Telecom Using Machine Learning in Big Data Platform](https://link.springer.com/article/10.1186/s40537-019-0191-6) `paper`
* [Customer Churn Prediction Using Improved Balanced Random Forests](https://www.sciencedirect.com/science/article/pii/S0957417408004326?casa_token=RZ4Wtn8dyHcAAAAA:KXvbYOoPP8zGdQfgGPrtAoBlye3bbH0-9Dtkk9wqKYut2hivlGlb5Kg49_3VI5fTxpLCL9jbNB8) `paper`
* [Dynamic Churn Prediction Framework with More Effective Use of Rare Event Data: The Case of Private Banking](https://www.sciencedirect.com/science/article/pii/S0957417414003595?casa_token=46wsLx6iXtkAAAAA:fjXJWiW0FCbFk1DyXV6s7LuYmu1jjU3xHq47z2CMnQvIOfsOyOyse3aZ8uBmkjeqOI2hWrikCq4) `paper`
* [Why You Should Stop Predicting Customer Churn and Start Using Uplift Models](https://www.sciencedirect.com/science/article/pii/S0020025519312022) `paper`

### Lifetime Value
* [A Deep Probabilistic Model for Customer Lifetime Value Prediction](https://arxiv.org/pdf/1912.07753.pdf) `paper`
* [An Engagement-Based Customer Lifetime Value System for E-commerce](https://www.kdd.org/kdd2016/papers/files/adf0755-vanderveldAbr.pdf) `paper`
* [Customer Lifetime Value Prediction Using Embeddings](https://arxiv.org/pdf/1703.02596.pdf) `paper`


# 14 Optimization Methods for Machine Learning

## Constrained Optimization
* [Lagrange Multiplier](https://en.wikipedia.org/wiki/Lagrange_multiplier) `wiki`
* [Karush–Kuhn–Tucker Conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) `wiki`

## Curve Fitting
* [Non-Linear Least Squares](https://en.wikipedia.org/wiki/Non-linear_least_squares) `wiki`

## Gradient Descent
* [An Overview of Gradient Descent Optimization Algorithms](https://arxiv.org/pdf/1609.04747.pdf) `paper`

## Root Finding
* [Bisection Method](https://en.wikipedia.org/wiki/Bisection_method) `wiki`
* [Secant Method](https://en.wikipedia.org/wiki/Secant_method) `wiki`
* [Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method) `wiki`


# 15 Data Structures and Algorithms

## Asymptotics Analysis
* [Solving Recurrences](https://docs.google.com/presentation/d/1L0cq2b7yb_n1bwyI2_BWHugGfZABxIu2r9ZXvTo4zuA/edit#slide=id.g820f38cd3e_0_0) `lecture notes`
* [Know Thy Complexities!](https://www.bigocheatsheet.com/) `blog`

## Array and List
* [Difference between Linked List and Arrays](https://www.faceprep.in/data-structures/linked-list-vs-array/) `blog`

## Stacks and Queues
* [Stack](https://en.wikipedia.org/wiki/Stack_%28abstract_data_type%29) `wiki`
* [Queue](https://en.wikipedia.org/wiki/Queue_%28abstract_data_type%29) `wiki`

## Hash Tables
* [Hashing!](https://docs.google.com/presentation/d/1GugHPgUl282bwviOm-K9VHdxSVKlYvi0qXFXZSCcfQY/edit#slide=id.g8b5bddc967_0_114) `lecture notes`

## Heaps and Priority Queues
* [Heap](https://en.wikipedia.org/wiki/Heap_%28data_structure%29) `wiki`
* [Priority Queue](https://en.wikipedia.org/wiki/Priority_queue) `wiki`

## Trees
* [Binary Search Trees & Red-Black Trees!](https://docs.google.com/presentation/d/14760WScwlpPwWW_Wi-aEWV5njmiVjliLvWLrdSzmGzQ/edit#slide=id.g8b93bcfe26_0_124) `lecture notes`
* [Tree Traversal](https://en.wikipedia.org/wiki/Tree_traversal) `wiki`
* [Trie](https://en.wikipedia.org/wiki/Trie) `wiki`

## Graphs
* [Graphs, BFS & DFS](https://docs.google.com/presentation/d/1c5wf2xvOqOmXORO0lAf11JAPUyzVap7gOKkD8V3UTU8/edit#slide=id.g8d231503f1_0_118) `lecture notes`
* [Topological Sort](https://en.wikipedia.org/wiki/Topological_sorting) `wiki`

## Binary Search
* [Binary Search Algorithm](https://en.wikipedia.org/wiki/Binary_search_algorithm) `wiki`
* [二分查找、二分边界查找算法的模板代码总结](https://segmentfault.com/a/1190000016825704) `blog`

### Sorting
* [Merge Sort](https://en.wikipedia.org/wiki/Merge_sort) `wiki`
* [Quick Sort](https://en.wikipedia.org/wiki/Quicksort) `wiki`
* [Heap Sort](https://en.wikipedia.org/wiki/Heapsort) `wiki`
* [Counting Sort](https://en.wikipedia.org/wiki/Counting_sort) `wiki`
* [Bucket Sort](https://en.wikipedia.org/wiki/Bucket_sort) `wiki`
* Just for fun. The most elegant but useless sorting algorithm: [Stooge Sort](https://en.wikipedia.org/wiki/Stooge_sort) `wiki`

## Backtracking
* [In-Depth Backtracking with LeetCode Problems - Part 1](https://medium.com/algorithms-and-leetcode/backtracking-e001561b9f28) `blog`
* [In-Depth Backtracking with LeetCode Problems - Part 2](https://medium.com/algorithms-and-leetcode/backtracking-with-leetcode-problems-part-2-705c9cc70e52) `blog`
* [In-Depth Backtracking with LeetCode Problems - Part 3](https://medium.com/algorithms-and-leetcode/in-depth-backtracking-with-leetcode-problems-part-3-b225f19e0d51) `blog`

## Shortest Path
* [Dijkstra’s algorithm](https://docs.google.com/presentation/d/1WTk02PXjmyHVjpu9SvDywIp5hAejZbAt4jrUtA8V6AE/edit#slide=id.g8dcdfac922_0_118) `lecture notes`
* [Bellman-Ford & Floyd-Warshall](https://docs.google.com/presentation/d/1j5_MKfTAhDwkk_XoA6cnVPOmIOQGrUnPFHbvYcnWFSs/edit#slide=id.g8ccafe7f33_0_118) `lecture notes`

## Dynamic Programming
* [Dynamic Programming](https://people.eecs.berkeley.edu/~vazirani/algorithms/chap6.pdf) `book chapter`
* [More Dynamic Programming!](https://docs.google.com/presentation/d/1IbGRskTKARU6eZOvZ-jHq8S4tncMMc0ZZmx75kI9gT4/edit#slide=id.g8d231503f1_0_118) `lecture notes`
* [Even More Dynamic Programming!](https://docs.google.com/presentation/d/1c5wf2xvOqOmXORO0lAf11JAPUyzVap7gOKkD8V3UTU8/edit#slide=id.g8d231503f1_0_118) `lecture notes`

# 16 Python Programming

## Built-In Data Structures
* [Data Structures](https://docs.python.org/3/tutorial/datastructures.html#) `documentation`

## List Comprehensions
* [List Comprehensions](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions) `documentation`

## Generators
* [Python Generators](https://www.programiz.com/python-programming/generator) `blog`

## Passing by Object Reference
* [Is Python Pass-by-Reference or Pass-by-Value?](https://robertheaton.com/2014/02/09/pythons-pass-by-object-reference-as-explained-by-philip-k-dick/) `forum`

## Decorators
* [Primer on Python Decorators](https://realpython.com/primer-on-python-decorators/) 

## Object-Oriented Programming
* [Object-Oriented Programming (OOP) in Python 3](https://realpython.com/python3-object-oriented-programming/) `blog`

