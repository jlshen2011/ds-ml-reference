# ds-ml-reference

# 1 Probability Theory and Statistical Inference

## Probability Theory

### Probability Space and Measure
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