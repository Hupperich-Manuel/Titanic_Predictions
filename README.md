# Titanic_Predictions
Predicted a binary outcome, either survived at the titanic or not, according to different features

The titanic is one of the most iconic and at the same time sad stories in the history of human beings. There are barely individuals who are not familiar with its story and how lucky some people where on that liner, because of certain charcteristics that they took with them. Wether they where kids or had a higher purchasing power, there was a pattern to follow when predicting the probability of getting a save boat, leaving unharmed the ship.

## 1. Explanatory Data Analysis

The cleaning of the data, is by far the most challenging part in most of the machine learning projects, since you can extremely improve (or harm) your model according to the individual features and the types of features you train your model with.

### 1.1. Feature Selection

For feature selection, we will go through two main aspects.

The first one will be focus on the identification of multicollinearity in the data and eliminate the explainatory variables which are highly correlated as they will double their effect/importance in the model, and this does not show the reality of the data. Therefore we drop that columns.


#### 1.1.1 Multicollinearity


#### 1.1.2 Domain Knowledge


## 2. Train the Model

Firstly, there was a need to see how ensemble modelling perfromed on the data, and see if we could identfy the outperfromers beyond the models. The way this was appoached was through a AUC analysis, through which it was easy to identify the realtionship between sensitity


## 2.1. Linear Discriminant Analysis

With the LDA we are going to model the conditional distribution of the independent variable Y (1,0), given the explainatory variables.

LDA reduces the dimensionallity of the data (similar to PCA), with the main difference that it focuses on maximizing the separability between the two classes.
In the case of a dataset with more than one explanatory variable, we will handle the maximization through a mean vextor and a covariance matrix, aiming to minimize the scatter.

To properly perform the LDA analysis, and getting so better results than with other classification algorithms, the data has to fullfil some assumptions:

* The classes need to be well separated, which can be difficult to observe in a multidimensional cunundrum.
* The observations have to follow a mulivariate Gaussian distribution, as it assumes that each feature follows a one-dimensional normal distribution **X ∼ N(μ, Σ)**, with **some** correlation between the pairs (but not so that it could harm the model). Therefore, there was a need to prove this assumption with the Jarque-Bera Normality Test.
[Jarque Bera]
[Heatmap and distribution picture]

in order to calculate the probability that a certain observation belongs to the _kth_ class, we denote the density function as follows: fk(X) ≡ Pr(X = x|Y = k). So, applying the Bayes theorem to this formula we end up getting 

![image](https://user-images.githubusercontent.com/67901472/143683703-e174f414-ecba-4c3f-b0d5-e6e486cbdfa3.png)



To justify the usage of the LDA, a part from having a AUC of 97%, there are three reasons to point put:
* Having binary class, <img src="https://render.githubusercontent.com/render/math?math=Y (0, 1)"> , they are considered to be well separated
* Since the number of observations is relatively small and the distribution of features is approximatelly normal, the choose of LDA returns better results that with other binary classification models. Every feature is significant in the Jarque-Bera test and the distribution of the observation is randomly distributed between 0 and 1 (after dimensionality reduction).
I,mages, jarque-bera and dispersion



In our example we will




