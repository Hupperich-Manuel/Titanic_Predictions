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

To justify the usage of the LDA, a part from having a AUC of 97%, i will point out three main points:
* Having binary class, <img src="https://render.githubusercontent.com/render/math?math=Y (0, 1)"> , thwey are considered to be well separated

In our example we will


![equation]$a^2+b^2=c^2$

<img src="https://render.githubusercontent.com/render/math?math=Y (0, 1)">




