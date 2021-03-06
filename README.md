# Titanic Predictions
Predicted a binary outcome, either survived at the titanic or not, according to different features

The titanic is one of the most iconic and at the same time saddest stories in the history of human beings. There are barely any individuals who are not familiar with its story and how lucky some people were on that liner, because of certain characteristics that they took with them. Whether they were kids or had a higher purchasing power, there was a pattern to follow when predicting the probability of getting a safe boat, leaving unharmed the ship.

## 1. Explanatory Data Analysis

The cleaning of the data is by far the most challenging part in most of the machine learning projects since you can extremely improve (or harm) your model according to the individual features and the types of features you train your model with.

### 1.1. Feature Selection

For feature selection, we will go through three main aspects.

The first approach tried to identify the features that would add no value to our classification, as they are not significant. Those features were the ID of the Passenger, the Name of this, the Embarked location, and the Ticket number. You might argue that the Ticket number could be an indicator of the purchasing power of the Passenger, however, there were other variables such as Fare, that actually delivered the Price of the ordered Cabin. Lastly, after having trained our model, there was evidence that the Cabin feature did not at a good job on the model since there were a lot of Null values contained in it (even replacing the values with dummy variables, this did not improve the performance).

The second one focuses on the identification of multicollinearity in the data and eliminates the explanatory variables which are highly correlated as they will double their effect/importance in the model, and this does not show the reality of the data. Therefore we drop those columns.

The third approach was more focused on the normalization of the data. Since the data combined Categorical and Continuous Variables, there was a need to rescale the data so that there were no imbalances while training it.
To do so, a pipeline was coded, that aimed to pass the data through a Standard Scaling (setting the mean = 0 and standard deviation = 1). Furthermore, Principal Component Analysis (PCA) was done, since the chosen algorithm could easily overfit the data, through the prior dimensionality reduction, we can tackle this issue (there could have been other dimensionality reducers as the regularized LDA, which makes use of the shrinkage method, but it worked without this more complex approach).


## 2. Train the Model

The way the model was approached went through an initial study of the data, observing its behavior. Lately, as it could be seen some patterns indicated to classifiers such as the Linear Discriminant Analysis (LDA), Logistic Regression, or the Bayes Classifier. Therefore an ensemble modeling seemed to be a straightforward method to quickly identify the performance of multiple algorithms at a time and then eliminate the underperforming ones. Finally, the model that performed the best on the given data ended up being LDA, so a deeper study needed to be done on this specific algorithm.

## 2.1. Linear Discriminant Analysis

With the LDA we are going to model the conditional distribution of the independent variable Y (1,0), given the explanatory variables X.
The idea behind LDA is to use the logistic regression modeling Pr(Y = k|X = x), but through a less direct approach, which is basically to model the distribution of the features X given Y Pr(X = x|Y = k) and then use the Bayes theorem to flip it to estimate Pr(Y = k|X = x). In other words, instead of predicting what is the probability of surviving or not given certain attributes, the selected algorithm rephrases the estimation saying what is the probability that a person has a specific profile, given that she/he has survived or not, and then since some assumptions are fulfilled, we can flip this assumption and estimate the probability for the test sample. As said, in order to get positive takeaways from the model the data has to follow some assumptions:

* The classes are well separated.

* The observations follow a multivariate Gaussian distribution, as it assumes that each feature follows a one-dimensional normal distribution **X ??? N(??, ??)**, with **some** correlation between the pairs.


![Screenshot_2](https://user-images.githubusercontent.com/67901472/143683714-b9fecc0f-4ecf-4ed8-aed7-8eeb4f73a739.png)    ![jarque bera](https://user-images.githubusercontent.com/67901472/143683856-ab8c8e00-d412-4ca3-8932-5c60c80f4ed8.png)


_Source: Seaborn Library Python 3.0._                                                  
_Source: Stats Jarque-Bera Test Python 3.0._

![Screenshot_1](https://user-images.githubusercontent.com/67901472/143683719-5d7ca72d-5478-4c43-9765-1856218aeca2.png)

_Source: Matplotlib Pyplot Library Python 3.0._


LDA reduces the dimensionality  of the data (similar to PCA), with the main difference that it focuses on maximizing the separability between the two classes.
In the case of a dataset with more than one explanatory variable, we will handle the maximization through a mean vector and a covariance matrix, aiming not only to maximize the means but to minimize the scatter.


In order to calculate the probability that a certain observation belongs to the _kth_ class, we denote the density function as follows: fk(X) ??? Pr(X = x|Y = k). So, applying the Bayes theorem to this formula we end up getting: 

![image](https://user-images.githubusercontent.com/67901472/143683703-e174f414-ecba-4c3f-b0d5-e6e486cbdfa3.png) 

Where:
- ??k is the prior probability that an observation belongs to _kth_ class (which is simple in this case as we compute the fraction of the training observations that belong to the _kth_ class).
- when it comes to the estimation of fk(X) we will have to see if we can approximate the Bayes classifier (for a given observation, which class maximizes pk(X), while at the same time reducing the error rate).
  * The density function fk(X), should be large if there is a high probability that the observation _X_ belongs to the _kth_ class, and vice versa.

Given that we have more than one feature we will denote the multivariate Gaussian density function as:

![image](https://user-images.githubusercontent.com/67901472/143763437-19bb1c64-05c6-4059-9b88-be887929e8bd.png)

Where:
- ?? is a vector of means of the variables
- ?? is the covariance matrix of the variables

Now we only have to replace this formula in the Bayes formula, which I previously mentioned, and see what is the maximum value of ??k(x) = xT ?????1??k ??? 1/2??Tk ?????1??k + log ??k (LDA depends on the linear combination of the elements of x) given an observation classified as a _kth_ class.

The decision boundaries are the only points where the probability to belong to one class is equal to one belonging to the other. In our case, we will have one decision boundary, as we only have 2 classes.

When it comes to the evaluation we can see how the error rate was minimized:

![Confusion Matrix](https://user-images.githubusercontent.com/67901472/143763875-63c8829d-7abd-4e46-a2ec-cd2812572c83.png)

_Source: Matplotlib Pyplot Library Python 3.0._

Looking at the sensitivity (True positive rate) and the specificity (false positive rate), which characterize the performance of our LDA, compared to other classifiers, we observe that it is almost perfect. The AUC is about 97%, which shows that there is a large increase in true positives with a little to no change in false positives, the model fits almost perfectly.

![image](https://user-images.githubusercontent.com/67901472/143763993-248f548d-6144-4859-9aeb-d7fe45c8457a.png)

_Source: Matplotlib Pyplot Library Python 3.0._

## 3. Code the LDA

Now we are entering into the interesting part of this study, the code.
We covered the whole code in a function that returns a dataframe with the predictions and the Passenger_ID. In this section, we will break up the code into its components explaining so each step.

The first part was the import of the different libraries and the datasets:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

df_train = pd.read_csv("train.csv", index_col=0)
df_test = pd.read_csv("test.csv", index_col=0)
gender_sub = pd.read_csv("gender_submission.csv", index_col=0)
```

Then as said previously, there was a need to drop the unnecessary columns and so weigh only the one who truly would add value to our model:

```python
#Feature Selection
df_test = pd.merge(df_test, gender_sub, left_index=True, right_index=True) #merges the test set to see if there are missing values
df_test['Sex'] = df_test['Sex'].replace("male", 0).replace("female", 1) #binary values for the 'Sex' feature
df_train['Sex'] = df_train['Sex'].replace("male", 0).replace("female", 1)
df_train.drop(['Name', 'Embarked', 'Ticket', 'Cabin'], 1, inplace=True) #dropping the irrelevant variables
df_test.drop(['Name', 'Embarked', 'Ticket','Cabin'], 1, inplace=True)
```

Since the data could have some Null values in its columns, there was a need to identify if the frequency of the Null Values was significant (basically to make a decision whether to eliminate the rows or not). Moreover, since 'Pclass' had three categorical variables, which referred to the ticket class the Passenger traveled with, dummy variables were created for simplification.


```python
#Was for training set and the test set
for i in df_test.columns:
    print(i, sum(df_test[i].isnull())/df_test.shape[0])
    
age = []

for i in df_train['Age']:
    if str(i) == 'nan':
        age.append(df_train['Age'].mean())
    else:
        age.append(i)

df_train['Age'] = age

df_train = pd.get_dummies(df_train, columns=['Pclass'])
df_test = pd.get_dummies(df_test, columns=['Pclass'])

age = []

for i in df_test['Age']:
    if str(i) == 'nan':
        age.append(df_test['Age'].mean())
    else:
        age.append(i)

df_test['Age'] = age

df_test.dropna(inplace=True)
```

Finally, the pandas DataFrame was converted into arrays to be trained on the sklearn libraries

```python
 train_X, train_y = np.array(df_train[df_train.columns[1:]]),  np.array(df_train[df_train.columns[0]])
 test_X, test_Y =  np.array(df_test[df_test.columns[:-1]]), np.array(df_test[df_test.columns[-1:]])

 train_Y = np.expand_dims(train_y, axis=0) #this converts the shape from (15,) into (15,1), which need to be done, otherwise the sklearn libraries will return an error

 train_Y = train_Y.T
```

Now the data was ready to be trained. To do a Pipeline was coded that Standardized, Reduced, and Trained the Data with the Linear Discriminant Analysis algorithm (with its standard parameters), through cross-validation of 5 and the return of the training score. After that, we fitted the training X features with the training Y ('Survived') and evaluated the predictions with the score.

```python
pipe = Pipeline([('scaler', StandardScaler()),('pca',PCA()),('model',LinearDiscriminantAnalysis())])

params = {'model__solver':['svd']}

search = GridSearchCV(pipe, param_grid=params, cv = 5, return_train_score=True)

search.fit(train_X, train_Y)
y_hat = search.predict(test_X)

score = search.score(test_X, test_Y)
#print("LDA Score: %.02f"%(score))
```
The score was 1.0 which is the highest possible value you can get, getting us to the top 170 performers of the Kaggle competition.

## 4. Conclusion

To conclude I would like to thank my partner [Igancio Gonzalez Granero](https://www.linkedin.com/in/ignacio-gonzalez-granero/), for joining this competition with me. To continuously seek new challenges that get us out of our comfort zone, and make us improve in every aspect of our lives are some of the values that define us. Through Data Science competitions, we are enabled to gain a lot of knowledge, which could not be more useful, when achieving the goals that each of us seeks to accomplish.

Have a nice day!

## 5. References

* [Stack Overflow](https://stackoverflow.com/)
* [Introduction to Statistical Learning](https://www.statlearning.com/)
* [Kaggle](https://www.kaggle.com/c/titanic/data?select=test.csv)





