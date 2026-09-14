# TRAINING MODELS

What is this chapter really about?

Chapter 4 answers a very important question:

`How does a machine-learning model actually learn its parameters from data?`

In previous chapters, you learned things like:

- what machine learning is,
- regression,
- classification,
- training/test sets,
- performance measures,
- overfitting and underfitting.

Now Géron goes deeper.

Instead of simply saying:

"We train a model."

this chapter asks:

`What happens mathematically when we train the model?`


# Linear Regression

Linear Regression is a supervised learning algorithm used to predict a numerical value.

It assumes that the relationship between the input features and the target value can be represented approximately by a straight line.

## Linear Regression Model

The general Linear Regression model is:

**ŷ = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ**

Where:

* ŷ = predicted value
* θ₀ = bias/intercept
* θ₁, θ₂, ..., θₙ = model parameters
* x₁, x₂, ..., xₙ = input features

The model can also be written as:

**ŷ = θᵀx**

The objective of training is to find the values of the parameters that produce predictions as close as possible to the actual values.

## Cost Function

A common cost function for Linear Regression is Mean Squared Error (MSE).

**MSE = (1/m) × Σ(ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²**

Where:

* m = number of training instances
* ŷ⁽ⁱ⁾ = prediction for instance i
* y⁽ⁱ⁾ = actual target value

The model is trained by finding the parameters that minimize the MSE.

## Main Idea

Linear Regression tries to find the line or hyperplane that best fits the training data.

The training process can therefore be summarized as:

**Data → Model → Predictions → Cost → Optimization → Best Parameters**

---


# The Normal Equation

The Normal Equation is a mathematical method for finding the optimal parameters of a Linear Regression model directly.

Instead of repeatedly improving the parameters using an optimization algorithm, the Normal Equation calculates the parameters in one mathematical operation.

## Formula

**θ = (XᵀX)⁻¹Xᵀy**

Where:

* θ = vector containing the model parameters
* X = matrix containing the training instances
* y = vector containing the target values
* Xᵀ = transpose of X
* (XᵀX)⁻¹ = inverse of XᵀX

## What It Does

The Normal Equation finds the values of θ that minimize the Mean Squared Error.

This means it finds the Linear Regression model that gives the best fit according to the MSE cost function.




The Normal Equation is a closed-form solution for Linear Regression.

**Normal Equation → Calculate optimal parameters directly**

---



# Computational Complexity

Computational complexity describes how the amount of computation required by an algorithm increases as the size of the problem increases.

This is important because an algorithm that works well on a small dataset may become very slow when the dataset becomes large.

## Normal Equation Complexity

The Normal Equation requires calculating:

**XᵀX**

and then finding its inverse.

If there are n features, the computational cost of matrix inversion grows approximately with:

**O(n³)**

This means that increasing the number of features can make the computation significantly more expensive.

## Gradient Descent

Gradient Descent does not require calculating a matrix inverse.

Its computational cost depends mainly on:

* Number of training instances
* Number of features
* Number of iterations

Gradient Descent can therefore be more practical when the number of features is very large.



The Normal Equation can be very convenient for smaller problems.

For large datasets or very high-dimensional problems, iterative optimization methods are generally more scalable.

---

# Gradient Descent

Gradient Descent is an optimization algorithm used to find the parameters of a model that minimize a cost function.

The basic idea is to start with some parameter values and gradually change them in the direction that reduces the cost.

---

Imagine standing on a mountain and wanting to reach the lowest point.

You look at the slope around you and move downhill.

Gradient Descent does something similar:

**Calculate slope → Move downhill → Calculate slope again → Repeat**

The goal is to reach a point where the cost function is as small as possible.

## Gradient

The gradient tells us the direction in which the cost function increases most rapidly.

Therefore, to reduce the cost, we move in the opposite direction of the gradient.

## Gradient Descent Update

The general update rule is:

**θ(next step) = θ − η × ∇θ MSE(θ)**

Where:

* θ = current model parameters
* η = learning rate
* ∇θ MSE(θ) = gradient of the cost function

## Learning Rate

The learning rate controls how large each step is.

### Learning rate too small

The algorithm takes very small steps.

Training may become extremely slow.

### Learning rate too large

The algorithm may jump over the minimum.

It may fail to converge and could even move farther away from the minimum.

### Appropriate learning rate

The algorithm gradually approaches a good minimum.

---

Gradient Descent is essentially:

**Find the direction of the slope and move in the opposite direction.**

---


# Batch Gradient Descent

Batch Gradient Descent calculates the gradient using the **entire training dataset** before making each parameter update.

## Process

For every iteration:

1. Calculate predictions for all training instances.
2. Calculate the gradient using all training instances.
3. Update the parameters.
4. Repeat.



## Summary

**Batch Gradient Descent = entire training set → one update**

It is suitable when the dataset is reasonably small and can fit comfortably into memory.

---



# Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) calculates the gradient using **one randomly selected training instance at a time**.

Instead of waiting to process the entire dataset, the model updates its parameters after each individual instance.

## Process

For each training instance:

1. Select one instance.
2. Calculate the prediction.
3. Calculate the gradient.
4. Update the parameters.
5. Continue with another instance.


## Learning Rate

SGD often uses a learning-rate schedule.

The learning rate starts relatively high and gradually decreases.

This allows the algorithm to move quickly at the beginning and make smaller adjustments later.

---

**Stochastic Gradient Descent = one training instance → one update**

---


# Mini-Batch Gradient Descent

Mini-Batch Gradient Descent is a compromise between Batch Gradient Descent and Stochastic Gradient Descent.

Instead of using:

* the entire dataset, or
* only one instance,

it uses a small group of training instances called a **mini-batch**.

## Process

For every update:

1. Select a small batch of training instances.
2. Calculate predictions.
3. Calculate the gradient.
4. Update the parameters.
5. Repeat.


## Comparison

| Method        | Data used per update     |
| ------------- | ------------------------ |
| Batch GD      | Entire training set      |
| Stochastic GD | One instance             |
| Mini-Batch GD | Small group of instances |



# Polynomial Regression

Polynomial Regression is a form of Linear Regression that can be used to model nonlinear relationships.

The idea is to add powers of the original features to the training data.

For example, instead of using only:

**x**

we can also use:

**x², x³, x⁴, ...**

## Example

A polynomial model could be:

**ŷ = θ₀ + θ₁x + θ₂x²**

Although the relationship between x and ŷ is nonlinear, the model is still linear with respect to its parameters θ₀, θ₁, and θ₂.

Therefore, Polynomial Regression can still be trained using Linear Regression techniques.

## Why Use Polynomial Regression?

A straight line may not be able to represent a dataset correctly.

Adding polynomial features allows the model to create curves.

## Risk of Overfitting

Using a polynomial degree that is too high can make the model extremely flexible.

The model may fit the training data very closely, including noise.

This can lead to overfitting.

## Main Idea

**Linear Regression + polynomial features = Polynomial Regression**

The polynomial degree must be chosen carefully.

---



# Learning Curves

Learning curves are graphs that show the performance of a model as the amount of training data increases.

They are useful for diagnosing:

* Underfitting
* Overfitting
* Whether adding more training data may help

## Training Error

Training error is measured using the training data.

As the training set becomes larger, the training error may increase because it becomes harder for the model to fit every example perfectly.

## Validation Error

Validation error is measured using data that was not used to train the model.

It helps estimate how well the model generalizes to unseen data.

## Underfitting

Typical signs:

* Training error is high.
* Validation error is also high.
* The two curves are relatively close.

This means the model is too simple or has insufficient flexibility.

Possible solutions include:

* Using a more complex model.
* Adding useful features.
* Reducing excessive regularization.

## Overfitting

Typical signs:

* Training error is very low.
* Validation error is significantly higher.

This means the model performs very well on training data but poorly on unseen data.

Possible solutions include:

* Getting more training data.
* Using a simpler model.
* Applying regularization.
* Reducing model complexity.

## Main Idea

Learning curves help us understand whether the model is:

**Underfitting → Overfitting → Generalizing well**

---


# Regularized Linear Models

Regularization is a technique used to reduce overfitting by restricting the complexity of a model.

The basic idea is to add a penalty to the cost function when the model parameters become too large.

## Why Regularization?

A model with very large parameter values can become overly sensitive to the training data.

Regularization encourages smaller parameter values.

This usually produces a simpler model that generalizes better.

## Important Rule

Regularization should generally be applied to the model's weights, but not to the bias/intercept.

## Main Regularization Methods

The chapter introduces:

1. Ridge Regression
2. Lasso Regression
3. Elastic Net
4. Early Stopping

## Feature Scaling

Feature scaling is important when using regularized models.

If features have very different scales, the regularization penalty can affect them unfairly.

Common scaling methods include:

* Standardization
* Min-Max scaling

## Main Idea

**Regularization = constrain model complexity to reduce overfitting**

---



# Ridge Regression

Ridge Regression is a regularized version of Linear Regression.

It adds an L2 penalty to the cost function.

## Ridge Cost Function

The Ridge cost function is:

$$
J(\theta) =
MSE(\theta)
+
\alpha \sum_{i=1}^{n}\theta_i^2
$$

The summation applies to the model weights, not the bias term.

## Alpha

α is the regularization strength.

### α = 0

Ridge Regression becomes ordinary Linear Regression.

### Small α

The model is only slightly regularized.

### Large α

The model is strongly regularized and the weights are pushed closer to zero.

## Effect of Ridge Regression

Ridge does not usually make coefficients exactly zero.

Instead, it gradually reduces their size.

This helps prevent individual features from having excessively large influence.


## Important Point

Feature scaling is important before applying Ridge Regression.

---



# Lasso Regression

Lasso stands for **Least Absolute Shrinkage and Selection Operator**.

It is another regularized version of Linear Regression.

Lasso uses an L1 penalty.

## Lasso Cost Function

$$
J(\theta) =
MSE(\theta)
+
\alpha \sum_{i=1}^{n}|\theta_i|
$$

The penalty is based on the absolute values of the model weights.

## Effect of Lasso

One important characteristic of Lasso is that it can reduce some feature weights exactly to zero.

When a coefficient becomes zero, that feature effectively disappears from the model.

Therefore, Lasso can perform **feature selection**.

## Alpha

The parameter α controls the strength of regularization.

### Small α

Weak regularization.

### Large α

Strong regularization and more coefficients may become zero.

## Ridge vs Lasso

| Ridge                      | Lasso                         |
| -------------------------- | ----------------------------- |
| Uses L2 penalty            | Uses L1 penalty               |
| Shrinks weights            | Can make weights exactly zero |
| Usually keeps all features | Can perform feature selection |

---

**Lasso = regularization + automatic feature selection**

---


# Elastic Net

Elastic Net combines the characteristics of Ridge Regression and Lasso Regression.

It uses both L1 and L2 regularization.

## Cost Function

$$
J(\theta) =
MSE(\theta)
+
r\alpha \sum_{i=1}^{n}|\theta_i|
+
\frac{1-r}{2}\alpha
\sum_{i=1}^{n}\theta_i^2
$$

Where:

* α = overall regularization strength
* r = balance between Lasso and Ridge
* r = 0 gives Ridge Regression
* r = 1 gives Lasso Regression

## Why Use Elastic Net?

Elastic Net provides a balance between:

* Ridge's ability to shrink coefficients.
* Lasso's ability to eliminate some coefficients.

It can be useful when there are several correlated features.

## Choosing Between Methods

The chapter recommends generally avoiding plain Linear Regression when the model is clearly overfitting.

Regularized models are often preferred.

Elastic Net can be a good compromise when you are unsure whether Ridge or Lasso is more appropriate.

## Main Idea

**Elastic Net = L1 regularization + L2 regularization**

---



# Early Stopping

Early Stopping is a regularization technique that stops training before the model begins to overfit.

It is particularly useful with iterative learning algorithms such as Gradient Descent.

## How It Works

During training, monitor the model's performance on a validation set.

Initially:

* Training error decreases.
* Validation error also decreases.

Eventually, the model may start overfitting.

At that point:

* Training error continues decreasing.
* Validation error begins increasing.

Early Stopping stops the training process around the point where validation performance is best.

## Example

Imagine validation error behaves like this:

**High → decreasing → minimum → increasing**

The best model is generally around the minimum validation error.

## Why It Works

The model is prevented from continuing to learn the noise and specific details of the training data.

---

**Train → monitor validation error → stop before overfitting**

Early Stopping is simple but very effective.



# Logistic Regression — Estimating Probabilities

Logistic Regression is a classification algorithm.



It can estimate the probability that an instance belongs to a particular class.

## Logistic Regression Model

The model first calculates a score:

**t = θᵀx**

This score is then passed through the logistic function, also called the sigmoid function.

## Sigmoid Function

$$
\hat{p} = \sigma(x^T\theta)
$$

$$
\sigma(t) = \frac{1}{1 + \exp(-t)}
$$

The sigmoid function converts any real-valued number into a value between 0 and 1.

Therefore:

**0 ≤ σ(t) ≤ 1**

This output can be interpreted as a probability.

## Interpretation

If:

**σ(t) ≈ 0**

the model is predicting a low probability of the positive class.

If:

**σ(t) ≈ 1**

the model is predicting a high probability of the positive class.

If:

**σ(t) = 0.5**

the model is exactly at the classification threshold.


---


# Logistic Regression — Training and Cost Function

Logistic Regression needs a cost function that encourages accurate probability predictions.

The cost for one training instance is called **log loss**.

## Log Loss

For one training instance:

$$
J(\theta) =
-\left[
y\log(\hat{p}) +
(1-y)\log(1-\hat{p})
\right]
$$

Where:

* y = actual class
* p̂ = predicted probability

## If y = 1

The cost becomes:

**−log(p̂)**

A prediction close to 1 produces a small cost.

A prediction close to 0 produces a very large cost.

## If y = 0

The cost becomes:

**−log(1 − p̂)**

A prediction close to 0 produces a small cost.

A prediction close to 1 produces a very large cost.

## Why Log Loss?

Log loss strongly penalizes confident incorrect predictions.

For example, predicting a probability of 0.99 for the wrong class produces a very large penalty.

This encourages the model to produce probabilities that are both accurate and appropriately confident.



# Logistic Regression — Decision Boundaries

A decision boundary separates different classes.

Logistic Regression calculates a probability and then uses a threshold to determine the predicted class.

## Classification Threshold

A common threshold is:

**0.5**

If:

**p ≥ 0.5**

predict the positive class.

If:

**p < 0.5**

predict the negative class.

Because:

**σ(0) = 0.5**

the decision boundary occurs when:

**θᵀx = 0**

## Example

Suppose a model predicts:

**p = 0.80**

The model classifies the instance as the positive class.

If:

**p = 0.20**

the model classifies it as the negative class.

## Decision Boundary

The decision boundary is the set of points where the model is equally likely to predict either class.

For a simple two-feature Logistic Regression model, the boundary can be a straight line.

For higher-dimensional data, it becomes a hyperplane.



# Softmax Regression

Softmax Regression, also called Multinomial Logistic Regression, is used for multiclass classification.

Unlike binary Logistic Regression, which predicts between two classes, Softmax Regression can classify an instance into one of several mutually exclusive classes.

## Important Condition

The classes must be mutually exclusive.

An instance should belong to one class rather than several classes simultaneously.

## Class Scores

The model calculates a score for every class.

For class k:

$$
s_k(x) = x^T\theta^{(k)}
$$

The scores are converted into probabilities using the Softmax function.

## Softmax Probability

$$
\hat{p}_k =
\sigma(s(x))_k =
\frac{\exp(s_k(x))}
{\sum_{j=1}^{K}\exp(s_j(x))}
$$

The probabilities of all classes add up to 1.

Therefore:

**p₁ + p₂ + ... + pₖ = 1**


## Prediction

The predicted class is the class with the highest estimated probability:

$$
\hat{y} = \arg\max_k \sigma(s(x))_k
$$

Since Softmax preserves the order of the scores, this is equivalent to:

$$
\hat{y} = \arg\max_k s_k(x)
$$

## Cross-Entropy Cost

Softmax Regression is trained using a cross-entropy cost function.

The objective is to make the probability of the correct class as high as possible.

## Example

Suppose a model predicts:

| Class   | Probability |
| ------- | ----------: |
| Class A |        0.10 |
| Class B |        0.70 |
| Class C |        0.20 |

The prediction is:

**Class B**

because it has the highest probability.

