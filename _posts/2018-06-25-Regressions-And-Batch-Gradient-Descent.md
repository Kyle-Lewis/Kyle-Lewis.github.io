---
layout: post
title: "Regressions and Batch Gradient Descent"
categories: [Machine Learning]
date: 2018-06-25
---

The Algorithm
A least squares regression algorithm is a means to fit functions to datasets by way of minimizing the error incrementally. For a given input vector X you form a guess h(Xi) which will have some square error Ei = (h(Xi) - Yi)2

where i just denotes the index of the input vector you are looking at. You can get the total Error by summing over i (that is, summing over all points in your input, using your guess function).

We can then try to minimize the error. We know from basic calc that the minima of a function will be found at a place where its derivative is equal to zero, so set:



Which is described in slightly different notation in Andrew Ng's lecture, he defines a stand in J(theta) function and finds its derivative specifically for a linear case.

The algorithm is then as follows: chose theta values (which are weights for polynomial terms in your guess, for now at least) and iteratively replace those theta values by

subtracting along the direction of the gradient
and in proportion to the gradient 
Where the gradient is defined in the formula above for one dimension at least. Take enough steps and your function will reach a minimum in error. It doesn't just "work" however, as you introduce more terms it becomes easy to get stuck in local minima as I will discuss below.

Code
The algorithm is very simple to implement. Most of my time was spent figuring out (again) how to get matplotlib and pyplot to do what I wanted. There was also some pre-processing which needed to be done; the data sets I found to play with, some Chinese weather data provided by UC Irvine, had a few negatives / nulls which needed to be interpolated. Additionally, I found it quite helpful when tuning alpha values to normalize target data sets.

Here's the algorithm itself though, after that work has been done:

[code language="python"]
def batchGradientDescentStep(Weights, InputVectors, TargetVector, Alphas):for weightIdx in range(len(Weights)):
		sumOfDerivativeErrors = 0.0
		for idx in range(len(InputVectors[0])):

				# Subtract from the weight the gradient in the associated Input direction
				# whose derivative is equal to the sum [ Err (not squared) * X_i ] where
				# X_i is the InputVector associated with the weight.

			if len(Weights) == 2:
				sumOfDerivativeErrors = tools.firstOrderPolyErrors(weightIdx, idx, Weights, InputVectors, TargetVector)
			elif len(Weights) == 3:
				sumOfDerivativeErrors = tools.secondOrderPolyErrors(weightIdx, idx, Weights, InputVectors, TargetVector)

		Weights[weightIdx] -= Alphas[weightIdx] * sumOfDerivativeErrors

	return Weights
[/code]
Where the polynomial error functions are just the derivative error function I defined above for specific order polynomials, and I just count the number of weights provided to determine which to use. You can see the rest on my GitHub under the LSR folder.

Results
I did a linear fit to a selection of temperature data which had somewhat of a constant run up. Then I expanded the data set to include the winter months, and did a second order polynomial fit to that data.


Linear regression on the first ~half of a years worth of data
There were some things which I hadn't thought of while taking notes from the online lectures. For one, the regression rates for different weights need to be different, very different in fact. In general it seems to me that lower order terms should be 'learned' faster than higher order, so that the general location of the proper minima is found. With a quick linear rate for example, the above regression got stuck with a constant offset around zero, as the line quickly found a slope intersecting the data and locking the constant weight from moving, as doing so would increase error (even though there would be a better solution found by doing so)

After working on an MCMC application  using randomness and Markov chains with detail balance I know a more probabilistic approach could potentially make quick work of all this alpha parameter tuning, and get rid of the local minima problem entirely. I'll hold off for now however.


I realized that I normalized the temperature data after making the gif . . . you get the idea though
Resources
I'm cross referencing quite a few resources while I get into this subject, but the core resources are lectures provided online. For this particular post:

The first of Andrew Ng's lectures on Machine Learning, provided by Stanford:
Lecture 1: Introduction
Lecture 2: linear regression, gradient descent
The third of Yaser Abu-Mostafa's lectures, provided by CalTech:
Lecture 3: The Linear Model
Data from UC Irvine's machine learning repository
The code for these regressions on my GitHub in the LSR folder