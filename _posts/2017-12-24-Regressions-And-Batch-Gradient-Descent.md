---
layout: post
title: "Regressions and Batch Gradient Descent"
categories: [Machine Learning]
date: 2017-12-24
image: images/gradient-descent/secondorderpoly.gif
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  CommonHTML: { scale: 150 },
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<h2 align="center">References</h2>

I'm cross referencing quite a few resources while I get into this subject, but the core resources are lectures provided online. For this particular post:

- The first of Andrew Ng's lectures on Machine Learning, provided by Stanford:

	- Lecture 1: <a href="https://www.youtube.com/watch?v=UzxYlbK2c7E&list=PLA89DCFA6ADACE599&index=1" target="_blank">Introduction</a>

	- Lecture 2: <a href="https://www.youtube.com/watch?v=5u4G23_OohI&list=PLA89DCFA6ADACE599&index=2" target="_blank">linear regression, gradient descent</a>

- The third of Yaser Abu-Mostafa's lectures, provided by CalTech:

	- Lecture 3: <a href="https://www.youtube.com/watch?v=FIbVs5GbBlQ&index=3&list=PLD63A284B7615313A" target="_blank">The Linear Model</a>

- Data from <a href="http://archive.ics.uci.edu/ml/index.php" target="_blank">UC Irvine's machine learning repository</a>

- The code for these regressions <a href="https://github.com/Hobbes1/MachineLearningProjects/tree/464d9dd47d9d2ab7ba62cfc863c1eb4c0d99b360/1_LSR" target="_blank">on my GitHub in the LSR folder</a>

<h2 align="center">Motivation</h2>

There are plenty of reasons these days to explore Machine Learning. For me, in reviewing the math behind MCMC methods in a previous project I'd been coming across a lot of people discussing the topic not for Computational Physics applications, but for Machine Learning. A quick look revealed that many of the more advanced topics in Machine Learning share a mathematical foundation with QM (high dimensionality and linear algebra) and at that point I was sold. So in some of the project posts I make from here on out, i'll be going through Andrew Ng and other freely available lecture resources and picking out examples to implement; hopefully getting better with CUDA as I go.

<h2 align="center">The Algorithm</h2>

A least squares regression algorithm is a means to fit functions to datasets by way of minimizing the error incrementally. For a given input vector $\vec{X}$ you form a guess $h(\vec{X}_i)$ which will have some square error $E_i = (h(\vec{X}_i) - Y_i)^2$

where $i$ just denotes the index of the input vector you are looking at. You can get the total Error by summing over $i$ (that is, summing over all points in your input, using your guess function).

We can then try to minimize the error. We know from basic calc that the minima of a function will be found at a place where its derivative is equal to zero, so set:

<div style="font-size: 150%;">
	$$ \frac{\partial E(h_{theta}(\vec{x}), \vec{y})}{\partial\theta} = 2\sum_{i=0}^n(h_{theta}(\vec{x_i}) - \vec{y_i}) \cdot \frac{\partial h_{theta})(\vec{x})}{\partial\theta} $$
</div>

Which is described in slightly different notation in Andrew Ng's lecture, he defines a stand in J(theta) function and finds its derivative specifically for a linear case.

The algorithm is then as follows: chose theta values (which are weights for polynomial terms in your guess, for now at least) and iteratively replace those theta values by

1) subtracting along the direction of the gradient
2) and in proportion to the gradient 

Where the gradient is defined in the formula above for one dimension at least. Take enough steps and your function will reach a minimum in error. It doesn't just "work" however, as you introduce more terms it becomes easy to get stuck in local minima as I will discuss below.

<h2 align="center">Code</h2>

The algorithm is very simple to implement. Most of my time was spent figuring out (again) how to get matplotlib and pyplot to do what I wanted. There was also some pre-processing which needed to be done; the data sets I found to play with, some Chinese weather data provided by UC Irvine, had a few negatives / nulls which needed to be interpolated. Additionally, I found it quite helpful when tuning alpha values to normalize target data sets.

Here's the algorithm itself though, after that work has been done:

<hr>
<div style="width:110%">

{% highlight python %}

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

{% endhighlight %}

</div>
<hr>

Where the polynomial error functions are just the derivative error function I defined above for specific order polynomials, and I just count the number of weights provided to determine which to use. You can see the rest on my GitHub under the LSR folder.

<h2 align="center">Results</h2>

I did a linear fit to a selection of temperature data which had somewhat of a constant run up. Then I expanded the data set to include the winter months, and did a second order polynomial fit to that data.

<figure>
	<img src="{{site.baseurl}}/images/gradient-descent/newlsr.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">Linear regression on the first ~half of a years worth of data</figcaption>
</figure>

There were some things which I hadn't thought of while taking notes from the online lectures. For one, the regression rates for different weights need to be different, very different in fact. In general it seems to me that lower order terms should be 'learned' faster than higher order, so that the general location of the proper minima is found. With a quick linear rate for example, the above regression got stuck with a constant offset around zero, as the line quickly found a slope intersecting the data and locking the constant weight from moving, as doing so would increase error (even though there would be a better solution found by doing so)

After working on an MCMC application  using randomness and Markov chains with detail balance I know a more probabilistic approach could potentially make quick work of all this alpha parameter tuning, and get rid of the local minima problem entirely. I'll hold off for now however.

<figure>
	<img src="{{site.baseurl}}/images/gradient-descent/secondorderpoly.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">I realized that I normalized the temperature data after making the gif . . . you get the idea though</figcaption>
</figure>

