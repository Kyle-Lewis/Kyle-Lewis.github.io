---
layout: post
title: "Logistic Regression"
categories: [Machine Learning]
date: 2018-03-05
image: images/logistic-regression/regression.gif
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

- Andrew ng lecture 3

- Github

<h2 align="center">Motivation</h2>

Logistic Regression is the first classification algorithm i've come across. It seems somewhat important so I figured it deserved an implementation such that I really "got it". It also turns out to be a special case of Softmax Regression, a more general classifier which I also intend on implementing. As I think will be typical of these projects, the post and code should be quite short, there's nothing fancy going on. 

<h2 align="center">The Algorithm</h2>

Where least squares regression attempts to find a line that fits data, logistic regression attempts to find a line that divides two classes of data. In fact it doesn't do this with a *line* at all, but by fitting a logistic function (now our hypothesis function, $h$) to the data:

<div style="font-size: 150%;">
	$$ h_{\theta}(x) = \frac{1}{1 + e^{-\theta^Tx}}$$
</div>

Where the vector $\theta$ contains a weight for every feature (basis vector) of the data. What this looks like as you move up in dimensionality is a division between two regions where the function is high (approaching 1) and low (approaching 0) with a smooth transition between the states. You would interpret the value of the function as the probability of the point being of that class.

<figure>
	<img src="{{site.baseurl}}/images/logistic-regression/sigmoid.png" style="padding-bottom:0.5em; width:50%; margin-left:auto; margin-right:auto; display:block;" />
	<img src="{{site.baseurl}}/images/logistic-regression/2d-sigmoid.png" style="padding-bottom:0.5em; width:50%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">The hypothesis sigmoid in one and two dimensions, I'll let you imagine more</figcaption>
</figure>

Typically another change is made. The regression is made such that the *log likelihood* of the *parameters $\theta$* is *maximized*. Whereas before, the *error* given the *data* and *parameters* was *minimized*. Honestly in this case the numbers make more sense than the words for me. Starting off by defining likelihood:

<div style="font-size: 150%;">
	$$ 
	\begin{align}L(\theta) & \equiv P(\vec{y} | x;\theta) \\
	&= \prod_{i=1}^mP(y^i | x^i;\theta)
	\end{align}
	$$
</div>

Which you would read "Likelihood of a set of weights $\theta$ is the probability of seeing the values (classes, $\vec{y}$) given the data points $x$ as parameterized by $\theta$". And in our case, we mean parameterized by $\theta$ in the logistic function above, though in general it could be parameterized by any function. The second line has broken out of the vector form; products are used when considering join probability, and here we are considering the joint probability of many points being of a certain class. For the logistic hypothesis function the probability of a point being of a certain class is split into two cases and can be combined with a powers trick:

<div style="font-size: 150%;">
	$$ 
	P(y=1 | x;\theta) = h_{\theta}(x)
	P(y=0 | x;\theta) = 1 - h_{\theta}(x)
	P(y | x;\theta) = h_{\theta}(x)^y(1 - h_{\theta}(x))^{1-y}
	$$
</div>

As with least squares regression we want to apply gradient descent, though this time it will be gradient *ascent* because we are maximizing a function. It is simply a change in sign. As before we want the derivative. Now at this point most people will determine the derivative *of the log* of the function for likelihood. In general I'll just say this is to make the required "learning rates" for the algorithm more tractable, as functions aren't blowing up so much. [This is a pretty good response with a little more explanation](https://stats.stackexchange.com/questions/174481/why-to-optimize-max-log-probability-instead-of-probability). Having accepted this, we can then get our derivative:

<div style="font-size: 150%;">
	$$
	L(\theta) = \prod_i h_{\theta}(x^i)^{y^i}(1 - h_{\theta}(x^i))^{1-y^i}

	\begin{align}log(L(\theta)) & = \sum_{i=1}^my^ilog(h_{\theta}(x^i) + (1-y^i)log(1-h_{\theta}(x^i))) \\
	&= \sum_{i, y_i=1}log(h_i) + \sum_{i, y=0}log(1-h_i) \equiv LCL
	\end{align}
	\text{Apply a derivative w.r.t. the j'th parameter weight \theta}
	\frac{\partial}{\partial\theta_j}log(L(\theta))
	\begin{align} = \sum_{i, y_i=1}\frac{\partial}{\partial\theta_j}log(h_i) + \sum_{i, y=0}\frac{\partial}{\partial\theta_j}log(1-h_i) \\
	&= \sum_{i, y_i=1}frac{1}{h_i}
	\end{align}
		\text{Taking h_{\theta}(x_i) \equiv h_i and splitting the sum:}

	$$
</div>








