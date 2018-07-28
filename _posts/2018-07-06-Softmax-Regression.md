---
layout: post
title: "Softmax Regression"
categories: [Machine Learning]
date: 2018-06-25
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

- <a href="https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf" target="_blank">A general introduction to the exponential family</a>, I believe this was extra material provided as a part of a course taught at Berkely, it was very helpful in expanding upon Ng's introduction.

- <a href="https://youtu.be/nLKOQfKLUks" target="_blank">Andrew Ng's lecture</a> on generalized linear models, in which he introduces Softmax Regression. 

- The Softmax code discussed here is on <a href="https://github.com/Kyle-Lewis/CudaSoftmax" target="_blank">my Github</a>

<h2 align="center">Motivation</h2>

Logistic regression only worked with two classes, surely we should be able to distinguish between more. This post will contain a description of the softmax regression algorithm with a proof that gets us from the hypothesis function to the weight update rule which comprises the algorithm. I'll also discuss the code for this project which utilizes CUDA to speed up the main update algorithm, as well as visualization functions. This particular implementation is general in the number of classes for our data, but specifically expects two input features, so that we can plot them easily. To grow the input feature set would be to grow a vector here or there. 

<h2 align="center">Exponential Family </h2>

As with all these regressions, we are attempting to apply gradient descent to the parameters of a hypothesis function such that likelihood is maximized. Before, the sigmoid function of logistic regression was ideal in that it had an output on $[0, 1]$ and could distinguish between two regions. It turns out that the sigmoid can be retrieved generally from inverting a certain *natural parameter* of a probability distribution which lives in the exponential family. The exponential family contains probability distributions parameterized by $\eta$ (the natural parameter) in the following form:

<div style="font-size: 150%;">
	$$ 
	p(y|\eta) = h(y)e^{\eta^TT(y)-A(\eta)}
	$$
</div>

Where the inclusion of the inner product lets us know that $\eta$ can generally be a vector; this turns out to be the case now that we are attempting to generate a model for multiple classes. There's quite a few terms here: $h(y)$, a normalization term, $T(y)$ the *sufficient statistic*, $A(\eta)$ the *cumulant function*, and our *natural parameter* $\eta$. I won't go into detail about the sufficient statistic or cumulant function here, they have very useful properties which are detailed in the Berkeley link. 

Often you want to represent the parameter vector $\eta$ as a linear combination of the form:
<div style="font-size: 150%;">
	$$ 
	\vec{\eta} = \phi(\theta) \\
	\text{and so:} \\
	p(y|\phi(\theta)) = h(y)e^{\phi(\theta)^TT(y)-A(\phi(\theta))}
	$$
</div>
Though of course there are cases where $\eta$ cannot be represented in this way, these cases are considered non-linear or of the *curved exponential family*. I'm also not dealing with those here. 

*If* however we can represent our probability in this way, our hypothesis function falls out of the exponential family form by inverting the parameter vector; now $\phi(\theta)$. All we have to do first is convert our probability function into the form of the exponential family. There's a really neat trick, simply taking the exponent of the log of the probability will get you most if not all the way there. For example, when classifying between two cases $y=0 \text{and} y=1$ in the previous post we used the Bernoulli distribution. Using the trick:

<div style="font-size: 150%;">
	$$ 
	\begin{align}p(y | \phi) & = \phi^y(1 - \phi)^{1-y} \\
	&= e^{y log(\phi) + log(1-\phi)} \\
	\end{align}
	$$
</div>

Which gives us the forms of the functions:

<div style="font-size: 150%;">
	\eta = log(\frac{\phi}{(1-\phi)}) \\
	T(y) = y \\
	A(\eta) = -log(1 - \phi) = log(1 + e^{\eta}) 
</div>

And inverting the equation for $\eta$:

<div style="font-size: 150%;">
	$$
	\phi = \frac{1}{1+e^{-\eta}}
	$$
</div>

Which was exactly the logistic hypothesis function from before. 

Now for multiple classes we have a new multinomial probability distrubution to try out. For $M$ trials on data comprised of $K$ classes 

<div style="font-size: 150%;">
	$$
	p(y | \phi) = \frac{M!}{\prod_{k=1}^K{y_k!}}\prod_{k=1}^K{\phi_k^{y_k}}
	$$
</div>

Now, Ng chooses to drop the scaling factor out front and focus only on the product of the natural parameters $\phi$. We can use the same trick on this form of the equation. We also break out the $K^{th}$ component of $phi$ from the rest of the sum to achieve a *minimal* representation for the distribution. This makes sense; as Ng describes, once we have up to $K-1$ terms, the $K$ term can be represented by one minus the sum of the rest. 

<div style="font-size: 150%;">
	$$
	p(y | \phi) = \exp{\sum_{k=1}^Ky_k\log{\phi_k}}
	$$
</div>


<h2 align="center">Code</h2>

<h2 align="center">Results</h2>

<figure>
	<img src="{{site.baseurl}}/images/softmax/softmax.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">Running Softmax regression slowly on some easy dataset+</figcaption>
</figure>

