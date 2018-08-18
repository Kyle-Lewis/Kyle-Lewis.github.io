---
layout: post
title: "Generative Learning and Bayes' theorem"
categories: [Machine Learning]
date: 2018-08-16
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

<h2 align="center">Motivation</h2><hr>

I'm getting around to a project implementing a Naive Bayes classifier and there's some preliminary work that's going to be helpful in transitioning from logistic and softmax regression algorithms. Hopefully *that* post can get straight to the code as a result. 

<h2 align="center">Generative vs Discriminate</h2><hr>

Discriminant learning algorithms (like logistic regression) try to map from input features to class lables; $P(Y|X=x)$ can be read: the probability of some data point being of a class $y$, *given* the features $\vec{x}$ of that point. Conversely, generative learning algorithms try to map classes to the distributions of their features; $P(X|Y=y)$. 

To me, the generative side of this symmetry is a little less intuitive. [Think of the geographic population distribution of the people who speak French.](https://en.wikipedia.org/wiki/Geographical_distribution_of_French_speakers) Already, the word distribution gives it away. "French speaking" is the class label and the geography, be it coordinates or country, is the feature set. You can imagine a good model for French speakers would have a high probability of finding that they *have the feature* of being from France, and then the DRC, and then Germany, and so on. 

It sounds wierd, why would we do such a thing? We want to do it because, having generated such a model, we can use [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) to retrieve what we typically want to predict, the probability that a person is a French speaker given that they are from a certain country (and of course, other features, if we want a good amount of predictive power). 

And its not hard to calculate!

<div style="font-size: 150%;">
	$$ 
	p(y|x) = \frac{p(x|y)p(y)}{p(x)}
	$$
</div>

In words:
<div style="font-size: 150%;">
	$$ 
	p(Person speaks French, given that they are from France) = \frac{p(Person is from France, given that they speak French)p(Person speaks french)}{p(Person is from France)}
	$$
</div>
Thinking of some extreme cases helps feeling our way around this relationship. Imagine a world where 99% of French speakers live in Australia; the numerator on the right becomes very small, and as a result, the feature of being from France wouldn't be a very strong predictor of speaking French. In another world where 99% of people live in France, the odds of French people speaking French almost becomes simply the odds that a person speaks French in general.

<h2 align="center">For Example, Gaussian Discriminate Analysis</h2><hr>

We do have to come up with some model for the distribution of features. Of course there are infinitely many choices to make here but at least one common choice is to model some dataset as Gaussian, everyone's favourite distribution. The general form of $k$ dimensional gaussians looks like the following: 

<div style="font-size: 150%;">
	$$ 
	P(x;\mu,\Sigma) = \frac{1}{(2\pi)^{n/2}\det{\Sigma}^{1/2}}exp\Big\{-\frac{1}{2}(\vec{x}-\vec{\mu})^T\Sigma^{-1}(\vec{x}-vec{\mu})\Big\}
	\text{Where:}
	\mu \text{Is a vector of the means for the distribution in each dimsension}
	\Sigma \text{Is a K by K matrix detoting the covariances between each axis}
	$$
</div>
If you aren't familiar with the concept of covariance, Ng has some really nice pictures in [his notes](http://cs229.stanford.edu/notes/cs229-notes2.pdf) that I won't bother reproducing. Simply put, the matrix describes the shape of the distribution, while the means describe the position. 


<h2 align="center">References</h2><hr>

- <a href="https://www.youtube.com/watch?v=qRJ3GKMOFrE&index=6&list=PLA89DCFA6ADACE599&t=0s" target="_blank">Andrew Ng's lecture</a> on generative learning algorithms, <a href="http://cs229.stanford.edu/notes/cs229-notes2.pdf" target="_blank"> and his notes</a>.

- Code for the demonstration on <a href="https://github.com/Kyle-Lewis/MachineLearningProjects/tree/master/4_GDA" target="_blank">my Github</a>

WIP