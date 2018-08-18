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

<h2 align="center">References</h2><hr>

- <a href="https://www.youtube.com/watch?v=qRJ3GKMOFrE&index=6&list=PLA89DCFA6ADACE599&t=0s" target="_blank">Andrew Ng's lecture</a> on generative learning algorithms, <a href="https://www.youtube.com/watch?v=qRJ3GKMOFrE&index=6&list=PLA89DCFA6ADACE599&t=0s" target="_blank"> and his notes</a>.

- Code for the demonstration on my GitHub: <a href="https://github.com/Kyle-Lewis/MachineLearningProjects/tree/master/4_GDA" target="_blank">my Github</a>

WIP