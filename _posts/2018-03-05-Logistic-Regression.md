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

Where the vector $\theta$ contains a weight for every feature (basis vector) of the data. What this looks like as you move up in dimensionality is a division between two regions where the function is high (approaching 1) and low (approaching 0) with a smooth transition between the states:

<figure>
	<img src="{{site.baseurl}}/images/logistic-regression/sigmoid.png" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<img src="{{site.baseurl}}/images/logistic-regression/2d-sigmoid.png" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">The hypothesis sigmoid in one and two dimensions, i'll let you imagine more</figcaption>
</figure>
