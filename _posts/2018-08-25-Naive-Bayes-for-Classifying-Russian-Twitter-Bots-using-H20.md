---
layout: post
title: "Classifying Russian Twitter Bots using Naive Bayes and H20"
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

[Before](https://kyle-lewis.github.io/machine%20learning/2018/08/16/Generative-Learning-and-Bayes-Theorem.html) I presented an application of Bayes Theorem in GDA. Bayes theorem can also be applied to text classification problems as well through a Naive Bayes classifier. If we were to classify a peice of text using word counts we would do so using a joint probability. We can also use the generative Bayes Classification rule to do so. At least one motivation for using Naive Bayes for this approach is that many other classifiers will fail for various reasons given such a large feature space, like the number of unique words in a given dataset. 

<h2 align="center">Naive Bayes in Text Classification</h2><hr>

Here's that joint probability I mentioned. Assuming there are $m$ unique words in our dataset, and class labels $y$:

<div style="font-size: 130%;">
	$$ 
	\begin{align}
	P(w_1,w_2,w_3 \ldots w_m | y) & = p(w_1 | y) \cdot p(w_2 | y \cdot x_1) \cdot p(w_3 | y \cdot x_1 \cdot x_2) \ldots \\
	\\
	\text{And, the big assumption:}
	\\
	& = p(w_1 | y) \cdot p(w_2 | y) \cdot p(w_3 | y) \ldots \\
	& = \prod_{i=1}^mp(x_i | y)
	\end{align}
	$$
</div>

The big assumption is that our features are all independent from eachother with respect to the classes $y$. As in, seeing the word "shop" does not increase the likelihood of seeing the word "buy" in an email (we of course know that it would). This assumption reduces the number of parameters in the model significantly and lets us throw something together very quickly.

Ng presents the case where $Y$ takes on two values zero or one, and each $w_i$ takes on values zero or one, representing simply the presense of a word in a document which can take on only two classes. If we want more than two classes, and also if we want to take into account instances where words appear more than once, it just ends up looking a little different when we place the Likelihood function with the assumption above into Bayes rule:

<div style="font-size: 130%;">
	$$ 
	\begin{align}
	\text{Generally, Bayes rule reads:} 
	\\ \\
	Posterior \space odds  &= \frac{Likelihood \space \cdot Prior \space odds}{Evidence} 
	\\ \\
	\text{For boolean class, boolean variables:}
	\\ \\
	P(Y=y_1|w_1 \ldots w_m) &= \frac{\prod_i^mP(w_i|Y = y_1) \cdot P(Y = y_1)}{\prod_i^mP(w_m|Y=y_1) \cdot P(Y = y_1) + \prod_i^mP(w_m|Y=y_0) \cdot P(Y = y_0)}
	\\ \\
	\text{For k discrete classes, and real valued variables:}
	\\ \\
	P(Y=y_k|w_{1,1} \ldots w_{m,l}) &= \frac{\prod_i^{ml}P(w_{il}|Y = y_k) \cdot P(Y = y_k)}{\sum_j^KP(Y=y_j)\cdot\prod_i^{ml}P(w_{ml}|Y=y_k)}
	\\
	\end{align}
	$$
</div>

In the last case I introduced a new index, $l$, for the word sums. $w_{il}$ indexes over each word, and over each count of each word, so we've added terms to fit, but fitting is already pretty fast for Naive Bayes so it's not much of a price to pay. I like the idea of the multinomial representation better, it allows for a "softness", maybe a few words appear once in a piece of text by chance, but multiple occurances of some words could have far more decisive power when determining the class of the text.

As with GDA the prediction rule is then to simply assign the class with the highest probability given the Likelihood, Prior, and Evidence terms. It turns out we can actually eliminate the Evidence term, as it is constant when looking for a maximum among $Y\in[y_0 \ldots y_k]$ :
<div style="font-size: 130%;">
	$$ 
	\begin{align}
	Y_{assigned} & = argmax_{y_k} { \frac{\prod_i^m P(w_i|Y = y_k) \cdot P(Y = y_k) } {\sum_j^K P(Y=y_j) \cdot \prod_i^m P(w_m|Y=y_k)} } \\
	& = argmax_{y_k} \prod_i^m P(w_i|Y = y_k) \cdot P(Y = y_k)
	\end{align}
	$$
</div>

So, in order to make predictions, we just need to know all of the Prior terms $P(Y = y_k)$ and all of the Likelihood terms $P(w_m|Y=y_k)$, in either the binomial or multinomial case. The maximum likelihood estimates for each of the Likelihood terms looks like:

<div style="font-size: 130%;">
	$$ 
	\phi_{ij, y = k} = P(W_i = w_{ij}|Y = y_k) = \frac{\sum_{ij}I{W_i == w_{ij} \land Y = y_k} } {\sum_k I{Y=y_k}}
	$$
</div>

Or in english, the percent of cases where the class was of class $k$ and you observed your feature $w_{ij}$. Likewise the estimate for the Priors is just the percent of cases where the class of the document is of the particular class:

<div style="font-size: 130%;">
	$$ 
	\phi_{y} = P(Y = y_k) = \frac{\sum_n I{Y=y_k}}{N}
	$$
</div>

At this point, everything is calculable! No regressions to run, we just have to calculate all of these terms for a large enough dataset, and then attempt to make predictions with them on remaining data.

<h2 align="center">References</h2><hr>

- <a href="http://www.cs.columbia.edu/~mcollins/em.pdf" target="_blank">Notes from a Columbia University course</a> focused on Naive Bayes and Maximum Likelihood Estimation.

- <a href="https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf" target="_blank"> A Chapter from a textbook written by Tom Mitchell at Carnegie Mellon</a> focused on Naive Bayes which also describes discrete valued variables and multiple classes. 

- <a href="https://youtu.be/qRJ3GKMOFrE?list=PLA89DCFA6ADACE599" target="_blank"> Andrew Ng's 5th lecture, on GDA, Naive Bayes, and Laplace Smoothing. 

- <a href="https://youtu.be/qyyJKd-zXRE?list=PLA89DCFA6ADACE599" target="_blank"> Also his 6th lecture, where he discusses multinomial Naive Bayes. 

- <a href="http://cs229.stanford.edu/notes/cs229-notes2.pdf" target="_blank"> And finally the accompanying notes for the course. 

- Code for the demonstration on <a href="https://github.com/Kyle-Lewis/MachineLearningProjects/tree/master/4_GDA" target="_blank">my Github</a>