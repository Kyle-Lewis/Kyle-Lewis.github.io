---
layout: post
title: "The Dual Problem, Kernels, and SMO"
categories: [Machine Learning]
date: 2018-10-6
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

I'm still working my way towards the implementation of a support vector machine, doing my best to leave no gaps in understanding. In the last post I discussed the general application of Lagrangian systems of equations to optimization problems. This approach (solving the resulting system of equations) would have been enough if the optimization problem found in SVMs yielded a system in which the solution lies directly on the intersection of all of the constraints. (Un)fortunately that is not the case, and we have to do something more interesting.

TL;DR:

Translating the previously discussed optimization problem into what is known as the "dual problem". It will turn out that the dual problem can be expressed in terms of inner products between feature vectors. There will *then* be some neat Kernel computations of inner products which will let us map feature spaces into higher dimensions relatively inexpensively. This will in turn allow the training steps of sequential minimal optimization, SMO (a gradient descent algorithm specific to constrianed quadratic problems) to be calculated quickly in high dimensional spaces, which we can use to build non-linear classifiers.

<h2 align="center">The Primal and Dual problems</h2><hr>

[Before](https://kyle-lewis.github.io/machine%20learning/2018/09/15/Constrained-Maximization-and-Lagrange-Multipliers.html) I discussed Lagrangian representations of optimization problems with inequality constraints. Generally, and in all of the referenced discussions of the primal and dual problems, we can also include strict(er) equality constraints $h_i(x)$. I'll also drop the explicit features $x$ and $y$, from here on out writing in terms of a generalized vector $\vec{x}$ of any dimensionality. Against my better judgement I will also drop the vector mark from this $x$. With these changes problem statement and resulting Lagrangian become:

<div style="font-size: 130%;">
	$$ 
	\min f(x) \\
	\begin{align}
	s.t. & g_i(x) \leq 0 \\
	& h_i(x) = 0 \\
	\\
	\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=i}^m \lambda_ig_i(x) + \sum_{i=1}^p\nu_ih_i(x) \\
	\end{align}
	$$
</div>

**The Primal Problem** is the maximization of the Lagrangian over the Lagrange multipliers $\lambda$ and $\nu$, holding all $\lambda_i \geq 0$. This quantity, which is now a function only of $x$, can then be minimized. *Wait weren't we minimizing f(x) directly? Doesn't this change our result?*. It turns out that it does not.

<div style="font-size: 130%;">
	$$ 
	\frac{min}{x}(\frac{max}{\lambda, \nu, \lambda_i \geq 0} (\mathcal{L}(x, \lambda, \nu))) = \frac{min}{x}\Theta_P(x)
	$$
</div>

The inner maximization problem, $\Theta_P(x)$ or the "primal objective" turns out to just be an encoding of the original minimization problem with some special behavior which captures the constraints:

<div style="font-size: 130%;">
	$$ 
	\begin{align}
	\Theta_P(x) & = \max_{\substack{\lambda, \nu, \lambda_i \geq 0} } \bigg[ f(x) + \sum_{i=i}^m \lambda_ig_i(x) + \sum_{i=1}^p\nu_ih_i(x) \\
	& = f(x) + \max_{\substack{\lambda, \nu, \lambda_i \geq 0} } \sum_{i=i}^m \lambda_ig_i(x) + \sum_{i=1}^p\nu_ih_i(x) \bigg] \\
	\end{align}
	$$
</div>

If, however, $x$ is *feasible* - it satisfies $g_i(x) \leq 0$ and $h_i(x) = 0$ - then the sum of inequality constraint terms are strictly negative or zero, and the equality constraints are of course zero having satisfied the constraints. Nothing fancy was done there; when the constraints are satisfied you simply have a sum of positive values times negative values, and that sum has to be less than or equal to zero. Because of these two results, for a feasible point $x\*$ the primal objective problem is exactly the original minimization problem. For *in*feasible points $x$ however, the violations - $g_i(x) \gre 0$, or $h_i(x) \neq 0$ - allows for either of the sums to go to infinity by choosing an arbitrarily large Lagrange multiplier:

<div style="font-size: 130%;">
	$$ 
	\begin{align}
	\Theta_P(x\*) & = f(x\*) + \max_{\substack{\lambda, \nu, \lambda_i \geq 0} } \bigg[ \underbrace{\sum_{i=i}^m \lambda_ig_i(x\*)}_\text{\leq 0} + \underbrace{\sum_{i=1}^p\nu_ih_i(x\*)}_\text{= 0} \bigg] \\
	\Theta_P(x^\dagger) & = f(x^\dagger) + \max_{\substack{\lambda, \nu, \lambda_i \geq 0} } \bigg[ \underbrace{\sum_{i=i}^m \lambda_ig_i(x^\dagger)}_\text{\leq 0} + \underbrace{\sum_{i=1}^p\nu_ih_i(x^\dagger)}_\text{= 0} \bigg] \\
	\text{or} \\
	& = f(x) + \begin{cases}
				0 & \text{if x is feasible} \\
				\inf & \text{if x is not feasible} \\
				\end{cases}
	\end{align}
	$$
</div>

We could describe the primal maximization as being a barrier function which prevents the consideration of infeasible points. For these, the whole system blows up as a result of the structure of our inner-maximization / outer-minimization problem. Finally, we define the feasible solution $P\* = \Theta_P(x\*)$




<h2 align="center">Kernels in Machine Learning and computing inner products</h2><hr>


<h2 align="center">Sequential Minimal Optimization</h2><hr>


<h2 align="center">References</h2><hr>

- <a href="http://cs229.stanford.edu/section/cs229-cvxopt2.pdf" target="_blank"> Notes from Stanford's cs299 course on convex optimization through the primal and dual problem. 

- <a href="http://cs229.stanford.edu/notes/cs229-notes3.pdf" target="_blank"> Related notes on Kernels taking advantage of the dual form, and SMO.

- <a href="https://www.youtube.com/watch?v=s8B4A5ubw6c&index=8&list=PLA89DCFA6ADACE599&t=1160s" target="_blank"> Andrew Ng's accompanying lecture for the same course. 

- <a href="https://pdfs.semanticscholar.org/2862/e7b8fefb209cdb4c47a1643f2af71cd67b00.pdf" target="_blank"> A review of Kernel methods in machine learning. 

- <a href="https://youtu.be/FJVmflArCXc" target="_blank"> Stanford's Stephen Boyd on the dual problem in convex optimization, offers some additional context with which to think of the problem. 

