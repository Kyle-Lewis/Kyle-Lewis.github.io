---
layout: post
title: "Constrained Maximization and Lagrange Multipliers"
categories: [Machine Learning]
date: 2018-09-15
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

Lessons on Support Vector Machines - at least the ones I've watched - jump straight into a lagrangian representation for a constrained miminization problem without much justification. I thought it would be valuable to prove to myself why this approach works, before going on to use it in SVM classification later.

<h2 align="center">Minimizing one function with one constraint</h2><hr>

I'll use a 2d parabola $$f(x,y) = x^2 + y^2$$ as the function I'm trying to minimize, with the inequality constraint of $$ y + \frac{1}{4}x - 1 \geq 0 $$. I'll call the line of the inequality $$ g(x,y) $$ and *for now* I'll be assuming that the minima will actually lie on this constrianing line and ignore the fact that it's an inequality.

To minimize, we want to decrease $$ f(x,y) $$ as long as we can while the still satisfying the inequality (intersecting with the constraint at least once). The minima will occur when we have just one single point left which satisfies our constraint 

<figure>
	<img src="{{site.baseurl}}/images/constrained-optimization/ShrinkingToMin3.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">Contours of the function to optimize, and the edge of the constraint $g$</figcaption>
</figure>

A fact we can use to solve for this point of interest is that tangent curves will have gradients which are proportional to eachother.

With the addition of the constraint equation itself $$ y + \frac{1}{4}x -1 = 0 $$ we then have 3 variables and 3 equations which we can solve. In this case we could just substitute our way to the solution, but with higher dimensional functions with increasing numbers of variables it helps to make use of [gaussian elimination](https://en.wikipedia.org/wiki/Gaussian_elimination) strategies. And of course, any software employing these methods can make use of LAPACK or Numpy's linalg module (*also lapack*) to solve the system:

<div style="font-size: 130%;">
	$$ 
	\begin{bmatrix}2.0 & 0.0 & -0.5 \\ 0.0 & 2.0 & -1.0 \\ 0.5 & 1.0 & 1.0 \end{bmatrix} \times 
	\left[\begin{array}{c} s_x \\ s_y \\ s_{\lambda} \end{array}\right]
	=
	\left[\begin{array}{c} 0 \\ 0 \\ 0 \end{array}\right]
	$$
</div>

<figure>
	<img src="{{site.baseurl}}/images/constrained-optimization/ModifyingConstraint.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">Gradually modifying the constraint and plotting the contour of the resulting minima</figcaption>
</figure>

The "Lagrangian" in the name of this method just refers to the form of the system of equations when written out to one side, and the "multipliers" are just what people call the extra variables we introduce, $\lambda$.

<div style="font-size: 130%;">
	$$ 
	\mathcal{L}(x, y, \lambda) \equiv f(x, y) - \lambda g(x, y) \\ 
	$$
</div>

Any constant term in the constraint is moved from the right to the left under the multiplier; such that $ \lambda (g(x, y) - b) = 0 $. This is one of the [KKT](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions) conditions known as complementary slackness. "Slackness" is used because when the constriant is found to be *more than satisfied* the behavior of the system allows for it; essentially allowing for some "cost" to accumulate as a result. 
 
<h2 align="center">What about when constraints don't matter?</h2><hr>

Now I'll let $g(x,y)$ be the inequality $y + \frac{1}{4}x - 1 \geq 0$ and not assume that the solution lies on the constraint boundary. With a constraint on the system that's not binding - the minima of the function lies within the constrained region but not on the boundary - the system of equations to solve becomes the usual $\nabla f(x, y) = \vec{0}$. This is entirely contained by the form of the Lagrangian. Taking the derivative of the Lagrangian and setting it to zero is conveniantly equivalent to stating the system of equations that we found above:

<div style="font-size: 130%;">
	$$ 
	\nabla \mathcal{L}(x, y, \lambda) \implies  
	\begin{cases}
		\nabla f(x, y) = \lambda g(x, y) \\
		g(x,y) = 0 
	\end{cases}
	$$
</div>

But if $lambda$ is zero, as is the case for a non-binding constraint, you are left with the usual minimization equation:

<div style="font-size: 130%;">
	$$ 
	\nabla \mathcal{L}(x, y, \lambda) = 0 \implies \nabla f(x, y) = 0 \\
	$$
</div>

<figure>
	<img src="{{site.baseurl}}/images/constrained-optimization/NonBindingConstraint.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">Disregarding the constraint when its non-binding</figcaption>
</figure>

We know to set $\lambda$ for the constraint to zero when the inequality constraint is *more than satisfied*; $g(x,y) > 0$. Think about it from a calculus perspective, if the value of the constraint is currently positive by any amount, then moving anywhere in the feature space by any infinitesimally small amount you will still have a $g(x,y)$ value that is greater than zero, and so we must be in the region enclosed by the inequality and not on the border - not bounded. That assumption wouldn't hold for non-continuous constraints, so don't use those unless you want to come up with clever workarounds. An equivalent reason for setting $\lambda$ to zero arises when discussing the lagrangian in terms of the primal and dual problem, which I'll get to later. 

<h2 align="center">What about multiple constraints?</h2><hr>

Lets say we had two constriants on the same function as before; $ \frac{1}{2}x + y + 2 \geq 0 $ and $ 2x + y + 4 \geq 0 $. We end up just adding to the lagrangian:

<div style="font-size: 130%;">
	$$ 
	\mathcal{L}(x, y, \lambda_{g_1}, \lambda_{g_2}) = f(x, y) - \lambda_{g_1} * g_1(x, y) - \lambda_{g_2} * g_2(x, y) \\
	\text{where:} \\
	g_1 = \frac{1}{2}x + y + 2 \\
	g_2 = 2x + y + 4 \\
	\text{and generally:}
	\mathcal{L}(x, y, \lambda_{g_1}, ... \lambda_{g_m}) = f(x, y) - \sum_i^m \lambda_{g_i} * g_i(x, y)
	$$
</div>

We then go and solve the system of derivatives just as we did before, we just have a larger matrix equation in front of us. 

<figure>
	<img src="{{site.baseurl}}/images/constrained-optimization/multipleConstriants.png" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">Two constraints on $f(x, y)$</figcaption>
</figure>

Unfortuneately we're not done. We can already imaging the awkward situation which results when we go ahead and add more constraints, with some binding and some non-binding, which may not have a conveniant shared solution of this form. Later, the dual optimization problem, the addition of a tolerance factor, and the application of sequantial minimalization methods *will* finally bring us to the point of solving an arbitrary system.


<h2 align="center">References</h2><hr>

- <a href="https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/lagrange-multipliers-and-constrained-optimization/v/lagrange-multiplier-example-part-1" target="_blank">Khan academy series on constrained optimization through lagrangian systems of equations. 

- <a href="http://www1.maths.leeds.ac.uk/~cajones/math2640/notes4.pdf" target="_blank"> Notes on inequality vs equality constraints, and binding vs non-binding constraints. 