---
layout: post
title: "Naive Bayes for Classifying Russian Twitter Trolls"
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

Lessons on Support Vector Machines - at least the ones I've watched - jump straight into a lagrangian representation for a constrained miminization problem without much justification. I thought it would be valuable to prove to myself why this approach works, and tie it to some physics problem to give it some meaning. 

<h2 align="center">Minimizing one function with one constraint</h2><hr>

I'll use a 2d parabola $$f(x,y) = x^2 + y^2$$ as the function I'm trying to minimize, with the inequality constraint of $$ y + \frac{1}{4}x \geq 1 $$. I'll call the line of the inequality $$ g(x,y) $$ and *for now* I'll be assuming that the minima will actually lie on this constrianing line and ignore the fact that it's an inequality.

// Plot of curves of the parabola, with the line of inequality. 

To minimize, we want to decrease $$ f(x,y) $$ as long as we can while the still satisfying the inequality (intersecting with the constraint at least once). The minima will occur when we have just one single point left which satisfies our constraint 

// Gif of shrinking x^2 + y^2 circles and the constraint, with a highlighted arc of the curve shrinking to a highlighted point of the curve at the minima where the curve is tangent to the constraint.

A fact we can use to solve for this point of interest is that tangent curves will have gradients which are proportional to eachother. Or:

<div style="font-size: 130%;">
	$$ 
	\nabla f(x,y) \propto \nabla g(x,y) \\
	\text{or, giving a weight to the ratio:}
	\nabla f(x,y) \probto \lambda \nabla g(x,y)
	$$
</div>

With the addition of the constraint equation itself $$ y + \frac{1}{4}x = 1 $$ we then have 3 variables and 3 equations which we can solve. In this case we could just substitute our way to the solution, but with higher dimensional functions with increasing numbers of variables it helps to make use of [gaussian elimination](https://en.wikipedia.org/wiki/Gaussian_elimination) strategies. And of course, any software employing these methods can make use of LAPACK or Numpy's linalg module to solve the system.

// Gif of the solution point moving as you change slope and offset of the constraining line, expand the circle x^2 + y^2 and display the solution point while you are at it. 

<h2 align="center">What about when constraints don't matter?</h2><hr>

If we can 


<h2 align="center">References</h2><hr>

- <a href="https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/lagrange-multipliers-and-constrained-optimization/v/lagrange-multiplier-example-part-1" target="_blank">Khan academy series on constrained optimization through lagrangian systems of equations. 

- <a href="http://www1.maths.leeds.ac.uk/~cajones/math2640/notes4.pdf" target="_blank"> Notes on inequality vs equality constraints, and binding vs non-binding constraints. 