---
layout: post
title: "Variational Monte Carlo in QM"
categories: [Physics, CUDA]
date: 2017-07-19
image: images/variational-mc/4-node-potential.gif
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

- An introduction to the <a href="http://pages.cs.wisc.edu/~jerryzhu/cs731/mcmc.pdf" target="_blank">Markov Chain detail balance</a>

- Or a <a href="https://en.wikipedia.org/wiki/Detailed_balance" target="_blank">wiki page on the topic</a> if you want to scratch that itch with less math.

- And finally, an introductory <a href="https://www.youtube.com/watch?time_continue=431&v=Ws63I3F7Moc" target="_blank">Khan video</a> provides a great example to introduce the topic.

<!-- - My {GitHub repo} with the 1 dimensional CPU only implementation -->

- And <a href="https://github.com/Hobbes1/CudaVariationalMonteCarloQM" target="_blank"> my GitHub repo for this project</a>

<h2 align="center">Introduction and Motivation</h2>

I wanted to revisit a quick topic which was presented to me during my undergrad because it turns out so called Markov Chains are in fact a very useful tool in certain branches of machine learning as well. Additionally, while the basic implementation and proof of concept is entirely possible without any accelerating hardware or algorithm, I still wanted to introduce CUDA to more problems; in this case by upping the dimensionality of the problem. I've also gotten CUDA and openGL to play nicely together so I can do away with the pesky third party post-simulation animations, as well as the limiting data transfers from my graphics devices to the host CPU.

<h2 align="center">The Markov Business</h2>

This project is an application of the *Metropolis Hastings Algorithm* which is of a class of Markov Chain Monte Carlo (MCMC) methods used for deriving probability distrubutions like those found in quantum mechanics. I've generally discussed the Markov Chain Monte Carlo in [my notes on the subject.](2017-12-17-Monte-Carlo-Markov-Chains-and-Detail-Balance ) The algorithm can be used to discover distributions, given that you know desired properties of the distribution.

<h2 align="center">Physics</h2>

So what distribution are we solving for, and what properties can we take advantage of? This will require *some* introductory QM work. Everything here can be found in an undergraduate level QM textbook, I'm partial to Griffiths. I'll cover stationary solutions of the Schr&ouml;dinger equation (in *nice* cases) and the *Variational Principle* which can be used to approximate solutions in some of the less than nice cases. It turns out MCMC will work to discover ground state distrubutions for potentials that we can't solve analytically. Which turns out to be most real potentials.

We'll start with the Schr&ouml;dinger equation itself, a second order differential equation: 

<div style="font-size: 150%;">
	$$
	i\hbar\frac{\partial \Psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2\Psi}{\partial x^2} + V\Psi
	$$
</div>

It describes the evolution of a wave-function, $ \Psi $ in time, in response to its initial conditions and the potential $V$. $ \Psi $ itself is a quantum mechanical description of a particle; no longer discrete but distributed in space with some probability of observing it in a given region. The wave-function itself is not *physical*, it's more of a mathematical artefact; what really matters are *observable* quantities like the *expected* position, which can be calculated $ \braket{x} = \integral_{-\inf}^{+\inf} x |\Psi(x,t)|^2dx$. 

There are any number of solutions to the equation, but it turns out that there exists a class of solutions which are time independant, and that these solutions are very important. Through separation of variables and some substitution the Schr&ouml;dinger equation can be split into two ordinary differential equations of time and of position:

<div style="font-size: 150%;">
	$$
	\Psi(x,t) = \psi(x)\phi(t)   \text{  where  }   \frac{d\phi}{dt} = -\frac{iE}{\hbar}\phi   \text{  and  }   -\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V\psi = E\psi 
	$$
</div>

You can solve the time dependant equation easily though integration and write out our new $ \psi $:

<div style="font-size: 150%;">
	$$
	\Psi(x,t) = \psi(x)e^{\frac{-iEt}{\hbar}}
	$$
</div>

Functions of this form have a stationary *probability density*. (That term under the integral before). The complex exponential terms cancel as per Euler's formula.

<div style="font-size: 150%;">
	$$
	|\Psi(x,t)|^2 = \Psi^*\Psi = \psi^*e^{\frac{iEt}{\hbar}}\psie^{\frac{-iEt}{\hbar}} = |\psi(x)|^2
	$$
</div>

So the wave functions which satisfy the product form of the Schr&ouml;dinger equation correspond to stationary states. Not only the expected position, but all observable quantities are constant for these states; the expected energy is constant, and of course momentum which is zero for a stationary state. *Most importantly* these stationary solutions are eigenfunctions - or eigenvectors if you like - of the observables of the state as represented by hermitian operators. This means that the general time-independant Schr&ouml;dinger equation, can be written as a linear combination of the stationary states written above. So,  Thats a lot of linear algebra in a few sentences, and unfortunately I won't be proving or going into more detail here. I'd recommend the back of Griffiths for a very brifef, physics-relevant view; or any math textbook on the subject if you are curious. 

Ok cool, if we've found the eigenstates of the system, we've in fact solved for states which are time dependant as well, as they can be written as a weighted sum of stationary states, which is of course normalized. However, it turns out that outside of [certain cases](https://en.wikipedia.org/wiki/List_of_quantum-mechanical_systems_with_analytical_solutions) we cannot solve for these states analytically. Instead we often have to rely on numerical methods. 

<h2 align="center">Code Samples</h2>

<hr>
<div style="width:110%">

{% highlight c++ %}

		/* Steps alternating even/odd "red/black" points in the path
		 * with the chance of acceptance being related to the change 
		 * in the action which the move brings to the path */
		// Note: Kernel Functions can't be defined within the class

__global__ void Step(float2* rawRingPoints,
					 float epsilon,
					 int mode,
					 float tau, 
					 float dt, 
					 int numPoints,
					 unsigned int redBlack)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numPoints)
		printf("ERROR: Out of rawRingPoints[] bounds, calling: %d, max: %d\n", idx, numPoints);
	
	// alternate steps, red/black:
	if((idx + redBlack) % 2 == 0)
	{
		return;
	}
	
	float2 newPoint = rawRingPoints[idx];

	curandState state;
	curand_init((unsigned long long)clock(), idx, 0, &state);

	float randX = (0.5 - curand_uniform(&state)) * 2.0 * epsilon;
	float randY = (0.5 - curand_uniform(&state)) * 2.0 * epsilon;
	newPoint.x = rawRingPoints[idx].x + randX;
	newPoint.y = rawRingPoints[idx].y + randY;

	// Run accept/deny on the move
	rawRingPoints[idx] = MCTest(rawRingPoints, newPoint, mode, tau, dt, numPoints, idx);
	__syncthreads();
}

{% endhighlight %}

</div>
<hr>

Another post in the making! I've developed CUDA accelerated and CPU side C++ code to perform approximations to quantum electrodynamics, known as Lattice QED. In particular there are codes to perform discrete path integration of an electron wavefunction in various potentials, as well as a Monte-Carlo approach to solving ground state and excited state energy levels and wavefunctions.

And a bit of a teaser for my results in a double well potential . . .

I've made up this potential. It's not a real potential. It just looks neat.

Populating a simple harmonic oscillator.