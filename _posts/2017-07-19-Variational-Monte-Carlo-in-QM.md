---
layout: post
title: "Variational Monte Carlo in QM"
categories: [Physics, CUDA]
date: 2017-07-19
image: images/nbody-cuda/4-node-potential.gif
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

An introduction to the Markov Chain detail balance
Or a wiki page if you want to scratch that itch with less math.
And finally, an introductory Khan video provides a great example to introduce the topic.
My {GitHub repo} with the 1 dimensional CPU only implementation
And {the repo for the CUDA implementation} in 2 dimensions and with openGL interoperability.
{introduction to variational method / QM at all?}

<h2 align="center">Introduction and Motivation</h2>

I wanted to revisit a quick topic which was presented to me during my undergrad because it turns out so called Markov Chains are in fact a very useful tool in certain branches of machine learning as well. Additionally, while the basic implementation and proof of concept is entirely possible without any accelerating hardware or algorithm, I still wanted to introduce CUDA to more problems; in this case by upping the dimensionality of the problem. I've also gotten CUDA and openGL to play nicely together so I can do away with the pesky third party post-simulation animations, as well as the limiting data transfers from my graphics devices to the host CPU.

<h2 align="center">Physics</h2>

Intro QM: explain shrodinger, wavefunctions, and expectation values (energy)
Explain Variational Principle / method

<h2 align="center">The Markov Business</h2>

You can read about the Markov Chain Monte Carlo in [my notes on the subject.](2017-12-17-Monte-Carlo-Markov-Chains-and-Detail-Balance )

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