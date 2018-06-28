---
layout: post
title: "N-Body Simulations for Galaxies - CUDA"
categories: [Physics, CUDA]
date: 2017-07-19
---

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  CommonHTML: {
    scale: 150
  }
});
</script>
<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

<h2 align="center">References</h2>

- The Disk-Bulge-Halo model for galaxies, stability, etc: <a href="https://arxiv.org/pdf/astro-ph/9502051.pdf" target="_blank">Kiujiken & Dubinski</a> 

- Their GalactICS package which I have used to populate galaxies <a href="http://adsabs.harvard.edu/abs/2011ascl.soft09011K" target="_blank">is publicly available</a>

- My <a href="https://github.com/Hobbes1/CudaGalaxies">GitHub repo</a> for this project

<h2 align="center">Introduction and Motivation</h2>

Following up after the simplified model in the previous post, I can now get into more high powered applications. The only "new physics" to be discussed at this point is the new model for the galaxies, which will be random populations of density functions which have been granted to us by *Kiujiken & Dubinski*. We also introduce dark matter to our galaxies, which turns out to be a necessity for stability.

The simulation on the other hand requires a new approach; using the parallelism of CUDA to cut down on the extremely expensive simulation times. With the number of masses I intended to use, the straightforward CPU based approach (without any fancy speed ups ) simulations of a reasonable time-scale would have taken lifetimes! Particularly on my poor little laptop. I'll be discussing the technology behind GPU computing and the CUDA based algorithm which allows for these simulations to be ran in hours, or minutes really, if you don't want to bother making smooth videos like I did.

There is quite a bit of motivation for this work for me personally, GPU computing opens doors to a new scope of computational science, in areas like fluid dynamics, quantum electrodynamics (QED), and molecular dynamics.

<h2 align="center">The Physics</h2>


I'll break this portion into descriptions of Kiujiken & Dubinski's models for the bulge, halo, and disk which I have generated to use as initial conditions. The bulge and halo are simpler, nearly spherically symmetric models which are approximately stable on their own, the rings are more complex due to their asymmetry. The functions I post below are distribution functions in energy which have density functions (the probability to be found in a region of phase space which one would expect to use to populate at random) associated with them through integration over 2 or 3 variables, depending on the model. The paper solves for these functions as well.

<h2 align="center">The Bulge</h2>

<figure>
	<img src="{{site.baseurl}}/images/nbody-cuda/bulge2.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">Stand alone King's bulge</figcaption>
</figure>

The bulge utilizes King's model, which can be thought of as a truncated Isothermal Sphere. Model's such as these achieve stability by requiring inward gravitational forces to match outward "pressure" when forming a distribution. One must make corrections to an infinity which arises at the origin, as well as the truncation which must be made in a simulation, without which the sphere would go to infinity and contain theoretically infinite mass. Here's what that ends up looking like in the paper:



These are not very pretty! What matters here are the desired ρb value specifying the central bulge density, and the relationships between ψc and σb and σ0, which determine how sharply the distribution cuts off. The author's chose values which gave a more centrally dense bulge than the halo distribution, likely to match observation.

<h2 align="center">The Disk</h2>

<figure>
	<img src="{{site.baseurl}}/images/nbody-cuda/disk.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">Stand alone Disk, with no central bodies to hold it stable</figcaption>
</figure>

The model for the disk expects there to be central mass, from the bulge and halo, in order to be stable. As such, masses closest to the center are given more energy such that they don't collapse inward. Simulating the thing alone the results in an expansion of these masses, but I've done it anyway for the sake of completion. The distribution function below has become a function of 3 energy variables, w.r.t. planar motions, angular momentum, and the additional z energy component which the paper models simply as an oscillation across the plane of the disk:



the density and frequency parameters with a tilde are chosen to match observations, in particular light densities and rotation speeds of real disk galaxies, and likely for stability as well when the disk model is mixed with the other two. Energies and frequencies as a function of Rc are dependent on the energy of a circular orbit with the desired Lz parameter.

<h2 align="center">The Halo</h2>

<figure>
	<img src="{{site.baseurl}}/images/nbody-cuda/halo.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">The dark matter halo, another ~ isothermal sphere with a softer decay</figcaption>
</figure>


If you can make out the now tiny axes (which have remained the same size) through the artifacts in the gif you can see the halo is massive in comparison to the visible galaxy. It's also literally massive, ~94% of the entire system using this combined model. The distribution function is not too different from that of the bulge, it decays more slowly and most importantly, it is not spherical. It's known as an Evan's model and has slightly flattened poles, converging slightly inward towards the disk:

$$
	f_{halo}(E,L_z^2) = 
		\begin{cases}
		[(AL_z^2 + B)e^(-E/\sigma_0^2)] * [e^(-E/\sigma_0^2) - 1] & \text{if $E < 0$,} \\
		0 & \text{otherwise.}
		\end{cases}
$$


As described, the parameters A, B, and C, correspond to the density scale, the "core" radius, and the flattening parameter which is characteristic of Evan's model.

<h2 align="center">All Together</h2>

<figure>
	<img src="{{site.baseurl}}/images/nbody-cuda/33kgalaxy7h.gif" style="padding-bottom:0.5em; width:60%; margin-left:auto; margin-right:auto; display:block;" />
	<figcaption style="text-align:center;">A stationary full galaxy, stable when put together as intended</figcaption>
</figure>



<h2 align="center">The Code</h2>

to be written . . .

the code is linked above, this will contain an introduction to CUDA and how it fits with common C++ programming, and a description of the n-body algorithm which differs only slightly from NVIDIA's provides sample. 

I'll discuss the possibility for a tree code implementation as well, and the additional orders of magnitude which it would bring in speed. 

<h2 align="center">Results</h2>

I'll be posting more on the results, including a simulated collision of these D.B.H. galaxies. For now here is a colorized visualization of the code running on a galaxy comprised of ~33 thousand points. The simulation ran on the order of hours; without any approximations like a tree-code it could have taken many years on the CPU.
