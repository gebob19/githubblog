---
title:  "Going with the Flow: An Introduction to Normalizing Flows"
description: "This blog post/tutorial dives deep into the theory and PyTorch code for Normalizing Flows"
categories: example
mathjax: true
---

![alt text](https://gebob19.github.io/assets/norm_flow/nf.png "Normalizing Flows (from R-NVP Paper)")

Normalizing Flows (NFs) {%cite rezende2015variational --file normflow %} learn an *invertible* mapping $$f: X \rightarrow Z$$, where $$X$$ is our data distribution and $$Z$$ is a chosen latent-distribution.

Normalizing Flows are part of the generative model family, which includes Variational Autoencoders (VAEs) {%cite vaebayes --file normflow %}, and Generative Adversarial Networks (GANs) {%cite NIPS2014_5423 --file normflow %}. Once we learn the mapping $$f$$, we generate data by sampling $$z \sim p_Z$$ and then applying the inverse transformation, $$f^{-1}(z) = x_{gen}$$. 

*Note*: $$p_Z(z)$$ is the probability density of sampling $$z$$ under the distribution $$Z$$.

In this blog to understand normalizing flows better, we will cover the algorithm's theory and implement a flow model in PyTorch. But first, let us flow through the advantages and disadvantages of normalizing flows.

*Note:* If you are not interested in the comparison between generative models you can skip to 'How Normalizing Flows Work'

## Why Normalizing Flows

With the amazing results shown by VAEs and GANs, why would you want to use Normalizing flows? We list the advantages below

*Note*: Most advantages are from the GLOW paper {%cite kingma2018glow --file normflow %}

- NFs optimize the exact log-likelihood of the data, log($$p_X$$) 
    - VAEs optimize the lower bound (ELBO)
    - GANs learn to fool a discriminator network
- NFs infer exact latent-variable values $$z$$, which are useful for downstream tasks 
  - The VAE infers a distribution over latent-variable values
  - GANs do not have a latent-distribution 
- Potential for memory savings, with NFs gradient computations scaling constant to their depth
    - Both VAE's and GAN's gradient computations scale linearly to their depth
- NFs require only an encoder to be learned
    - VAEs require encoder and decoder networks 
    - GANs require generative and discriminative networks

But remember what mother says, "There ain't no such thing as a free lunch". 

Some of the downsides of normalizing flows are as follows, 

- The requirements of invertibility and efficient Jacobian calculations restrict model architecture 
    - more on this later... 
- Less resources/research on NFs compared to other generative models
    - The reason for this blog! 
- NFs generative results are still behind VAEs and GANs

Now let us get dirty in some theory!

# How Normalizing Flows Work

In this section, we understand the heart of Normalizing Flows. 

## Probability Distribution Change of Variables

Consider a random variable $$X \in \mathbb{R}^d$$ (our data distribution) and an invertable transformation $$ f: \mathbb{R}^d \mapsto \mathbb{R}^d$$ 

Then there is a random variable $$Z \in \mathbb{R}^d$$ which $$f$$ maps $$X$$ to. 

Furthermore,

$$P(X = x) = P(f(X) = f(x)) = P(Z = z)\tag{0}$$

Now consider some interval $$\beta$$ over $$X$$. Then there exists some interval $$\beta^{\prime}$$ over $$Z$$ such that, 

$$P(X \in \beta) = P(Z \in \beta^{\prime})\tag{1}$$

$$\int_{\beta} p_X dx = \int_{\beta^{\prime}} p_Z dz\tag{2}$$

For the sake of simplicity, we consider a single region.

$$ dx \cdot p_X(x) = dz \cdot p_Z(z) \tag{3}$$

$$ p_X(x) = \mid\dfrac{dz}{dx}\mid \cdot p_Z(z) \tag{4}$$

*Note:* We apply the absolute value to maintain the equality since by the probability axioms $$p_X$$ and $$p_Z$$ will always be positive.

$$ p_X(x) = \mid\dfrac{df(x)}{dx}\mid \cdot p_Z(f(x)) \tag{5}$$

$$ p_X(x) = \mid det(\dfrac{df}{dx}) \mid \cdot p_Z(f(x)) \tag{6}$$ 

*Note:* We use the determinant to generalize to the multivariate case ($$d > 1$$)

$$ \log(p_X(x)) = \log(\mid det(\dfrac{df}{dx}) \mid) + \log(p_Z(f(x))) \tag{7}$$ 

Tada! To model our random variable $$X$$, we need to maximize the right-hand side of equation (7). 

Breaking the equation down:
- $$ \log(\mid det(\dfrac{df}{dx}) \mid) $$ is the amount of stretch/change $$f$$ applies to the probability distribution $$p_X$$. 
    - This term is the log determinant of the Jacobian matrix ($$\dfrac{df}{dx}$$). We refer to the determinant of the Jacobian matrix as the Jacobian.

- $$\log(p_Z(f(x)))$$ constrains $$f$$ to transform $$x$$ to the distribution $$p_Z$$. 

Since there are no constraints on $$Z$$ we can choose $$p_Z$$! Usually, we choose $$p_Z$$ to be gaussian. 

Now I know what your thinking, as a reader of this blog you strive for greatness and say, 
> 'Brennan, a single function does not satisfy me. I have a hunger for more.'  

## Applying multiple functions sequentially 

Fear not my readers! I will show you how we can sequentially apply multiple functions.

Let $$z_n$$ be the result of sequentially applying $$n$$ functions to $$x \sim p_X$$.

$$ z_n = f_n \circ \dots \circ f_1(x) \tag{8}$$

$$ f = f_n \circ \dots \circ f_1 \tag{9}$$

Using the handy dandy chain rule, we can modify equation (7) with equation (8) to get equation (10) as follows. 

$$ \log(p_X(x)) = \log(\mid det(\dfrac{df}{dx}) \mid) + \log(p_Z(f(x))) \tag{7}$$ 

$$ \log(p_X(x)) = \log(\prod_{i=1}^{n} \mid det(\dfrac{dz_i}{dz_{i-1}}) \mid) + \log(p_Z(f(x)))\tag{10}$$

Where $$x \triangleq z_0$$ for conciseness.

$$ \log(p_X(x)) = \sum_{i=1}^{n} \log(\mid det(\dfrac{dz_i}{dz_{i-1}}) \mid) + \log(p_Z(f(x))) \tag{11}$$

We want the Jacobian term to be easy to compute since we will need to compute it $$n$$ times.

To efficiently compute the Jacobian, the functions $$f_i$$ (corresponding to $$z_i$$) are chosen to have a lower or upper triangular Jacobian matrix. Since the determinant of a triangular matrix is the product of its diagonal, which is easy to compute.

Now that you understand the general theory of Normalizing flows, lets flow through some PyTorch code.

# The Family of Flows 

For this post we will be focusing on, real-valued non-volume preserving flows (R-NVP) {% cite dinh2016density --file normflow %}. 

Though there are many other flow functions out and about such as NICE {% cite dinh2014nice --file normflow %}, and GLOW {%cite kingma2018glow --file normflow %}. For keeners wanting to learn more, I will show you to the 'More Resources' section at the bottom of this post which includes blog posts with more flows which may interest you. 

# R-NVP Flows 

We consider a single R-NVP function $$f: \mathbb{R}^d \rightarrow \mathbb{R}^d$$, with input $$\mathbf{x} \in \mathbb{R}^d$$ and output $$\mathbf{z} \in \mathbb{R}^d$$. 

To quickly recap, in order to optimize our function $$f$$ to model our data distribution $$p_X$$, we want to know the forward pass $$f$$, and the Jacobian $$\mid det(\dfrac{df}{dx}) \mid$$. 

We then will want to know the inverse of our function $$f^{-1}$$ so we can transform a sampled latent-value $$z \sim p_Z$$ to our data distribution $$p_X$$, generating new samples!

## Forward Pass

$$f(\mathbf{x}) = \mathbf{z}\tag{12}$$

The forward pass is a combination of copying values while stretching and shifting the others. First we choose some arbitrary value $$k$$ which satisfies $$0 < k < d$$ to split our input.

R-NVPs forward pass is then the following 

$$ \mathbf{z}_{1:k} = \mathbf{x}_{1:k} \tag{13}$$

$$ \mathbf{z}_{k+1:d} = \mathbf{x}_{k+1:d} \odot \exp(\sigma(\mathbf{x}_{1:k})) + \mu(\mathbf{x}_{1:k})\tag{14}$$

Where $$\sigma, \mu: \mathbb{R}^k \rightarrow \mathbb{R}^{d-k}$$ and are any arbitrary functions. Hence, we will choose $$\sigma$$ and $$\mu$$ to both be deep neural networks. Below is PyTorch code of a simple implementation.

<script src="https://gist.github.com/gebob19/1c10929c2b8a7089321e29c4c33dca4a.js"></script>

## Log Jacobian

The Jacobian matrix $$\dfrac{df}{d\mathbf{x}}$$ of this function will be 

$$\begin{bmatrix}I_d & 0 \\
\frac{d z_{k+1:d}}{d \mathbf{x}_{1:k}} &   \text{diag}(\exp[\sigma(\mathbf{x}_{1:k})])   \end{bmatrix}  \tag{15}$$

The log determinant of such a Jacobian Matrix will be 

$$\log(\det(\dfrac{df}{d\mathbf{x}})) = \log(\prod_{i=1}^{d-k} \mid\exp[\sigma_i(\mathbf{x}_{1:k})]\mid) \tag{16}$$

$$\log(\mid\det(\dfrac{df}{d\mathbf{x}})\mid) = \sum_{i=1}^{d-k} \log(\exp[\sigma_i(\mathbf{x}_{1:k})]) \tag{17}$$

$$\log(\mid\det(\dfrac{df}{d\mathbf{x}})\mid) = \sum_{i=1}^{d-k} \sigma_i(\mathbf{x}_{1:k}) \tag{18}$$

<script src="https://gist.github.com/gebob19/8dc1fe38b73fd350ff63b81f5947111a.js"></script>

## Inverse 

$$f^{-1}(\mathbf{z}) = \mathbf{x}\tag{19}$$

One of the benefits of R-NVPs compared to other flows is the ease of inverting $$f$$ into $$f^{-1}$$, which we formulate below using the forward pass of equation (14)

$$ \mathbf{x}_{1:k} = \mathbf{z}_{1:k} \tag{20}$$

$$ \mathbf{x}_{k+1:d} = (\mathbf{z}_{k+1:d} - \mu(\mathbf{x}_{1:k})) \odot \exp(-\sigma(\mathbf{x}_{1:k})) \tag{21}$$

$$ \Leftrightarrow \mathbf{x}_{k+1:d} = (\mathbf{z}_{k+1:d} - \mu(\mathbf{z}_{1:k})) \odot \exp(-\sigma(\mathbf{z}_{1:k})) \tag{22}$$

<script src="https://gist.github.com/gebob19/4458074fa1e804ad14e704a4e246c3ec.js"></script>

## Summary

And voilà, the recipe for R-NVP is complete! 

To summarize we now know how to compute $$f(\mathbf{x})$$, $$\log(\mid\det(\dfrac{df}{d\mathbf{x}})\mid)$$, and $$f^{-1}(\mathbf{z})$$. 

Below is the full jupyter notebook with PyTorch code for model optimization and data generation.

[Jupyter Notebook](https://github.com/gebob19/introduction_to_normalizing_flows)

*Note:* In the notebook the multilayer R-NVP flips the input before a forward/inverse pass for a more expressive model. 

### Optimizing Model

$$ \log(p_X(x)) = \log(\mid det(\dfrac{df}{dx}) \mid) + \log(p_Z(f(x))) $$ 

$$ \log(p_X(x)) = \sum_{i=1}^{n} \log(\mid det(\dfrac{dz_i}{dz_{i-1}}) \mid) + \log(p_Z(f(x)))$$

<script src="https://gist.github.com/gebob19/7440c0c0473749f7c3fed67ee3e25962.js"></script>

### Generating Data from Model

$$ z \sim p_Z $$

$$ x_{gen} = f^{-1}(z) $$

<script src="https://gist.github.com/gebob19/f453a654da8ff5ecd41978b9ce6b9fc8.js"></script>

# Conclusion

In summary, we learned how to model a data distribution to a chosen latent-distribution using an invertible function $$f$$. We used the change of variables formula to discover that to model our data we must maximize the Jacobian of $$f$$ while also constraining $$f$$ to our latent-distribution. We then extended this notion to sequentially applying multiple functions $$f_n \circ \dots \circ f_1(x)$$. Lastly, we learned about the theory and implementation of the R-NVP flow. 

Thanks for reading! 

Question? Criticism? Phrase? Advice? Topic you want to be covered? Leave a comment in the section below!

Want more content? Follow me on [Twitter](https://twitter.com/brennangebotys)! 

# References 

{% bibliography --cited --file normflow %}

## More Resources

- Indepth analysis of more recent flows: [https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)

- More flows and their equations: [http://akosiorek.github.io/ml/2018/04/03/norm_flows.html](http://akosiorek.github.io/ml/2018/04/03/norm_flows.html)

- Tensorflow Normalizing Flow Tutorial: [https://blog.evjang.com/2018/01/nf1.html](https://blog.evjang.com/2018/01/nf1.html)

- Video resource on the change of variables formulation: [https://www.youtube.com/watch?v=OeD3RJpeb-w](https://www.youtube.com/watch?v=OeD3RJpeb-w)
