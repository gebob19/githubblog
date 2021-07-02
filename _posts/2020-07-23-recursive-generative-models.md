---
title:  "Generative Models: Recursive Edition"
description: "In this tutorial we look at generative models which use recursive networks (RNN, LSTM, GRU, etc.) to generate time-series data."
categories: example
mathjax: true
---

Generative Adversarial Networks (GANs) have shown great results in computer vision but how do they perform when applied to time-series data? Following this, do Convolutional Neural Networks (CNNs) or do Recursive Neural Networks (RNNs) achieve the best results? 

In this post, we discuss GAN implementations which aim to generate time-series data including, C-RNN-GANs {%cite mogren2016c --file rnn_gen %}, RC-GANs {%cite esteban2017real --file rnn_gen %} and TimeGANs {%cite yoon2019time --file rnn_gen %}. Lastly, we implement RC-GAN and generate stock data. 

# Basic GAN Intro 

There are many great resources on GANs so I only provide an introduction here. 

GANs include a generator and a discriminator. The generator takes latent variables as input (usually values sampled from a normal distribution) and outputs generated data. The discriminator takes the data (real or generated/fake) as input and learns to discriminate between the two. 

The gradients of the discriminator are used both to improve the discriminator and improve the generator.

Here's a nice picture for the more visually inclined from a wonderful [blog](https://robotronblog.com/2017/09/05/gans/).

<div align="center">
<img src="https://robotronblog.files.wordpress.com/2017/09/g1.jpg" alt="GAN-description" width="600" class="center"/>
</div>

and a nice equation for the more equation-y inclined where $$D$$ is the discriminator and $$G$$ is the generator.

$$\min_G \max_D \mathbb{E}_{x\sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z\sim p_z(z)}[\log(1 - D(G(z)))]$$

# C-RNN-GAN

The first paper we investigate is 'Continuous recurrent neural networks with adversarial training' (C-RNN-GAN) {%cite mogren2016c --file rnn_gen %}. 

The generative model takes a latent variable concatenated with the previous output as input. Data is then generated using an RNN and a fully connected layer.

<!-- <div align="center">
<img src="https://gebob19.github.io/assets/recursive_gan/c-rnn.png" alt="C-RNN-GAN" width="600" class="center"/>
</div> -->
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/c-rnn6.png" alt="C-RNN-GAN" class="center"/>
</div>

Note: In the paper, `start` is initialized from Uniform [-1, 1].

The discriminator is a bi-directional RNN followed by a fully connected layer. 

The generator is implemented in PyTorch as follows, 

<script src="https://gist.github.com/gebob19/b379123b493fb5db035d93c171947e0b.js"></script>

# RC-GAN

The next paper is 'Real-Valued (Medical) Time Series Generation With Recurrent Conditional GANs' {%cite esteban2017real --file rnn_gen %}.

RC-GAN's generator's input consists of a sequence of latent variables. 

The paper also introduces a 'conditional' GAN, where conditional/static information ($$c$$) is concatenated to the latent variables and used as input to improve training.

<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/cr-gan.png" alt="CR-GAN" class="center"/>
</div>

The discriminator is the same as in C-RNN-GAN but is not bi-directional.

The implementation is as follows,

<script src="https://gist.github.com/gebob19/bcbe223c0ae39412ebe93a6fe8c23048.js"></script>


# Time-GAN

TimeGan {%cite yoon2019time --file rnn_gen %} is the most recent approach, which aims to maximize the similarities between embeddings of real data and fake data. 

First, the generator ($$G$$) creates embeddings ($$\hat{h_t} = G(\hat{h_{t-1}}, z_t)$$) from latent variables while the embedding network ($$E$$) encodes real data ($$h_t = E(h_{t-1}, x_t)$$). The Discriminator ($$D$$) then discriminates between real and fake embeddings. While the Recovery network ($$R$$) reconstructs the real data (creating $$\hat{x_t}$$) from its respective embedding. 

This leads to 3 losses 

- Embedding difference (Goal: Similar embeddings for real and fake data)

$$L_S = \mathbb{E}_{x_{1:T} \sim p} \sum_t || h_t - G(h_{t-1}, z_t) ||  $$

Notice: $$G$$ takes $$h_{t-1}$$ as input, NOT $$\hat{h_{t-1}}$$

- Recovery Score (Goal: meaningful embeddings for real data)

$$L_R = \mathbb{E}_{x_{1:T} \sim p} \sum_t ||x_t - \tilde{x_t} ||  $$

- Discriminator Score

$$L_U = \mathbb{E}_{x_{1:T} \sim p} \sum_t log(y_t) +  \mathbb{E}_{x_{1:T} \sim \hat{p}} \sum_t log(1 - \hat{y_t})$$

<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/timegan2.png" alt="Time-GAN" class="center"/>
</div>

Note: Similar to the previous paper, the paper talks about static/context features which can be used throughout the training process (E.g the label (1, 2, ..., 9) when generating the MNIST dataset). To simplify this post, I chose to sweep this little detail under the blogpost rug. 

To complete the optimization, the total loss is weighed by two hyperparameters $$\lambda$$ and $$\eta$$ (whos values were found to be non-significant). Leading to the following...

$$\min_{E, R} \lambda L_S + L_R $$ 

$$\min_{G} \eta L_S + \max_{D} L_U $$ 

## Empirical Results

Below are the results comparing time-series focused, generative models. We can see that TimeGAN performs the best across all datasets with RC-GAN close behind. For a more detailed explanation of the data, refer to the paper.

<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/results.png" alt="table results from TimeGAN" class="center" width="400" height="100"/>
</div>


# RC-GAN + Stock Data 

Since both RC-GAN and TimeGAN show similar results and RC-GAN is a much simpler approach we will implement and investigate RC-GAN. 

### Generator and Discriminator 

<script src="https://gist.github.com/gebob19/201691dca85d9e766a9b5b896824dc44.js"></script>

### Training Loop 

<script src="https://gist.github.com/gebob19/4f95f82c80f8ff7f1122c5897a6db877.js"></script>


## Visualizing Stock Data 

Before we generate stock data, we need to understand how stock data is visualized. 

Every day, the price which the stock opened and closed at, and the highest and lowest price the stock reached that day is represented using a candlestick.

If the stock closed higher than it opened, the candle is filled green. If the stock closed lower than it opened, then the candle is filled red. 

Nice!

<div align="center" width="600" height="300">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/candlesticks.jpg" alt="candlestick_model" class="center" width="600" height="300"/>
</div>

### Examples

The model was trained with the GOOGLE price data split into 30-day parts (used in the TimeGAN paper).

Below are some generated data along with low-dimension analysis using T-SNE. 

<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/ex/s1.png" alt="examples" class="center" width="400" height="100"/>
</div>
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/ex/s2.png" alt="examples" class="center" width="400" height="100"/>
</div>
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/ex/s3.png" alt="examples" class="center" width="400" height="100"/>
</div>
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/ex/tsne.png" alt="examples" class="center" width="400" height="100"/>
</div>

Though it looks that the examples overlap through a T-SNE visualization, they do not always look realistic. 

<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/ex/s4.png" alt="tsne-overlap" class="center" width="400" height="100"/>
</div>

## Feature Association 

We can also investigate what the learned features associate with by shifting the axis values around in latent space. Since we trained our model with a $$z$$ dimension of 10 we can shift the value of each of these dimensions and see how it changes the generated stock data. 

### [Original Generated Data]
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/og.png" alt="original-data" class="center" width="700" height="200"/>
</div>

## Shifting Noise Axis Values [-1, -0.5, +0.5, +1]


### Index 0
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features0.png" alt="feature" class="center" width="700" height="200"/>
</div>

### Index 1
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features1.png" alt="feature" class="center" width="700" height="200"/>
</div>

### Index 2
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features2.png" alt="feature" class="center" width="700" height="200"/>
</div>

### Index 3
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features3.png" alt="feature" class="center" width="700" height="200"/>
</div>

### Index 4
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features4.png" alt="feature" class="center" width="700" height="200"/>
</div>

### Index 5
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features5.png" alt="feature" class="center" width="700" height="200"/>
</div>

### Index 6
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features6.png" alt="feature" class="center" width="700" height="200"/>
</div>

### Index 7
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features7.png" alt="feature" class="center" width="700" height="200"/>
</div>

### Index 8
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features8.png" alt="feature" class="center" width="700" height="200"/>
</div>

### Index 9
<div align="center" width="500" height="100">
<img src="https://raw.githubusercontent.com/gebob19/gebob19.github.io/source/assets/recursive_gan/features/features9.png" alt="feature" class="center" width="700" height="200"/>
</div>

There is also a [notebook](https://github.com/gebob19/RNN_stock_generation) which contains all the code needed to test this out for yourself!

If you enjoyed the post, feel free to follow me on [Twitter](https://twitter.com/brennangebotys) for updates on new posts! 

# References 

{% bibliography --cited --file rnn_gen %}
