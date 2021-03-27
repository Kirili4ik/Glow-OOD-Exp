### Glow Performance on Out-of-Domain data experiment in PyTorch

This repository is dedicated to training and experimenting with glow model described in [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)

The model is adapted from [rosinality](https://github.com/rosinality/glow-pytorch) and is written in GLOW.py, trained on CelebA dataset. There are some random generated examples with different temperatures for sampling:

![alt text](https://github.com/Kirili4ik/Glow-PyTorch/blob/main/pictures/samples.png "Samples")

The main experiment is described in [Why Normalizing Flows Fail to Detect Out-of-Distribution Data](https://arxiv.org/abs/2006.08545) and can be found in main_exp.ipynb. For OOD pictures SVHN dataset is used. 

Here you can see the distributions of LogLikelihoods:

![alt text](https://github.com/Kirili4ik/Glow-PyTorch/blob/main/pictures/distributions.png "Distributions")

We can kinda proove authors' findings:

> We argue that flows are biased towards learning graphical properties of the data such as local pixel correlations (e.g. nearby pixels usually have similar colors) rather than semantic properties of the data (e.g. what objects are shown in the image).

There are pictures with the highest LogLikelihood in OOD data and with least LogLikelihoods in train/test sets compared:

![alt text](https://github.com/Kirili4ik/Glow-PyTorch/blob/main/pictures/exp1.png "exp1")

And there are the reversed (lowest for OOD data and highest for train/test sets):

![alt text](https://github.com/Kirili4ik/Glow-PyTorch/blob/main/pictures/exp2.png "exp2")

It is not hard to see that blurry images from Out-of-Domain with "usually similar colors for nearby pixels" have the highest LogLikelihoods even though there are no faces at all! On the other hand, faces with contrast backgrounds of a lot of different colors score the lowest by the model.

Thx [mseitzer](https://github.com/mseitzer/pytorch-fid) for implementing Inception.py for FID calculation.

Thx [WanbB](https://wandb.ai/) for convenient logging and beautiful report writing tool.
