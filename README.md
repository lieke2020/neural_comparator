# neural_comparator
Unsupervised Anti-Hebbian network for biological matching (based on https://doi.org/10.1162/NECO_a_00424)

This file is Python 3.8 code based on Ludueña, G. A., & Gros, C. (2013). 
A self-organized neural comparator. Neural Computation, 25(4), 1006–1028. 
https://doi.org/10.1162/NECO_a_00424
The network is able to learn a sclar match value that is high when two inputs are different and low when two inputs are the same.
There is no feedback, the learning fully depends on the pre- and postsynaptic activities.

This code is my own and was not written in collaboration with G. A. Ludueña and C. Gros.
Questions or comments regarding this file can be addressed to lieke.c@nin.knaw.nl

-------------------
The class defined in this file implements the neural comparator Matchnet;
Entry point for the model is the 'step()' function which defined what the matchnet
does during a single time step; this entails: 

1-  The construction of an input vector based on the trial type (match or non-match) 

2-  A feedforward pass that computes the match value x4

3-  A local anti-Hebbian learning rule that updates the weights of each neuron

The cosine similarity is used to inspect when two vectors are defined to be different or similar.
This code plots the network output and cosine similarity over the last section of the timesteps for evaluation.

The addition of the CIFAR-10 dataset is my own and is outside the scope of the paper.
I have extracted 10000 activity vectors of the second fully connected layer from the PyTorch
network that was improved by Caleb Woy. 
https://www.kaggle.com/whatsthevariance/pytorch-cnn-cifar10-68-70-test-accuracy
It is a first attempt to learn a neural comparator with real-life data.

@author: Lieke Ceton
