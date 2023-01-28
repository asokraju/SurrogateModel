# Developing Surrogate models for dynamical systems with improved accuracy while decreasing the computational cost and complexity, using deep neural networks

## Problem formulation:
Consider the following system
$$x_{k+1}=f(x_k, u_k)$$
Find suitable mappings $y=\phi(x)$ and $w=\psi(x,u)$ such that the above system can be represented as Linear Time-Invariant system (LTI):
$$y_{k+1}=Ay_{k}+B w_k.$$

Since these are nonlinear maps, we also need to find their inverse mappings, $x=\phi^{-1}(y)$ and $u=\psi^{-1}(y,w)$.

The problem of developing surrogate models for dynamical systems is an important task in many fields of science and engineering. The goal is to find a simplified, approximate model of a complex system that captures its essential dynamics, while reducing the computational cost and complexity. The use of deep neural networks (DNNs) for this purpose has gained significant attention in recent years due to their ability to learn complex nonlinear mappings.

The specific problem I've formulated is to find mappings $y=phi(x)$ and $w=psi(x,u)$ that allow the nonlinear system $x_{k+1}=f(x_k, u_k)$ to be represented as a Linear Time-Invariant system (LTI) $y_{k+1}=Ay_{k}+B w_k$. Additionally, I aim to find the inverse mappings $x=phi^-1(y)$ and $u=psi^-1(y,w)$. These mappings allow us to transform the input-output behavior of the original system into a more interpretable and computationally efficient LTI system, which can be used for control, prediction, or other purposes.

To approximate these mappings, I propose to use DNNs, specifically a network structure inspired by autoencoders. Autoencoders are a type of neural network that are trained to learn a compact, low-dimensional representation of the input data, known as the bottleneck or latent representation. This structure is particularly well-suited to the problem of model reduction because it allows the network to learn a compact, interpretable representation of the input-output behavior of the original system. Motivated by the autoencoders, I use the following network

<img src="https://github.com/asokraju/SurrogateModel/blob/e8876c9a2e79792d6a2da7ea15e952a0848a014d/tools/nn.PNG" width="500" align="center">

The objective is to minimize the following loss:
$$L(x_k, u_k, x_{k+1}) = \|x_{k+1}-\hat x_{k+1}\|_2^2 + \|u_k-\hat{u}_k\|_2^2$$
which can be simplifed to as

$$L(x_k, u_k, x_{k+1})=\|x_{k+1}-\hat{\phi}^{-1}\left(A\hat{\phi}(x_k)+B\hat{\psi}(x_k, u_k)\right)\|_2^2 + \|u_k-\hat \psi^{-1}(\hat \phi(x_k),\hat \psi(x_k,u_k))\|_2^2$$

where $\hat{\psi}$ represents the approximated function of $\psi$. Note that the  loss function only needs current state ($x_k$), action ($u_k$) and the next state ($x_{k+1}$).


Used  tensorflow subclassing to model and train the neural network. Below represents the neural network snap-shot.

<img src="https://github.com/asokraju/SurrogateModel/blob/7e983e5db480da07a210fac92aec8a233fd32ac6/tools/index.png" width="400" align="center">

In order to train the network, I would need to collect a large dataset of inputs and outputs from the original system, which can be used to train the network to learn the desired mappings. Once the network is trained, it can be used to predict the outputs of the original system for new inputs, or to map inputs and outputs to the reduced LTI system.

It is important to note that the accuracy of the surrogates model is highly dependent on the quality of the training data and the architecture of the neural network. Choosing appropriate neural network architectures and training algorithms, as well as pre-processing and cleaning the data is crucial for achieving good performance.