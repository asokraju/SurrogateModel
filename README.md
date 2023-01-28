# SurrogateModel
Developing Surrogate models for Physics based systems with improved accuracy while decreasing the computational cost and complexity, using deep neural networks.

Problem formulation:
Consider the following system
$$x_{k+1}=f(x_k, u_k)$$
Final suitable mappings $y=\phi(x)$ and $w=\psi(x,u)$ such that the above system can be represented as Linear Time-Invariant system (LTI):
$$y_{k+1}=Ay_{k}+B w_k.$$

Since these are nonlinear maps, we also need to find their inverse mappings, $x=\phi^{-1}(y)$ and $u=\psi^{-1}(y,w)$.

We aim to use deep-neural networks to approximate these functions. Motivated by the autoencoders, we use the following network

<img src="https://github.com/asokraju/SurrogateModel/blob/e8876c9a2e79792d6a2da7ea15e952a0848a014d/tools/nn.PNG" width="400" align="center">

The objective is to minimize the following loss:
$$L(x_k, u_k, x_{k+1}) = \|x_{k+1}-\hat x_{k+1}\|_2^2 + \|u_k-\hat{u}_k\|_2^2$$
which can be simplifed to as

$$L(x_k, u_k, x_{k+1})=\|x_{k+1}-\hat{\phi}^{-1}\left(A\hat{\phi}(x_k)+B\hat{\psi}(x_k, u_k)\right)\|_2^2 + \|u_k-\hat \psi^{-1}(\hat \phi(x_k),\hat \psi(x_k,u_k))\|_2^2$$

where $\hat{\psi}$ represents the approximated function of $\psi$. Note that the  loss function only needs current state ($x_k$), action ($u_k$) and the next state ($x_{k+1}$).


Used  tensorflow subclassing to model and train the neural network. Below represents the neural network snap-shot.

<img src="https://github.com/asokraju/SurrogateModel/blob/7e983e5db480da07a210fac92aec8a233fd32ac6/tools/index.png" width="400" align="right">



