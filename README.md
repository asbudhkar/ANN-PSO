# ANN-PSO

## Traning Neural Network with Particle Swarm Optimization instead of Gradient Descent
- Motivation
  - Gradient Descent requires differentiable activation function to calculate derivates making it slower than feedforward
  - To speed up backprop lot of memory is required to store activations
  - Backpropagation is strongly dependent on weights and biases initialization. A bad choice can lead to stagnation at local minima and so a suboptimal solution is found.
  - Backpropagation cannot make full use of parallelization due to its sequential nature

- Advantages of PSO
  - PSO does not require a differentiable or continous function
  - PSO leads to better convergence and is less likely to get stuck at local minima
  - Faster on GPU

## Hybrid Approach PSO+Backprop
- The performance of PSO deteriorates as the dimensionality of search space increases. 
- For huge network use PSO for initialization and then use gradient descent

## Variants
- Local best
  - Robust against local optimas
  - Slow convergence
- Global best
  - Fast convergence

## Environment
 - Ubuntu 
 - Nvidia GPU with CUDA 9
 - Python3
 - Tensorflow
 - QtCreator-QML

## How to run
Testing with IRIS dataset
- GUI   
Execute nnui in terminal

- Terminal  
python clinn.py [OPTION]  
  - --gbest: GlobalBest Factor - Global best for PSO. The higher it is, the more the
particles stick together. Control Particles from going from too further.  
  - --lbest: LocalBest Factor - Local best for PSO. The higher it is, the less the
particle will move towards the global best. And lesser chance of convergence. Set it to higher values if you wish to increase the exploration. Set a high value if using for initialization and not training.  
  - --veldec: Velocity Decay: Decay in velocity after each position update. The decay prevents the network weights
going too far away.  
  - --mv: Velocity Max - Maximum velocity for a particle along a dimension.  
  - --mvdec: Velocity Max Decay - Multiplier for Max Velocity with each update.    
  - --lr: Learning Rate: Learning Rate if Hybrid Approach. A much higher learning rate can be used with hybrid approach for learning weights.  
  - --pno: Number of Particles - Number of particles.  
  - --iter: Number of Iterations.  
  - --bs: Batchsize for training.  
  - --hybrid: Use Adam along with PSO.  
  - --lr: Learning Rate if Hybrid Approach.  
  - --hl: Hidden layers for the network.  
  - --lbpso: Use Local Best Variant of PSO.  
