# Summary
In this report, I describe training result using **DQN**.

# Learning Algorithm
Source code of learning algorithm is placed in `model.py' and 'agent.py'.

## 1. model.py
this is the deep learning model, I used two layer neural network as Q-Value Estimator.
Hidden layers are composed of ``State -> 64 -> ReLU -> 64 -> ReLU -> Action``

## 2. agent.py
agent.py is the reinforment learning agent to provide action for state



# parameters after training
the parameters from training saved in ./checkpoint.pth
