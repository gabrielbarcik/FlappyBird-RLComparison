## Playing Flappy Bird Using Deep Q Learning DQN and Proximal Policy Optimization PPO

This repository aims to analyze and compare the performance of two agents based on PPO and DQN algorithms playing the popular game Flappy Bird. 

We have adapted an existing environment based on pygame() FlappyBird to the known RL library gym, developped by OpenAI. 

Our agents are based on the algorithms developped by openAI and made avaialable in the github of 'stable-baselines'.

We also train the agent using the DQN architecture implemented using tensorflow for comparison

### Benchmarking
To test and bechmark our implementation we used the Google Compute Platform. It was the ideal solution, as the training of these networks are usually very compute intensive and higly dependent on specifific hardware, such as GPUs

### Aknowledgements
The work described here wouldn't be possible without the following repositories
1. hill-a/stable-baselines
2. floodsung/Gym-Flappy-Bird
3. yenchenlin/DeepLearningFlappyBird: 
And all the papers described in our report.



