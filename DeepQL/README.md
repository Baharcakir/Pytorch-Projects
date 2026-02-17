# Deep Q-Learning with CartPole (PyTorch)

This project implements a **Deep Q-Network (DQN)** from scratch using **PyTorch** and trains it on the classic **CartPole-v1** environment from Gymnasium.

The goal is to understand how **reinforcement learning**, **experience replay**, and **target networks** work together to solve a control problem.

---

## Environment

**CartPole-v1 (from gymnasium)**

* Observation space: 4 continuous values
  *(cart position, cart velocity, pole angle, pole angular velocity)*
* Action space: 2 discrete actions
  *(push left, push right)*
* Episode ends when the pole falls or time limit is reached.

Loaded via:

```python
env = gym.make("CartPole-v1", render_mode="human")
```

---

## State Representation

Each state is a vector of length **4**:

```python
state, info = env.reset()
n_observations = len(state)
```

Converted to a PyTorch tensor:

```python
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
```

---

## Replay Memory (Experience Replay)

A replay buffer stores transitions:

```python
Transition = (state, action, next_state, reward)
```

Implemented using a deque:

```python
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
```

**Why?**

* Breaks correlation between consecutive samples
* Stabilizes training
* Improves sample efficiency

---

## DQN Model Architecture

A simple fully-connected neural network:

```
Input (4)
↓
Linear(4 → 128) + ReLU
↓
Linear(128 → 128) + ReLU
↓
Linear(128 → 2)
```

Implemented as:

```python
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
```

Outputs **Q-values** for each action.

---

## Epsilon-Greedy Action Selection

Balances **exploration vs exploitation**:

```python
eps = eps_end + (eps_start - eps_end) * exp(-steps_done / eps_decay)
```

* With probability ε → random action
* Otherwise → greedy action from policy network

This ensures early exploration and later exploitation.

---

## Target Network

Two networks are used:

* **Policy network** → updated every step
* **Target network** → slowly updated

Soft update rule:

```python
θ_target = τ * θ_policy + (1 - τ) * θ_target
```

This prevents unstable learning caused by moving targets.

---

## Optimization Step

For each batch:

1. Sample transitions from replay memory
2. Compute current Q-values
3. Compute target Q-values using target network
4. Minimize Huber loss

```python
loss = SmoothL1Loss(Q(s,a), r + γ * max(Q_target(s')))
```

---

## Training Setup

| Parameter       | Value       |
| --------------- | ----------- |
| Environment     | CartPole-v1 |
| Episodes        | 100         |
| Batch size      | 128         |
| Replay memory   | 10,000      |
| Discount (γ)    | 0.99        |
| Learning rate   | 1e-4        |
| Optimizer       | AdamW       |
| Epsilon start   | 0.9         |
| Epsilon end     | 0.05        |
| Epsilon decay   | 1000        |
| Target update τ | 0.005       |
| Device          | CPU         |

---

## Training Loop

At each step:

* Select action (epsilon-greedy)
* Interact with environment
* Store transition in memory
* Sample batch
* Optimize model
* Soft update target network

```python
memory.push(state, action, next_state, reward)
optimize_model()
```

---

## Visualization

Training progress is plotted as:

* Episode duration
* Moving average (last 100 episodes)

This shows how the agent improves over time.

---

## Results

After training:

* The agent learns to balance the pole
* Episode lengths steadily increase
* Typically reaches **200+ steps per episode**

Which means the task is essentially solved.

---

## Key Concepts Demonstrated

* Reinforcement Learning
* Q-Learning
* Deep Q-Networks (DQN)
* Experience Replay
* Target Networks
* Epsilon-Greedy Exploration
* Huber Loss
* Gradient Clipping
* Online learning loop

---

## Limitations

* Uses only CPU
* No Double DQN
* No Dueling Network
* No Prioritized Replay
* Fixed hyperparameters
* Only 100 episodes

---

## Possible Improvements

* Train on GPU (CUDA)
* Increase number of episodes (500–1000)
* Implement Double DQN
* Add Dueling Architecture
* Use Prioritized Replay
* Save & load trained models
* Plot reward instead of duration
* Add evaluation mode (no exploration)

---

## How to Run

Install dependencies:

```bash
pip install gymnasium torch matplotlib
```

Run the notebook:

```bash
jupyter notebook DeepQL.ipynb
```

---

## Conclusion

This project shows how a **Deep Q-Network can be built from scratch** using PyTorch.

It demonstrates that:

* Neural networks can approximate Q-functions
* Experience replay stabilizes learning
* Target networks prevent divergence
* Even simple architectures can solve classic RL problems

This is a strong foundation for moving on to:

* Double DQN
* Dueling DQN
* PPO / A2C
* Atari games
* Continuous control with DDPG / SAC
