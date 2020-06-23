# Proximal Policy Optimization
#### Implementation of PPO Algorithm in TF2

Notes: https://shivamshakti.dev/posts/ppo

## Usage

- Create a virtual environment for Python (I use [this](https://gist.github.com/shakti365/c8384d421ace17a6586f5b8733d5705c) setup)

- Install the dependencies

  ```
  pip install -r requirements.txt
  ```

- Run the training script

  ```
  cd src
  python main.py # Uses `MountainCarContinuous-v0` by default
  ```

- Run the evaluation script

  ```
  python play.py --model_name <PATH_TO_SAVED_MODEL>
  ```

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)

- [Open AI: Spinning Up - PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

- [Reinforcement Learning Coach - PPO](https://nervanasystems.github.io/coach/components/agents/policy_optimization/ppo.html)
