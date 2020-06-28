import gym
import argparse
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf

from ppo import PPOClipped
from env import ContinuousCartPoleEnv

tf.keras.backend.set_floatx('float64')

logging.basicConfig(level='INFO')

parser = argparse.ArgumentParser(description='SAC')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--env_name', type=str, default='MountainCarContinuous-v0',
                    help='name of the gym environment with version')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--verbose', type=bool, default=False,
                    help='log execution details')
parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch sample size for training')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to run backprop in an episode')
parser.add_argument('--model_path', type=str, default='../data/models/',
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default=f'{str(datetime.utcnow().date())}-{str(datetime.utcnow().time())}',
                    help='name of the saved model')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')


if __name__=='__main__':

    # Parse the arguments.
    args = parser.parse_args()

    # Instantiate the environment.
    tf.random.set_seed(args.seed)

    writer = tf.summary.create_file_writer(args.model_path + args.model_name + '/summary')

    # Instantiate the environment.
    if args.env_name == "ContinuousCartPoleEnv":
        env = ContinuousCartPoleEnv()
    else:
        env = gym.make(args.env_name)
    env.seed(args.seed)
    state_space = env.observation_space.shape[0]
    # TODO: fix this when env.action_space is not `Box`
    action_space = env.action_space.shape[0]

    # Initialize policy and value function parameters.
    ppo = PPOClipped(writer, action_space)

    # Reset global tracking variables

    # Start epoch.
    for epoch in range(args.epochs):

        # Reset epoch variables.
        current_state = env.reset()
        done = False
        episode_reward = 0.0
        transitions = []

        # Run an episode.
        while not done:

            if args.render:
                env.render()

            # sample action from policy
            current_state_ = np.array(current_state, ndmin=2).reshape(1, -1)
            action, log_pi = ppo.policy(current_state_)

            action_ = action.numpy()[0]
            log_pi_ = log_pi.numpy()[0]

            # Execute action, observe next state and reward
            next_state, reward, done, _ = env.step(action_)
            episode_reward += reward
            transitions.append([current_state_, action_, log_pi_, reward,
                                next_state])

            current_state = next_state

        print(f"Episode Reward: {episode_reward}")

        # Update policy and value function parameter.
        policy_loss, value_loss = ppo.update(transitions)

        # If the KL divergence between the old and new policy crosses threshold
        # then do early stoppping.

        # Log epoch summary.
        print(f"Epoch: {epoch}")
        print(f"Policy Loss: {policy_loss}")
        print(f"Value Loss: {value_loss}")

        # Save model.

