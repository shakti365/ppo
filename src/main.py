import gym
import argparse
import logging
import numpy as np
from datetime import datetime
import tensorflow as tf
import json

from ppo import PPOClipped
from env import ContinuousCartPoleEnv

tf.keras.backend.set_floatx('float64')

logging.basicConfig(level='INFO')

parser = argparse.ArgumentParser(description='PPO')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--env_name', type=str, default='MountainCarContinuous-v0',
                    help='name of the gym environment with version')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--verbose', type=bool, default=False,
                    help='log execution details')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to run backprop in an episode')
parser.add_argument('--model_path', type=str, default='../data/models/',
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default=f'{str(datetime.utcnow().date())}-{str(datetime.utcnow().time())}',
                    help='name of the saved model')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--lambd', type=float, default=0.99,
                    help='discount factor for future rewards')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--epsilon', type=float, default=0.2,
                    help='clipping value')
parser.add_argument('--horizon', type=int, default=100,
                    help='max step per episode to calculate TD return')


if __name__=='__main__':

    # Parse the arguments.
    args = parser.parse_args()

    # Instantiate the environment.
    tf.random.set_seed(args.seed)

    writer = tf.summary.create_file_writer(args.model_path + args.model_name + '/summary')

    with open(args.model_path + args.model_name + '/param.json', 'w') as f:
        f.write(json.dumps(args.__repr__()))


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
    ppo = PPOClipped(writer, action_space, learning_rate=args.learning_rate,
                     epsilon=args.epsilon, gamma=args.gamma, lambd=args.lambd)

    # Reset global tracking variables

    # Start epoch.
    for epoch in range(args.epochs):

        # Reset epoch variables.
        current_state = env.reset()
        done = False
        episode_reward = 0.0
        transitions = []
        episode_step = 0

        # Run an episode.
        while not done:

            if args.render:
                env.render()

            # sample action from policy
            current_state_ = np.array(current_state, ndmin=2).reshape(1, -1)
            action, log_pi, mu, sigma = ppo.policy(current_state_)

            if args.env_name == "ContinuousCartPoleEnv":
                action_ = np.clip(action.numpy()[0], -1, 1)
            else:
                action_ = action.numpy()[0]
            log_pi_ = log_pi.numpy()[0]

            # Execute action, observe next state and reward
            next_state, reward, done, _ = env.step(action_)
            episode_reward += reward

            if done:
                not_terminal = 0
            else:
                not_terminal = 1

            next_state_ = np.array(next_state, ndmin=2).reshape(1, -1)
            transitions.append([current_state_, action_, log_pi_, mu, sigma, reward,
                                next_state_, not_terminal])

            if args.verbose:
                logging.info(f"""\
                             current_state: {current_state_}\n\
                             action: {action_}\n\
                             reward: {reward}\n\
                             next_state: {next_state_}\n""")

            current_state = next_state

            episode_step += 1
            if episode_step == args.horizon:
                break

        # Update policy and value function parameter.
        policy_loss, value_loss = ppo.update(transitions, epoch)

        # Log summaries
        with writer.as_default():
            tf.summary.scalar("policy_loss", policy_loss.numpy(), epoch)
            tf.summary.scalar("value_loss", value_loss.numpy(), epoch)
            tf.summary.scalar("episode_reward", episode_reward, epoch)

        # If the KL divergence between the old and new policy crosses threshold
        # then do early stoppping.

        # Log epoch summary.
        print(f"Epoch: {epoch}")
        print(f"Policy Loss: {policy_loss}")
        print(f"Value Loss: {value_loss}")
        print(f"Episode Reward: {episode_reward}")

        # Save model.
        ppo.policy.save_weights(args.model_path + args.model_name + '/model')

