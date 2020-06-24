import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import copy

tf.keras.backend.set_floatx('float64')

EPSILON = 1e-8


class Actor(Model):

    def __init__(self, output_dim, hidden_dims):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_layers = [layers.Dense(layer, activation=tf.nn.relu) for layer in hidden_dims]
        self.mean_layer = layers.Dense(self.output_dim)
        # Standard deviation doesn't depend on state
        self.log_stddev = tf.Variable(initial_value=0.0, trainable=True)

    def call(self, state):
        # Pass input through all hidden layers
        inp = state
        for layer in self.hidden_layers:
            inp = layer(inp)

        # Generate mean output
        mu = self.mean_layer(inp)

        # Convert log stddev to stddev
        sigma = tf.exp(self.log_stddev)

        # Use re-parameterization trick to stochastically sample action from
        # the policy network. First, sample from a Normal distribution of
        # sample size as the action and multiply it with stdev
        dist = tfp.distributions.Normal(mu, sigma)
        action = dist.sample()

        # Calculate log probability of the action
        log_pi = dist.log_prob(action)

        # TODO: use tanh squashing trick as in SAC if required

        return action, log_pi

    @property
    def trainable_variables(self):
        variables = []
        for layer in self.hidden_layers:
            variables.extend(layer.trainable_variables)
        variables.extend(self.mean_layer.trainable_variables)
        variables.append(self.log_stddev)
        return variables


class Critic(Model):

    def __init__(self, hidden_dims):
        super().__init__()
        self.hidden_layers = [layers.Dense(layer, activation=tf.nn.relu) for layer in hidden_dims]
        self.output_layer = layers.Dense(1, activation=tf.nn.relu)

    def call(self, state):
        # Pass input through all hidden layers
        inp = state
        for layer in self.hidden_layers:
            inp = layer(inp)

        # Generate mean output
        value = self.output_layer(inp)

        return value

    @property
    def trainable_variables(self):
        variables = []
        for layer in self.hidden_layers:
            variables.extend(layer.trainable_variables)
        variables.extend(self.output_layer.trainable_variables)
        return variables


class PPOClipped:

    def __init__(self,
                 writer,
                 action_dim,
                 policy_hidden_dims=[64, 64],
                 value_hidden_dims=[64, 64],
                 learning_rate=1e-4,
                 gamma=0.99):
        """Implementation of PPO with Clipped object"""
        self.policy = Actor(action_dim, policy_hidden_dims)
        self.value = Critic(value_hidden_dims)

        self.writer = writer
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate)



