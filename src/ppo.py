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
        self.log_stddev = tf.Variable(initial_value=0.0, trainable=True,
                                      dtype=tf.float64)

    def call(self, state, action=None):
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

        if action is None:
            action = dist.sample()

        # Calculate log probability of the action
        log_pi = dist.log_prob(action)

        return action, log_pi, mu, sigma

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
                 gamma=0.99,
                 lambd=0.9,
                 epsilon=0.2):
        """Implementation of PPO with Clipped object"""
        self.policy = Actor(action_dim, policy_hidden_dims)
        self.value = Critic(value_hidden_dims)

        self.writer = writer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambd = lambd
        self.epsilon = epsilon

        self.policy_optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        self.value_optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    def update(self, transitions, epoch):
        """Does a backprop on policy and value networks"""

        with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:

            R = 0.0
            policy_loss = 0.0
            policy_loss_ = []
            value_loss = 0.0
            value_loss_ = []
            num_samples = 0.0
            advantage = 0
            for transition in reversed(transitions):
                current_state, action, log_pi_old, _, _, reward, next_state, done = transition

                R = self.gamma * R + reward

                # GAE
                V_s = self.value(current_state)
                V_s_next = self.value(next_state)
                td_residual = reward + self.gamma * V_s_next * done - V_s

                advantage = self.gamma * self.lambd * advantage + td_residual

                # Compute policy loss
                _, log_pi, _, _ = self.policy(current_state, action)

                is_ratio = tf.exp(log_pi - log_pi_old)


                unclipped = is_ratio * advantage
                clipped = tf.cond(advantage < 0,
                                  lambda: tf.multiply(1-self.epsilon, advantage),
                                  lambda: tf.multiply(1+self.epsilon, advantage))

                policy_loss -= tf.math.minimum(clipped, unclipped)
                value_loss += tf.pow(V_s - R, 2)
                num_samples = 1

            value_objective = (1/num_samples) * value_loss
            policy_objective = (1/num_samples) * policy_loss

        policy_vars = self.policy.trainable_variables
        policy_grads = [tf.clip_by_value(grad, -1, 1) for grad in
                        tape.gradient(policy_objective, policy_vars)]
        self.policy_optimizer.apply_gradients(zip(policy_grads, policy_vars))

        with self.writer.as_default():
            for var, grad in zip(policy_vars, policy_grads):
                tf.summary.histogram(f"policy_grad-{var.name}", grad, epoch)
                tf.summary.histogram(f"policy_var-{var.name}", var, epoch)

        value_vars = self.value.trainable_variables
        value_grads = [tf.clip_by_value(grad, -1, 1) for grad in
                       tape.gradient(value_objective, value_vars)]
        self.value_optimizer.apply_gradients(zip(value_grads, value_vars))

        with self.writer.as_default():
            for var, grad in zip(value_vars, value_grads):
                tf.summary.histogram(f"value_grad-{var.name}", grad, epoch)
                tf.summary.histogram(f"value_var-{var.name}", var, epoch)

        return policy_loss, value_loss

