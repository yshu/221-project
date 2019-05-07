import tensorflow as tf
import tensorflow.contrib.layers as layers

from core.deep_q_learning import DQN

class Linear(DQN):
    """
    Implement Fully Connected
    """
    def add_placeholders_op(self):
        h, w, c = list(self.env.observation_space.shape)
        self.s = tf.placeholder(tf.uint8, shape=[None, h, w, c*self.config.state_history])
        self.a = tf.placeholder(tf.int32, shape=[None])
        self.r = tf.placeholder(tf.float32, shape=[None])
        self.sp = tf.placeholder(tf.uint8, shape=[None, h, w, c*self.config.state_history])
        self.done_mask = tf.placeholder(tf.bool, shape=[None])
        self.lr = tf.placeholder(tf.float32, shape=())

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions
        Implement a fully connected with no hidden layer (linear
        approximation with bias) using tensorflow.
        """
        num_actions = self.env.action_space.n
        with tf.variable_scope(scope, reuse=reuse):
            flatten = layers.flatten(state)
            out = layers.fully_connected(flatten, num_actions, activation_fn=None)
        return out


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network
        """
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        op = [tf.assign(t_vars[i], q_vars[i]) for i in range(len(q_vars))]
        self.update_target_op = tf.group(*op)

    def add_loss_op(self, q, target_q):
        """
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2
        """
        num_actions = self.env.action_space.n
        gamma = self.config.gamma * tf.reduce_max(target_q, axis=1)
        q_samp = tf.where(self.done_mask, self.r, self.r+gamma)
        q_s = tf.reduce_sum(q*tf.one_hot(self.a, num_actions), axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(q_samp, q_s))

    def add_optimizer_op(self, scope):
        optimizer = tf.train.AdamOptimizer(self.lr)
        scope_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grads_and_vars = optimizer.compute_gradients(self.loss, scope_vars)
        if self.config.grad_clip:
            clipped_grads_and_vars = [(tf.clip_by_norm(v[0], self.config.clip_val),v[1]) for v in grads_and_vars]
        self.train_op = optimizer.apply_gradients(clipped_grads_and_vars)
        gradients = [grad for grad in grads_and_vars]
        self.grad_norm = tf.global_norm(gradients)


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        num_actions = self.env.action_space.n
        with tf.variable_scope(scope, reuse=reuse):
            conv = layers.conv2d(state, 32, 8, stride=4)
            conv = layers.conv2d(conv, 64, 4, stride=2)
            conv = layers.conv2d(conv, 64, 3, stride=1)
            flatten = layers.flatten(conv)
            out = layers.fully_connected(flatten, 512)
            out = layers.fully_connected(out, num_actions, activation_fn=None)
        return out        