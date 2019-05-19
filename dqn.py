import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
from progressbar import Progbar
from util import *
import os
import numpy as np
import sys
from gym import wrappers
from collections import deque


from utils.replay_buffer import ReplayBuffer

class config():
    env_name           = "Pong-v0"
    record             = True
    output_path        = "results/pong_dqn_6/"
    model_output       = output_path + "model.weights/"
    log_path           = output_path + "log.txt"
    plot_output        = output_path + "scores.png"
    record_path        = output_path + "monitor/"
    saving_freq        = 250000
    log_freq           = 50
    eval_freq          = 250000
    record_freq        = 250000

    num_episodes_test  = 100
    nsteps_train       = 5000000
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 10000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    skip_frame         = 4
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.1
    eps_nsteps         = 1000000
    learning_start     = 50000
    soft_epsilon       = 0.05

class DQN(object):
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.build()
        
    def add_placeholders_op(self):
        h, w, c = list(self.env.observation_space.shape)
        self.s = tf.placeholder(tf.uint8, shape=[None, h, w, c*self.config.state_history])
        self.a = tf.placeholder(tf.int32, shape=[None])
        self.r = tf.placeholder(tf.float32, shape=[None])
        self.sp = tf.placeholder(tf.uint8, shape=[None, h, w, c*self.config.state_history])
        self.done_mask = tf.placeholder(tf.bool, shape=[None])
        self.lr = tf.placeholder(tf.float32, shape=())
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")

    def q_network_op(self, state):
        num_actions = self.env.action_space.n
        conv = layers.conv2d(state, 32, 8, stride=4)
        conv = layers.conv2d(conv, 64, 4, stride=2)
        conv = layers.conv2d(conv, 64, 3, stride=1)
        flatten = layers.flatten(conv)
        out = layers.fully_connected(flatten, 512)
        out = layers.fully_connected(out, num_actions, activation_fn=None)
        return out

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
        loss = tf.reduce_mean(tf.squared_difference(q_samp, q_s))
        return loss
 
    def build(self):
        self.add_placeholders_op()
        
        with tf.variable_scope('q', reuse=False):
            s = tf.cast(self.s, tf.float32)/255. #[0,255] -> [0,1]
            self.q = self.q_network_op(s)
            
        with tf.variable_scope('target_q', reuse=False):
            sp = tf.cast(self.sp, tf.float32)/255. #[0,255] -> [0,1]
            self.target_q = self.q_network_op(sp)

        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
        t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_q')
        op = [tf.assign(t_vars[i], q_vars[i]) for i in range(len(q_vars))]
        self.update_target_op = tf.group(*op)
        
        self.loss = self.add_loss_op(self.q, self.target_q)
        
        optimizer = tf.train.AdamOptimizer(self.lr)
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
        grads_and_vars = optimizer.compute_gradients(self.loss, q_vars)
        self.train_op = optimizer.apply_gradients(grads_and_vars)
        
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
        
class train_DQN(DQN):
    def __init__(self, env, config):
        DQN.__init__(self, env, config)
        self.logger = get_logger(config.log_path)
        self.avg_reward = 0
        self.progress = Progbar(target=self.config.nsteps_train)
        
    def get_log(self, exp_schedule, lr_schedule, t, loss_eval, max_q_values, rewards):
        if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and (t % self.config.learning_freq == 0)):
            self.avg_reward = np.mean(rewards)
            max_q = np.mean(max_q_values)
            exp_schedule.update(t)
            lr_schedule.update(t)
            if len(rewards) > 0:
                self.progress.update(t + 1, values=[("Loss", loss_eval), ("Avg_R", self.avg_reward), 
                                     ("Max_R", np.max(rewards)), ("eps", exp_schedule.epsilon), 
                                     ("Max_Q", max_q), ("lr", lr_schedule.epsilon)])

        elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
            sys.stdout.write("\rLearning not start yet: {}/{}...".format(t, self.config.learning_start))
            sys.stdout.flush()
            
    def train_step(self, t, replay_buffer, lr):
        loss_eval = 0

        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(self.config.batch_size)
            model_spec = {self.s: s_batch,
                          self.a: a_batch,
                          self.r: r_batch,
                          self.sp: sp_batch, 
                          self.done_mask: done_mask_batch,
                          self.lr: lr, 
                          self.avg_reward_placeholder: self.avg_reward, }
            loss_eval, summary, _ = self.sess.run([self.loss, self.all_summary, self.train_op], feed_dict=model_spec)

            self.file_writer.add_summary(summary, t)

        if t % self.config.target_update_freq == 0:
            self.sess.run(self.update_target_op)

        if (t % self.config.saving_freq == 0):
            self.saver.save(self.sess, self.config.model_output)

        return loss_eval
            
    def train(self, exp_schedule, lr_schedule):
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)

        t = last_eval = last_record = 0
        scores_eval = [] # scores for plot
        scores_eval += [self.evaluate()]
        
        while t < self.config.nsteps_train:
            sum_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1

                # replay memory stuff
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action_values = self.sess.run(self.q, feed_dict={self.s: [q_input]})[0]
                best_action = np.argmax(action_values)
                q_values = action_values
                action = exp_schedule.get_action(best_action)

                max_q_values.append(max(q_values))
                q_values += list(q_values)
                new_state, reward, done, info = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                loss_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)
                self.get_log(exp_schedule, lr_schedule, t, loss_eval, max_q_values, rewards)
                sum_reward += reward
                if done or t >= self.config.nsteps_train: break

            rewards.append(sum_reward)          

            if t > self.config.learning_start:
                if last_eval > self.config.eval_freq:
                    last_eval = 0
                    scores_eval += [self.evaluate()]

                elif self.config.record and (last_record > self.config.record_freq):
                    self.logger.info("Recording...")
                    last_record =0
                    self.record()

        self.logger.info("*** Training is done.")
        self.saver.save(self.sess, self.config.model_output)
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)
   
    def evaluate(self, env=None, num_episodes=None):
        if env is None: env = self.env
        if num_episodes is None:
            self.logger.info("Evaluating...")
            num_episodes = self.config.num_episodes_test

        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = []
        for i in range(num_episodes):
            sum_reward = 0
            state = env.reset()
            while True:
                idx     = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()
                action = self.env.action_space.sample()
                if self.config.soft_epsilon < np.random.random():
                    action = np.argmax(self.sess.run(self.q, feed_dict={self.s: [q_input]})[0])
                new_state, reward, done, info = env.step(action)
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state
                sum_reward += reward
                if done: break
            rewards.append(sum_reward)     

        avg_reward = np.mean(rewards)
        if num_episodes > 1: self.logger.info("Average reward: {:04.2f}".format(avg_reward))
        return avg_reward

    def record(self):
        record_env = gym.wrappers.Monitor(self.env, self.config.record_path, video_callable=lambda x: True, resume=True)
        self.evaluate(record_env, 1)

    def run(self, exp_schedule, lr_schedule):
        self.sess = tf.Session()
        self.all_summary = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(config.output_path, self.sess.graph)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.run(model.update_target_op)

        self.saver = tf.train.Saver()

        # model
        self.train(exp_schedule, lr_schedule)

        if self.config.record:
            self.record()
            
if __name__ == '__main__':
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    if not os.path.exists(config.model_output):
        os.makedirs(config.model_output)

    env = gym.make(config.env_name)
    env = MaxAndSkipWrapper(env, skip=config.skip_frame)
    env = ResizeWrapper(env, preprocess=greyscale, shape=(80, 80, 1))

    eps_schedule = LinearExploration(env, config.eps_begin, 
                   config.eps_end, config.eps_nsteps)

    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
                   config.lr_nsteps)

    model = train_DQN(env, config)
    model.run(eps_schedule, lr_schedule)