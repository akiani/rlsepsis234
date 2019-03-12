import datetime
import numpy as np
import random as rnd
import pandas as pd
import os
import sys
import time
import tensorflow as tf
import tensorflow.contrib.layers as layers

from collections import deque
from core.deep_q_learning import DQN
from utils.general import get_logger, Progbar, export_plot
from utils.test_env import EnvTest
from envs.off_policy_env import EnvOffPol
from utils.replay_buffer import ReplayBuffer

from configs.final_dqn import config


class LinearSchedule(object):
    """
    Linear Schedule is useful for on model training epsilon annealing.
    """

    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t):
        """
        Updates epsilon

        Args:
            t: int
                frame number
        """

        if t > self.nsteps:
            self.epsilon = self.eps_end
        else:
            self.epsilon = self.eps_begin + (
                self.eps_end - self.eps_begin) / self.nsteps * t


class LinearExploration(LinearSchedule):
    """
    Useful for on model training e-greedy behavior policy.
    """

    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            nsteps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)

    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action

        Args:
            best_action: int 
                best action according some policy
        Returns:
            an action
        """
        use_random = rnd.random() < self.epsilon
        if use_random:
            return self.env.action_space.sample()
        else:
            return best_action


class FinalDQN(DQN):
    """
    DQN for Sepsis policy prediction, based on A. Raghu arXiv:1711.09602v1 [cs.AI]

    This is a derivation of code from Stanford CS234 Assignment2.

    Search for "tding" for some customization notes.
    """

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        state_shape = list(self.env.observation_space.shape)

        self.s = tf.placeholder(
            tf.uint8,
            shape=(None, state_shape[0], state_shape[1],
                   state_shape[2] * self.config.state_history))
        self.a = tf.placeholder(tf.int32, shape=(None))
        self.r = tf.placeholder(tf.int32, shape=(None))
        self.sp = tf.placeholder(
            tf.uint8,
            shape=(None, state_shape[0], state_shape[1],
                   state_shape[2] * self.config.state_history))
        self.done_mask = tf.placeholder(tf.bool, shape=(None))
        self.lr = tf.placeholder(tf.float32, shape=None)
        self.is_training = tf.placeholder(tf.bool, name='is_training')

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Impletemts Dueling Q network with 2 hidden layers each.

        V estimate has shape = (batch_size, 1)
        A estimate has shape = (batch_size, num_actions)

        Q = V + A - Avg(A)

        Batch normalization is applied after each hidden layer.

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n

        with tf.variable_scope(scope, reuse=reuse):
            if self.config.use_batch_norm:
                v_l1 = tf.layers.batch_normalization(
                    tf.layers.dense(
                        inputs=tf.layers.flatten(state),
                        units=128,
                        activation=lambda x: tf.nn.leaky_relu(x, alpha=0.5)),
                    training=self.is_training)
                v_l2 = tf.layers.batch_normalization(
                    tf.layers.dense(
                        inputs=v_l1,
                        units=128,
                        activation=lambda x: tf.nn.leaky_relu(x, alpha=0.5)),
                    training=self.is_training)
                v_out = tf.layers.dense(inputs=v_l2, units=1)
                a_l1 = tf.layers.batch_normalization(
                    tf.layers.dense(
                        inputs=tf.layers.flatten(state),
                        units=128,
                        activation=lambda x: tf.nn.leaky_relu(x, alpha=0.5)),
                    training=self.is_training)
                a_l2 = tf.layers.batch_normalization(
                    tf.layers.dense(
                        inputs=a_l1,
                        units=128,
                        activation=lambda x: tf.nn.leaky_relu(x, alpha=0.5)),
                    training=self.is_training)
                a_out = tf.layers.dense(inputs=a_l2, units=num_actions)
                # Q = V + A - E(A)
                out = tf.keras.backend.repeat_elements(
                    v_out, num_actions,
                    -1) + a_out - tf.keras.backend.repeat_elements(
                        tf.reduce_mean(a_out, axis=-1, keepdims=True),
                        num_actions, -1)
            else:
                v_l1 = tf.layers.dense(
                    inputs=tf.layers.flatten(state),
                    units=128,
                    activation=lambda x: tf.nn.leaky_relu(x, alpha=0.5))
                v_l2 = tf.layers.dense(
                    inputs=v_l1,
                    units=128,
                    activation=lambda x: tf.nn.leaky_relu(x, alpha=0.5))
                v_out = tf.layers.dense(inputs=v_l2, units=1)
                a_l1 = tf.layers.dense(
                    inputs=tf.layers.flatten(state),
                    units=128,
                    activation=lambda x: tf.nn.leaky_relu(x, alpha=0.5))
                a_l2 = tf.layers.dense(
                    inputs=a_l1,
                    units=128,
                    activation=lambda x: tf.nn.leaky_relu(x, alpha=0.5))
                a_out = tf.layers.dense(inputs=a_l2, units=num_actions)
                # Q = V + A - E(A)
                out = tf.keras.backend.repeat_elements(
                    v_out, num_actions,
                    -1) + a_out - tf.keras.backend.repeat_elements(
                        tf.reduce_mean(a_out, axis=-1, keepdims=True),
                        num_actions, -1)
        return out

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/programmers_guide/variable_scope

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        q_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        with tf.variable_scope(target_q_scope, reuse=tf.AUTO_REUSE):
            print(q_scope)
            print(target_q_scope)
            all_ops = [
                tf.assign(
                    tf.get_variable(
                        v.name.replace(q_scope + "/", "", 1).split(":")[0],
                        shape=v.shape), v) for v in q_vars
            ]
        self.update_target_op = tf.group(*all_ops)

    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n

        print(q)
        print(target_q)
        print(self.config.gamma)
        q_samp = tf.where(
            self.done_mask, tf.cast(self.r, tf.float32),
            tf.add(
                tf.cast(self.r, tf.float32),
                tf.scalar_mul(self.config.gamma, tf.reduce_max(
                    target_q, axis=1))))
        q_pred = tf.reduce_sum(
            tf.multiply(tf.one_hot(self.a, num_actions, dtype=tf.float32), q),
            axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(q_samp, q_pred))

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm

        tding:
        Add batch normalization.
        Args:
            scope: (string) scope name, that specifies if target network or not
        """

        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        grads_and_vars = optimizer.compute_gradients(
            self.loss,
            var_list=tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        if self.config.grad_clip:
            grads_and_vars = [(tf.clip_by_norm(grad, self.config.clip_val),
                               var) for grad, var in grads_and_vars]

        # batch normalization
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.apply_gradients(grads_and_vars)
        self.grad_norm = tf.global_norm(
            [grad for grad, var in grads_and_vars if grad is not None])

    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        tding: override and customize step for off policy env
        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size,
                                     self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0  # time control of nb of steps
        scores_eval = []  # list of scores computed at iteration time
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train: self.env.render()
                # replay memory stuff
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                if self.config.train_off_policy:
                    # tding: action is no longer based on e-greedy. it's returned by the env.
                    # return val info was removed
                    # action produced from e-greedy above is overridden.
                    # USE THIS FOR OFF POLICY
                    new_state, action, reward, done, ids = self.env.step()
                else:
                    # perform action in env
                    # USE THIS FOR MODEL BASED
                    action = exp_schedule.get_action(best_action)
                    new_state, reward, done, info = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer,
                                                       lr_schedule.epsilon)

                # logging stuff
                if ((t > self.config.learning_start)
                        and (t % self.config.log_freq == 0)
                        and (t % self.config.learning_freq == 0)):
                    self.update_averages(rewards, max_q_values, q_values,
                                         scores_eval)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(
                            t + 1,
                            exact=[("Loss", loss_eval),
                                   ("Avg_R", self.avg_reward),
                                   ("Max_R", np.max(rewards)),
                                   ("eps", exp_schedule.epsilon),
                                   ("Grads", grad_eval), ("Max_Q", self.max_q),
                                   ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (
                        t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(
                        t, self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            if (t > self.config.learning_start) and (last_eval >
                                                     self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

            if (t > self.config.learning_start) and self.config.record and (
                    last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        # Run testing for off policy
        if self.config.train_off_policy:
            self.validate()
        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training

        tding:
        for off policy environment, since there is no model, agent cannot
        interact with model for evaluation.
        """
        if self.config.train_off_policy:
            self.logger.info("Running off policy, not evaluating...")
            return 0

        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size,
                                     self.config.state_history)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test: env.render()

                # store last state in buffer
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.get_action(q_input)

                # perform action in env
                new_state, reward, done, info = env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward

    def validate(self, env=None, num_episodes=None):
        """
        For the off policy env, validate and compare the actions chosen by the
        agent and by physicians.
        """
        if not self.config.train_off_policy:
            self.logger.info("Not running off policy, not validating...")
            return 0

        self.logger.info("Validating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size,
                                     self.config.state_history)
        rewards = []

        env.init_validate()
        res = []
        action_map = {}
        count = 0
        for i in range(5):
            for j in range(5):
                action_map[count] = [i, j]
                count += 1
        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            res_episode = []
            while True:
                if self.config.render_test: env.render()

                # store last state in buffer
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action_pred = self.get_action(q_input)

                # perform action in env
                new_state, action_real, reward, done, ids = env.step()

                subject_id = ids[0]
                hadm_id = ids[1]
                icustay_id = ids[2]
                interval_start_time = ids[3]
                interval_end_time = ids[4]
                # sofa for comparing results.
                sofa = state[0, 0, 37]

                # store in replay memory
                replay_buffer.store_effect(idx, action_real, reward, done)
                state = new_state

                iv_pred, vaso_pred = action_map[action_pred]
                iv_real, vaso_real = action_map[action_real]
                res_episode.append([
                    subject_id, hadm_id, icustay_id, interval_start_time,
                    interval_end_time, sofa, iv_pred, vaso_pred, iv_real,
                    vaso_real
                ])
                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            if total_reward != -15 and total_reward != 15:
                self.logger.info("reward is not +- 15")
            mortal = 0
            if total_reward == -15:
                mortal = 1
            for i in range(len(res_episode)):
                res_episode[i].append(mortal)
                #print("actions: {}".format(res_episode[i]))
            res += res_episode

        # print(res)
        output = pd.DataFrame(
            res,
            columns=[
                'subject_id', 'hadm_id', 'icustay_id', 'interval_start_time',
                'interval_end_time', 'sofa', 'iv_pred', 'vaso_pred', 'iv_real',
                'vaso_real', 'died'
            ])
        output.to_csv(
            os.path.join(
                self.config.output_path,
                "pred_real_compare" + datetime.datetime.fromtimestamp(
                    time.time()).strftime('%Y-%m-%dT%H:%M:%S')))

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    if config.train_off_policy:
        env = EnvOffPol("data")
    else:
        env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, config.eps_end,
                                     config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # train model
    model = FinalDQN(env, config)
    model.run(exp_schedule, lr_schedule)
