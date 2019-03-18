import numpy as np
import pandas as pd
import os


class ActionSpace(object):
    def __init__(self, n):
        self.n = n


class ObservationSpace(object):
    def __init__(self, shape):
        self.shape = shape


class EnvOffPol(object):
    """
    Modified from Stanford CS234 Assignment 2
    Column indices
    0       subject_id
    [1,4]   misc
    [5,50]  features
    51      action
    52      hospital_expire_flag
    53      reward
    54      row_id
    55      row_id_next
    """

    def __init__(self, data_dir):
        self.action_space = ActionSpace(25)
        # hacky way to reuse replay_buffer etc.
        self.observation_space = ObservationSpace([1, 1, 46])
        self.data_dir = data_dir
        self.init_train()

    def _process_sessions(self):
        self.index = 0
        prev_stay_id = 0
        self.p_starting_indices = []
        for i in range(len(self.samples)):
            if self.samples[i, 0] != prev_stay_id:
                self.p_starting_indices.append(i)
            prev_stay_id = self.samples[i, 0]
        self.ps_index = 0
        print("There are {} distinct sessions with {} records".format(
            len(self.p_starting_indices), len(self.samples)))

    def init_train(self):
        self.samples = pd.read_csv(
            os.path.join(self.data_dir,
                         'train_state_action_reward_df.csv')).values
        self._process_sessions()

    def init_validate(self):
        self.samples = pd.read_csv(
            os.path.join(self.data_dir,
                         'test_state_action_reward_df.csv')).values
        self._process_sessions()

    def _features(self, index):
        """
        len = 46
        """
        return self.samples[index, 5:51]

    def _new_state(self):
        if self._is_terminal():
            return np.zeros(46)
        return self._features(self.index + 1)

    def _action(self):
        return self.samples[self.index, 51]

    def _reward(self):
        return self.samples[self.index, 53]

    def _is_terminal(self):
        if self.index == len(self.samples) - 1:
            return True
        if self.samples[self.index, 0] != self.samples[self.index + 1, 0]:
            return True
        return False

    def _id_prefix(self):
        return self.samples[self.index, 0:5]

    def reset(self):
        """
        calling reset jumps to the next patient.
        """
        if self.ps_index >= len(self.p_starting_indices):
            self.ps_index = 0
            print(
                "Off Policy Env patients exhausted... returning to first patient."
            )
        self.index = self.p_starting_indices[self.ps_index]
        # There's one sample in the dataset with 2 end of episode rewards. Skip it
        if (self.samples[self.index, 0] == 32701):
            print("Skipping defective sample subject_id {}".format(
                self.samples[self.index, 0]))
            self.ps_index += 1
            self.index = self.p_starting_indices[self.ps_index]

        #print("Onto patient {} at record {} of {}".format(
        #    self.samples[self.index, 0], self.index, len(self.samples)))
        self.ps_index += 1
        return np.reshape(
            self._features(self.index), self.observation_space.shape)

    def step(self):
        """
        returns next_state, action, reward, done
        """
        if self.index >= len(self.samples):
            self.index = 0
            print(
                "Off Policy Env samples exhausted... returning to first sample."
            )
        res = (np.reshape(self._new_state(), self.observation_space.shape),
               self._action(), self._reward(), self._is_terminal(),
               self._id_prefix())
        #print("sample {} of {}, patient {}".format(self.index, len(
        #    self.samples), self.samples[self.index, 0]))
        self.index += 1
        return res

    def render(self):
        print(self.samples[self.index])


def test_reward(env):
    """
    Test to ensure the env is sane.
    Verify that rewards only come at the end of one episode, episodes are
    terminated correctly.

    """
    size = len(env.samples)
    print("env has {} samples".format(size))
    env.reset()
    episode_rec = []
    episode_reward = 0
    for i in range(size):
        next_state, action, reward, is_terminal, id_pref = env.step()
        episode_rec += (id_pref, action, reward, next_state, is_terminal)
        episode_reward += reward
        if is_terminal:
            env.reset()
            if episode_reward > 15:
                print("found episode with irregular reward {}: {}".format(
                    episode_reward, episode_rec))
            episode_rec = []
            episode_reward = 0


if __name__ == '__main__':
    env = EnvOffPol('data3')
    test_reward(env)
    env.init_validate()
    test_reward(env)
