import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import random
import math
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from utils import *
from envs.gridworld_env import GridworldEnv

# env = gym.make("MiniGrid-Fetch-5x5-N2-v0", render_mode="human")
# env_obs = FlatObsWrapper(env)
# observation, info = env_obs.reset(seed=123)


class DP:
    def __init__(self, args):
        self.args = args
        self.env = GridworldEnv()
        self.actions = self.env.action_space.n
        tmp = np.array(np.where(self.env.start_grid_map != 1))
        self.gridsize = self.env.start_grid_map.shape[1]
        self.states = (tmp[0] * self.gridsize + tmp[1]).tolist()

    def cal_v(self, state_index, V, policy):
        A_num = self.actions
        state = self.states[state_index]
        pos = state2pos(state, self.gridsize)
        v = 0
        for action in range(A_num):
            self.env.change_start_state(pos)
            next_pos, r, terminated, _ = self.env.step(action)
            next_state = pos2state(next_pos, self.gridsize)
            v += policy[state_index][action] * (r + self.args.gamma * V[self.states.index(next_state)])
        return v

    def run(self):
        policy = np.ones((len(self.states), self.actions)) * (1 / self.actions)  # initialize as uniform policy
        rewards = []
        end_steps = []
        epoch = 0  # update policy per epoch
        while True:
            epoch += 1
            V = self.policy_evaluation(policy)
            update_policy = self.policy_improve(V)
            if (update_policy == policy).all():
                break
            policy = copy.copy(update_policy)
            reward, end_step = self.run_with_current_policy(policy)
            rewards.append(reward)
            end_steps.append(end_step)
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
        ax1.plot(rewards)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid()

        ax2.plot(end_steps)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps before terminate')
        ax2.grid()
        plt.show()
        self.demo(policy)
        return policy

    def policy_evaluation(self, policy):
        V = np.zeros(len(self.states))
        # evaluate policy
        step = 0
        while True:
            delta = 0
            _ = self.env.reset()
            for s in range(len(self.states)):
                Vs = self.cal_v(s, V, policy)
                delta = max(delta, np.abs(Vs - V[s]))
                V[s] = Vs
            if delta < self.args.theta:
                break
            step += 1
        return V

    def policy_improve(self, V):
        policy = np.zeros([len(self.states), self.actions]) / self.actions  # initialize as uniform policy
        for s in range(len(self.states)):
            state = self.states[s]
            pos = state2pos(state, self.gridsize)
            Q = np.zeros(self.actions)
            for a in range(self.actions):
                self.env.change_start_state(pos)
                next_pos, r, terminated, _ = self.env.step(a)
                next_state = pos2state(next_pos, self.gridsize)
                Q[a] += r + self.args.gamma * V[self.states.index(next_state)]
            action = np.argmax(Q)
            # Greedyly take best action.
            policy[s][action] = 1.0
        return policy

    def run_with_current_policy(self, policy):
        _ = self.env.reset()
        reward = 0
        step = 0
        terminated = False
        pos = self.env.agent_start_state
        state = pos2state(pos, self.gridsize)
        action = np.argmax(policy[self.states.index(state)])
        while (not terminated) and (step < self.args.maxstep):
            step += 1
            pos, r, terminated, _ = self.env.step(action)
            state = pos2state(pos, self.gridsize)
            reward += r
            action = np.argmax(policy[self.states.index(state)])
        return reward, step

    def demo(self, policy):
        self.env.verbose = True
        _ = self.env.reset()
        terminated = False
        pos = self.env.agent_start_state
        state = pos2state(pos, self.gridsize)
        action = np.argmax(policy[self.states.index(state)])
        while not terminated:
            pos, r, terminated, _ = self.env.step(action)
            state = pos2state(pos, self.gridsize)
            action = np.argmax(policy[self.states.index(state)])


class DRO:
    def __init__(self, args):
        self.args = args
        self.env = GridworldEnv()
        self.actions = self.env.action_space.n
        tmp = np.array(np.where(self.env.start_grid_map != 1))
        self.gridsize = self.env.start_grid_map.shape[1]
        self.states = (tmp[0] * self.gridsize + tmp[1]).tolist()
        self.dataset = ReplayBuffer(self.args)
        self.rewardmap = np.zeros((len(self.states), self.actions))
        self.datafreq = np.zeros((len(self.states), self.actions, len(self.states)))

    def make_dataset(self):
        # sample all state-action first to make sure P(s,a)>0
        for s in range(len(self.states)):
            state = self.states[s]
            pos = state2pos(state, self.gridsize)
            for action in range(self.actions):
                self.env.change_start_state(pos)
                next_pos, r, terminated, _ = self.env.step(action)
                next_state = pos2state(next_pos, self.gridsize)
                self.rewardmap[s][action] = r
                self.dataset.add(state, action, next_state, r, terminated)

        # randomly add state-action
        Done = False
        while not Done:
            _ = self.env.reset()
            max_step = 10
            for s in range(len(self.states)):
                step = 0
                state = self.states[s]
                pos = state2pos(state, self.gridsize)
                self.env.change_start_state(pos)
                while step < max_step:
                    step += 1
                    action = np.random.choice(range(self.actions))
                    next_pos, r, terminated, _ = self.env.step(action)
                    next_state = pos2state(next_pos, self.gridsize)
                    self.dataset.add(state, action, next_state, r, terminated)
                    state = next_state
                if self.dataset.__len__() == self.args.buffersize:
                    Done = True
                    break
                print(self.dataset.__len__())

    def make_datafreq(self):
        for s in range(len(self.states)):
            for a in range(self.actions):
                for s_next in range(len(self.states)):
                    self.datafreq[s, a, s_next] = self.dataset.inquire_num(self.states[s],a,self.states[s_next])

    def run(self):
        # sample offline dataset
        self.make_dataset()
        self.make_datafreq()
        V, policy = self.policy_iter()
        self.demo(policy)
        return policy

    def policy_iter(self):
        V = np.zeros(len(self.states))
        epoch = 0  # update policy per epoch
        max_epoch = 100
        rewards = []
        end_steps = []
        record = np.zeros((max_epoch, len(self.states)))
        while True:
            record[epoch, :] = V
            epoch += 1
            delta = 0
            V_pre = copy.copy(V)
            policy = np.zeros([len(self.states), self.actions]) / self.actions
            for s in range(len(self.states)):
                Q = np.zeros(self.actions)
                for a in range(self.actions):
                    center = self.datafreq[s][a] / np.sum(self.datafreq[s][a])
                    # radius = self.get_radius(s, a)
                    radius = 0.2
                    solution = TV_opt(V_pre, center, radius)
                    Q[a] += self.rewardmap[s][a] + self.args.gamma * solution
                Vs = np.max(Q)
                policy[s][np.argmax(Q)] = 1.0
                V[s] = Vs
            reward, end_step = self.run_with_current_policy(policy)
            rewards.append(reward)
            end_steps.append(end_step)
            delta = max(delta, np.mean(np.abs(V_pre - V)))
            print(epoch, delta)
            if epoch > max_epoch-1:
                break
        np.savetxt('csvs/dro_value.csv', record, delimiter=',')
        plt.figure()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
        ax1.plot(rewards)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid()

        ax2.plot(end_steps)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps before terminate')
        ax2.grid()
        plt.show()
        return V, policy

    def get_radius(self, s, a):
        radius = math.sqrt(math.log(len(self.states)/self.args.delta)/(2*np.sum(self.datafreq[s][a])))
        if radius > 2:
            return 2
        else:
            return radius

    def run_with_current_policy(self, policy):
        _ = self.env.reset()
        reward = 0
        step = 0
        terminated = False
        pos = self.env.agent_start_state
        state = pos2state(pos, self.gridsize)
        action = np.argmax(policy[self.states.index(state)])
        while (not terminated) and (step < self.args.maxstep):
            step += 1
            pos, r, terminated, _ = self.env.step(action)
            state = pos2state(pos, self.gridsize)
            reward += r
            action = np.argmax(policy[self.states.index(state)])
        return reward, step

    def demo(self, policy):
        self.env.verbose = True
        _ = self.env.reset()
        terminated = False
        pos = self.env.agent_start_state
        state = pos2state(pos, self.gridsize)
        action = np.argmax(policy[self.states.index(state)])
        while not terminated:
            pos, r, terminated, _ = self.env.step(action)
            state = pos2state(pos, self.gridsize)
            action = np.argmax(policy[self.states.index(state)])


class ReplayBuffer(object):
    def __init__(self, args):
        self._storage = []
        self._maxsize = args.buffersize
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, current_state, action, next_state, reward, done):
        data = (current_state, action, next_state, reward, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize  # queue

    def _encode_sample(self, idxes):
        current_states, actions, next_states, rewards, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            current_state, action, next_state, reward, done = data
            current_states.append(current_state)
            actions.append(action)
            next_states.append(next_states)
            rewards.append(reward)
            dones.append(done)
        return np.array(current_states), np.array(actions), np.array(next_states), np.array(rewards), np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def inquire_num(self, s, a, s_next):
        data = np.array(self._storage)[:, :3]
        idx = np.intersect1d(np.where(data[:, 0] == s), np.where(data[:, 1] == a))
        sa = data[idx,:]
        num = np.array(np.where(sa[:,2] == s_next)).size
        return num



