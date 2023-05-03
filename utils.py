import numpy as np
import argparse
import matplotlib.pyplot as plt
import cvxpy as cp
import math

phi = 0.49
tra = 3
total_step = 2000

All_states = list(range(0, 15))
All_actions = list(range(0, 30))
S_n = len(All_states)
A_n = len(All_actions)

R = 0.4


def parser_from_dict(dic):
    parser = argparse.ArgumentParser()
    for k, v in dic.items():
        parser.add_argument("--" + k, default=v)
    args = parser.parse_args()

    return args


def pos2state(pos, gridsize):
    return pos[0]*gridsize + pos[1]


def state2pos(state, gridsize):
    return (state//gridsize, state%gridsize)


def normal_vec(vectr):
    for k in range(len(vectr)):
        if vectr[k] < 0:
            vectr[k] = -1 * vectr[k]
    sum = np.sum(vectr)
    for i in range(len(vectr)):
        vectr[i] = vectr[i] / sum
    return np.array(vectr)


sig = np.random.random()
rewards = np.ones((S_n, A_n))
kernels = np.ones((S_n, A_n, S_n))

# initilsize prob kernels
for s in All_states:
    for a in All_actions:
        kernels[s, a] = normal_vec(np.random.normal(np.random.random(), 1000 * sig * sig, S_n))  # init prob from normal Gaussian distribution


# set reward
for s in range(S_n):
    for a in range(A_n):
        rewards[s, a] = np.random.normal(np.random.random(), 1000 * sig * sig)   # init reward from normal Gaussian distribution


# get transition prob
def pro(s, a, t):
    return kernels[s, a, t]


# get instant reward
def reward(s, a):
    return rewards[s, a]


# init policy
uniform_policy = np.array([1 / A_n] * A_n)


# choose action following uniform policy
def choose_actoin():
    return np.random.choice(All_actions, p=uniform_policy.ravel())


# get next state
def step(ST, AC):
    trans_kernel = kernels[ST, AC]
    return np.random.choice(All_states, p=trans_kernel.ravel())


def V_Q(Q):
    V_inner = []
    for kk in range(len(Q)):
        V_inner.append(np.max(Q[kk]))
    return np.array(V_inner)


def span(v, w):
    v = np.array(v)
    w = np.array(w)
    sp_max = np.max(v - w)
    sp_min = np.min(v - w)
    return sp_max - sp_min


def ep_dis(sample_list):
    sample_num = len(sample_list)
    ep_distribution = [0] * S_n
    for k in range(sample_num):
        ep_distribution[(sample_list[k])] = ep_distribution[(sample_list[k])] + 1 / sample_num
    return ep_distribution


def TV_opt(V, center, radius):
    x = V.size
    objective = cp.Maximize((center @ (V - x) - radius * (cp.max(V - x) - cp.min(V - x))))
    constraints = [0 <= x]
    prob = cp.Problem(objective, constraints)
    prob.solve()  # Returns the optimal value.
    return prob.value
