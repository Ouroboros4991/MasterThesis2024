# Based on the code found here: https://github.com/lweitkamp/option-critic-pytorch/blob/master/option_critic.py
"""Functionality shared across option critic agents.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

import math
import itertools
import numpy as np


def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs


def __get_action_distribution(state, model, option):
    """Get the action distribution for the given state and option.

    Args:
        state: State to calculate the action distribution for.
        model: Model to calculate the action distribution with.
        option: Option to calculate the action distribution for.

    Returns:
        array: Numpy array with the action distribution.
    """
    logits = state.data @ model.options_W[option] + model.options_b[option]
    action_dist = (logits / model.temperature).softmax(dim=-1)
    return action_dist.detach().numpy()


def __hellinger_regulizer(state, model) -> float:
    """Calculate the hellinger distance between the intra-option policies of the model
    for the given state.
    This as defined in the paper "Disentangling Options with Hellinger Distance Regularizer"

    Args:
        states: The states for which to calculate the hellinger distance.
        model: The option critic model to calculate the distance for.

    Returns:
        float: Helling distance loss
    """
    possible_option_combinations = list(
        itertools.combinations([i for i in range(model.num_options)], 2)
    )
    l_hd_reg = 0
    for comb in possible_option_combinations:
        p, q = comb
        p_dist = __get_action_distribution(state, model, p)
        q_dist = __get_action_distribution(state, model, q)
        summation = np.sum((np.sqrt(p_dist) - np.sqrt(q_dist)) ** 2)
        hd = math.sqrt(summation) / math.sqrt(2)
        l_hd_reg += hd
    l_hd_reg = l_hd_reg / len(possible_option_combinations)
    return l_hd_reg


def critic_loss(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options = torch.LongTensor(options).to(model.device)
    rewards = torch.FloatTensor(rewards).to(model.device)
    masks = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = obs.squeeze(0) # model.get_state(obs).squeeze(0)
    Q = model.get_Q(states)

    # the update target contains Q_next, but for stable learning we use prime network for this
    # next_states_prime = model_prime.get_state(next_obs).squeeze(0)
    next_states_prime = next_obs.squeeze(0)
    next_Q_prime = model_prime.get_Q(next_states_prime)  # detach?

    # Additionally, we need the beta probabilities of the next state
    # next_states = model.get_state(next_obs).squeeze(0)
    next_states = next_obs.squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * (
        (1 - next_options_term_prob) * next_Q_prime[batch_idx, options]
        + next_options_term_prob * next_Q_prime.max(dim=-1)[0]
    )

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err    


def critic_loss_w_option_reward(model, model_prime, data_batch, args):
    obs, options, rewards, option_rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    options = torch.LongTensor(options).to(model.device)
    rewards = torch.FloatTensor(rewards).to(model.device)
    option_rewards = torch.FloatTensor(option_rewards).to(model.device)
    masks = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = obs.squeeze(0) # model.get_state(obs).squeeze(0)
    Q = model.get_Q(states)

    # the update target contains Q_next, but for stable learning we use prime network for this
    # next_states_prime = model_prime.get_state(next_obs).squeeze(0)
    next_states_prime = next_obs.squeeze(0)
    next_Q_prime = model_prime.get_Q(next_states_prime)  # detach?

    # Additionally, we need the beta probabilities of the next state
    # next_states = model.get_state(next_obs).squeeze(0)
    next_states = next_obs.squeeze(0)
    next_termination_probs = model.get_terminations(next_states).detach()
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = option_rewards + masks * args.gamma * (
        (1 - next_options_term_prob) * next_Q_prime[batch_idx, options]
        + next_options_term_prob * next_Q_prime.max(dim=-1)[0]
    )

    # to update Q we want to use the actual network, not the prime
    td_err = (Q[batch_idx, options] - gt.detach()).pow(2).mul(0.5).mean()
    return td_err   

def actor_loss(
    obs, option, logp, entropy, reward, done, next_obs, model, model_prime,
    args,
    option_densities: dict=None,
):
    state = obs # model.get_state(obs)
    next_state = next_obs # model.get_state(next_obs)
    next_state_prime = next_obs # model_prime.get_state(next_obs)

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option].detach()
    # option_term_prob = model.get_terminations(state)
    # next_option_term_prob = model.get_terminations(next_state).detach()

    Q = model.get_Q(state).detach().squeeze()
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    # Target update gt
    gt = reward + (1 - done) * args.gamma * (
        (1 - next_option_term_prob) * next_Q_prime[option]
        + next_option_term_prob * next_Q_prime.max(dim=-1)[0]
    )

    # The termination loss
    # Starts from the termination probability
    # Increases when the Q value of the option is close to the max Q value
    # + the termination reg
    # So the termination reg compensates when the Q value is much lower
    # than the max Q value
    # The third term causes the loss to only take into account 
    # the termination loss when the episode is not done
    termination_loss = (
        option_term_prob
        * (Q[option].detach() - Q.max(dim=-1)[0].detach() + args.termination_reg)
        * (1 - done)
    )
   
    # if option_densities:
    #     sd_option_density = np.std(option_densities[option])
    #     termination_loss = termination_loss + sd_option_density

    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    if args.hd_reg:
        hd_reg = __hellinger_regulizer(state, model)
        actor_loss += hd_reg
    return actor_loss


