# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:29:34 2024

@author: Thinkpad
"""

from Investregion import *
import random
import gym
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence
from torch.distributions import Distribution, Normal, Categorical, Independent, RelaxedBernoulli
from IPython.display import clear_output
import matplotlib.pyplot as plt
import argparse
import os
import math
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from collections import deque
from ROA import *

import sys
from pathlib import Path


os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["OMP_NUM_THREADS"] = "1"
GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
# args = parser.parse_args()


# Check if running in a Jupyter notebook
if 'ipykernel_launcher' in sys.argv[0]:
    args = parser.parse_args(args=[])  # Avoid parsing errors in Jupyter
else:
    args = parser.parse_args()

# print(f"Train: {args.train}, Test: {args.test}")

# 
def get_save_path(model_path):
    # 
    model_dir = os.path.dirname(model_path)  # "./model"
    
    # athlib?
    reward_filename = f"rewards_{Path(model_path).stem}.npy"  # ?
    save_path = Path(model_dir) / reward_filename  # 
    
    # ?
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    return str(save_path)

def sequence_signature(sequence):
    return hash(tuple(tuple(sorted(stage)) if isinstance(stage, (list, tuple)) else stage for stage in sequence))           

# Reward scaling / penalties (keep training scale consistent)
REWARD_SCALE_DIV = 100.0
INCOMPLETE_R_STEP = -5000.0
ALPHA_INIT = 0.1
ALPHA_MIN = 1e-4
ALPHA_MAX = 0.5

class ReplayBuffer:
    def __init__(self, capacity, device='cpu'):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = device

    def push(self, state, action, reward, next_state, done):
        state = state.cpu() if isinstance(state, torch.Tensor) else torch.FloatTensor(state)
        action = action.cpu() if isinstance(action, torch.Tensor) else torch.FloatTensor(action)
        reward = reward.cpu() if isinstance(reward, torch.Tensor) else torch.FloatTensor([reward])
        next_state = next_state.cpu() if isinstance(next_state, torch.Tensor) else torch.FloatTensor(next_state)
        done = torch.BoolTensor([bool(done.cpu())]) if isinstance(done, torch.Tensor) else torch.BoolTensor([bool(done)])

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(self.device),
            torch.stack(actions).to(self.device),
            torch.stack(rewards).to(self.device),
            torch.stack(next_states).to(self.device),
            torch.stack(dones).to(self.device)
        )

def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)

class TransformerPolicyNetwork(nn.Module):
    def __init__(self, num_regions, hidden_size, k, num_gcn_layers=2, device='cuda', period_num=None):
        super(TransformerPolicyNetwork, self).__init__()
        
        self.num_regions = num_regions
        self.k = k
        self.hidden_size = hidden_size
        self.device = device
        self.period_num = period_num
        
        # Node features: invest_state, invested_ratio, current_step, baseline_demand
        self.node_feature_dim = 4
        
        # Global self-attention encoder with dynamic region interactions.
        self.node_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )
        self.region_embed = nn.Embedding(num_regions, hidden_size)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
            for _ in range(max(1, num_gcn_layers))
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(max(1, num_gcn_layers))])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 4 * hidden_size),
                nn.ReLU(),
                nn.Linear(4 * hidden_size, hidden_size),
            )
            for _ in range(max(1, num_gcn_layers))
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(max(1, num_gcn_layers))])

        # Residual path: preserve region-wise raw information.
        self.raw_feat_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.res_fuse = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # Global token encoder (CLS-style) to avoid mean-pooling bottleneck.
        self.global_proj = nn.Sequential(
            nn.Linear(3, hidden_size),  # current_step, invested_ratio, demand_mean
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # ?
        # 1?(k1)
        self.num_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, k)
        )
        
        # 2
        self.region_node_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)   # ogit
        )
    
        self.to(device)
        
    def log_prob_of_action(self, state, action, mask=None, k_vals=None, selected_indices=None):
        """
        Compute log pi(a|s) for a given MultiBinary action under our 'k + subset' policy.

        Inputs:
          state:  torch or np, shape [B,T,F] / [B,F] / [F]
          action: torch or np, shape [B,T,N] / [B,N] / [N], values in {0,1}
          mask:   valid mask [B,T,N] / [B,N] / [N] (1 valid, 0 invalid)
          k_vals: optional sampled k values [B,T] / [B] / scalar
          selected_indices: optional ordered picks [B,T,k] / [B,k] / [k]

        Returns:
          log_prob: torch.Tensor shape [B,T] (or squeezed)
        """
        # to tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        elif not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action).to(self.device)
        else:
            action = action.to(self.device).float()

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().to(self.device)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.FloatTensor(mask).to(self.device)
            else:
                mask = mask.to(self.device)

        if selected_indices is not None:
            if isinstance(selected_indices, np.ndarray):
                selected_indices = torch.from_numpy(selected_indices).long().to(self.device)
            elif not isinstance(selected_indices, torch.Tensor):
                selected_indices = torch.as_tensor(selected_indices, dtype=torch.long, device=self.device)
            else:
                selected_indices = selected_indices.to(self.device).long()

        if k_vals is not None:
            if isinstance(k_vals, np.ndarray):
                k_vals = torch.from_numpy(k_vals).long().to(self.device)
            elif not isinstance(k_vals, torch.Tensor):
                k_vals = torch.as_tensor(k_vals, dtype=torch.long, device=self.device)
            else:
                k_vals = k_vals.to(self.device).long()

        orig_dim = state.dim()
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
            action = action.unsqueeze(0).unsqueeze(0)
            if mask is not None: mask = mask.unsqueeze(0).unsqueeze(0)
            if selected_indices is not None:
                selected_indices = selected_indices.unsqueeze(0).unsqueeze(0)
            if k_vals is not None:
                k_vals = k_vals.view(1, 1)
        elif state.dim() == 2:
            state = state.unsqueeze(1)
            if action.dim() == 2: action = action.unsqueeze(1)
            if mask is not None and mask.dim() == 2: mask = mask.unsqueeze(1)
            if selected_indices is not None and selected_indices.dim() == 2:
                selected_indices = selected_indices.unsqueeze(1)
            if k_vals is not None and k_vals.dim() == 1:
                k_vals = k_vals.unsqueeze(1)

        B, T, _ = state.shape

        num_logits, region_logits = self.forward(state)  # [B,T,k], [B,T,N]
        scores = region_logits

        # valid
        if mask is None:
            invested = state[:, :, 2:self.num_regions+2]
            valid = (1.0 - invested).clamp(0, 1)
        else:
            valid = mask.clamp(0, 1)

        # observed k from sampled trajectory if available, fallback to action sum.
        if k_vals is not None:
            k_obs = k_vals.long().clamp(min=1, max=self.k)
        elif selected_indices is not None:
            k_obs = (selected_indices >= 0).sum(dim=-1).long().clamp(min=1, max=self.k)
        else:
            k_obs = action.sum(dim=-1).long().clamp(min=1, max=self.k)  # [B,T]
        k_idx = (k_obs - 1).clamp(min=0, max=self.k-1)              # [B,T]

        # log p(k)
        num_dist = torch.distributions.Categorical(logits=num_logits)
        logp = num_dist.log_prob(k_idx)  # [B,T]

        eps = 1e-12
        if selected_indices is not None:
            # Exact ordered subset log-prob under sequential without-replacement sampling.
            logp_subset = torch.zeros((B, T), device=self.device, dtype=torch.float32)
            for b in range(B):
                for t in range(T):
                    kk = int(min(k_obs[b, t].item(), self.k))
                    if kk <= 0:
                        continue
                    remaining = valid[b, t].clone()
                    for pick_idx in range(kk):
                        j = int(selected_indices[b, t, pick_idx].item())
                        if j < 0 or j >= self.num_regions:
                            logp_subset[b, t] += -50.0
                            continue
                        masked_scores = scores[b, t].masked_fill(remaining < 0.5, -1e9)
                        probs = torch.softmax(masked_scores, dim=-1)
                        if remaining[j] < 0.5:
                            logp_subset[b, t] += -50.0
                        else:
                            logp_subset[b, t] += torch.log(probs[j] + eps)
                            remaining[j] = 0.0
        else:
            # Fallback approximation (unordered subset) for compatibility.
            masked_scores = scores.masked_fill(valid < 0.5, -1e9)
            probs = torch.softmax(masked_scores, dim=-1)  # [B,T,N]
            logp_subset = (action * torch.log(probs + eps)).sum(dim=-1)  # [B,T]
            invalid_chosen = (action * (1.0 - valid)).sum(dim=-1)        # [B,T]
            logp_subset = logp_subset + (invalid_chosen * -50.0)

        logp = logp + logp_subset

        if orig_dim == 1:
            return logp[0, 0]
        if orig_dim == 2:
            return logp[0]
        return logp

        
    def sample_action_and_logprob(self, state, mask=None, deterministic=False, return_selected=False, return_k=False):
        """
        For PPO with MultiBinary actions.
        Distribution:
          1) sample k ~ Categorical(num_logits)  -> k in {1..self.k}
          2) sample k regions WITHOUT replacement using sequential categorical draws
             from masked region probabilities (softmax over logits on remaining set)
        Returns:
          action_binary: torch.float32 [B,T,N] (or squeezed)
          log_prob:      torch.float32 [B,T]   (sum of log probs)
        """
        # ---- to tensor & shape ----
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().to(self.device)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.FloatTensor(mask).to(self.device)
            else:
                mask = mask.to(self.device)

        original_dim = state.dim()
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)  # [1,1,F]
            if mask is not None:
                mask = mask.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2:
            state = state.unsqueeze(1)               # [B,1,F]
            if mask is not None:
                mask = mask.unsqueeze(1)

        B, T, _ = state.shape

        num_logits, region_logits = self.forward(state)  # [B,T,k], [B,T,N]

        # region logits -> scores (we'll use softmax for without-replacement draws)
        # NOTE: sigmoid is for independent Bernoulli; here we need a categorical score.
        scores = region_logits  # [B,T,N]

        # valid mask: if mask not provided, block already-invested ones based on state
        if mask is None:
            invested = state[:, :, 2:self.num_regions+2]          # [B,T,N]
            valid = (1.0 - invested).clamp(0, 1)                  # [B,T,N]
        else:
            valid = mask.clamp(0, 1)

        # sample k
        num_dist = torch.distributions.Categorical(logits=num_logits)  # [B,T]
        if deterministic:
            k_idx = torch.argmax(num_logits, dim=-1)  # [B,T] in 0..k-1
        else:
            k_idx = num_dist.sample()                 # [B,T]
        k_val = k_idx + 1                             # [B,T] in 1..k

        # build action by sequential without-replacement sampling
        action = torch.zeros((B, T, self.num_regions), device=self.device)
        selected = torch.full((B, T, self.k), -1, dtype=torch.long, device=self.device)
        logp_subset = torch.zeros((B, T), dtype=torch.float32, device=self.device)

        for b in range(B):
            for t in range(T):
                valid_idx = torch.nonzero(valid[b, t] > 0.5).squeeze(1)
                if valid_idx.numel() == 0:
                    continue

                kk = int(k_val[b, t].item())
                kk = min(kk, valid_idx.numel(), self.k)

                remaining = valid[b, t].clone()
                for pick_idx in range(kk):
                    masked_scores = scores[b, t].clone().masked_fill(remaining < 0.5, -1e9)
                    probs = torch.softmax(masked_scores, dim=-1)

                    if deterministic:
                        j = torch.argmax(probs).item()
                    else:
                        j = torch.distributions.Categorical(probs=probs).sample().item()

                    action[b, t, j] = 1.0
                    selected[b, t, pick_idx] = j
                    logp_subset[b, t] += torch.log(probs[j] + 1e-12)
                    remaining[j] = 0.0

        logp_total = num_dist.log_prob(k_idx) + logp_subset

        if original_dim == 1:
            if return_selected and return_k:
                return action[0, 0], logp_total[0, 0], selected[0, 0], k_val[0, 0]
            if return_selected:
                return action[0, 0], logp_total[0, 0], selected[0, 0]
            if return_k:
                return action[0, 0], logp_total[0, 0], k_val[0, 0]
            return action[0, 0], logp_total[0, 0]
        if original_dim == 2:
            if return_selected and return_k:
                return action[0], logp_total[0], selected[0], k_val[0]
            if return_selected:
                return action[0], logp_total[0], selected[0]
            if return_k:
                return action[0], logp_total[0], k_val[0]
            return action[0], logp_total[0]
        if return_selected and return_k:
            return action, logp_total, selected, k_val
        if return_selected:
            return action, logp_total, selected
        if return_k:
            return action, logp_total, k_val
        return action, logp_total

    
    def preprocess_state(self, state):
        if isinstance(state, torch.Tensor):
            if state.device != self.device:
                state = state.to(self.device)
        else:
            state = torch.FloatTensor(state).to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2:
            state = state.unsqueeze(1)

        batch_size, time_steps, _ = state.size()

        current_step = state[:, :, 0].unsqueeze(-1)
        invest_state = state[:, :, 2:self.num_regions+2]
        demand_base = state[:, :, self.num_regions+2:2*self.num_regions+2]
        if demand_base.size(-1) != self.num_regions:
            demand_base = torch.zeros_like(invest_state)

        if self.period_num is not None and self.period_num > 0:
            current_step = current_step / float(self.period_num)

        invested_ratio = torch.sum(invest_state, dim=2, keepdim=True) / self.num_regions
        demand_mean = demand_base.mean(dim=2, keepdim=True)

        node_features = torch.cat([
            invest_state.view(batch_size, time_steps, self.num_regions, 1),
            invested_ratio.expand(-1, -1, self.num_regions).view(batch_size, time_steps, self.num_regions, 1),
            current_step.expand(-1, -1, self.num_regions).view(batch_size, time_steps, self.num_regions, 1),
            demand_base.view(batch_size, time_steps, self.num_regions, 1),
        ], dim=3)

        global_features = torch.cat([current_step, invested_ratio, demand_mean], dim=-1)  # [B,T,3]
        return node_features, global_features, batch_size, time_steps

   
    def forward(self, state):
        x_nodes, global_features, batch_size, time_steps = self.preprocess_state(state)
        x_raw = x_nodes.view(-1, self.node_feature_dim)

        x = x_nodes.view(batch_size * time_steps, self.num_regions, self.node_feature_dim)
        x = self.node_proj(x)
        region_ids = torch.arange(self.num_regions, device=self.device).unsqueeze(0)
        x = x + self.region_embed(region_ids)

        bt = batch_size * time_steps
        x_global = self.global_proj(global_features.view(bt, 3)).unsqueeze(1)
        x_global = x_global + self.cls_token.expand(bt, -1, -1)
        x = torch.cat([x_global, x], dim=1)  # [BT, N+1, H]

        for attn, anorm, ffn, fnorm in zip(self.attn_layers, self.attn_norms, self.ffn_layers, self.ffn_norms):
            x_attn, _ = attn(x, x, x, need_weights=False)
            x = anorm(x + x_attn)
            x = fnorm(x + ffn(x))

        x_global_out = x[:, 0, :]            # [BT,H]
        x_region_out = x[:, 1:, :]           # [BT,N,H]
        x = x_region_out.contiguous().view(-1, self.hidden_size)

        # Residual fusion to preserve region-wise raw information.
        x_raw = self.raw_feat_encoder(x_raw)
        x = self.res_fuse(torch.cat([x, x_raw], dim=-1))
        
        # batch_size, time_steps, num_regions, hidden_size]
        x = x.view(batch_size, time_steps, self.num_regions, -1)

        # 1. k logits from CLS token readout (no mean pooling bottleneck).
        cls_feature = x_global_out.view(batch_size, time_steps, self.hidden_size)
        num_logits = self.num_head(cls_feature)  # [batch_size, time_steps, k]
        
        # 2. 
        region_logits = self.region_node_head(x).squeeze(-1)  # [B,T,N]
        
        return num_logits, region_logits
    
    def get_action(self, state, mask=None, deterministic=False):
        # ---- to tensor & shape ----
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().to(self.device)
            elif not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
            else:
                mask = mask.to(self.device)

        orig_dim = state.dim()
        if orig_dim == 1:
            state = state.unsqueeze(0).unsqueeze(0)  # [1,1,F]
            if mask is not None: mask = mask.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 2:
            state = state.unsqueeze(1)               # [B,1,F]
            if mask is not None: mask = mask.unsqueeze(1)

        B, T, _ = state.shape

        with torch.no_grad():
            k_logits, r_logits = self.forward(state)       # [B,T,K], [B,T,N]
            r_probs = torch.sigmoid(r_logits)              # [B,T,N]

            if mask is not None:
                r_probs = r_probs * mask

            r_probs = r_probs.clamp(1e-6, 1 - 1e-6)

            actions = torch.zeros((B, T, self.num_regions), device=self.device)

            for t in range(T):
                # 1. ?
                if deterministic:
                    k_pred = torch.argmax(k_logits[:, t, :], dim=-1) + 1
                else:
                    k_pred = torch.distributions.Categorical(logits=k_logits[:, t, :]).sample() + 1
                
                # 2. [] ""
                # ?<= ( * k) + ?
                period_num = self.period_num
                if period_num is not None:
                    current_step = state[:, t, 0].long()
                    
                    # (?mask ask=1 )
                    if mask is not None:
                        left_regions = mask[:, t, :].sum(dim=-1).long()
                    else:
                        # asktate[2:]?
                        left_regions = self.num_regions - state[:, t, 2:self.num_regions+2].sum(dim=-1).long()
                    
                    # (?
                    # total_steps = period_num + 1.  steps_after_this = total_steps - (current_step + 1)
                    steps_after = (period_num - current_step).clamp(min=0)
                    
                    # ?
                    future_capacity = steps_after * self.k
                    
                    # ?
                    must_pick = (left_regions - future_capacity).clamp(min=0)
                    
                    #  1 ?(?
                    floor_k = torch.max(must_pick, torch.tensor(1, device=self.device))
                    
                    # ?k: ? ? ?
                    k_final = torch.max(k_pred, floor_k)
                    
                    # ?k
                    limit = torch.min(torch.tensor(self.k, device=self.device), left_regions)
                    k_final = torch.min(k_final, limit)
                else:
                    k_final = k_pred
                k_val = k_final
  
                # choose top-k by probability among valid
                for b in range(B):
                    if mask is not None:
                        valid_idx = torch.nonzero(mask[b, t] > 0.5).squeeze(-1)
                    else:
                        valid_idx = torch.arange(self.num_regions, device=self.device)

                    if valid_idx.numel() == 0:
                        continue

                    kk = int(min(k_val[b].item(), valid_idx.numel()))
                    if kk <= 0:
                        continue
                        
                    # Get probabilities for valid regions
                    probs_valid = r_probs[b, t, valid_idx]

                    if deterministic:
                        # GREEDY: Pick top-k highest probs
                        topk_indices = torch.topk(probs_valid, kk, largest=True).indices
                        chosen = valid_idx[topk_indices]
                    else:
                        # SAMPLING: Pick based on probability distribution (Weighted Random)
                        # Normalize to sum to 1 for Categorical
                        probs_valid_norm = probs_valid / probs_valid.sum()
                        
                        # Sample without replacement logic
                        # (Simple approximation: sample k times, or use Multinomial if k>1)
                        try:
                            # torch.multinomial handles sampling without replacement 
                            # if replacement=False
                            sampled_indices = torch.multinomial(probs_valid_norm, kk, replacement=False)
                            chosen = valid_idx[sampled_indices]
                        except RuntimeError:
                            # Fallback if numerical errors occur (e.g. all probs 0)
                            topk_indices = torch.topk(probs_valid, kk, largest=True).indices
                            chosen = valid_idx[topk_indices]

                    actions[b, t, chosen] = 1.0


        # ---- squeeze back & to numpy ----
        if orig_dim == 1:
            a = actions[0, 0]
        elif orig_dim == 2:
            a = actions[:, 0]      # [B,N]
        else:
            a = actions            # [B,T,N] (?env ?

        return a.detach().cpu().numpy().astype(np.float32)
    
    def evaluate(self, state, mask=None, temperature=0.5):
        """
        Returns:
        action_st: [B,T,N]  (0/1 with straight-through gradients)
        logp:      [B,T]    log prob of the sampled action (consistent with sampling)
        """
        # ---- to tensor & shape ----
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        elif mask is not None:
            mask = mask.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
            if mask is not None: mask = mask.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2:
            state = state.unsqueeze(1)
            if mask is not None: mask = mask.unsqueeze(1)

        B, T, _ = state.shape
        temp = max(float(temperature), 1e-3)
        k_logits, r_logits = self.forward(state)  # k_logits:[B,T,K], r_logits:[B,T,N]

        # masked region scores for sampling
        if mask is None:
            invested = state[:, :, 2:self.num_regions+2]
            valid = (1.0 - invested).clamp(0, 1)
        else:
            valid = mask.clamp(0, 1)

        action_out = torch.zeros(B, T, self.num_regions, device=self.device)
        logp_out   = torch.zeros(B, T, device=self.device)

        for t in range(T):
            dist_k = torch.distributions.Categorical(logits=k_logits[:, t, :] / temp)  # [B,K]
            k_idx = dist_k.sample()  # [B]
            k_val = k_idx + 1        # [B] in 1..K
            logp_k_all = dist_k.log_prob(k_val.long() - 1)  # [B]

            # ---- enforce feasibility (must_pick) ----
            period_num = self.period_num
            if period_num is not None:
                current_step = state[:, t, 0].long()
                left_regions = valid[:, t, :].sum(dim=-1).long()
                steps_after = (period_num - current_step).clamp(min=0)
                future_capacity = steps_after * self.k
                must_pick = (left_regions - future_capacity).clamp(min=0)
                floor_k = torch.max(must_pick, torch.zeros_like(must_pick) + 1)
                k_val = torch.max(k_val, floor_k)
                limit = torch.min(torch.zeros_like(left_regions) + self.k, left_regions)
                k_val = torch.min(k_val, limit)
                k_val = torch.where(left_regions == 0, torch.zeros_like(k_val), k_val)

            # straight-through soft action proxy (for Q gradient)
            masked_scores = r_logits[:, t, :].masked_fill(valid[:, t, :] < 0.5, -1e9)
            a_soft = torch.softmax(masked_scores / temp, dim=-1)

            a_hard = torch.zeros_like(a_soft)
            for b in range(B):
                valid_idx = torch.nonzero(valid[b, t] > 0.5).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue
                kk = int(min(k_val[b].item(), valid_idx.numel()))
                if kk <= 0:
                    continue

                remaining = valid[b, t].clone()
                logp_subset = 0.0
                for _ in range(kk):
                    masked_scores_bt = r_logits[b, t].masked_fill(remaining < 0.5, -1e9)
                    probs_bt = torch.softmax(masked_scores_bt / temp, dim=-1)
                    j = torch.distributions.Categorical(probs=probs_bt).sample().item()
                    a_hard[b, j] = 1.0
                    logp_subset = logp_subset + torch.log(probs_bt[j] + 1e-12)
                    remaining[j] = 0.0

                # log p(k) uses the final enforced k
                if k_val[b].item() > 0:
                    logp_k = logp_k_all[b]
                else:
                    logp_k = torch.tensor(0.0, device=self.device)
                logp_out[b, t] = torch.clamp(logp_k + logp_subset, -20, 20)

            # straight-through: hard action with soft gradient
            a_st = a_hard - a_soft.detach() + a_soft
            action_out[:, t, :] = a_st

        return action_out, logp_out

class DiscreteSACQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.action_dim = action_dim
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  
        )
    
    def forward(self, state, action):
        if state.dim() == 1: state = state.unsqueeze(0)
        if action.dim() == 1: action = action.unsqueeze(0)
        assert state.size(0) == action.size(0)
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class TransformerSACQNetwork(nn.Module):
    """
    Transformer critic for TSAC:
      Q(s, a) with region-wise token encoding aligned to TransformerPolicyNetwork.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2, num_heads=8, device='cuda', period_num=None):
        super().__init__()
        self.state_dim = state_dim
        self.num_regions = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.period_num = period_num
        self.node_feature_dim = 5  # invest_state, action, invested_ratio, current_step, demand_base

        self.node_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.region_embed = nn.Embedding(action_dim, hidden_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),  # current_step, invested_ratio, demand_mean, action_ratio
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            for _ in range(max(1, num_layers))
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(1, num_layers))])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.ReLU(),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            for _ in range(max(1, num_layers))
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(1, num_layers))])

        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.to(device)

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device).float()

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        elif not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device).float()

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        B = state.size(0)
        if action.size(0) != B:
            raise ValueError("TransformerSACQNetwork expects state/action batch size match")

        n = self.num_regions
        current_step = state[:, 0:1]
        if self.period_num is not None and self.period_num > 0:
            current_step = current_step / float(self.period_num)
        invest_state = state[:, 2:n+2]
        demand_base = state[:, n+2:2*n+2]
        if demand_base.size(-1) != n:
            demand_base = torch.zeros_like(invest_state)

        action = action[:, :n]
        invested_ratio = invest_state.mean(dim=-1, keepdim=True)
        demand_mean = demand_base.mean(dim=-1, keepdim=True)
        action_ratio = action.mean(dim=-1, keepdim=True)

        node_feat = torch.stack([
            invest_state,
            action,
            invested_ratio.expand(-1, n),
            current_step.expand(-1, n),
            demand_base,
        ], dim=-1)  # [B,N,5]

        x_nodes = self.node_proj(node_feat)
        rid = torch.arange(n, device=self.device).unsqueeze(0)
        x_nodes = x_nodes + self.region_embed(rid)

        global_feat = torch.cat([current_step, invested_ratio, demand_mean, action_ratio], dim=-1)
        x_global = self.global_proj(global_feat).unsqueeze(1) + self.cls_token.expand(B, -1, -1)

        x = torch.cat([x_global, x_nodes], dim=1)  # [B,N+1,H]
        for attn, anorm, ffn, fnorm in zip(self.attn_layers, self.attn_norms, self.ffn_layers, self.ffn_norms):
            x_attn, _ = attn(x, x, x, need_weights=False)
            x = anorm(x + x_attn)
            x = fnorm(x + ffn(x))

        cls_out = x[:, 0, :]
        combined_feat = torch.cat([cls_out, global_feat], dim=-1)
        return self.q_head(combined_feat)
             

class MLPPolicyNetwork(nn.Module):

    def __init__(self, state_dim, num_regions, k, hidden_size=256, device="cuda", period_num=None):
        super().__init__()
        self.num_regions = num_regions
        self.k = k
        self.hidden_size = hidden_size
        self.device = device
        self.input_dim = state_dim
        self.period_num = period_num

        self.base_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
        )

        # unified output: first k are num_logits, last num_regions are region_logits
        self.unified_head = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, k + num_regions),
        )

        self.to(device)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Use Xavier initialization for stability
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        state: [B, state_dim]
        returns:
          num_logits:   [B, k]
          region_logits:[B, num_regions]   (LOGITS, not probs)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.base_net(state)
        out = self.unified_head(x)  # [B, k+N]
        num_logits = out[..., : self.k]
        region_logits = out[..., self.k :]
        return num_logits, region_logits

    def get_action(self, state, mask=None, deterministic=False):
        """
        SAC execution action for env.step (NO grad).
        Uses the same sampling definition as evaluate() to avoid
        behavior/target-policy mismatch in off-policy learning.
        """
        with torch.no_grad():
            actions, _ = self.evaluate(
                state, mask, temperature=1.0, deterministic=deterministic
            )

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy().astype(np.float32)
        if isinstance(actions, np.ndarray):
            if actions.ndim == 3 and actions.shape[0] == 1 and actions.shape[1] == 1:
                actions = actions[0, 0]
            elif actions.ndim == 2 and actions.shape[0] == 1:
                actions = actions[0]
        return actions
    
    def evaluate(self, state, mask=None, temperature=0.5, deterministic=False):
        """
        Returns:
        action_st: [B,T,N]  (0/1 with straight-through gradients)
        logp:      [B,T]    log prob of the sampled action (consistent with sampling)
        """
        # ---- to tensor & shape ----
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device)
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        elif mask is not None:
            mask = mask.to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
            if mask is not None: mask = mask.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2:
            state = state.unsqueeze(1)
            if mask is not None: mask = mask.unsqueeze(1)

        B, T, _ = state.shape
        temp = max(float(temperature), 1e-3)
        k_logits, r_logits = self.forward(state)  # k_logits:[B,T,K], r_logits:[B,T,N]

        if mask is None:
            invested = state[:, :, 2:self.num_regions+2]
            valid = (1.0 - invested).clamp(0, 1)
        else:
            valid = mask.clamp(0, 1)

        action_out = torch.zeros(B, T, self.num_regions, device=self.device)
        logp_out   = torch.zeros(B, T, device=self.device)

        for t in range(T):
            dist_k = torch.distributions.Categorical(logits=k_logits[:, t, :] / temp)  # [B,K]
            if deterministic:
                k_idx = torch.argmax(k_logits[:, t, :], dim=-1)
            else:
                k_idx = dist_k.sample()  # [B]
            k_val = k_idx + 1        # [B] in 1..K
            logp_k_all = dist_k.log_prob(k_val.long() - 1)  # [B]

            # ---- enforce feasibility (must_pick) ----
            period_num = self.period_num
            if period_num is not None:
                current_step = state[:, t, 0].long()
                left_regions = valid[:, t, :].sum(dim=-1).long()
                steps_after = (period_num - current_step).clamp(min=0)
                future_capacity = steps_after * self.k
                must_pick = (left_regions - future_capacity).clamp(min=0)
                floor_k = torch.max(must_pick, torch.zeros_like(must_pick) + 1)
                k_val = torch.max(k_val, floor_k)
                limit = torch.min(torch.zeros_like(left_regions) + self.k, left_regions)
                k_val = torch.min(k_val, limit)
                k_val = torch.where(left_regions == 0, torch.zeros_like(k_val), k_val)

            masked_scores = r_logits[:, t, :].masked_fill(valid[:, t, :] < 0.5, -1e9)
            a_soft = torch.softmax(masked_scores / temp, dim=-1)

            a_hard = torch.zeros_like(a_soft)
            for b in range(B):
                valid_idx = torch.nonzero(valid[b, t] > 0.5).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue
                kk = int(min(k_val[b].item(), valid_idx.numel()))
                if kk <= 0:
                    continue

                remaining = valid[b, t].clone()
                logp_subset = 0.0
                for _ in range(kk):
                    masked_scores_bt = r_logits[b, t].masked_fill(remaining < 0.5, -1e9)
                    probs_bt = torch.softmax(masked_scores_bt / temp, dim=-1)
                    if deterministic:
                        j = torch.argmax(probs_bt).item()
                    else:
                        j = torch.distributions.Categorical(probs=probs_bt).sample().item()
                    a_hard[b, j] = 1.0
                    logp_subset = logp_subset + torch.log(probs_bt[j] + 1e-12)
                    remaining[j] = 0.0

                if k_val[b].item() > 0:
                    logp_k = logp_k_all[b]
                else:
                    logp_k = torch.tensor(0.0, device=self.device)
                logp_out[b, t] = torch.clamp(logp_k + logp_subset, -20, 20)

            a_st = a_hard - a_soft.detach() + a_soft
            action_out[:, t, :] = a_st

        if T == 1:
            action_out = action_out.squeeze(1)   # [B,1,N] -> [B,N]

        return action_out, logp_out

# Backward-compatible alias for older imports.
GCNPolicyNetwork = TransformerPolicyNetwork

class TSAC_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim, k, max_epochs=2000, period_num=None):
        self.replay_buffer = replay_buffer
        self.device = device
        self.reward_mean = torch.zeros(1).to(self.device)
        self.reward_std = torch.ones(1).to(self.device)
        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.k = k
        
        # Symmetric Transformer critics for TSAC (match Transformer actor inductive bias).
        self.soft_q_net1 = TransformerSACQNetwork(
            state_dim, action_dim, hidden_dim, num_layers=2, num_heads=8, device=self.device, period_num=period_num
        ).to(self.device).float()
        self.soft_q_net2 = TransformerSACQNetwork(
            state_dim, action_dim, hidden_dim, num_layers=2, num_heads=8, device=self.device, period_num=period_num
        ).to(self.device).float()
        self.target_soft_q_net1 = copy.deepcopy(self.soft_q_net1).float()
        self.target_soft_q_net2 = copy.deepcopy(self.soft_q_net2).float()
        
        # Transformer Policy
        self.num_regions = action_dim 
        self.policy_net = TransformerPolicyNetwork(self.num_regions, hidden_dim, k, period_num=period_num).to(self.device)
        
        # ?alpha
        self.log_alpha = torch.tensor([math.log(ALPHA_INIT)], dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
    
        # 
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        # 
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        # ?
        soft_q_lr = 1e-4
        policy_lr = 3e-4
        alpha_lr = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr, weight_decay=1e-5)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr, weight_decay=1e-5)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.policy_delay = 2
        self.update_step = 0
        self.last_policy_loss = 0.0
        self.last_log_prob_norm = 0.0

    def set_episode(self, episode):
        return

    def generate_mask(self, state):
        if state.dim() == 3:
            invest_state = state[:, :, 2:self.num_regions + 2]
        else:
            invest_state = state[:, 2:self.num_regions + 2]
        # Create mask: 1 where investment is 0, 0 otherwise
        mask = (invest_state == 0).float()
        return mask

    def update(self, batch_size, reward_scale=1.0, auto_entropy=True, target_entropy=None, gamma=0.99, soft_tau=5e-3):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        if target_entropy is None:
            target_entropy = -float(max(1, self.k))
        
        # Calculate temperature
        current_temp = max(0.4, 1.2 - (self.current_epoch / self.max_epochs))
        
        state = state.float().to(self.device)
        action = action.float().to(self.device)
        next_state = next_state.float().to(self.device)
        reward = reward.float().to(self.device)
        done = done.float().to(self.device)
        if reward.dim() == 1:
            reward = reward.view(-1, 1)
        if done.dim() == 1:
            done = done.view(-1, 1)
        
        # Get Current Q Values
        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        
        # Training Q Function
        self.alpha = self.log_alpha.exp()

        # ---- Target Q Calculation ----
        next_mask = self.generate_mask(next_state)
        with torch.no_grad():
            next_action, next_log_prob = self.policy_net.evaluate(next_state, next_mask, temperature=current_temp)
            if next_action.dim() == 3:
                next_action = next_action.squeeze(1)
            if next_log_prob.dim() == 2:
                next_log_prob = next_log_prob.squeeze(1)
            next_log_prob_term = next_log_prob.unsqueeze(-1)
            
        next_q1 = self.target_soft_q_net1(next_state, next_action)
        next_q2 = self.target_soft_q_net2(next_state, next_action)
        target_q_min = torch.min(next_q1, next_q2)
        
        target_q_value = reward * reward_scale + (1 - done) * gamma * (target_q_min - self.alpha * next_log_prob_term)
        
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), max_norm=1.0)
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), max_norm=1.0)
        self.soft_q_optimizer2.step()
        
        # ---- Delayed actor / alpha update ----
        self.update_step += 1
        do_actor_update = (self.update_step % self.policy_delay == 0)
        q_slice = torch.min(predicted_q_value1, predicted_q_value2)
        policy_loss_value = self.last_policy_loss
        log_prob_norm_value = self.last_log_prob_norm

        if do_actor_update:
            mask = self.generate_mask(state)
            current_action, log_prob = self.policy_net.evaluate(state, mask, temperature=current_temp)
            if current_action.dim() == 3:
                current_action = current_action.squeeze(1)
            if log_prob.dim() == 2:
                log_prob = log_prob.squeeze(1)

            predict_q1 = self.soft_q_net1(state, current_action)
            predict_q2 = self.soft_q_net2(state, current_action)
            predicted_new_q_value = torch.min(predict_q1, predict_q2)

            log_prob_term = log_prob.unsqueeze(-1)
            q_slice = predicted_new_q_value
            logp_slice = log_prob_term
            policy_loss = (self.alpha * logp_slice - q_slice).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
            self.policy_optimizer.step()

            policy_loss_value = policy_loss.item()
            log_prob_norm_value = logp_slice.mean().item()
            self.last_policy_loss = policy_loss_value
            self.last_log_prob_norm = log_prob_norm_value

            if auto_entropy:
                alpha_loss = -(self.log_alpha * (logp_slice.detach() + target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(math.log(ALPHA_MIN), math.log(ALPHA_MAX))
                self.alpha = self.log_alpha.exp().detach()
            else:
                self.alpha = 1.
        
        # Soft Update Targets
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_((1.0 - soft_tau) * target_param.data + soft_tau * param.data)

        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_((1.0 - soft_tau) * target_param.data + soft_tau * param.data)

        return {
                    "q_loss": (q_value_loss1.item() + q_value_loss2.item()) / 2,
                    "policy_loss": policy_loss_value,
                    "avg_q": q_slice.mean().item(),
                    "avg_alpha": self.alpha.item() if hasattr(self.alpha, "item") else self.alpha,
                    "log_prob_norm": log_prob_norm_value,
                }

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        # PU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # GPUPU?
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1', map_location=device))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2', map_location=device))
        self.policy_net.load_state_dict(torch.load(path+'_policy', map_location=device))

        # ?
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()
    
        # CPUPU?
        self.soft_q_net1.to(device)
        self.soft_q_net2.to(device)
        self.policy_net.to(device)


class SAC_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim, k, batch_size, max_epochs=2000,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.0, device='cuda', period_num=None):
        
        self.reward_mean = torch.zeros(1, dtype=torch.float32).to(device)  
        self.reward_std = torch.ones(1, dtype=torch.float32).to(device)
        
        self.policy_net = MLPPolicyNetwork(state_dim, action_dim, k, hidden_dim, period_num=period_num).to(device).float()
        self.soft_q_net1 = DiscreteSACQNetwork(state_dim, action_dim, hidden_dim).to(device).float()
        self.soft_q_net2 = DiscreteSACQNetwork(state_dim, action_dim, hidden_dim).to(device).float()
        
        self.target_soft_q_net1 = copy.deepcopy(self.soft_q_net1).float()
        self.target_soft_q_net2 = copy.deepcopy(self.soft_q_net2).float()
        
        self.actor_optim = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q_optim1 = optim.Adam(self.soft_q_net1.parameters(), lr=1e-4, weight_decay=1e-5)
        self.q_optim2 = optim.Adam(self.soft_q_net2.parameters(), lr=1e-4, weight_decay=1e-5)
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.num_regions = action_dim 
        self.device = device
        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.k = k
        
        self.target_entropy = -float(max(1, k))
        # ?alpha
        self.log_alpha = torch.tensor([math.log(ALPHA_INIT)], dtype=torch.float32, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        # Keep a higher temperature floor to avoid exploration collapse.
        self.alpha_min = max(ALPHA_MIN, 5e-3)
        self.alpha_max = ALPHA_MAX
        # self.alpha_optim = None
        
        self.replay_buffer = replay_buffer
        self.policy_delay = 2
        self.update_step = 0
        self.last_policy_loss = 0.0
        self.last_log_prob_norm = 0.0
        
    def update(self, batch_size, reward_scale=1.0):
        current_temp = max(0.4, 1.2 - (self.current_epoch / self.max_epochs))
        # ---- sample batch ----
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = states.float().to(self.device)
        actions = actions.float().to(self.device)
        next_states = next_states.float().to(self.device)

        rewards = rewards.float().to(self.device)
        dones = dones.float().to(self.device)

        # ---- force shapes ----
        if rewards.dim() == 1:
            rewards = rewards.view(-1, 1)   # [B,1]
        if dones.dim() == 1:
            dones = dones.view(-1, 1)       # [B,1]

        # =========================================================
        # 1) Critic (Q) update
        # =========================================================
        with torch.no_grad():
            next_mask = self.generate_mask(next_states)  # should be torch tensor [B, action_dim]
            if isinstance(next_mask, np.ndarray):
                next_mask = torch.from_numpy(next_mask).float().to(self.device)
            else:
                next_mask = next_mask.float().to(self.device)

            next_actions, next_log_probs = self.policy_net.evaluate(next_states, next_mask, temperature=current_temp)

            target_q1 = self.target_soft_q_net1(next_states, next_actions)  # [B,1]
            target_q2 = self.target_soft_q_net2(next_states, next_actions)  # [B,1]
            target_q_min = torch.min(target_q1, target_q2)                  # [B,1]

            next_log_probs_term = next_log_probs

            # SAC target: r + (1-d) * gamma * (Qmin - alpha * logpi)
            target_q = rewards*reward_scale + (1.0 - dones) * self.gamma * (target_q_min - self.alpha * next_log_probs_term)  # [B,1]

        current_q1 = self.soft_q_net1(states, actions)  # [B,1]
        current_q2 = self.soft_q_net2(states, actions)  # [B,1]

        q_loss1 = F.mse_loss(current_q1, target_q)
        q_loss2 = F.mse_loss(current_q2, target_q)

        self.q_optim1.zero_grad(set_to_none=True)
        q_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), 1.0)
        self.q_optim1.step()

        self.q_optim2.zero_grad(set_to_none=True)
        q_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), 1.0)
        self.q_optim2.step()

        self.update_step += 1
        do_actor_update = (self.update_step % self.policy_delay == 0)
        q_new_min = torch.min(current_q1, current_q2)
        policy_loss_value = self.last_policy_loss
        log_prob_norm_value = self.last_log_prob_norm

        if do_actor_update:
            # =========================================================
            # 2) Actor update (delayed)
            # =========================================================
            for p in self.soft_q_net1.parameters():
                p.requires_grad_(False)
            for p in self.soft_q_net2.parameters():
                p.requires_grad_(False)

            mask = self.generate_mask(states)
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().to(self.device)
            else:
                mask = mask.float().to(self.device)

            new_actions, log_probs = self.policy_net.evaluate(states, mask, temperature=current_temp)
            q1_new = self.soft_q_net1(states, new_actions)
            q2_new = self.soft_q_net2(states, new_actions)
            q_new_min = torch.min(q1_new, q2_new)

            log_prob_term = log_probs
            policy_loss = (self.alpha * log_prob_term - q_new_min).mean()

            self.actor_optim.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.actor_optim.step()

            for p in self.soft_q_net1.parameters():
                p.requires_grad_(True)
            for p in self.soft_q_net2.parameters():
                p.requires_grad_(True)

            policy_loss_value = policy_loss.item()
            log_prob_norm_value = log_prob_term.mean().item()
            self.last_policy_loss = policy_loss_value
            self.last_log_prob_norm = log_prob_norm_value

            # =========================================================
            # 3) Alpha / entropy temperature update (delayed with actor)
            # =========================================================
            if self.alpha_optim is not None:
                alpha_loss = -(self.log_alpha * (log_prob_term.detach() + self.target_entropy)).mean()

                self.alpha_optim.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optim.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(math.log(self.alpha_min), math.log(self.alpha_max))
                self.alpha = self.log_alpha.exp().detach()
        
        self.alpha = self.log_alpha.exp().item()

        # =========================================================
        # 4) Soft update target critics
        # =========================================================
        with torch.no_grad():
            for param, target_param in zip(self.soft_q_net1.parameters(), self.target_soft_q_net1.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for param, target_param in zip(self.soft_q_net2.parameters(), self.target_soft_q_net2.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

        return {
                    "q_loss": (q_loss1.item() + q_loss2.item()) / 2,
                    "policy_loss": policy_loss_value,
                    "avg_q": q_new_min.mean().item(),
                    "avg_alpha": self.alpha.item() if hasattr(self.alpha, "item") else self.alpha,
                    "log_prob_norm": log_prob_norm_value
                }

    def generate_mask(self, state):
        batch_size, _ = state.shape
        invest_state = state[:, 2:self.num_regions + 2]
        # Create mask: 1 where investment is 0, 0 otherwise
        mask = (invest_state == 0).float()
        return mask

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

        self.soft_q_net1.load_state_dict(torch.load(path+'_q1', map_location=device))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2', map_location=device))
        self.policy_net.load_state_dict(torch.load(path+'_policy', map_location=device))
        
        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()
    
        self.soft_q_net1.to(device)
        self.soft_q_net2.to(device)
        self.policy_net.to(device)


def plot(rewards,file_name):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig(str(file_name)+'.png')
    # plt.show()

def swap_stages(sequence):
    if len(sequence) <= 1:
        return copy.deepcopy(sequence)
        
    new_seq = copy.deepcopy(sequence)
    i, j = random.sample(range(len(new_seq)), 2)
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq

def replace_regions(sequence):
    if len(sequence) <= 1:
        return copy.deepcopy(sequence)
        
    new_seq = copy.deepcopy(sequence)
    
    stage_indices = random.sample(range(len(new_seq)), 2)
    stage1, stage2 = stage_indices
    
    if not new_seq[stage1] or not new_seq[stage2]:
        return new_seq

    region1_idx = random.randint(0, len(new_seq[stage1])-1)
    region2_idx = random.randint(0, len(new_seq[stage2])-1)
    
    # 
    new_seq[stage1][region1_idx], new_seq[stage2][region2_idx] = \
        new_seq[stage2][region2_idx], new_seq[stage1][region1_idx]
    
    return new_seq

def redistribute_regions(sequence, period_num, k):

    if len(sequence) <= 1:
        return copy.deepcopy(sequence)
        
    new_seq = copy.deepcopy(sequence)
    

    source_stage = random.randint(0, len(new_seq)-1)
    

    if len(new_seq[source_stage]) <= 1:
        return new_seq
    

    region_idx = random.randint(0, len(new_seq[source_stage])-1)
    region = new_seq[source_stage].pop(region_idx)
    

    if len(new_seq) < period_num and random.random() < 0.3:

        insert_pos = random.randint(0, len(new_seq))
        new_seq.insert(insert_pos, [region])
    else:

        target_stage = random.randint(0, len(new_seq)-1)
        if len(new_seq[target_stage]) < k:
            new_seq[target_stage].append(region)
        else:
            replace_idx = random.randint(0, k-1)
            replaced_region = new_seq[target_stage][replace_idx]
            new_seq[target_stage][replace_idx] = region
            

            new_seq[source_stage].append(replaced_region)
    

    new_seq = [stage for stage in new_seq if stage]
    
    return new_seq


def verify_sequence(sequence, region_num):
    all_regions = set(range(1, region_num + 1))
    included = set()
    
    for stage in sequence:
        stage_set = set(stage)

        if len(stage) != len(stage_set):
            return False

        if not stage_set.isdisjoint(included):
            return False
        included.update(stage_set)
    

    return included == all_regions


def pure_SAC_train(file_path, model_path, k, max_episodes, f_a=0.0, p1=1.0, p2=1.0, distribution='gamma', warmup_episodes=20):
    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    period_num = len(allarea_set) - 1
    c_wr, c_ir = strike_price(allarea_set[0].region_dict, p1, p2)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, period_num, c_wr, c_ir)
    env = InvestEnv_Train(region_num, allarea_set, roa)
    
    # ?Buffer ?Trainer
    replay_buffer = ReplayBuffer(int(1e6))
    state_dim = env.observation_space.shape[0]
    action_dim = env.num_regions
    hidden_dim = 256
    batch_size = 32  # LBatch
    
    trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_dim, k, batch_size, period_num=period_num)
    
    # 
    history = {"rewards": [], "q_loss": [], "policy_loss": [], "entropy": [], "alpha": []}
    eps_start, eps_end = 0.35, 0.05
    eps_decay_end = max(1, int(max_episodes * 0.7))
    incomplete_recent = deque(maxlen=100)
    
    print(f"Start PURE SAC Training | Episodes={max_episodes} | K={k}")
    start_time = time.time()
    best_reward = -float('inf')

    for epi in range(1, max_episodes + 1):
        trainer.current_epoch = epi
        if epi <= eps_decay_end:
            eps_ratio = (epi - 1) / max(1, (eps_decay_end - 1))
            eps_greedy = eps_start + (eps_end - eps_start) * eps_ratio
        else:
            eps_greedy = eps_end
        
        # === 2.  (Collection) ===
        state = env.reset(allarea_set)
        ep_states, ep_actions, ep_rewards, ep_next_states, ep_dones = [], [], [], [], []
        V_prev = 0.0
        
        for _ in range(period_num + 1):
            mask = env.generate_mask()
            # eterministic=False 
            if random.random() < eps_greedy:
                valid_idx = [idx for idx, m in enumerate(mask) if m > 0.5]
                action = np.zeros(action_dim, dtype=np.float32)
                if len(valid_idx) > 0:
                    kk = random.randint(1, min(k, len(valid_idx)))
                    chosen = random.sample(valid_idx, kk)
                    action[chosen] = 1.0
            else:
                action = trainer.policy_net.get_action(state, mask, deterministic=False)
            
            next_state, _, done, _ = env.step(action)
            
            current_seq_now = env.invest_sequence
            V_now, _ = roa.sequence_valuation(current_seq_now, distribution)
            r_step = V_now - V_prev
            V_prev = V_now

            ep_states.append(state)
            ep_actions.append(action)
            ep_next_states.append(next_state)
            ep_dones.append(done)
            ep_rewards.append(r_step)
            
            state = next_state
            if done: break
            
        # === 3.  (Evaluation) ===
        # ?
        current_seq = env.invest_sequence
        incomplete = not verify_sequence(current_seq, region_num)
        incomplete_recent.append(1 if incomplete else 0)
        if incomplete:
            if len(ep_rewards) > 0:
                ep_rewards = [INCOMPLETE_R_STEP] * len(ep_rewards)
            final_reward = -1.0
        else:
            val, exec_times = roa.sequence_valuation(current_seq, distribution)
            if f_a > 0:
                r_t = reward_calculate(current_seq, roa, exec_times)
                final_reward = (val * (1 - f_a) + f_a * sum(r_t.values()))
            else:
                final_reward = max(-1, val)
            
        history['rewards'].append(final_reward)
        if final_reward > best_reward:
            best_reward = final_reward

        # === 4. Terminal correction + scaling ===
        if (not incomplete) and len(ep_rewards) > 0:
            ep_rewards[-1] += (final_reward - V_prev)
        rewards_np = np.array(ep_rewards, dtype=np.float32) / REWARD_SCALE_DIV
        
        # === 5.  Buffer ===
        for i in range(len(ep_states)):
            replay_buffer.push(
                ep_states[i], ep_actions[i], rewards_np[i], ep_next_states[i], ep_dones[i]
            )
            
        # === 6.  (Update) ===
        stats = None
        # uffer?
        if epi > warmup_episodes and len(replay_buffer.buffer) > batch_size * 2:
            # episode
            for _ in range(2):
                stats = trainer.update(batch_size, reward_scale=1.0)
                
        # === 7.  ===
        if epi % 10 == 0:
            incomplete_ratio = 100.0 * (sum(incomplete_recent) / max(1, len(incomplete_recent)))
            log_str = f"Ep {epi} | Reward: {final_reward:.0f} | Best: {best_reward:.0f}"
            if stats:
                log_str += (f" | QL: {stats.get('q_loss',0):.2f} | "
                            f"Ent: {-stats.get('log_prob_norm',0):.2f} | "  # ?
                            f"Alpha: {stats.get('avg_alpha',0):.3f}")
                history['q_loss'].append(stats.get('q_loss',0))
                history['entropy'].append(-stats.get('log_prob_norm',0))
            log_str += f" | Eps: {eps_greedy:.3f}"
            log_str += f" | Incomplete(100): {incomplete_ratio:.1f}%"
            print(log_str)

    trainer.save_model(model_path)
    return history

def pure_TSAC_train(file_path, model_path, k, max_episodes, f_a=0.0, p1=1.0, p2=1.0, distribution='gamma', warmup_episodes=20):
    # 1. ?
    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    period_num = len(allarea_set) - 1
    min_steps = math.ceil(region_num / k)
    if period_num + 1 < min_steps:
        print(f"Warning: period_num={period_num} too short for region_num={region_num}, k={k}. "
              f"Need at least {min_steps} steps to cover all regions.")
    c_wr, c_ir = strike_price(allarea_set[0].region_dict, p1, p2)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, period_num, c_wr, c_ir)
    env = InvestEnv_Train(region_num, allarea_set, roa)
    
    replay_buffer = ReplayBuffer(int(1e6))
    state_dim = env.observation_space.shape[0]
    action_dim = env.num_regions
    hidden_dim = 256
    batch_size = 64
    
    trainer = TSAC_Trainer(replay_buffer, state_dim, action_dim, hidden_dim, k, max_epochs=max_episodes, period_num=period_num)
    
    eps_start, eps_end = 0.35, 0.05
    eps_decay_end = max(1, int(max_episodes * 0.7))
    
    history = {"rewards": [], "q_loss": [], "policy_loss": [], "entropy": [], "alpha": []}
    
    print(f"Start PURE TSAC Training | Episodes={max_episodes}")
    best_reward = -float('inf')

    for epi in range(1, max_episodes + 1):
        trainer.current_epoch = epi
        if epi <= eps_decay_end:
            eps_ratio = (epi - 1) / max(1, (eps_decay_end - 1))
            eps_greedy = eps_start + (eps_end - eps_start) * eps_ratio
        else:
            eps_greedy = eps_end
        trainer.set_episode(epi)
        
        # === 2.  () ===
        state = env.reset(allarea_set)
        ep_data = {'s':[], 'a':[], 'r':[], 'ns':[], 'd':[]}
        
        # []: ?
        V_prev = 0.0
        
        for _ in range(period_num + 1):
            mask = env.generate_mask()
            # TSAC
            if random.random() < eps_greedy:
                valid_idx = [idx for idx, m in enumerate(mask) if m > 0.5]
                action = np.zeros(action_dim, dtype=np.float32)
                if len(valid_idx) > 0:
                    kk = random.randint(1, min(k, len(valid_idx)))
                    chosen = random.sample(valid_idx, kk)
                    action[chosen] = 1.0
            else:
                action = trainer.policy_net.get_action(state, mask, deterministic=False)
            
            next_state, _, done, _ = env.step(action)
            
            # []:  (Marginal Reward)
            # ?
            current_seq_now = env.invest_sequence
            # ?
            V_now, _ = roa.sequence_valuation(current_seq_now, distribution)
            # ?
            r_step = V_now - V_prev
            V_prev = V_now
            
            ep_data['s'].append(state)
            ep_data['a'].append(action)
            ep_data['ns'].append(next_state)
            ep_data['d'].append(done)
            
            # []: ?r_step?0
            ep_data['r'].append(r_step)
            
            state = next_state
            if done: break

        # === 3. ?Elite  ===
        # ?( History ?Elite  f_a )
        current_seq = env.invest_sequence

        #  r_step ?
        incomplete = not verify_sequence(current_seq, region_num)
        if incomplete:
            if len(ep_data['r']) > 0:
                ep_data['r'] = [INCOMPLETE_R_STEP] * len(ep_data['r'])
            final_reward = -1.0
        else:
            val, exec_times = roa.sequence_valuation(current_seq, distribution)
            if f_a > 0:
                r_t = reward_calculate(current_seq, roa, exec_times)
                final_reward = (val * (1 - f_a) + f_a * sum(r_t.values()))
            else:
                final_reward = max(-1, val)
            if len(ep_data['r']) > 0:
                ep_data['r'][-1] += (final_reward - V_prev)
        
        history['rewards'].append(final_reward)
        if final_reward > best_reward:
            best_reward = final_reward
            
        # === 4. ?Buffer ===
        # []:  Backward Discounting  (for i in reversed...)
        #  ep_data['r']  LSTM  r_step
        
        # ?incomplete?
        rewards_np = np.array(ep_data['r'], dtype=np.float32) / REWARD_SCALE_DIV
        
        # Push transitions
        for i in range(len(ep_data['s'])):
            replay_buffer.push(
                ep_data['s'][i], ep_data['a'][i], rewards_np[i], ep_data['ns'][i], ep_data['d'][i]
            )
        
        # === 5.  ===
        stats = None
        if epi > warmup_episodes and len(replay_buffer.buffer) > batch_size * 2:
            # TSAC
            for _ in range(2):
                stats = trainer.update(
                    batch_size,
                    reward_scale=1.0,
                    auto_entropy=True
                )

        # === 6.  ===
        if epi % 10 == 0:
            log_str = f"Ep {epi} | Reward: {final_reward:.0f} | Best: {best_reward:.0f}"
            if stats:
                log_str += (f" | QL: {stats.get('q_loss',0):.2f} | "
                            f"Ent: {-stats.get('log_prob_norm',0):.2f} | "
                            f"Alpha: {stats.get('avg_alpha',0):.3f}")
                history['q_loss'].append(stats.get('q_loss',0))
                history['entropy'].append(-stats.get('log_prob_norm',0))
            log_str += f" | Eps: {eps_greedy:.3f}"
            print(log_str)
            
    trainer.save_model(model_path)
    return history

def sa_ts_train(file_path, model_path, k, max_episodes, f_a=0.2, distribution='gamma'):
    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    period_num = len(allarea_set) - 1
    c_wr, c_ir = strike_price(allarea_set[0].region_dict)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, period_num, c_wr, c_ir)
    env = InvestEnv_Train(region_num, allarea_set, roa)
    rewards_history = []

    # SA
    T_init = 100.0
    T_final = 1e-1
    cr = 0.85
    nbr_outer_iter = int(math.log(T_final / T_init) / math.log(cr))
    nbr_inner_iter = int(max_episodes / nbr_outer_iter)
    start_time = time.time()
    run_time = 0

    # TS
    tabu_tenure = 7  # ?
    tabu_list = deque(maxlen=tabu_tenure)
    aspiration_criterion = float('-inf')  # 

    # ?
    s_list = random_sequence_generation(region_num, k, 1, period_num)
    s = s_list[0]
    # ?
    sequence_value, execution_times = roa.sequence_valuation(s, distribution)
    if sequence_value < 0:
        sequence_value = -1
    if f_a > 0:
        reward_t_dict = reward_calculate(s, roa, execution_times)
        cost_obj = (sequence_value * (1-f_a) + f_a * sum(reward_t_dict.values()))
    else:
        cost_obj = sequence_value

    s_best = copy.deepcopy(s)
    best_cost = cost_obj

    # 
    current_signature = sequence_signature(s)
    tabu_list.append(current_signature)

    T_current = T_init
    outer_counter = 0
    epi = 0

    # ?
    while outer_counter < nbr_outer_iter and epi < max_episodes:
        for inner_counter in range(nbr_inner_iter):
            epi += 1
            rewards_history.append(cost_obj)
            # 
            op_choice = random.random()
            if op_choice < 0.1:
                s_prime = swap_stages(s)
            elif op_choice < 0.2:
                s_prime = replace_regions(s)
            else:
                s_prime = redistribute_regions(s, period_num, k)

            # ?
            if not verify_sequence(s_prime, region_num) or len(s_prime) > period_num:
                continue

            # 
            prime_signature = sequence_signature(s_prime)

            # 
            sequence_value_prime, execution_times_prime = roa.sequence_valuation(s_prime, distribution)
            if sequence_value_prime < 0:
                sequence_value_prime = -1
            if f_a > 0:
                reward_t_dict_prime = reward_calculate(s_prime, roa, execution_times_prime)
                cost_obj_prime = (sequence_value_prime * (1-f_a) + f_a * sum(reward_t_dict_prime.values()))
            else:
                cost_obj_prime = sequence_value_prime

            # 
            is_tabu = prime_signature in tabu_list
            meets_aspiration = cost_obj_prime > aspiration_criterion

            if is_tabu and not meets_aspiration:
                continue  # ?

            # ?
            tabu_list.append(prime_signature)

            # ?
            if cost_obj_prime > best_cost:
                aspiration_criterion = cost_obj_prime * 1.1

            # 
            delta_cost = cost_obj_prime - cost_obj

            # Metropolis
            if delta_cost > 0 or random.random() < math.exp(delta_cost / T_current):
                s = copy.deepcopy(s_prime)
                cost_obj = cost_obj_prime

                # 
                if cost_obj > best_cost:
                    s_best = copy.deepcopy(s)
                    best_cost = cost_obj

            # print(f"Episode {epi} | Reward: {cost_obj:.2f} | Tabu size: {len(tabu_list)} | Best: {best_cost:.2f}")
            print(f"Episode {epi} | Reward: {cost_obj:.2f} | Invest_sequence: {s} | Average Reward: {sum(rewards_history)/len(rewards_history):.2f}")

            # ?
            run_time = time.time() - start_time

        # 
        T_current *= cr
        outer_counter += 1

    np.save(get_save_path(model_path), rewards_history)
    return s_best, best_cost, run_time

def sa_ts_TSAC_train(file_path, model_path, k, max_episodes, train=True,
                      f_a=0.0, p1=1, p2=1, distribution='gamma', pretrain_episodes=100, warmup_episodes=100):
    
    if train and pretrain_episodes > 0:
        print(f"=== Phase 1: Pre-training RL for {pretrain_episodes} episodes ===")
        # ?pure_TSAC_train
        # ?model_path
        pure_TSAC_train(file_path, model_path, k, pretrain_episodes, 
                         f_a, p1, p2, distribution)
        print("=== Phase 1 Complete. Model saved. Starting Phase 2: SA/TS Hybrid ===")

    # Init Buffers
    replay_buffer_size = int(1e6)
    replay_buffer = ReplayBuffer(replay_buffer_size) 
    
    # Load Environment
    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    period_num = len(allarea_set) - 1 
    min_steps = math.ceil(region_num / k)
    if period_num + 1 < min_steps:
        print(f"Warning: period_num={period_num} too short for region_num={region_num}, k={k}. "
              f"Need at least {min_steps} steps to cover all regions.")
    c_wr, c_ir = strike_price(allarea_set[0].region_dict, p1, p2)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, period_num, c_wr, c_ir)
    env = InvestEnv_Train(region_num, allarea_set, roa)

    # Init Trainer
    state_dim = env.observation_space.shape[0]
    action_dim = env.num_regions
    hidden_dim = 256
    
    # Hyperparams
    batch_size = 64
    
    trainer = TSAC_Trainer(replay_buffer, state_dim, action_dim, hidden_dim, k, period_num=period_num)
    
    try:
        trainer.load_model(model_path)
        print("Loaded model for Hybrid phase.")
    except:
        print("No pre-trained model found, starting from scratch (or random).")

    # Logging Lists
    history = {
        "rewards": [],      # (Tabu)
        "q_loss": [],
        "policy_loss": [],
        "avg_q": [],
        "entropy": [], 
        "alpha": []
    }

    # SA/TS Params
    T_init, T_final, cr = 100.0, 1e-1, 0.85
    nbr_outer_iter = int(math.log(T_final / T_init) / math.log(cr))
    nbr_inner_iter = max(1, int(max_episodes / nbr_outer_iter))
    
    tabu_list = deque(maxlen=7)
    aspiration_criterion = float('-inf')
    
    # Initial Solution
    state = env.reset(allarea_set)
    for _ in range(period_num + 1):
        mask = env.generate_mask()
        action = trainer.policy_net.get_action(state, mask,deterministic=True)
        state, _, done, _ = env.step(action)
        # if done: break
    s = env.invest_sequence

    incomplete_init = not verify_sequence(s, region_num)
    if incomplete_init:
        cost_obj = -1.0
    else:
        sequence_value, execution_times = roa.sequence_valuation(s, distribution)
        sequence_value = max(-1, sequence_value)
        cost_obj = sequence_value if f_a == 0 else (sequence_value*(1-f_a) + f_a*sum(reward_calculate(s, roa, execution_times).values()))
        
    s_best = copy.deepcopy(s)
    best_cost = cost_obj
    tabu_list.append(sequence_signature(s))
    
    T_current = T_init
    epi = 0
    
    print(f"Start Hybrid Training | T_init={T_init}")

    while epi < max_episodes:
        for _ in range(nbr_inner_iter):
            epi += 1
            trainer.current_epoch = epi + pretrain_episodes #  epoch 
            
            
            # --- Neighbor Generation ---
            op_choice = random.random()
            if op_choice < 0.25: s_prime = swap_stages(s)
            elif op_choice < 0.5: s_prime = replace_regions(s)
            else:
                state = env.reset(allarea_set)
                for _ in range(period_num + 1):
                    mask = env.generate_mask()
                    action = trainer.policy_net.get_action(state, mask,deterministic=False)
                    state, _, done, _ = env.step(action)
                    # if done: break
                s_prime = env.invest_sequence
                print(s_prime)

            incomplete_prime = not verify_sequence(s_prime, region_num)
            if incomplete_prime:
                cost_obj_prime = -1.0
                execution_times_prime = {}
            else:
                sequence_value_prime, execution_times_prime = roa.sequence_valuation(s_prime, distribution)
                sequence_value_prime = max(-1, sequence_value_prime)
                cost_obj_prime = sequence_value_prime if f_a == 0 else (sequence_value_prime*(1-f_a) + f_a*sum(reward_calculate(s_prime, roa, execution_times_prime).values()))
            
            #Teacher Forcing  -  RL
            is_new_best = cost_obj_prime > best_cost
            if train and (is_new_best or random.random() < 0.01):
                state = env.reset(allarea_set)
                ep_data = {'s':[], 'a':[], 'r':[], 'ns':[], 'd':[]}
                
                # Teacher Forcing path
                invest_sequence = s_prime
                seq_partial = [[] for _ in range(len(invest_sequence))]
                V_prev = 0.0

                for step in range(period_num + 1):
                    mask = env.generate_mask()
                    # Construct action from teacher sequence
                    action = np.zeros(env.num_regions)
                    for h, t_exec in execution_times_prime.items():
                        if t_exec == env.t:
                            for r_id in invest_sequence[h]: action[r_id-1] = 1
                            seq_partial[h] = list(invest_sequence[h])
                    
                    next_state, _, done, _ = env.step(action)
                    
                    # Incremental Reward (Marginal Reward)
                    V_now, _ = roa.sequence_valuation(seq_partial, distribution)
                    r_step = V_now - V_prev
                    V_prev = V_now
                    
                    ep_data['s'].append(state)
                    ep_data['a'].append(action)
                    ep_data['r'].append(r_step) # []  r_step
                    ep_data['ns'].append(next_state)
                    ep_data['d'].append(done)
                    
                    state = next_state
                    # if done: break
                
                #  r_step ?
                if incomplete_prime and len(ep_data['r']) > 0:
                    ep_data['r'] = [INCOMPLETE_R_STEP] * len(ep_data['r'])

                # === REWARD STANDARDIZATION === ()
                rewards_np = np.array(ep_data['r'], dtype=np.float32)
                ep_data['r'] = (rewards_np / REWARD_SCALE_DIV).tolist()

                # Push to Buffer
                for i in range(len(ep_data['s'])):
                    replay_buffer.push(
                        ep_data['s'][i], ep_data['a'][i], ep_data['r'][i], ep_data['ns'][i], ep_data['d'][i]
                    )
                
                if epi > warmup_episodes and len(replay_buffer.buffer) > batch_size:
                    for _ in range(2): # 
                        stats = trainer.update(batch_size, auto_entropy=True)
                        if stats:
                            history['q_loss'].append(stats.get('q_loss', 0))
                            history['policy_loss'].append(stats.get('policy_loss', 0))
                            history['avg_q'].append(stats.get('avg_q', 0))
                            history['entropy'].append(-stats.get('log_prob_norm', 0))
                            history['alpha'].append(stats.get('avg_alpha', 0))

                if is_new_best:
                    print(f"*** Teacher Forcing: RL learnt from new best {cost_obj_prime:.2f} ***")
                
            # Tabu Logic
            history['rewards'].append(cost_obj) # Track Tabu objective
            
            sig = sequence_signature(s_prime)
            is_tabu = sig in tabu_list
            if (not is_tabu) or (cost_obj_prime > aspiration_criterion):
                tabu_list.append(sig)
                if cost_obj_prime > best_cost:
                    best_cost = cost_obj_prime
                    s_best = copy.deepcopy(s_prime)
                    aspiration_criterion = cost_obj_prime * 1.1
                
                delta = cost_obj_prime - cost_obj
                if delta > 0 or random.random() < math.exp(delta / T_current):
                    s = copy.deepcopy(s_prime)
                    cost_obj = cost_obj_prime
            
            # Print Inspectors every 10 eps
            if epi % 10 == 0:
                log_str = f"Ep {epi} | Tabu R: {cost_obj:.0f}"
                if train and len(history['q_loss']) > 0:
                    log_str += (f" | QL: {history['q_loss'][-1]:.3f} | "
                                f"Ent: {history['entropy'][-1]:.3f} | "
                                f"Alpha: {history['alpha'][-1]:.3f}")
                print(log_str)

        T_current *= cr
        if epi >= max_episodes: break

    if train: trainer.save_model(model_path)
    return s_best, best_cost, history

def sa_ts_SAC_train(file_path, model_path, k, max_episodes, train = True, f_a = 0.0, p1 = 1.0, p2 = 1.0, distribution = 'gamma', warmup_episodes=100):
    replay_buffer = ReplayBuffer(int(1e6))
    
    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    period_num = len(allarea_set) - 1
    c_wr, c_ir = strike_price(allarea_set[0].region_dict, p1, p2)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, period_num, c_wr, c_ir)
    env = InvestEnv_Train(region_num, allarea_set, roa) 

    state_dim = env.observation_space.shape[0]
    action_dim = env.num_regions
    hidden_dim = 256
    batch_size = 32 # Increased
    
    trainer = SAC_Trainer(replay_buffer, state_dim, action_dim, hidden_dim, k, batch_size, period_num=period_num)
    if not train: trainer.load_model(model_path)
    
    # Init Metrics
    history = {
        "rewards": [], "q_loss": [], "policy_loss": [], 
        "avg_q": [], "entropy": [], "alpha": []
    }
    
    # SA Params
    T_init, T_final, cr = 100.0, 1e-1, 0.85
    nbr_outer_iter = int(math.log(T_final / T_init) / math.log(cr))
    nbr_inner_iter = int(max_episodes / nbr_outer_iter)
    tabu_list = deque(maxlen=7)
    aspiration_criterion = float('-inf')

    # Initial solution
    state = env.reset(allarea_set)
    for _ in range(period_num + 1):
        mask = env.generate_mask()
        action = trainer.policy_net.get_action(state, mask)
        state, _, done, _ = env.step(action)
        # if done: break
    s = env.invest_sequence
    
    sequence_value, execution_times = roa.sequence_valuation(s, distribution)
    cost_obj = max(-1, sequence_value)
    
    s_best = copy.deepcopy(s)
    best_cost = cost_obj
    tabu_list.append(sequence_signature(s))
    
    T_current = T_init
    epi = 0
    print(f"Start SAC Training | K={k} | Regions={region_num}")

    while epi < max_episodes:
        for _ in range(nbr_inner_iter):
            epi += 1
            
            # Neighbor Gen (Simplified)
            op_choice = random.random()
            if op_choice < 0.05: s_prime = swap_stages(s)
            elif op_choice < 0.1: s_prime = replace_regions(s)
            else:
                s_list = []
                vals = []
                for _ in range(2):
                    state = env.reset(allarea_set)
                    for _ in range(period_num + 1):
                        mask = env.generate_mask()
                        action = trainer.policy_net.get_action(state, mask)
                        state, _, done, _ = env.step(action)
                        # if done: break
                    s_list.append(env.invest_sequence)
                    val, _ = roa.sequence_valuation(env.invest_sequence, distribution)
                    vals.append(val)
                s_prime = s_list[np.argmax(vals)]

            # Eval
            sequence_value_prime, execution_times_prime = roa.sequence_valuation(s_prime, distribution)
            cost_obj_prime = max(-1, sequence_value_prime)

            # RL Collection
            if train:
                state = env.reset(allarea_set)
                ep_data = {'s':[], 'a':[], 'r':[], 'ns':[], 'd':[]}
                
                # Teacher path
                for h, t_exec in execution_times_prime.items():
                    # Just simulate the path to get states
                    pass 
                
                # Re-run env to collect transition data matching s_prime
                for step in range(period_num + 1):
                    mask = env.generate_mask()
                    action = np.zeros(env.num_regions)
                    # Map s_prime to current step
                    current_portfolio = []
                    for h, t_exec in execution_times_prime.items():
                        if t_exec == env.t:
                            current_portfolio = s_prime[h]
                            break
                    for r in current_portfolio: action[r-1] = 1
                    
                    next_state, reward, done, _ = env.step(action)
                    
                    ep_data['s'].append(state)
                    ep_data['a'].append(action)
                    ep_data['r'].append(reward) # placeholder
                    ep_data['ns'].append(next_state)
                    ep_data['d'].append(done)
                    
                    state = next_state
                    # if done: break
                
                # Backward Discount
                disc_r = cost_obj_prime
                for i in reversed(range(len(ep_data['r']))):
                    ep_data['r'][i] = disc_r
                    disc_r *= 0.95
                    
                # === REWARD STD ===
                rewards_np = np.array(ep_data['r'])
                ep_data['r'] = (rewards_np / 1000.0).tolist()

                # Push
                for i in range(len(ep_data['s'])):
                    replay_buffer.push(ep_data['s'][i], ep_data['a'][i], ep_data['r'][i], 
                                       ep_data['ns'][i], ep_data['d'][i])
                
                # Update
                stats = None
                if epi > warmup_episodes and len(replay_buffer.buffer) > batch_size * 2:
                     stats = trainer.update(batch_size, reward_scale=1.0) # Assumes fixed update returns dict
                
                if stats:
                    history['q_loss'].append(stats.get('q_loss', 0)) # SAC uses q_loss1
                    history['policy_loss'].append(stats.get('policy_loss', 0))
                    history['entropy'].append(-stats.get('log_prob_norm', 0))
                    history['alpha'].append(stats.get('avg_alpha', 0))


            # Tabu Acceptance
            history['rewards'].append(cost_obj)
            sig = sequence_signature(s_prime)
            if (sig not in tabu_list) or (cost_obj_prime > aspiration_criterion):
                tabu_list.append(sig)
                if cost_obj_prime > best_cost:
                    best_cost = cost_obj_prime
                    s_best = copy.deepcopy(s_prime)
                    aspiration_criterion = cost_obj_prime * 1.1
                
                delta = cost_obj_prime - cost_obj
                if delta > 0 or random.random() < math.exp(delta / T_current):
                    s = copy.deepcopy(s_prime)
                    cost_obj = cost_obj_prime

            # Logging
            if epi % 10 == 0:
                log_str = f"Ep {epi} | SAC | R: {cost_obj:.0f} | Best: {best_cost:.0f}"
                if train and len(history['q_loss']) > 0:
                    log_str += (f" | QL: {history['q_loss'][-1]:.3f} | "
                                f"Ent: {history['entropy'][-1]:.3f} | "
                                f"Alpha: {history['alpha'][-1]:.3f}")
                print(log_str)

        T_current *= cr
        if epi >= max_episodes: break
        
    if train: trainer.save_model(model_path)
    return s_best, best_cost, history

def split(lst, max_parts, max_size):
    # print('max_size:',max_size)
    n = len(lst)
    if n == 0:
        return []

    # ?
    if max_parts * max_size < n:
        raise ValueError("max_parts ?max_size ?")

    while True:
        sublists = []
        start = 0

        while start < n:
            remaining = n - start
            max_len = min(max_size, remaining)

            # 1ax_len
            size = random.randint(1, max_len)

            end = start + size
            sublists.append(lst[start:end])
            start = end

            # ?max_parts?
            if len(sublists) > max_parts:
                break

        # 
        if len(sublists) <= max_parts and sum(len(sub) for sub in sublists) == n:
            return sublists
        # 

#periodkegion
def Myopia_policy(file_path, k, f_a = 0.0, p1 = 1, p2 = 1, REVERSE = False, distribution = 'gamma'):
    start_time = time.perf_counter()
    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    period_num = len(allarea_set) - 1
    c_wr, c_ir = strike_price(allarea_set[0].region_dict, p1, p2)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, period_num, c_wr, c_ir)
    region_dict = allarea_set[0].region_dict
    region_list = list(region_dict.values())
    region_list_sorted = sorted(region_list, key=lambda r: r.d, reverse = REVERSE)
    result_sublists = split(region_list_sorted, period_num, k)
    sequence = []
    n = 0
    for sublist in result_sublists:
        portfolio = []
        for region in sublist:
            portfolio.append(region.id)
        sequence.append(portfolio)
    sequence_value, exec_times = roa.sequence_valuation(sequence, distribution)
    # mean_npv = roa.Future_NPV(sequence, exec_times, distribution)
    mean_npv = 0
    obj_reward = sequence_value * (1-f_a) + mean_npv * f_a
    run_time = time.perf_counter() - start_time
    return sequence, obj_reward, run_time


def Myopia_policy_k(file_path, k, f_a=0.0, p1=1, p2=1,
                          REVERSE=False, distribution='gamma'):

    start_time = time.perf_counter()

    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    T = len(allarea_set) - 1  # period_num

    c_wr, c_ir = strike_price(allarea_set[0].region_dict, p1, p2)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, T, c_wr, c_ir)

    region_list = list(allarea_set[0].region_dict.values())
    region_list_sorted = sorted(region_list, key=lambda r: r.d, reverse=(not REVERSE))

    H_needed = math.ceil(region_num / k)
    if H_needed > (T + 1):
        raise ValueError(f"T={T} too short for N={region_num}, k={k}: need at least {H_needed} periods.")

    sequence = []
    ptr = 0
    for _ in range(H_needed):
        sequence.append([r.id for r in region_list_sorted[ptr: ptr + k]])
        ptr += k

    #  portfolio = T+1?portfolio?
    while len(sequence) < (T + 1):
        sequence.append([])

    sequence_value, exec_times = roa.sequence_valuation(sequence, distribution)

    mean_npv = 0.0
    obj_reward = sequence_value * (1 - f_a) + mean_npv * f_a
    run_time = time.perf_counter() - start_time
    return sequence, obj_reward, run_time


def Allin_policy(file_path):
    start_time = time.perf_counter()
    allarea_set = load_variable_from_file(file_path)
    region_dict = allarea_set[0].region_dict
    region_list = list(region_dict.values())
    sequence = []
    portfolio = []
    for region in region_list:
        portfolio.append(region.id)
    sequence.append(portfolio)
    Exec_times = {0:0}
    return sequence, Exec_times

def reward_calculate(invest_sequence, roa, exec_times, n_simulations=300, distribution = 'gamma'):
    npv_results = []
    for _ in range(n_simulations):
        npv_p = {}
        invested_portfolios = []
        for t in range(roa.T+1):
            invested_regions = []
            npv_t = 0
            #ortfolio?
            for h, portfolio in enumerate(invest_sequence):
                # print('portfolio:',portfolio)
                if exec_times[h] == t:
                    invested_portfolios.append(portfolio)
                    break
            if len(invested_portfolios) > 0:
                for portfolio in invested_portfolios:
                    for region_id in portfolio:
                        invested_regions.append(region_id)  #
            sub_invested_regions = copy.deepcopy(invested_regions)
            for h in range(len(invested_portfolios)-1, -1, -1):
                h_portfolio = invested_portfolios[h]
                #Calculate demand after investment (including new regions)
                demand_after_in = 0; demand_after_out = 0;
                for origin_id in sub_invested_regions:
                    origin_region = roa.region_dict[origin_id]
                    demand_after_in += calculate_od_demand(t, len(sub_invested_regions), origin_region, origin_id, distribution)
                    for dest_id in sub_invested_regions:
                        if dest_id != origin_id:
                            demand_after_out += calculate_od_demand(t, len(sub_invested_regions), origin_region, dest_id, distribution) 
                
                for region_id in h_portfolio:
                    sub_invested_regions.remove(region_id)
                #Calcute demand before investment
                demand_before_in = 0; demand_before_out = 0;
                if len(sub_invested_regions) > 0:
                    for origin_id in sub_invested_regions:
                        origin_region = roa.region_dict[origin_id]
                        # Internal demand within each invested region
                        demand_before_in += calculate_od_demand(t, len(sub_invested_regions), origin_region, origin_id, distribution)
                        # Inter-region OD demands
                        for dest_id in sub_invested_regions:
                            if dest_id != origin_id:
                                demand_before_out += calculate_od_demand(t, len(sub_invested_regions), origin_region, dest_id, distribution)
                    # print("demand_before_in:",demand_before_in,"demand_before_out:",demand_before_out)
    
                # print("demand_after_in:",demand_after_in,"demand_after_out:",demand_after_out)
                incremental_demand = 0
                incremental_demand = demand_after_in + demand_after_out - demand_before_in - demand_before_out
                npv_t += incremental_demand - (roa.c_wr + 2*(len(sub_invested_regions) - 1)*roa.c_ir)
            discount_factor = (1 + roa.discount_rate) ** (-t)
            npv_p[t] = discount_factor * npv_t
        npv_results.append(npv_p)
    reward_t_dict = {}
    for t in range(roa.T+1):
        total_reward = 0
        for npv_p in npv_results:
            total_reward += npv_p[t]
        reward_t_dict[t] = total_reward / len(npv_results)
    return  reward_t_dict

class VNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # <--- CRITICAL FIX
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # <--- CRITICAL FIX
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)  # [B]

class TransformerValueNetwork(nn.Module):
    def __init__(self, state_dim, num_regions, hidden_dim=256, num_layers=2, num_heads=8, device="cuda", period_num=None):
        super().__init__()
        self.state_dim = state_dim
        self.num_regions = num_regions
        self.hidden_dim = hidden_dim
        self.device = device
        self.period_num = period_num
        self.node_feature_dim = 4

        self.node_proj = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.region_embed = nn.Embedding(num_regions, hidden_dim)

        self.global_proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            for _ in range(max(1, num_layers))
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(1, num_layers))])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.ReLU(),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            for _ in range(max(1, num_layers))
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(max(1, num_layers))])

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.to(device)

    def _to_tensor(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device).float()
        return state

    def _preprocess(self, state):
        state = self._to_tensor(state)
        orig_dim = state.dim()
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)  # [1,1,S]
        elif state.dim() == 2:
            state = state.unsqueeze(1)               # [B,1,S]

        B, T, _ = state.shape
        n = self.num_regions

        current_step = state[:, :, 0].unsqueeze(-1)  # [B,T,1]
        if self.period_num is not None and self.period_num > 0:
            current_step = current_step / float(self.period_num)
        invest_state = state[:, :, 2:n+2]            # [B,T,N]
        demand_base = state[:, :, n+2:2*n+2]         # [B,T,N]
        if demand_base.size(-1) != n:
            demand_base = torch.zeros_like(invest_state)

        invested_ratio = invest_state.mean(dim=-1, keepdim=True)      # [B,T,1]
        demand_mean = demand_base.mean(dim=-1, keepdim=True)          # [B,T,1]
        global_feat = torch.cat([current_step, invested_ratio, demand_mean], dim=-1)  # [B,T,3]

        node_feat = torch.cat([
            invest_state.unsqueeze(-1),                                        # [B,T,N,1]
            invested_ratio.unsqueeze(2).expand(-1, -1, n, -1),                 # [B,T,N,1]
            current_step.unsqueeze(2).expand(-1, -1, n, -1),                   # [B,T,N,1]
            demand_base.unsqueeze(-1),                                          # [B,T,N,1]
        ], dim=-1)  # [B,T,N,4]

        return state, orig_dim, B, T, node_feat, global_feat

    def forward(self, state):
        _, orig_dim, B, T, node_feat, global_feat = self._preprocess(state)

        bt = B * T
        node_feat = node_feat.view(bt, self.num_regions, self.node_feature_dim)
        x_nodes = self.node_proj(node_feat)
        rid = torch.arange(self.num_regions, device=self.device).unsqueeze(0)
        x_nodes = x_nodes + self.region_embed(rid)

        raw_global_flat = global_feat.view(bt, 3)
        x_global = self.global_proj(raw_global_flat).unsqueeze(1)
        x_global = x_global + self.cls_token.expand(bt, -1, -1)

        x = torch.cat([x_global, x_nodes], dim=1)  # [BT, N+1, H]
        for attn, anorm, ffn, fnorm in zip(self.attn_layers, self.attn_norms, self.ffn_layers, self.ffn_norms):
            x_attn, _ = attn(x, x, x, need_weights=False)
            x = anorm(x + x_attn)
            x = fnorm(x + ffn(x))

        cls_out = x[:, 0, :]
        combined_feat = torch.cat([cls_out, raw_global_flat], dim=-1)
        v = self.value_head(combined_feat).view(B, T)
        if orig_dim == 1:
            return v[0, 0]
        if orig_dim == 2:
            return v[:, 0]
        return v


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiBinaryKPolicy(nn.Module):
    def __init__(self, state_dim, num_regions, k, hidden_dim=256, device="cuda"):
        super().__init__()
        self.state_dim = state_dim
        self.num_regions = num_regions
        self.k = k
        self.device = device

        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.k_head = nn.Linear(hidden_dim, k)          # logits for k
        self.r_head = nn.Linear(hidden_dim, num_regions)    # logits for each region

        self.to(device)

    def forward(self, s):
        x = self.body(s)
        k_logits = self.k_head(x)          # [B, k]
        r_logits = self.r_head(x)          # [B, N]
        return k_logits, r_logits

    def log_prob_of_action(self, s, action, mask=None, k_val=None, selected_indices=None):
        """
        Log pi(a|s) under the same sequential-without-replacement sampler used in rollout.
        If selected_indices is provided, recompute exact step-wise log-prob in the sampled order.
        """
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float().to(self.device)
        if isinstance(selected_indices, np.ndarray):
            selected_indices = torch.from_numpy(selected_indices).long().to(self.device)

        if not isinstance(s, torch.Tensor):
            s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        else:
            s = s.to(self.device).float()
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device).float()
        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
        elif mask is not None:
            mask = mask.to(self.device).float()
        if selected_indices is not None and not isinstance(selected_indices, torch.Tensor):
            selected_indices = torch.as_tensor(selected_indices, dtype=torch.long, device=self.device)
        elif selected_indices is not None:
            selected_indices = selected_indices.to(self.device).long()

        orig_dim = s.dim()
        if s.dim() == 1:
            s = s.unsqueeze(0).unsqueeze(0)
            action = action.unsqueeze(0).unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0).unsqueeze(0)
            if selected_indices is not None:
                selected_indices = selected_indices.unsqueeze(0).unsqueeze(0)
        elif s.dim() == 2:
            s = s.unsqueeze(1)
            if action.dim() == 2:
                action = action.unsqueeze(1)
            if mask is not None and mask.dim() == 2:
                mask = mask.unsqueeze(1)
            if selected_indices is not None and selected_indices.dim() == 2:
                selected_indices = selected_indices.unsqueeze(1)

        B, T, _ = s.shape
        k_logits, r_logits = self.forward(s)

        if mask is None:
            invested = s[:, :, 2:self.num_regions+2]
            valid = (1.0 - invested).clamp(0, 1)
        else:
            valid = mask.clamp(0, 1)

        if k_val is None:
            k_val = action.sum(dim=-1).long().clamp(1, self.k)
        else:
            if not isinstance(k_val, torch.Tensor):
                k_val = torch.as_tensor(k_val, dtype=torch.long, device=self.device)
            else:
                k_val = k_val.to(self.device).long()
            if k_val.dim() == 1:
                k_val = k_val.view(B, T)
        k_idx = (k_val - 1).long().clamp(0, self.k - 1)
        k_logp = F.log_softmax(k_logits, dim=-1).gather(-1, k_idx.unsqueeze(-1)).squeeze(-1)

        logp_seq = torch.zeros((B, T), dtype=torch.float32, device=self.device)
        for b in range(B):
            for t in range(T):
                kk = int(k_val[b, t].item())
                if kk <= 0:
                    continue
                remaining = valid[b, t].clone()
                step_logp = 0.0

                if selected_indices is not None:
                    picks = selected_indices[b, t]
                else:
                    picks = torch.nonzero(action[b, t] > 0.5, as_tuple=False).squeeze(-1)
                    if picks.numel() < kk:
                        logp_seq[b, t] = -50.0
                        continue

                for j_step in range(min(kk, picks.numel() if picks.dim() > 0 else 0)):
                    j = int(picks[j_step].item())
                    if j < 0:
                        break
                    if j >= self.num_regions or remaining[j] < 0.5:
                        step_logp = step_logp - 50.0
                        continue
                    masked_scores = r_logits[b, t].masked_fill(remaining < 0.5, -1e9)
                    probs = torch.softmax(masked_scores, dim=-1)
                    step_logp = step_logp + torch.log(probs[j] + 1e-12)
                    remaining[j] = 0.0

                logp_seq[b, t] = step_logp

        out = k_logp + logp_seq
        if orig_dim == 1:
            return out[0, 0]
        if orig_dim == 2:
            return out[:, 0]
        return out

    def sample_action_and_logprob(self, state, mask=None, deterministic=False, return_selected=False):
        if isinstance(state, np.ndarray): state = torch.from_numpy(state).float().to(self.device)
        if state.dim() == 1: state = state.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2: state = state.unsqueeze(1)
        
        if mask is not None:
            if isinstance(mask, np.ndarray): mask = torch.from_numpy(mask).float().to(self.device)
            if mask.dim() == 1: mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 2: mask = mask.unsqueeze(1)

        B, T, _ = state.shape
        
        k_logits, r_logits = self.forward(state) # [B,T,k], [B,T,N]
        
        scores = r_logits.clone() 
        
        if mask is None:
            invested = state[:, :, 2:self.num_regions+2]
            valid = (1.0 - invested).clamp(0, 1)
        else:
            valid = mask.clamp(0, 1)

        num_dist = torch.distributions.Categorical(logits=k_logits)
        if deterministic:
            k_idx = torch.argmax(k_logits, dim=-1)
        else:
            k_idx = num_dist.sample()
        k_val = k_idx + 1

        action = torch.zeros((B, T, self.num_regions), device=self.device)
        selected = torch.full((B, T, self.k), -1, dtype=torch.long, device=self.device)
        
        for b in range(B):
            for t in range(T):
                valid_idx = torch.nonzero(valid[b, t] > 0.5).squeeze(1)
                if valid_idx.numel() == 0: continue
                
                kk = int(min(k_val[b, t].item(), valid_idx.numel()))
                remaining = valid[b, t].clone()

                for pick_idx in range(kk):
                    masked_scores = scores[b, t].clone()
                    masked_scores = masked_scores.masked_fill(remaining < 0.5, -1e9)
                    
                    probs = torch.softmax(masked_scores, dim=-1)
                    
                    if deterministic:
                        j = torch.argmax(probs).item()
                    else:
                        j = torch.distributions.Categorical(probs=probs).sample().item()
                    
                    action[b, t, j] = 1.0
                    if pick_idx < self.k:
                        selected[b, t, pick_idx] = j
                    remaining[j] = 0.0

        
        state_flat = state.view(-1, self.state_dim)
        action_flat = action.view(-1, self.num_regions)
        mask_flat = valid.view(-1, self.num_regions)
        k_val_flat = k_val.view(-1)
        
        selected_flat = selected.view(-1, self.k)
        logp_total = self.log_prob_of_action(
            state_flat, action_flat, mask_flat, k_val_flat, selected_indices=selected_flat
        )
        
        logp_total = logp_total.view(B, T)
        
        if state.size(0) == 1 and state.size(1) == 1:
            if return_selected:
                return action[0, 0], logp_total[0, 0], selected[0, 0]
            return action[0, 0], logp_total[0, 0]
        if state.size(1) == 1:
            if return_selected:
                return action.squeeze(1), logp_total.squeeze(1), selected.squeeze(1)
            return action.squeeze(1), logp_total.squeeze(1)
            
        if return_selected:
            return action, logp_total, selected
        return action, logp_total
    
    def entropy(self, s, mask=None):
        """
        Approx entropy: H(k) +  H(Bernoulli(p_i))  (independent approx)
        """
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float().to(self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float().to(self.device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)

        k_logits, r_logits = self.forward(s)
        k_prob = F.softmax(k_logits, dim=-1).clamp_min(1e-8)
        Hk = -(k_prob * torch.log(k_prob)).sum(dim=-1)

        p = torch.sigmoid(r_logits)
        if mask is not None:
            p = p * mask
        p = p.clamp(1e-6, 1 - 1e-6)
        Hb = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).sum(dim=-1)

        return (Hk + Hb).mean()

class VNetworkLSTM(nn.Module):
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu):
        super(VNetworkLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self._state_dim = state_space
        self._action_dim = action_space
        self.activation = activation
        self.device = device
        
        self.linear1 = nn.Linear(self._state_dim, hidden_dim)
        
        self.linear2 = nn.Linear(self._state_dim + self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        
        self.linear3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1) # Output Scalar V
        
        self.linear4.apply(linear_weights_init)
        self.to(device)
        
    def forward(self, state, last_action, hidden_in=None):
        """
        state: [B, T, S]
        last_action: [B, T, A]
        hidden_in: (h0, c0)
        """
        state = state.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        
        fc_branch = self.activation(self.linear1(state))
        
        lstm_input = torch.cat([state, last_action], -1)
        lstm_input = self.activation(self.linear2(lstm_input))
        
        lstm_branch, lstm_hidden = self.lstm1(lstm_input, hidden_in)
        
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)
        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)
        
        x = x.permute(1, 0, 2)
        return x, lstm_hidden
    
class PPOTrainer(nn.Module):
    def __init__(
        self,
        state_dim,
        num_regions,
        k,
        hidden_dim=256,
        device="cuda",
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.001,
        value_coef=0.5,
        ppo_epochs=4,
        minibatch_size=32,
        max_grad_norm=1.0,
    ):
        super().__init__()
        self.device = device

        self.policy_net = MultiBinaryKPolicy(state_dim, num_regions, k, hidden_dim, device=device)
        self.value_net = VNetwork(state_dim, hidden_dim).to(device)

        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm

        self.opt_pi = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.opt_v = optim.Adam(self.value_net.parameters(), lr=lr)

    def compute_gae_torch(self, rewards, values, dones, last_value):
        """
        rewards: [T]
        values:  [T]
        dones:   [T] (0/1)
        last_value: scalar tensor
        """
        T = rewards.size(0)
        adv = torch.zeros(T, device=self.device)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
            next_value = values[t]
        ret = adv + values
        return adv, ret

    def update(self, states, actions, masks, old_logp, ret, adv, selected_indices=None):
        states = states.float()
        actions = actions.float()
        if masks is not None:
            masks = masks.float()
        if selected_indices is not None:
            selected_indices = selected_indices.long()
        old_logp = old_logp.float()
        ret = ret.float()
        adv = adv.float()
    
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        T = states.size(0)
        idx = torch.randperm(T, device=self.device)

        total_pi_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0

        for _ in range(self.ppo_epochs):
            for start in range(0, T, self.minibatch_size):
                mb = idx[start:start+self.minibatch_size]
                s_mb = states[mb]
                a_mb = actions[mb]
                m_mb = masks[mb] if masks is not None else None
                sel_mb = selected_indices[mb] if selected_indices is not None else None
                old_mb = old_logp[mb]
                adv_mb = adv[mb]
                ret_mb = ret[mb]

                new_logp = self.policy_net.log_prob_of_action(s_mb, a_mb, m_mb, selected_indices=sel_mb)

                ratio = torch.exp(new_logp - old_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) *adv_mb
                pi_loss = -torch.min(surr1, surr2).mean()

                v_pred = self.value_net(s_mb)
                v_loss = F.mse_loss(v_pred, ret_mb)

                ent = self.policy_net.entropy(s_mb, m_mb)
                loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * ent

                self.opt_pi.zero_grad()
                self.opt_v.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

                self.opt_pi.step()
                self.opt_v.step()

                total_pi_loss += float(pi_loss.item())
                total_v_loss += float(v_loss.item())
                total_ent += float(ent.item())

        return total_pi_loss / (self.ppo_epochs + 1e-8), total_v_loss / (self.ppo_epochs + 1e-8), total_ent / (self.ppo_epochs + 1e-8)
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path + '_policy')
        torch.save(self.value_net.state_dict(), path + '_value')
    
    def load_model(self, path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy_net.load_state_dict(torch.load(path + '_policy', map_location=device))
        self.value_net.load_state_dict(torch.load(path + '_value', map_location=device))
        
        self.policy_net.eval()
        self.value_net.eval()
        
        self.policy_net.to(device)
        self.value_net.to(device)


class TPPOTrainer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, k, device='cuda', period_num=None):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.policy_net = TransformerPolicyNetwork(action_dim, hidden_dim, k, period_num=period_num)
        
        self.value_net = TransformerValueNetwork(
            state_dim=state_dim,
            num_regions=action_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            device=device,
            period_num=period_num,
        ).to(device)
        
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.current_episode = 0
        self.max_episodes = 1
        self.ppo_epochs = 4
        self.max_grad_norm = 0.5
        
        self.opt_pi = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.opt_v = optim.Adam(self.value_net.parameters(), lr=1e-4)

    def set_coinvest_W(self, W_np):
        return

    def set_episode(self, episode, max_episodes):
        self.current_episode = int(episode)
        self.max_episodes = max(1, int(max_episodes))

    def get_value(self, state, last_action=None, hidden_in=None):
        with torch.no_grad():
            v = self.value_net(state)
        return v, None

    def compute_gae(self, rewards, values, dones, last_value):
            """
            Calculates GAE (Generalized Advantage Estimation)
            """
            gae = 0
            returns = []
            advantages = []

            values = values.float()
            last_value = last_value.float()

            values = torch.cat([values, last_value.unsqueeze(0)]) # [T+1]

            for step in reversed(range(len(rewards))):
                delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
                gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[step])

            return torch.tensor(advantages, dtype=torch.float32, device=self.device), \
                   torch.tensor(returns, dtype=torch.float32, device=self.device)
    
    def update(self, states, actions, old_log_probs, returns, advantages, masks, selected_indices=None, k_vals=None):
        """
        PPO Update Function
        states: [B, T, S]
        """
        total_pi_loss = 0
        total_v_loss = 0
        total_ent = 0
        total_co = 0
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            new_log_probs = self.policy_net.log_prob_of_action(
                states, actions, masks, k_vals=k_vals, selected_indices=selected_indices
            )
            new_log_probs = new_log_probs.view(-1)
            old_log_probs_view = old_log_probs.view(-1)
            advantages_view = advantages.view(-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs_view)
            surr1 = ratio * advantages_view
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_view
            pi_loss = -torch.min(surr1, surr2).mean()
            
            values = self.value_net(states).reshape(-1)
            returns_view = returns.reshape(-1)
            
            v_loss = F.smooth_l1_loss(values, returns_view)
            
            entropy = -new_log_probs.mean()

             
            loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * entropy
            
            self.opt_pi.zero_grad()
            self.opt_v.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.opt_pi.step()
            self.opt_v.step()
            
            total_pi_loss += pi_loss.item()
            total_v_loss += v_loss.item()
            total_ent += entropy.item()

        return (
            total_pi_loss / self.ppo_epochs,
            total_v_loss / self.ppo_epochs,
            total_ent / self.ppo_epochs
        )
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path + '_policy')
        torch.save(self.value_net.state_dict(), path + '_value')

    def load_model(self, path):
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net.load_state_dict(torch.load(path + '_policy', map_location=dev))
        self.value_net.load_state_dict(torch.load(path + '_value', map_location=dev))
        self.policy_net.eval()
        self.value_net.eval()
        self.policy_net.to(dev)
        self.value_net.to(dev)


def _seq_to_action(step_seq, num_regions):
    action = np.zeros(num_regions, dtype=np.float32)
    if step_seq is None:
        return action
    for r in step_seq:
        rid = int(r) - 1
        if 0 <= rid < num_regions:
            action[rid] = 1.0
    return action


def _action_to_selected_and_k(action_np, k_max):
    idx = np.where(np.asarray(action_np) > 0.5)[0].astype(np.int64).tolist()
    k_obs = len(idx)
    if k_obs <= 0:
        return None, None
    selected = np.full((k_max,), -1, dtype=np.int64)
    use_k = min(k_obs, k_max)
    selected[:use_k] = np.asarray(idx[:use_k], dtype=np.int64)
    return selected, use_k


def ts_TPPO_train(file_path, model_path, k, max_episodes, train=True,
                  f_a=0.0, p1=1, p2=1, distribution='gamma',
                  pretrain_episodes=100, warmup_episodes=100):
    if train and pretrain_episodes > 0:
        print(f"=== Phase 1: Pre-training TPPO for {pretrain_episodes} episodes ===")
        pure_TPPO_train(file_path, model_path, k, pretrain_episodes, f_a, p1, p2, distribution)
        print("=== Phase 1 Complete. Starting Phase 2: TPPO + Tabu Search ===")

    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    period_num = len(allarea_set) - 1
    min_steps = math.ceil(region_num / k)
    if period_num + 1 < min_steps:
        print(f"Warning: period_num={period_num} too short for region_num={region_num}, k={k}. "
              f"Need at least {min_steps} steps to cover all regions.")

    c_wr, c_ir = strike_price(allarea_set[0].region_dict, p1, p2)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, period_num, c_wr, c_ir)
    env = InvestEnv_Train(region_num, allarea_set, roa)

    state_dim = env.observation_space.shape[0]
    action_dim = env.num_regions
    hidden_dim = 256
    trainer = TPPOTrainer(state_dim, action_dim, hidden_dim, k, device=device, period_num=period_num)
    trainer.ppo_epochs = 2

    try:
        trainer.load_model(model_path)
        print("Loaded pre-trained TPPO model for TS phase.")
    except Exception:
        print("No pre-trained TPPO model found, starting from scratch.")

    history = {"rewards": [], "pi_loss": [], "v_loss": [], "entropy": []}

    tabu_list = deque(maxlen=7)
    aspiration_criterion = float('-inf')

    episodes_per_batch = 1
    batch_buffer = []

    state = env.reset(allarea_set)
    for _ in range(period_num + 1):
        mask = env.generate_mask()
        action = trainer.policy_net.get_action(state, mask, deterministic=True)
        state, _, _, _ = env.step(action)
    s = env.invest_sequence

    incomplete_init = not verify_sequence(s, region_num)
    if incomplete_init:
        cost_obj = -1.0
    else:
        sequence_value, execution_times = roa.sequence_valuation(s, distribution)
        sequence_value = max(-1, sequence_value)
        if f_a == 0:
            cost_obj = sequence_value
        else:
            cost_obj = sequence_value * (1 - f_a) + f_a * sum(reward_calculate(s, roa, execution_times).values())

    s_best = copy.deepcopy(s)
    best_cost = cost_obj
    tabu_list.append(sequence_signature(s))

    print(f"Start TPPO+Tabu Search | Episodes={max_episodes}")
    for epi in range(1, max_episodes + 1):
        trainer.set_episode(epi + pretrain_episodes, max_episodes + pretrain_episodes)

        op_choice = random.random()
        if op_choice < 0.25:
            s_prime = swap_stages(s)
        elif op_choice < 0.5:
            s_prime = replace_regions(s)
        else:
            state = env.reset(allarea_set)
            for _ in range(period_num + 1):
                mask = env.generate_mask()
                action = trainer.policy_net.get_action(state, mask, deterministic=False)
                state, _, _, _ = env.step(action)
            s_prime = env.invest_sequence

        incomplete_prime = not verify_sequence(s_prime, region_num)
        execution_times_prime = {}
        if incomplete_prime:
            cost_obj_prime = -1.0
        else:
            sequence_value_prime, execution_times_prime = roa.sequence_valuation(s_prime, distribution)
            sequence_value_prime = max(-1, sequence_value_prime)
            if f_a == 0:
                cost_obj_prime = sequence_value_prime
            else:
                cost_obj_prime = sequence_value_prime * (1 - f_a) + f_a * sum(
                    reward_calculate(s_prime, roa, execution_times_prime).values()
                )

        is_new_best = cost_obj_prime > best_cost
        if train and (is_new_best or random.random() < 0.01):
            state_np = env.reset(allarea_set)
            V_prev = 0.0
            seq_partial = [[] for _ in range(period_num + 1)]
            ep_states, ep_actions, ep_masks = [], [], []
            ep_selected, ep_kvals = [], []
            ep_rewards, ep_dones, ep_values = [], [], []

            for _step in range(period_num + 1):
                mask_np = env.generate_mask()
                action_np = np.zeros(env.num_regions, dtype=np.float32)

                for h, t_exec in execution_times_prime.items():
                    if t_exec == env.t:
                        action_np = _seq_to_action(s_prime[h], env.num_regions)
                        if h < len(seq_partial):
                            seq_partial[h] = list(s_prime[h])
                        break

                next_state_np, _, done, _ = env.step(action_np)

                valid_seq = [stage for stage in seq_partial if stage]
                if valid_seq:
                    V_now, _ = roa.sequence_valuation(valid_seq, distribution)
                else:
                    V_now = 0.0
                r_step = V_now - V_prev
                V_prev = V_now

                selected_np, k_obs = _action_to_selected_and_k(action_np, k)
                if selected_np is not None:
                    ep_states.append(torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
                    ep_actions.append(torch.as_tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
                    ep_masks.append(torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
                    ep_selected.append(torch.as_tensor(selected_np, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0))
                    ep_kvals.append(torch.as_tensor(k_obs, dtype=torch.long, device=device).view(1, 1))
                    ep_rewards.append(r_step)
                    ep_dones.append(float(done))
                    with torch.no_grad():
                        v_step = trainer.value_net(torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0))
                    ep_values.append(float(v_step.item()))

                state_np = next_state_np

            if incomplete_prime and len(ep_rewards) > 0:
                ep_rewards = [INCOMPLETE_R_STEP] * len(ep_rewards)

            if len(ep_states) > 0:
                ep_rewards_t = torch.tensor(ep_rewards, dtype=torch.float32, device=device) / REWARD_SCALE_DIV
                ep_values_t = torch.tensor(ep_values, dtype=torch.float32, device=device)
                ep_dones_t = torch.tensor(ep_dones, dtype=torch.float32, device=device)
                with torch.no_grad():
                    last_val = trainer.value_net(
                        torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    ).item()
                adv, ret = trainer.compute_gae(ep_rewards_t, ep_values_t, ep_dones_t,
                                               torch.tensor(last_val, dtype=torch.float32, device=device))

                states_cat = torch.cat(ep_states, dim=1)
                actions_cat = torch.cat(ep_actions, dim=1)
                masks_cat = torch.cat(ep_masks, dim=1)
                selected_cat = torch.cat(ep_selected, dim=1)
                kvals_cat = torch.cat(ep_kvals, dim=1)
                old_log_probs = trainer.policy_net.log_prob_of_action(
                    states_cat, actions_cat, masks_cat, k_vals=kvals_cat, selected_indices=selected_cat
                ).detach()

                batch_buffer.append({
                    "states": states_cat,
                    "actions": actions_cat,
                    "masks": masks_cat,
                    "selected_indices": selected_cat,
                    "k_vals": kvals_cat,
                    "old_log_probs": old_log_probs,
                    "returns": ret.unsqueeze(0),
                    "advantages": adv.unsqueeze(0),
                })

                if epi > warmup_episodes and len(batch_buffer) >= episodes_per_batch:
                    b_states = torch.cat([b["states"] for b in batch_buffer], dim=0)
                    b_actions = torch.cat([b["actions"] for b in batch_buffer], dim=0)
                    b_masks = torch.cat([b["masks"] for b in batch_buffer], dim=0)
                    b_selected = torch.cat([b["selected_indices"] for b in batch_buffer], dim=0)
                    b_k_vals = torch.cat([b["k_vals"] for b in batch_buffer], dim=0)
                    b_old_log_probs = torch.cat([b["old_log_probs"] for b in batch_buffer], dim=0)
                    b_returns = torch.cat([b["returns"] for b in batch_buffer], dim=0)
                    b_advantages = torch.cat([b["advantages"] for b in batch_buffer], dim=0)

                    pi_loss, v_loss, ent = trainer.update(
                        b_states, b_actions, b_old_log_probs, b_returns, b_advantages, b_masks,
                        selected_indices=b_selected, k_vals=b_k_vals
                    )
                    history["pi_loss"].append(pi_loss)
                    history["v_loss"].append(v_loss)
                    history["entropy"].append(ent)
                    batch_buffer = []

        history["rewards"].append(cost_obj)

        sig = sequence_signature(s_prime)
        is_tabu = sig in tabu_list
        if (not is_tabu) or (cost_obj_prime > aspiration_criterion):
            tabu_list.append(sig)
            if cost_obj_prime > best_cost:
                best_cost = cost_obj_prime
                s_best = copy.deepcopy(s_prime)
                aspiration_criterion = cost_obj_prime * 1.1

            s = copy.deepcopy(s_prime)
            cost_obj = cost_obj_prime

        if epi % 10 == 0:
            log_str = f"Ep {epi} | Tabu R: {cost_obj:.0f}"
            if train and len(history["pi_loss"]) > 0:
                log_str += (f" | PL: {history['pi_loss'][-1]:.3f}"
                            f" | VL: {history['v_loss'][-1]:.3f}"
                            f" | Ent: {history['entropy'][-1]:.3f}")
            print(log_str)

    if train:
        trainer.save_model(model_path)
    return s_best, best_cost, history
        
def pure_TPPO_train(file_path, model_path, k, max_episodes, f_a=0.0, p1=1.0, p2=1.0, distribution='gamma', warmup_episodes=20):
    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    period_num = len(allarea_set) - 1
    c_wr, c_ir = strike_price(allarea_set[0].region_dict, p1, p2)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, period_num, c_wr, c_ir)
    env = InvestEnv_Train(region_num, allarea_set, roa)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.num_regions
    hidden_dim = 256
    
    trainer = TPPOTrainer(state_dim, action_dim, hidden_dim, k, device=device, period_num=period_num)
    trainer.ppo_epochs = 2
    
    history = {"rewards": [], "pi_loss": [], "v_loss": [], "entropy": []}
    
    episodes_per_batch = 4
    batch_buffer = []
    
    print(f"Start PURE TPPO (Transformer+LSTM) Training | Episodes={max_episodes}")
    best_reward = -float('inf')

    for epi in range(1, max_episodes + 1):
        trainer.set_episode(epi, max_episodes)
        state_np = env.reset(allarea_set)
        state = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) # [1,1,S]
        
        last_action_np = np.zeros(action_dim)
        last_action = torch.as_tensor(last_action_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        
        ep_data = {
            'states': [], 'actions': [], 'masks': [],
            'selected_indices': [], 'k_vals': [],
            'log_probs': [], 'rewards': [], 'values': [], 'dones': []
        }
        
        V_prev = 0.0
        seq_partial = [[] for _ in range(period_num + 1)]
        
        for step in range(period_num + 1):
            mask_np = env.generate_mask()
            mask = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, selected_idx, k_val = trainer.policy_net.sample_action_and_logprob(
                    state, mask, deterministic=False, return_selected=True, return_k=True
                )
                value = trainer.value_net(state)
            
            action_np = action.cpu().numpy().flatten()
            next_state_np, _, done, _ = env.step(action_np)
            
            current_portfolio = [i+1 for i, v in enumerate(action_np) if v > 0.5]
            seq_partial[step] = current_portfolio
            valid_seq = [stage for stage in seq_partial if stage]
            if valid_seq:
                V_now, _ = roa.sequence_valuation(valid_seq, distribution)
            else:
                V_now = 0.0
            r_step = V_now - V_prev
            V_prev = V_now
            
            ep_data['states'].append(state)
            ep_data['actions'].append(action)
            ep_data['masks'].append(mask)
            ep_data['selected_indices'].append(selected_idx)
            ep_data['k_vals'].append(k_val)
            ep_data['log_probs'].append(log_prob.detach())
            ep_data['values'].append(value.item())
            ep_data['rewards'].append(r_step)
            ep_data['dones'].append(float(done))
            
            state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) 
            last_action = action
            
            if done: break
            
        current_seq = env.invest_sequence
        incomplete = not verify_sequence(current_seq, region_num)
        if incomplete:
            plot_reward = -1.0
            if len(ep_data['rewards']) > 0:
                ep_data['rewards'] = [INCOMPLETE_R_STEP] * len(ep_data['rewards'])
        else:
            final_val, _ = roa.sequence_valuation(current_seq, distribution)
            if f_a > 0:
                r_t = reward_calculate(current_seq, roa, {})
                plot_reward = (final_val * (1 - f_a) + f_a * sum(r_t.values()))
            else:
                plot_reward = max(-1, final_val)
            if len(ep_data['rewards']) > 0:
                ep_data['rewards'][-1] += (plot_reward - V_prev)

        ep_rewards = torch.tensor(ep_data['rewards'], device=device) / 1000.0 # Scaling
        ep_values = torch.tensor(ep_data['values'], device=device)
        ep_dones = torch.tensor(ep_data['dones'], device=device)
        
        with torch.no_grad():
            last_val = trainer.value_net(state).item()
            
        adv, ret = trainer.compute_gae(ep_rewards, ep_values, ep_dones, torch.tensor(last_val, dtype=torch.float32, device=device))
        
        batch_buffer.append({
            'states': torch.cat(ep_data['states'], dim=1),       # [1, T, S]
            'actions': torch.cat(ep_data['actions'], dim=1),     # [1, T, A]
            'masks': torch.cat(ep_data['masks'], dim=1),
            'selected_indices': torch.cat(ep_data['selected_indices'], dim=1),
            'k_vals': torch.cat(ep_data['k_vals'], dim=1),
            'old_log_probs': torch.stack(ep_data['log_probs']).view(1, -1),
            'returns': ret.unsqueeze(0),
            'advantages': adv.unsqueeze(0)
        })

            
        history['rewards'].append(plot_reward)
        if plot_reward > best_reward: best_reward = plot_reward
        
        if epi > warmup_episodes and epi % episodes_per_batch == 0:
            
            b_states = torch.cat([b['states'] for b in batch_buffer], dim=0) # [B, T, S]
            b_actions = torch.cat([b['actions'] for b in batch_buffer], dim=0)
            b_masks = torch.cat([b['masks'] for b in batch_buffer], dim=0)
            b_selected = torch.cat([b['selected_indices'] for b in batch_buffer], dim=0)
            b_k_vals = torch.cat([b['k_vals'] for b in batch_buffer], dim=0)
            b_old_log_probs = torch.cat([b['old_log_probs'] for b in batch_buffer], dim=0)
            b_returns = torch.cat([b['returns'] for b in batch_buffer], dim=0)
            b_advantages = torch.cat([b['advantages'] for b in batch_buffer], dim=0)
             
            pi_loss, v_loss, ent = trainer.update(
                b_states, b_actions, b_old_log_probs, b_returns, b_advantages, b_masks,
                selected_indices=b_selected, k_vals=b_k_vals
            )
            
            batch_buffer = [] # 
            
            history['pi_loss'].append(pi_loss)
            history['v_loss'].append(v_loss)
            history['entropy'].append(ent)
              
            print(f"Ep {epi} | Reward: {plot_reward:.0f} | Best: {best_reward:.0f} | "
                  f"PL: {pi_loss:.3f} | VL: {v_loss:.3f} | Ent: {ent:.3f} | ")
        elif epi % 10 == 0:
            print(f"Ep {epi} | Reward: {plot_reward:.0f} | Best: {best_reward:.0f}")
            
    trainer.save_model(model_path)
    return history

def pure_PPO_train(file_path, model_path, k, max_episodes, f_a=0.0, p1=1.0, p2=1.0, distribution='gamma', warmup_episodes=20):
    allarea_set = load_variable_from_file(file_path)
    region_num = len(allarea_set[0].region_dict)
    period_num = len(allarea_set) - 1
    c_wr, c_ir = strike_price(allarea_set[0].region_dict, p1, p2)
    roa = CompoundOptionAnalysis(allarea_set[0].region_dict, period_num, c_wr, c_ir)
    env = InvestEnv_Train(region_num, allarea_set, roa)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.num_regions
    hidden_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trainer = PPOTrainer(state_dim, action_dim, k, hidden_dim, device=device)
    trainer.ppo_epochs = 2
    
    episodes_per_batch = 1
    batch_buffer = [] 
    history = {"rewards": [], "pi_loss": [], "v_loss": [], "entropy": []}
    
    print(f"Start PURE PPO (Dense Reward) Training | Episodes={max_episodes}")
    best_reward = -float('inf')

    for epi in range(1, max_episodes + 1):
        state_np = env.reset(allarea_set)
        state = torch.as_tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0) 
        
        ep_states, ep_actions, ep_masks = [], [], []
        ep_selected = []
        ep_logprobs, ep_rewards, ep_dones = [], [], []
        ep_values = []
        
        V_prev = 0.0
        seq_partial = [[] for _ in range(period_num + 1)]
        
        plot_reward = 0
        done = False
        step_count = 0
        
        while not done:
            mask_np = env.generate_mask()
            mask = torch.as_tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, selected_idx = trainer.policy_net.sample_action_and_logprob(
                    state, mask, deterministic=False, return_selected=True
                )
                value = trainer.value_net(state)
            
            action_np = action.cpu().numpy().flatten()
            
            next_state_np, _, done, _ = env.step(action_np)
            
            current_portfolio = [i+1 for i, v in enumerate(action_np) if v > 0.5]
            
            if step_count < len(seq_partial):
                seq_partial[step_count] = current_portfolio
            
            try:
                valid_seq = [stage for stage in seq_partial[:step_count+1] if stage]
                if valid_seq and len(valid_seq) > 0:
                    V_now, _ = roa.sequence_valuation(valid_seq, distribution)
                else:
                    V_now = 0.0
            except Exception:
                V_now = V_prev # Fallback
            
            r_step = V_now - V_prev
            V_prev = V_now
            
            if done:
                current_seq = env.invest_sequence
                
                if not current_seq or len(current_seq) == 0:
                    plot_reward = 0.0
                    r_step = 0.0
                else:
                    try:
                        final_val, exec_times = roa.sequence_valuation(current_seq, distribution)
                        
                        if f_a > 0:
                            r_t = reward_calculate(current_seq, roa, exec_times)
                            true_objective = (final_val * (1 - f_a) + f_a * sum(r_t.values()))
                        else:
                            true_objective = max(-1, final_val)
                        
                        plot_reward = true_objective
                        r_step += (true_objective - V_prev)
                    except ValueError as e:
                        print(f"Warning: Valuation failed at step {step_count}: {e}")
                        plot_reward = V_prev
                        r_step = 0.0

            ep_states.append(state)
            
            if action.dim() == 1:
                ep_actions.append(action.unsqueeze(0))
            else:
                ep_actions.append(action)
                
            ep_masks.append(mask)
            ep_selected.append(selected_idx.unsqueeze(0) if selected_idx.dim() == 1 else selected_idx)
            ep_logprobs.append(log_prob.detach())
            ep_values.append(value.detach())
            ep_rewards.append(r_step)
            ep_dones.append(float(done))
            
            state = torch.as_tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
            step_count += 1
            
            if step_count >= period_num + 1:
                break
        
        current_seq = env.invest_sequence
        incomplete = (not current_seq) or (not verify_sequence(current_seq, region_num))
        if incomplete:
            plot_reward = -1.0
            if len(ep_rewards) > 0:
                ep_rewards = [INCOMPLETE_R_STEP] * len(ep_rewards)

        history['rewards'].append(plot_reward)
        if plot_reward > best_reward: best_reward = plot_reward
        
        ep_rewards_t = torch.tensor(ep_rewards, dtype=torch.float32, device=device) / 1000.0
        ep_values_t = torch.cat(ep_values)
        ep_dones_t = torch.tensor(ep_dones, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            last_val = trainer.value_net(state).item() if not done else 0.0
            
        adv, ret = trainer.compute_gae_torch(ep_rewards_t, ep_values_t, ep_dones_t, torch.tensor(last_val, device=device))
        
        batch_buffer.append({
            'states': torch.cat(ep_states),
            'actions': torch.cat(ep_actions),
            'masks': torch.cat(ep_masks),
            'selected_indices': torch.cat(ep_selected),
            'old_log_probs': torch.stack(ep_logprobs).squeeze(),
            'returns': ret,
            'advantages': adv
        })

        if epi > warmup_episodes and epi % episodes_per_batch == 0:
            b_states = torch.cat([b['states'] for b in batch_buffer], dim=0)
            b_actions = torch.cat([b['actions'] for b in batch_buffer], dim=0)
            b_masks = torch.cat([b['masks'] for b in batch_buffer], dim=0)
            b_selected = torch.cat([b['selected_indices'] for b in batch_buffer], dim=0)
            b_old_log_probs = torch.cat([b['old_log_probs'] for b in batch_buffer], dim=0)
            b_returns = torch.cat([b['returns'] for b in batch_buffer], dim=0)
            b_advantages = torch.cat([b['advantages'] for b in batch_buffer], dim=0)
            
            pi_loss, v_loss, ent = trainer.update(
                b_states, b_actions, b_masks, b_old_log_probs, b_returns, b_advantages, selected_indices=b_selected
            )
            batch_buffer = []
            
            history['pi_loss'].append(pi_loss)
            history['v_loss'].append(v_loss)
            history['entropy'].append(ent)
            
            print(f"Ep {epi} | Reward: {plot_reward:.0f} | Best: {best_reward:.0f} | PL: {pi_loss:.3f} | VL: {v_loss:.3f} | Ent: {ent:.3f}")
        elif epi % 10 == 0:
            print(f"Ep {epi} | Reward: {plot_reward:.0f} | Best: {best_reward:.0f}")

    trainer.save_model(model_path)
    return history

def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    
    
