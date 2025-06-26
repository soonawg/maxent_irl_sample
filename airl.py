import torch
import torch.nn as nn

# AIRL 구현에 필요한 핵심 구성요소들

class RewardFunction(nn.Module):
    """보상 함수 g_θ(s,a)"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Linear(state_dim + action_dim, 1)
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class ShapingFunction(nn.Module):
    """상태 가치 함수 h_ψ(s) - potential shaping"""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Linear(state_dim, 1)
    
    def forward(self, state):
        return self.net(state)

class AIRLDiscriminator(nn.Module):
    """AIRL 판별자 - 핵심 공식"""
    def __init__(self, reward_fn, shaping_fn, gamma=0.99):
        super().__init__()
        self.g = reward_fn      # 보상 함수
        self.h = shaping_fn     # shaping 함수
        self.gamma = gamma
    
    def forward(self, s, a, s_next, log_pi):
        # AIRL 핵심 공식: f = g(s,a) + γh(s') - h(s)
        f = self.g(s, a) + self.gamma * self.h(s_next) - self.h(s)
        
        # D = exp(f) / (exp(f) + π(a|s))
        return torch.sigmoid(f - log_pi)

# 필요한 것들:
# 1. 전문가 시연 데이터 (s, a, s')
# 2. 정책 π_φ(a|s) 
# 3. 판별자 업데이트: max_θ,ψ E_expert[log D] + E_policy[log(1-D)]
# 4. 정책 업데이트: max_φ E_policy[log D] (학습된 보상으로 RL)

expert_data = None  # (states, actions, next_states)
policy = None       # π_φ(a|s)
discriminator = None # AIRL discriminator
optimizer_d = None  # 판별자 옵티마이저  
optimizer_p = None  # 정책 옵티마이저
