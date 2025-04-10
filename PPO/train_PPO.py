import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# 🧱 Actor-Critic 신경망 정의
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # 공통 feature 추출 계층
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU()
        )
        # 정책 (행동 확률) 계층
        self.policy = nn.Sequential(
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )
        # 가치 함수 계층
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

# 🧮 GAE로 리턴과 어드밴티지 계산
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantage = 0
    returns = []
    advantages = []
    values = values + [0]  # 마지막 상태의 가치를 0으로 추가
    for t in reversed(range(len(rewards))):
        # Temporal difference (TD) 오차
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        # GAE 누적
        advantage = delta + gamma * lam * (1 - dones[t]) * advantage
        returns.insert(0, advantage + values[t])  # 리턴 = 어드밴티지 + 가치
        advantages.insert(0, advantage)
    return returns, advantages

# 🔒 PPO Clipped Objective 손실 함수
def ppo_loss(old_log_probs, new_log_probs, advantages, clip_eps=0.2):
    # 확률 비율 (new/old)
    ratio = torch.exp(new_log_probs - old_log_probs)
    # 클리핑된 손실
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(ratio * advantages, clipped).mean()

# ▶️ 학습 루프
def train():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    for episode in range(1000):
        obs = env.reset()
        done = False

        # Rollout 데이터 저장
        obs_list, action_list, reward_list = [], [], []
        logprob_list, value_list, done_list = [], [], []

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            probs, value = model(obs_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # 환경 상호작용
            next_obs, reward, done, _ = env.step(action.item())

            # Rollout 저장
            obs_list.append(obs_tensor)
            action_list.append(action)
            reward_list.append(reward)
            logprob_list.append(log_prob)
            value_list.append(value.item())
            done_list.append(done)

            obs = next_obs

        # GAE로 리턴 및 어드밴티지 계산
        returns, advantages = compute_gae(reward_list, value_list, done_list)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        # 텐서 변환
        obs_tensor = torch.stack(obs_list)
        action_tensor = torch.tensor(action_list)
        old_logprob_tensor = torch.stack(logprob_list)

        # 여러 번 업데이트 (에폭 반복)
        for _ in range(4):
            # forward pass
            probs, values = model(obs_tensor)
            dist = Categorical(probs)
            new_logprobs = dist.log_prob(action_tensor)

            # 손실 계산
            pg_loss = ppo_loss(old_logprob_tensor, new_logprobs, advantages)
            vf_loss = nn.functional.mse_loss(values.squeeze(), returns)
            entropy = dist.entropy().mean()

            loss = pg_loss + 0.5 * vf_loss - 0.01 * entropy

            # 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 로깅
        total_reward = sum(reward_list)
        print(f"Episode {episode}, Total reward: {total_reward}")

    env.close()

# 🏁 실행
train()
