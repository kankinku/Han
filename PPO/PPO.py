import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## 디바이스 설정 ##################################
print("============================================================================================")
# 디바이스를 CPU 또는 CUDA로 설정
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()  # GPU 메모리 초기화
    print("디바이스 설정: " + str(torch.cuda.get_device_name(device)))
else:
    print("디바이스 설정: cpu")
print("============================================================================================")


################################## PPO 정책 ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []  # 행동 리스트
        self.states = []  # 상태 리스트
        self.logprobs = []  # 로그 확률 리스트
        self.rewards = []  # 보상 리스트
        self.state_values = []  # 상태 가치 리스트
        self.is_terminals = []  # 종료 여부 리스트
    
    def clear(self):  # 버퍼 초기화
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space  # 연속 행동 공간 여부
        
        if has_continuous_action_space:
            self.action_dim = action_dim  # 행동 차원
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)  # 행동 분산 초기화
        # 액터 정의
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),  # 상태 입력
                            nn.Tanh(),  # 활성화 함수
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),  # 행동 출력
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),  # 상태 입력
                            nn.Tanh(),  # 활성화 함수
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),  # 행동 확률 출력
                            nn.Softmax(dim=-1)  # 확률 분포 생성
                        )
        # 크리틱 정의
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),  # 상태 입력
                        nn.Tanh(),  # 활성화 함수
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)  # 상태 가치 출력
                    )
        
    def set_action_std(self, new_action_std):  # 행동 표준편차 설정
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("경고: 이산 행동 공간 정책에서 ActorCritic::set_action_std() 호출")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):  # 순전파는 구현되지 않음
        raise NotImplementedError
    
    def act(self, state):  # 행동 선택
        if self.has_continuous_action_space:
            action_mean = self.actor(state)  # 행동 평균 계산
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # 공분산 행렬 생성
            dist = MultivariateNormal(action_mean, cov_mat)  # 다변량 정규 분포
        else:
            action_probs = self.actor(state)  # 행동 확률 계산
            dist = Categorical(action_probs)  # 범주형 분포

        action = dist.sample()  # 행동 샘플링
        action_logprob = dist.log_prob(action)  # 로그 확률 계산
        state_val = self.critic(state)  # 상태 가치 추정

        return action.detach(), action_logprob.detach(), state_val.detach()  # 결과 반환
    
    def evaluate(self, state, action):  # 평가 함수
        if self.has_continuous_action_space:
            action_mean = self.actor(state)  # 행동 평균 계산
            
            action_var = self.action_var.expand_as(action_mean)  # 행동 분산 확장
            cov_mat = torch.diag_embed(action_var).to(device)  # 공분산 행렬 생성
            dist = MultivariateNormal(action_mean, cov_mat)  # 다변량 정규 분포
            
            # 단일 행동 환경 처리
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)  # 행동 확률 계산
            dist = Categorical(action_probs)  # 범주형 분포
        action_logprobs = dist.log_prob(action)  # 로그 확률 계산
        dist_entropy = dist.entropy()  # 분포 엔트로피 계산
        state_values = self.critic(state)  # 상태 가치 추정
        
        return action_logprobs, state_values, dist_entropy  # 결과 반환


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space  # 연속 행동 공간 여부

        if has_continuous_action_space:
            self.action_std = action_std_init  # 초기 행동 표준편차

        self.gamma = gamma  # 할인율
        self.eps_clip = eps_clip  # 클리핑 파라미터
        self.K_epochs = K_epochs  # 업데이트 에폭 수
        
        self.buffer = RolloutBuffer()  # 롤아웃 버퍼 초기화

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)  # 현재 정책
        self.optimizer = torch.optim.Adam([  # Adam 최적화
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)  # 이전 정책
        self.policy_old.load_state_dict(self.policy.state_dict())  # 초기화 시 현재 정책 복사
        
        self.MseLoss = nn.MSELoss()  # 평균 제곱 오차 손실 함수

    def set_action_std(self, new_action_std):  # 행동 표준편차 설정
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("경고: 이산 행동 공간 정책에서 PPO::set_action_std() 호출")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):  # 행동 표준편차 감소
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate  # 표준편차 감소
            self.action_std = round(self.action_std, 4)  # 소수점 4자리 반올림
            if (self.action_std <= min_action_std):  # 최소값 이하로 떨어지면
                self.action_std = min_action_std
                print("액터 출력 action_std를 min_action_std로 설정: ", self.action_std)
            else:
                print("액터 출력 action_std 설정: ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("경고: 이산 행동 공간 정책에서 PPO::decay_action_std() 호출")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):  # 행동 선택
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)  # 상태를 텐서로 변환
                action, action_logprob, state_val = self.policy_old.act(state)  # 이전 정책으로 행동 선택

            self.buffer.states.append(state)  # 버퍼에 저장
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()  # 행동 반환
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)  # 상태를 텐서로 변환
                action, action_logprob, state_val = self.policy_old.act(state)  # 이전 정책으로 행동 선택
            
            self.buffer.states.append(state)  # 버퍼에 저장
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()  # 행동 반환

    def update(self):  # 정책 업데이트
        # 몬테카를로 방식으로 리턴 추정
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:  # 에피소드 종료 시
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)  # 할인된 보상 계산
            rewards.insert(0, discounted_reward)
            
        # 보상 정규화
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # 평균 0, 표준편차 1로 정규화

        # 리스트를 텐서로 변환
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # 어드밴티지 계산
        advantages = rewards.detach() - old_state_values.detach()

        # K 에폭 동안 정책 최적화
        for _ in range(self.K_epochs):
            # 이전 행동 및 가치 평가
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # 상태 가치 차원 조정
            state_values = torch.squeeze(state_values)
            
            # 확률 비율 계산 (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 대리 손실 계산
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # PPO 클리핑 목표 손실
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # 경사 하강 단계
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 새로운 가중치를 이전 정책에 복사
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 버퍼 초기화
        self.buffer.clear()
    
    def save(self, checkpoint_path):  # 모델 저장
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):  # 모델 로드
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))