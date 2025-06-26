import numpy as np

# 전문가 feature expectation 계산
def compute_expert_feature_expectations(expert_trajectories, feature_fn):
    fe = np.zeros(feature_fn.dim)
    for traj in expert_trajectories:
        for state in traj:
            fe += feature_fn(state)
    return fe / len(expert_trajectories)

# 보상 함수: 상태 -> 보상 (가중치 * feature)
class LinearReward:
    def __init__(self, dim):
        self.weights = np.random.randn(dim)

    def __call__(self, state, feature_fn):
        return np.dot(self.weights, feature_fn(state))

# 상태 -> softmax 정책을 통한 trajectory 샘플링
def sample_trajectory(env, reward_fn, feature_fn, max_steps=200):
    state = env.reset()
    traj = []
    for _ in range(max_steps):
        rewards = []
        actions = []
        for a in range(env.action_space.n):
            env_copy = copy_env(env)  # 환경 복사 필요
            next_state, _, _, _, _ = env_copy.step(a)
            r = reward_fn(next_state, feature_fn)
            rewards.append(r)
            actions.append(a)
        probs = softmax(rewards)
        action = np.random.choice(actions, p=probs)
        next_state, _, done, _, _ = env.step(action)
        traj.append(state)
        state = next_state
        if done:
            break
    return traj

# MaxEnt IRL 업데이트 핵심 루프
def train_maxent_irl(env, expert_trajectories, feature_fn, iterations=100, lr=0.01):
    reward_fn = LinearReward(feature_fn.dim)
    expert_fe = compute_expert_feature_expectations(expert_trajectories, feature_fn)

    for i in range(iterations):
        sampled_fe = np.zeros_like(expert_fe)
        for _ in range(len(expert_trajectories)):
            traj = sample_trajectory(env, reward_fn, feature_fn)
            for state in traj:
                sampled_fe += feature_fn(state)
        sampled_fe /= len(expert_trajectories)

        grad = expert_fe - sampled_fe
        reward_fn.weights += lr * grad

        if i % 10 == 0:
            print(f"[{i}] Reward weight update norm: {np.linalg.norm(grad):.4f}")

    return reward_fn
