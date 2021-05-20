import torch
from torch.nn import functional as F
import numpy as np

import stable_baselines3 as sb
import gym
from stable_baselines3.common import logger

from utils import ENV

class M_DQN(sb.DQN):
    def train(self, gradient_steps: int, batch_size: int = 100, entropy_tau = 0.01, alpha=0.9, lo=-1) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)

                # calculate entropy term with logsumexp trick to stay in log space
                logsum = torch.logsumexp((next_q_values - next_q_values.max(1)[0].unsqueeze(-1))/entropy_tau, 1).unsqueeze(-1)
                tau_log_pi_next = next_q_values - next_q_values.max(1)[0].unsqueeze(-1) - entropy_tau*logsum
                pi_target = F.softmax(next_q_values/entropy_tau, dim=1)

                regularized_next_q_values = (pi_target * (next_q_values - tau_log_pi_next) * (1 - replay_data.dones)).sum(1)
                regularized_next_q_values = (self.gamma * regularized_next_q_values).unsqueeze(-1)
                
                # calculate munchausen addon with logsumexp trick and q_net_target
                current_q_values = self.q_net_target(replay_data.observations).detach()
                current_v_values = current_q_values.max(1)[0].unsqueeze(-1)
                logsum = torch.logsumexp((current_q_values - current_v_values)/entropy_tau, 1).unsqueeze(-1)
                log_pi = current_q_values - current_v_values - entropy_tau*logsum
                munchausen_ = log_pi.gather(1, replay_data.actions.long())
                munchausen_clipped = alpha * munchausen_.clip(min=lo, max=0)

                target_q_values = replay_data.rewards + munchausen_clipped + regularized_next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss", np.mean(losses))


print('Starting training...')
env = gym.make(ENV)
env.seed(0)
sb.common.utils.set_random_seed(1)

model = M_DQN("MlpPolicy", env, verbose=0, tensorboard_log=f'output/{env.spec.id}/', 
            learning_rate = lambda x: x * 4e-3 + (1-x) * 1e-3 if x > 0.55 else 1e-9,
            buffer_size = 50000, tau=1.0, batch_size=256, target_update_interval = 4000, max_grad_norm=1.0,
            train_freq = 2, gradient_steps = 1,
            exploration_fraction = 0.3, exploration_final_eps = 0.08,
            learning_starts=1000, policy_kwargs={'net_arch': [256, 256]})

# model that beats DQN at least once :)))
# model = M_DQN("MlpPolicy", env, verbose=0, tensorboard_log=f'output/{env.spec.id}/', 
#             buffer_size = 16000, tau=1, batch_size=256, target_update_interval = 10000, max_grad_norm=1,
#             train_freq=1, learning_starts=1000, policy_kwargs={'net_arch': [256, 256]})

model.learn(total_timesteps=1000000, tb_log_name = "M_DQN", log_interval = 5)
model.save(f'output/{env.spec.id}-mdqn-1')

print('Starting evaluation...')
model = M_DQN.load(f'output/{env.spec.id}-mdqn')

G = []
for _ in range(30):
    obs = env.reset()
    # env.render()
    done = False
    cur = 0
    while not done:
        action, _states = model.predict(obs)
        obs, r, done, info = env.step(action)
        cur += r
        # env.render()
    G.append(cur)
    print(cur)
print(sum(G) / len(G))
