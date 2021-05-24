import gym
import stable_baselines3 as sb
from stable_baselines3.common import logger

from utils import ENV

import torch
from torch.nn import functional as F

class M_A2C(sb.A2C):
    def train(self, entropy_tau = 1e-5, munchausen_alpha = 1e-2, lo=-1) -> None:
        """
        Update policy using torche currently gatorchered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            # TODO: avoid second computation of everytorching because of torche gradient
            values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in torche original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()

            next_state_entropy = entropy_tau * self.gamma * self.policy.evaluate_actions(torch.from_numpy(self._last_obs), torch.empty(0))[-1].detach()
            munchausen_clipped = munchausen_alpha * entropy_tau * log_prob.clip(min=lo, max=0).detach()
            target_values = rollout_data.returns + munchausen_clipped + next_state_entropy

            # Value loss using torche TD(gae_lambda) target
            value_loss = F.mse_loss(target_values, values)

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/entropy_loss", entropy_loss.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/value_loss", value_loss.item())

print('Starting training...')
env = gym.make(ENV)
env.seed(1)
sb.common.utils.set_random_seed(1)

model = M_A2C("MlpPolicy", env, verbose=0, tensorboard_log=f'output/{env.spec.id}/', 
        # learning_rate = lambda x: 9e-4 * x + (1-x) * 5e-4,
        n_steps=8, 
        )

# model.learn(total_timesteps=100000, tb_log_name = "M_A2C", log_interval = 5, )
# model.save(f'output/{env.spec.id}-m_a2c')

print('Starting evaluation...')
model = M_A2C.load(f'output/{env.spec.id}-m_a2c-1')

import pandas as pd
from tqdm import trange

G = []
for _ in trange(100):
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
print(pd.Series(G).describe())
# pd.Series(G).to_csv(f'data/{env.spec.id}/eval-ma2c.csv')
