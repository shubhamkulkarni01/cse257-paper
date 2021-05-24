import gym
import stable_baselines3 as sb

from utils import ENV

class DQN(sb.DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with torch.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)

                target_q_values = replay_data.rewards + \
                        (1 - replay_data.dones) * self.gamma * next_q_values.max(1)[0].unsqueeze(-1)

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
env.seed(1)
sb.common.utils.set_random_seed(1)

model = sb.DQN("MlpPolicy", env, verbose=0,  tensorboard_log=f'output/{env.spec.id}/', 
            buffer_size = 16000, tau=1, batch_size=256, target_update_interval = 10000, max_grad_norm=1,
            train_freq=1, learning_starts=1000, policy_kwargs={'net_arch': [256, 256]})

# model.learn(total_timesteps=1000000, log_interval = 5)
# model.save(f'output/{env.spec.id}-dqn')

print('Starting evaluation...')
model = DQN.load(f'output/{env.spec.id}-dqn')


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
# pd.Series(G).to_csv(f'data/{env.spec.id}/eval-dqn.csv')
