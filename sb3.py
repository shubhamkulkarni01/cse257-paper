import gym
import stable_baselines3 as sb

env = gym.make('CartPole-v0')
model = sb.DQN("MlpPolicy", env, verbose=0, tensorboard_log='output/', 
        buffer_size = 10000, tau=1, batch_size=256, target_update_interval = 10000, max_grad_norm=1,
        train_freq=1, learning_starts=100, policy_kwargs={'net_arch': [256, 256]})
model.learn(total_timesteps=100000, log_interval = 5)

model.save('output/CartPole-v0-dqn')

model.load('output/CartPole-v0-dqn')

for _ in range(1):
    obs = env.reset()
    env.render()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
        env.render()
