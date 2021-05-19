import gym
import stable_baselines3 as sb

env = gym.make('CartPole-v1')
model = sb.DQN("MlpPolicy", env, verbose=0, tensorboard_log=f'output/{env.spec.id}/', 
            buffer_size = 16000, tau=1, batch_size=256, target_update_interval = 10000, max_grad_norm=1,
            train_freq=1, learning_starts=1000, policy_kwargs={'net_arch': [256, 256]})

# model = sb.DQN("MlpPolicy", env, verbose=0, tensorboard_log=f'output/{env.spec.id}/')
model.learn(total_timesteps=100000, log_interval = 5)

model.save(f'output/{env.spec.id}-dqn')

model.load(f'output/{env.spec.id}-dqn')

for _ in range(3):
    obs = env.reset()
    env.render()
    done = False
    G = 0
    while not done:
        action, _states = model.predict(obs)
        obs, r, done, info = env.step(action)
        G += r
        env.render()
    print(G)
