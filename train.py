import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 解决OpenMP冲突
#tensorboard --logdir runs
import os
import time
from datetime import datetime
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from PPO import PPO
from rocket import Rocket

def train():
    print("============================================================================================")

    env_name = "RocketLanding"
    task = 'landing'

    max_ep_len = 1000
    max_training_timesteps = int(6e6)

    print_freq = max_ep_len * 10
    save_model_freq = int(1e5)

    update_timestep = max_ep_len * 4
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lam = 0.95
    lr_actor = 0.0003
    lr_critic = 0.001

    print("Training environment:", env_name)

    env = Rocket(max_steps=max_ep_len, task=task, rocket_type='starship')
    state_dim = env.state_dims
    action_dim = env.action_dims

    log_dir = f"./runs/RocketLanding_{datetime.now().strftime('%Y%m%d_%H%M')}"
    writer = SummaryWriter(log_dir=log_dir)

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, lam=lam)

    directory = "./PPO_preTrained/RocketLanding/"
    os.makedirs(directory, exist_ok=True)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at:", start_time)

    time_step = 0
    i_episode = 0
    print_running_reward = 0
    print_running_episodes = 0

    try:
        while time_step <= max_training_timesteps:
            state = env.reset()
            current_ep_reward = 0

            for t in range(1, max_ep_len + 1):
                action = ppo_agent.select_action(state)
                state, reward, done, _ = env.step(action)

                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step += 1
                current_ep_reward += reward

                if time_step % update_timestep == 0:
                    ppo_agent.update()

                if done:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1
            i_episode += 1

            if i_episode % 10 == 0:
                avg_reward = print_running_reward / print_running_episodes
                print(f"Episode: {i_episode} | Timestep: {time_step} | Avg Reward: {avg_reward:.2f}")
                writer.add_scalar("Reward/Average", avg_reward, time_step)
                writer.add_scalar("Reward/Episode", current_ep_reward, time_step)
                print_running_reward = 0
                print_running_episodes = 0

            if time_step % save_model_freq == 0:
                checkpoint_path = directory + f"PPO_RocketLanding_{i_episode}.pth"
                ppo_agent.save(checkpoint_path)
                print(f"Model saved at timestep {time_step}")

    except KeyboardInterrupt:
        print("\n\n训练被手动中断，正在保存当前模型...")

    # 训练结束或中断时，保存最终模型
    final_model_path = directory + "PPO_RocketLanding_FINAL.pth"
    ppo_agent.save(final_model_path)
    print(f"最终模型已保存至: {final_model_path}")
    print(f"总训练时间: {datetime.now().replace(microsecond=0) - start_time}")

    writer.close()
    print("Training finished.")

if __name__ == '__main__':
    train()