import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os
import torch
from PPO import PPO
from rocket import Rocket

def test():
    print("============================================================================================")
    print("开始测试PPO火箭着陆模型...")

    # ====================== 参数设置 ======================
    env_name = "RocketLanding"
    task = 'landing'                    # 'landing' 或 'hover'
    rocket_type = 'starship'            # 'starship' 或 'falcon'

    max_ep_len = 1000                   # 每episode最大步数
    total_test_episodes = 10            # 测试多少局（可自行修改）
    render = True                       # 是否实时显示画面
    frame_delay = 8                     # 画面延迟（毫秒），调大画面会慢一点，便于观察

    # 加载的模型路径（请改成你实际保存的模型文件名）
    model_path = "./PPO_preTrained/RocketLanding/PPO_RocketLanding_FINAL.pth"

    # ====================== 创建环境和代理 ======================
    env = Rocket(max_steps=max_ep_len, task=task, rocket_type=rocket_type)
    state_dim = env.state_dims
    action_dim = env.action_dims

    ppo_agent = PPO(state_dim, action_dim)

    # 加载训练好的模型
    if os.path.exists(model_path):
        ppo_agent.load(model_path)
        print(f"成功加载模型: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        return

    # ====================== 开始测试 ======================
    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):
        state = env.reset()
        ep_reward = 0

        print(f"\n--- Episode {ep} 开始 ---")

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render(wait_time=frame_delay)   # 调用环境自带的渲染函数

            if done:
                break

        test_running_reward += ep_reward
        print(f"Episode {ep} 结束 | 奖励: {ep_reward:.2f} | 步数: {t}")

    # ====================== 最终统计 ======================
    avg_test_reward = test_running_reward / total_test_episodes
    print("\n============================================================================================")
    print(f"测试完成！共测试 {total_test_episodes} 局")
    print(f"平均奖励: {avg_test_reward:.2f}")
    print("============================================================================================")

if __name__ == '__main__':
    test()