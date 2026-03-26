# Rocket Recycling with PPO

**基于 PPO 的火箭回收连续控制项目**  
使用 GitHub 开源环境 **[jiupinjia/rocket-recycling](https://github.com/jiupinjia/rocket-recycling)** 实现 SpaceX Starship 风格的 2D 刚体火箭回收任务（belly-flop 机动 + 垂直软着陆）。

### 项目说明
本项目**完全基于**开源仓库 [jiupinjia/rocket-recycling](https://github.com/jiupinjia/rocket-recycling) 的仿真环境，实现了 PPO 算法（离散动作空间），成功完成了从高空自由落体到精确姿态控制与软着陆的完整决策流程。

### 核心贡献
- 将原始环境接入 PPO 框架，构建 Actor-Critic 网络（Categorical 策略采样）
- 实现 GAE 优势函数估计、Clip surrogate 目标、熵正则项
- 通过 RolloutBuffer + 多轮更新，实现了从 -300 奖励到稳定400的收敛

### 训练结果
- 最终累计奖励从 -300 显著提升至400
- 成功实现稳定软着陆（姿态控制 + 垂直着陆）

### 如何复现
```bash
# 1. 克隆本仓库
git clone https://github.com/Robinsuperm/PPO-rocket-recycling.git

# 2. 安装依赖
pip install -r requirements.txt

# 3. 开始训练
python train.py
