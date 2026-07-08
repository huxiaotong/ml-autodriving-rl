# ML Autodriving RL

学习目标：理解自动驾驶强化学习从系统地图、模型训练、仿真、奖励函数、评估、sim-to-real、真实部署，到 AWS 最小训练项目的完整工程流程。

## 内容结构

```text
docs/
  README.md
  stage-01-system-map.md
  ...
  stage-09-capstone-project-plan.md
  deepracer-comparison-and-migration.md

code/
  README.md
  configs/
  training/
  infra/
  tools/
```

## 推荐入口

- 学习路线索引：[docs/README.md](docs/README.md)
- 第 9 阶段项目计划：[docs/stage-09-capstone-project-plan.md](docs/stage-09-capstone-project-plan.md)
- AWS 最小训练项目代码：[code/README.md](code/README.md)
- DeepRacer 对比与切换方案：[docs/deepracer-comparison-and-migration.md](docs/deepracer-comparison-and-migration.md)

## 项目说明

`code/` 中的内容是学习脚手架，用 `highway-env + PPO + AWS` 跑通最小训练、评估和模型注册流程。它不是可部署到真实车辆的自动驾驶系统。

