# Autodriving RL 学习路线索引

这个目录记录从“理解自动驾驶强化学习”到“用 AWS 跑通一个最小训练项目”的学习材料。整体目标不是研究 RL 算法细节，而是理解训练、评估、部署和数据闭环的工程过程。

---

## 阅读顺序

| 阶段 | 文档 | 重点 |
| --- | --- | --- |
| 1 | [建立整体地图](stage-01-system-map.md) | 自动驾驶不是一个模型，而是感知、预测、规划、控制、安全和云端闭环的系统 |
| 2 | [理解不同模型训练的基本语言](stage-02-model-training-language.md) | 感知、预测、规划、控制、安全、场景挖掘、RL policy 各自怎么训练 |
| 3 | [理解为什么自动驾驶需要仿真](stage-03-simulation-and-validation.md) | 仿真、场景库、离线回放、验证链路 |
| 4 | [看懂一个最小训练流程](stage-04-minimal-training-flow.md) | 环境、observation、action、reward、训练循环、checkpoint、评估 |
| 5 | [重点理解奖励函数](stage-05-reward-function.md) | reward 组件、权重、reward hacking、版本管理 |
| 6 | [理解训练后的评估](stage-06-post-training-evaluation.md) | 固定评估场景、多指标、gate、失败案例、回归测试 |
| 7 | [理解仿真到现实的差距](stage-07-sim-to-real-gap.md) | sim-to-real gap、domain randomization、真实日志回放、shadow mode |
| 8 | [理解真实部署流程](stage-08-real-deployment-flow.md) | 模型治理、打包、车端运行时、灰度、回滚、监控、数据回流 |
| 9 | [做一个小项目来串起来](stage-09-capstone-project-plan.md) | 用 `highway-env + PPO + AWS` 串起训练、评估、注册和风险判断 |

补充模板：

| 文档 | 用途 |
| --- | --- |
| [第 9 阶段项目报告模板](stage-09-project-report-template.md) | 记录实验结果、gate 结论、sim-to-real 风险和下一步 |
| [与 AWS DeepRacer 的比较和切换方案](deepracer-comparison-and-migration.md) | 比较当前小项目和 DeepRacer，并说明如果切换到 DeepRacer 要做什么 |

---

## 代码入口

第 9 阶段项目代码统一放在：

```text
../code/
```

核心内容：

```text
code/
  README.md
  configs/
    highway-ppo-v001.yaml
    highway-ppo-conservative.yaml
    highway-ppo-speedy.yaml
  training/
    train.py
    evaluate.py
    Dockerfile
  infra/
    cloudformation/foundation.yaml
    scripts/*.sh
  tools/
    summarize_evaluations.py
```

运行入口：

```bash
cd /Users/xiaotonghu/Documents/ML/autodriving-rl/code
```

---

## 当前结构约定

- 第 1-8 阶段主要是概念讲义。
- 第 4 阶段讲“最小训练流程是什么”，不再承载具体代码说明。
- 第 9 阶段是实际项目入口，复用 `code/` 中的 IaC、训练、评估和实验脚本。
- `code/` 里的项目是学习脚手架，不是可部署的真实自动驾驶系统。

---

## 目前已实现

- AWS 最小基础设施 IaC：S3、ECR、SageMaker execution role。
- SageMaker Training 训练入口。
- SageMaker Processing 评估入口。
- Model Registry 候选模型注册脚本。
- 三组 reward 配置：baseline、conservative、speedy。
- 增强版 `evaluation_report.json`：metrics、gates、decision、failure reasons、per-episode metrics。
- 多实验汇总脚本：`tools/summarize_evaluations.py`。
- 第 9 阶段项目报告模板。

---

## 已知刻意重复

这些主题会在多个阶段重复出现，但角度不同：

- `Model Registry`：第 2 阶段讲模型训练语言，第 4/9 阶段讲最小项目，第 8 阶段讲真实部署治理。
- `Greengrass / Kubernetes / EKS Anywhere`：第 1 阶段讲边缘编排边界，第 8 阶段讲真实部署运行时。
- `仿真和评估`：第 3 阶段讲为什么需要，第 6 阶段讲训练后评估体系，第 9 阶段落到项目实践。

这些不是冲突，而是为了让同一个工程概念在不同上下文中逐步加深。
