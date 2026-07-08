# 第 9 阶段项目报告模板

项目名称：Highway Lane-Change RL on AWS

日期：

负责人：

---

## 1. 项目目标

说明本项目要验证什么：

```text
用 highway-env + PPO + AWS 最小训练流水线，跑通自动驾驶 RL 策略的训练、评估、模型注册和风险判断流程。
```

明确本项目不验证什么：

```text
这不是可部署到真实车辆的自动驾驶系统。
```

---

## 2. 任务定义

环境：

```text
highway-v0
```

Agent：

```text
高速公路自车
```

目标：

```text
安全、高效、稳定地行驶，并在需要时做变道决策。
```

---

## 3. Observation / Action / Reward

Observation：

```text
填写 observation features
```

Action：

```text
填写 action space
```

Reward：

```text
填写 reward version 和主要权重
```

---

## 4. AWS 架构

使用的 AWS 服务：

| 服务 | 用途 |
| --- | --- |
| S3 | 配置、模型产物、评估报告 |
| ECR | 训练和评估镜像 |
| SageMaker Training | 云上训练 |
| SageMaker Processing | 离线评估 |
| Model Registry | 候选模型注册 |
| CloudWatch | 日志 |

---

## 5. 实验配置

| 实验 | 配置文件 | reward 版本 | 目的 |
| --- | --- | --- | --- |
| baseline | `code/configs/highway-ppo-v001.yaml` | reward-v001-baseline | 对照 |
| conservative | `code/configs/highway-ppo-conservative.yaml` | reward-v002-conservative | 更保守 |
| speedy | `code/configs/highway-ppo-speedy.yaml` | reward-v003-speedy | 更追求速度 |

---

## 6. 评估指标

| 指标 | 说明 |
| --- | --- |
| average_reward | 平均累计 reward |
| success_rate | 未碰撞完成 episode 的比例 |
| collision_rate | 碰撞比例 |
| unsafe_distance_rate | 危险跟车距离比例 |
| average_speed | 平均速度 |
| hard_brake_count_per_100_episodes | 每 100 episodes 急刹次数 |
| lane_change_count_per_episode | 平均每 episode 变道次数 |

---

## 7. 实验结果

| 实验 | average_reward | success_rate | collision_rate | unsafe_distance_rate | average_speed | decision |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| conservative | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| speedy | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

---

## 8. Gate 结论

Gate 配置：

```yaml
min_success_rate:
max_collision_rate:
max_unsafe_distance_rate:
max_hard_brake_count_per_100_episodes:
```

结论：

```text
pass / fail / needs_review
```

失败原因：

```text
填写 failure_reasons
```

---

## 9. 失败案例分析

记录最重要的失败行为：

```text
例如：碰撞、危险距离、频繁变道、急刹、卡住不动。
```

可能原因：

```text
reward、场景、动作空间、训练步数、评估场景不足等。
```

---

## 10. sim-to-real 风险

| 风险 | 当前状态 | 下一步 |
| --- | --- | --- |
| 传感器输入 | 使用结构化仿真状态 | 未来加入真实传感器或更复杂仿真 |
| 车辆动力学 | 简化 | 未来用更真实动力学 |
| 交通行为 | 简化 | 增加 aggressive / distracted drivers |
| 安全层 | 未集成 | 增加 safety layer |
| 车端部署 | 未覆盖 | 未来做 shadow mode 或边缘推理模拟 |

---

## 11. 最终判断

这个模型是否进入下一阶段：

```text
否 / 是，但仅限继续仿真评估 / 是，可进入 shadow mode 设计
```

理由：

```text
填写理由。
```

---

## 12. 下一步

- 增强 evaluation report。
- 增加固定 evaluation suite。
- 增加 regression suite。
- 做 reward 对比。
- 升级到 MetaDrive 或 CARLA。
- 增加安全层。

