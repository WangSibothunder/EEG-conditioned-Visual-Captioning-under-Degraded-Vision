# Goal: 分段完成 EEG-enhanced VLM 课程设计实验报告

你现在的任务不是继续做实验，而是完成课程设计报告写作。根目录已经有 `实验报告.md`，其中包含报告总线、章节规划、实验主线、模型结果和写作要求。请严格阅读该文件，并按照 LaTeX 模板分段写作，最终形成一份约 2 万字的中文课程设计报告。

## 一、必须先读取的文件

1. `实验报告.md`：总大纲和主线。
2. `outputs/deep_gen_evlm/FINAL_DEEP_GEN_EVLM_REPORT.md`：最终生成式 EVLM 报告，如果存在。
3. `outputs/deep_gen_evlm/ALL_DEEP_GEN_METRICS.csv`：生成式指标，如果存在。
4. A2 final、VTF、Route5、best-of-N 相关 summary/metrics/examples 文件，如果存在。
5. 当前 LaTeX 模板：`main.tex`、`setup.tex`、`sections/*.tex`。

不要凭空编造不存在的指标。所有具体数值必须来自已有 metrics/report 文件；找不到就写 `TODO: 待填入实际结果`。

## 二、最终报告定位

报告题目：

```text
基于 EEG 增强视觉语言模型的退化图像语义识别与生成式描述研究
```

核心主线：

```text
最初目标：EEG + image → caption
困难：小样本 EEG 不适合直接开放生成
转向：CLIP semantic space 作为中间桥梁
主模型：A2 temporal-spectral-spatial semantic fusion
扩展：VTF token-level EEG-enhanced vision tokens
生成：Route5 Qwen2-VL LoRA generative EVLM
结果：A2 是最强定量模型，Route5 是生成式 EVLM 原型
```

不要把报告写成“乱试模型记录”。要写成一条有逻辑的研究演进路线。

## 三、篇幅要求

总字数目标：18,000–22,000 中文字。

大致分配：

- 摘要：500–700 字
- 第 1 章 绪论：2200–2600 字
- 第 2 章 相关技术与理论基础：2800–3300 字
- 第 3 章 数据集与任务定义：1800–2200 字
- 第 4 章 系统总体设计：2200–2600 字
- 第 5 章 A2 EEG 语义融合模型：3000–3500 字
- 第 6 章 Token-level EEG-enhanced VLM：2200–2700 字
- 第 7 章 生成式 EVLM 设计：2800–3500 字
- 第 8 章 实验结果与分析：3500–4500 字
- 第 9 章 工程实现与代码结构：1200–1800 字
- 第 10 章 总结与展望：900–1200 字

## 四、分阶段写作计划

不要一次性乱写完。按以下阶段逐步完成，每阶段写完后检查编译。

### Phase 1：绪论与理论基础

修改：

```text
sections/01_intro.tex
sections/02_related_work.tex
```

目标：约 5000–6000 字。

必须写清：

- 视觉语言模型背景
- 视觉退化问题
- EEG 作为视觉语义增强信号的意义
- CLIP 与生成式 VLM 的区别
- EEG 视觉解码特点
- 视觉退化建模
- LoRA 参数高效微调

### Phase 2：数据集、任务定义、系统总体架构

修改：

```text
sections/03_dataset_task.tex
sections/04_system_design.tex
```

目标：约 4000–4800 字。

必须写清：

- Thought2Text/EEGCVPR 数据基本情况
- image-level split 防泄漏
- clean/lowres16/mixed/occlusion50/strong_blur/strong_noise
- vision_only/real_eeg/shuffled_eeg/random_eeg/eeg_only
- Top-1、Top-5、Class Hit、valid caption rate 等指标
- 系统模块划分
- 实验流程

### Phase 3：A2 主定量模型

修改：

```text
sections/05_a2_semantic_fusion.tex
```

目标：约 3000–3500 字。

必须写清：

- 为什么设计 temporal-spectral-spatial EEG encoder
- A2 EEG encoder 的时间、频谱、空间通道建模
- CLIP text prototypes
- EEG-image semantic fusion
- real-vs-shuffled/random 对照训练思想
- A2 为什么是主定量模型

### Phase 4：VTF 与生成式 EVLM

修改：

```text
sections/06_vtf_token_fusion.tex
sections/07_generative_evlm.tex
```

目标：约 5000–6000 字。

必须写清：

- 从 pooled embedding 到 token-level 的动机
- CLIP ViT visual tokens `[B,50,512]`
- EEG tokens `[B,4,512]`
- visual-to-EEG cross-attention
- enhanced vision tokens
- 早期 Qwen soft prompt 为什么失败
- GRU decoder baseline 的作用和局限
- Route1–Route5 探索
- 最终 Route5 Qwen2-VL LoRA 结构
- caption target ablation
- best-of-N reranking

### Phase 5：实验结果与分析

修改：

```text
sections/08_experiments.tex
```

目标：约 4000–5000 字。

必须从真实 metrics 文件填入：

- A2 主结果
- A2 与 residual/prototype/P2/P2A2 等对比
- VTF token-level 结果
- Route5 Qwen2-VL 结果
- LoRA r8/r16 对比
- T1/T3 caption target 对比
- best-of-N reranking 结果
- qualitative examples：reflex camera、canoe、parachute、bolete mushroom 等

不能编造数值。缺失则写 TODO。

### Phase 6：工程实现、总结、摘要、参考文献

修改：

```text
sections/09_engineering.tex
sections/10_conclusion.tex
sections/00_abstract.tex
sections/11_references.tex
sections/A_appendix_code.tex
```

目标：约 2500–3500 字。

必须写清：

- 项目目录结构
- 核心代码模块
- 训练评估流程
- LoRA 依赖、显存、caption target 清洗等工程问题
- 总结三层成果：A2、VTF、Route5
- 不足与展望

## 五、图表要求

图片可以先用占位符，不需要立刻画。

但是必须保留以下图表位置：

- 图 2-1：CLIP 与生成式 VLM 区别示意图
- 图 4-1：系统总体架构图
- 图 4-2：实验流程图
- 图 5-1：A2 EEG encoder 结构图
- 图 5-2：A2 semantic fusion 流程图
- 图 6-1：VTF cross-attention 结构图
- 图 7-1：Route5 Qwen2-VL generative EVLM 架构图
- 图 8-1：A2 不同退化条件结果图
- 图 8-2：生成式 EVLM 结果对比图
- 图 8-3：qualitative caption 示例图

图必须后期自己绘制，不要直接抠网上图。

## 六、写作风格要求

1. 使用中文，专有名词可保留英文。
2. 不要夸大成“脑电直接生成任意语言”。
3. 统一表述为：

```text
小样本 EEG-enhanced generative VLM prototype
```

4. A2 是主定量模型，Route5 是生成式 EVLM 原型。
5. 结果部分要诚实写：生成式结果有提升，但仍不如 A2 作为纯 semantic classifier 稳定。
6. 每个模块都要写“动机—结构—实现—作用—结果”。
7. 不要写成流水账。

## 七、硬性检查

每完成一个 Phase 后，运行：

```bash
xelatex main.tex
xelatex main.tex
```

修复 LaTeX 编译错误。

最终必须保证：

- `main.tex` 能用 XeLaTeX 编译；
- 图、表、代码清单有编号和标题；
- 缺失图片用占位符；
- 具体实验数值来自真实 metrics；
- 最终报告主线一致。
