# EEG-enhanced VLM 课程设计报告 LaTeX 模板

## 编译方式

推荐使用 XeLaTeX：

```bash
xelatex main.tex
xelatex main.tex
```

如果安装了 latexmk：

```bash
latexmk -xelatex main.tex
```

## 字体说明

模板不包含任何字体文件。编译时会优先使用 Windows 常见字体 SimSun/SimHei；若没有则尝试 Noto Serif CJK SC / Noto Sans CJK SC；再回退到 TeX Live 自带 Fandol 字体。

## 格式说明

- 正文：宋体，小四，1.5 倍行距。
- 一级标题：黑体，二号，居中。
- 二级标题：黑体，小三。
- 图、表、代码清单均有编号和标题。
- 图片暂时用占位框，后期替换为自己绘制的图。

## 写作建议

DeepSeek/Claude 分阶段写作时，直接修改 `sections/*.tex`。不要一次性重写 `main.tex` 和 `setup.tex`。

建议顺序：

1. `01_intro.tex` + `02_related_work.tex`
2. `03_dataset_task.tex` + `04_system_design.tex`
3. `05_a2_semantic_fusion.tex`
4. `06_vtf_token_fusion.tex` + `07_generative_evlm.tex`
5. `08_experiments.tex`
6. `09_engineering.tex` + `10_conclusion.tex` + 摘要 + 参考文献

## 注意

课程模板要求图不能直接抠图，建议使用 draw.io/PPT/Python/Mermaid 自己绘制后导出 PDF/PNG 放入 `figures/`。
