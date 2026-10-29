> 最新目录规则（2026-10-29，按用户后续要求）：大学公开课程直接放在仓库根目录，例如 `CME295/`、`CS188/`，不再添加 `Stanford/`、`Berkeley/` 等大学父目录。`Other_Courses/` 保留。根 README 按知识主题介绍学习内容，课程与大学作为来源信息。本条覆盖下文按大学分类的旧目录示例和相关流程。

# class_for_AI — 课程学习仓库总规则

你正在操作的工作区 `class_for_AI` 是我的**课程学习专用仓库**。

这个文件夹会与我本地电脑上的课程学习文件对应，并最终同步到 GitHub，用于长期保存、整理和展示我的课程学习记录。

你的核心职责是：

> 帮我按照课程组织和维护学习资料，包括笔记、课程作业、练习题、课程实验、课程 Project、课程代码和学习总结。

---

# 1. 工作区边界

这是最重要的规则。

## 允许放入本仓库的内容

只要内容属于某一门明确的“课程”，就可以放入本仓库。

课程可以来自：

- 我的大学正式课程
- Stanford / MIT / Berkeley / CMU 等大学公开课程
- Coursera / edX / Udacity 等平台课程
- YouTube 系统课程
- 教材配套课程
- 其他具有明确课程结构的系统学习内容

例如：

- CS189
- CS229
- CS231N
- CS224N
- CS336
- CS149
- MIT 6.5940
- CMU 11-785
- Daniel Bourke PyTorch Course
- 学校的数据结构课程
- 学校的数字逻辑课程

这些课程中产生的：

- Notes
- Homework
- Exercises
- Labs
- Course Projects
- Assignments
- Code
- Reading Notes
- Exam Review
- Course Summary

都属于本仓库。

---

# 2. 严格禁止放入本仓库的内容

**这个仓库不是我的通用 AI 项目仓库，也不是科研仓库。**

以下内容禁止放入 `class_for_AI`：

## 独立 GitHub 项目

例如我自己从 GitHub 找到并复现的：

- nano-vLLM
- vLLM
- SGLang
- llama.cpp
- MiniGPT
- LLaVA
- 自己寻找的开源项目
- 独立工程项目

即使这些项目用于学习，只要它们**不是某门课程明确要求的 Course Project / Assignment**，都不能放入这里。

---

## 独立论文复现

例如：

- FlashAttention 论文复现
- LoRA 论文复现
- CLIP 论文复现
- MyVLM
- Yo'LLaVA
- personalized LLM 论文复现
- 自己选择的 arXiv / conference paper reproduction

这些属于科研 / 独立实践，不属于课程学习仓库。

即使我是在“学习”这些内容，也不能因为学习目的而放进 `class_for_AI`。

---

## 独立个人项目

例如：

- 自己设计的 AI App
- Personal AI
- Multi-Agent 系统
- 游戏 AI
- 个性化模型实验
- AI Systems 独立实验
- 自己开发的工具

除非它是某门课程明确要求完成的课程项目，否则不能放入这里。

---

# 3. 判断原则

当你不知道某个内容是否应该进入本仓库时，使用下面这个判断：

> “这个东西是否可以明确归属于某一门课程？”

如果答案是：

**可以，而且属于课程教学、课程作业、课程实验或课程项目**

→ 可以放入。

如果答案是：

**这是我自己额外找的 GitHub 项目 / 论文 / 科研实践 / 独立项目**

→ 不允许放入。

如果存在歧义，不要擅自放入仓库，应提醒我确认。

---

# 4. 顶层目录组织原则

仓库以“课程”为最主要组织单位。

推荐整体结构：

```text
class_for_AI/
│
├── README.md
│
├── University/
│   ├── Course_Name/
│   └── ...
│
├── Stanford/
│   ├── CS189/
│   ├── CS229/
│   ├── CS224N/
│   ├── CS231N/
│   ├── CS336/
│   └── ...
│
├── MIT/
│   ├── 6.5940/
│   └── ...
│
├── Berkeley/
│   └── ...
│
├── CMU/
│   └── ...
│
└── Other_Courses/
    └── ...
```

学校正式课程统一放入：

```text
University/
```

海外大学公开课程优先按照大学分类：

```text
Stanford/
MIT/
Berkeley/
CMU/
...
```

无法明确归类到大学的系统课程放入：

```text
Other_Courses/
```

不要为了分类创建大量没有实际内容的空文件夹。

---

# 5. 每门课程的标准目录

创建一门新课程时，优先采用下面的结构：

```text
Course_Name/
│
├── README.md
│
├── Notes/
│
├── Exercises/
│
├── Assignments/
│
├── Labs/
│
├── Projects/
│
├── Code/
└── Resources/
```

但不要机械创建所有目录。

如果课程没有 Labs，就不需要 `Labs/`。

如果课程没有 Project，就不需要 `Projects/`。

只创建实际需要的目录。

---

# 6. 各目录含义

## Notes/

用于课程笔记。

例如：

```text
Notes/
├── Lecture_01_Introduction.md
├── Lecture_02_Linear_Models.md
├── Lecture_03_Optimization.md
└── ...
```

笔记应该尽可能按照 Lecture / Chapter / Topic 组织。

---

## Exercises/

用于我为了掌握课程内容主动做的练习。

包括：

- 课程配套练习
- 自己根据课程知识完成的小题
- 数学推导
- 编程练习
- Concept Check

例如：

```text
Exercises/
├── Linear_Regression/
├── Probability/
└── Optimization/
```

---

## Assignments/

课程正式布置的：

- Homework
- Problem Set
- Assignment

例如：

```text
Assignments/
├── HW01/
├── HW02/
└── HW03/
```

每个 Assignment 可以包含：

```text
HW01/
├── README.md
├── solution.md
├── src/
└── figures/
```

---

## Labs/

课程实验。

例如：

```text
Labs/
├── Lab01/
├── Lab02/
└── ...
```

---

## Projects/

只允许保存：

> **该课程明确要求或明确属于该课程体系的 Course Project。**

不能把独立 GitHub 项目或者论文复现塞进这里。

---

## Code/

用于不属于具体 Homework / Lab / Project 的课程示例代码或学习代码。

例如：

```text
Code/
├── linear_regression.py
├── neural_network.ipynb
└── transformer_example.py
```

---

## Resources/

保存课程相关辅助资料。

例如：

- syllabus
- cheatsheet
- reading list
- 官方课程资料链接
- 自己整理的资源索引

尽量不要直接提交体积巨大的公开视频、模型权重或数据集。

---

# 7. 每门课程必须有 README

每门课程根目录应尽量维护：

```text
README.md
```

README 推荐包含：

```markdown
# Course Name

## Course Information

- University:
- Course:
- Instructor:
- Semester / Version:
- Official Website:
- Status:

## Why I Learn This Course

简单说明学习目的。

## Progress

- [x] Lecture 01
- [x] Lecture 02
- [ ] Lecture 03

## Contents

### Notes
课程笔记。

### Exercises
练习。

### Assignments
课程作业。

### Labs
实验。

### Projects
课程项目。

## What I Learned

随着课程学习逐渐补充。

## Status

In Progress / Completed / Paused
```

不要伪造我尚未完成的学习进度。

---

# 8. 根目录 README

仓库根目录的 `README.md` 用于描述整个课程学习仓库。

它应该说明：

1. 这是我的课程学习记录仓库；
2. 内容主要包括课程笔记、练习、作业、实验和课程 Project；
3. 不包含独立论文复现和独立 GitHub 项目；
4. 按大学 / 课程进行分类；
5. 可以维护课程学习状态。

可以维护类似：

```markdown
## Courses

| Course | Institution | Area | Status |
|---|---|---|---|
| CS189 | UC Berkeley | Machine Learning | In Progress |
| CS229 | Stanford | Machine Learning | Planned |
| CS336 | Stanford | LLM | Planned |
```

只有我实际开始或计划记录的课程才加入，不要凭空添加大量课程。

---

# 9. 文件命名规范

优先使用：

```text
Lecture_01_Introduction.md
Lecture_02_Linear_Regression.md

HW01/
HW02/

Lab01/
Lab02/

Project_Name/
```

文件夹和代码文件优先使用英文。

正文笔记允许中文、英文或者中英混合。

不要随意使用：

```text
新建文件夹
未命名.md
test123.py
aaa.md
笔记1.md
最终版2.md
```

命名必须能直接反映内容。

---

# 10. Markdown 笔记规范

整理课程笔记时：

- 保留我的原始理解和内容；
- 可以改善 Markdown 排版；
- 可以修复明显的格式问题；
- 可以建立标题层级；
- 可以增加目录；
- 可以整理公式格式；
- 可以整理代码块；
- 可以整理图片引用；
- 可以帮助建立知识结构。

但不要擅自把我的笔记大规模改写成另一份教程。

尤其不能因为“看起来更完整”就添加大量我实际上没有学过的内容。

如果补充内容来自你的解释，应明确区分：

```markdown
> Additional Note:
```

或者类似标记。

---

# 11. 代码处理原则

课程代码应：

- 保留可运行性；
- 不随意改变课程要求；
- 不为了“优化代码”而重写我的作业；
- 保留我原本的实现思路；
- 可以修复明显的目录、import、格式和工程组织问题；
- 如果发现 bug，可以指出，但不要未经允许大规模替换我的实现。

对于课程 Assignment：

如果任务涉及需要我独立完成的作业，优先帮助我理解、调试和解释，而不是直接把整个答案替换掉。

---

# 12. Jupyter Notebook

对于 `.ipynb`：

如果课程本身使用 Notebook，可以正常保存。

推荐：

```text
Assignments/HW01/HW01.ipynb
Labs/Lab01/Lab01.ipynb
Code/Transformer_Demo.ipynb
```

不要无意义地把所有代码都转换成 Notebook。

---

# 13. 图片与附件

课程图片推荐放入对应内容附近，例如：

```text
Notes/
├── Lecture_03_CNN.md
└── assets/
    ├── cnn_architecture.png
    └── convolution_example.png
```

或者：

```text
Assignments/HW01/
├── README.md
├── solution.md
└── figures/
```

不要把所有课程的图片全部堆在仓库根目录。

---

# 14. 大文件规则

GitHub 仓库应保持轻量。

默认不要提交：

- 大型 Dataset
- Model Weights
- Checkpoints
- Cache
- Python 虚拟环境
- build 文件
- 大型视频
- 临时文件

例如：

```text
.venv/
venv/
__pycache__/
.ipynb_checkpoints/
*.pt
*.pth
*.ckpt
cache/
tmp/
```

如果课程确实需要大型数据，应优先：

- README 中记录下载方式；
- 提供下载链接；
- 提供脚本；
- 使用 `.gitignore`。

---

# 15. Git 安全原则

这个目录会同步到 GitHub，因此所有操作必须考虑 Git 历史。

不要：

- 随意删除大量已有文件；
- 强制覆盖我的笔记；
- 使用危险 Git 命令；
- 擅自执行 `git reset --hard`；
- 擅自执行 force push；
- 删除不确定用途的文件。

涉及大规模：

- 删除
- 移动
- 重命名
- 重构

前应先告诉我计划。

---

# 16. Commit 原则

如果我让你帮助提交 Git commit，commit message 应清晰反映课程与内容。

推荐格式：

```text
CS189: add lecture 3 notes

CS229: update optimization exercises

CS231N: add assignment 1 solutions

University-DataStructure: add binary tree lab
```

避免：

```text
update
fix
123
new files
test
```

---

# 17. 不要混淆“课程 Project”和“独立 Project”

这是长期最重要的分类规则之一。

例如：

## 可以进入仓库

如果 CS336 课程要求：

```text
Build a Transformer Language Model
```

那么：

```text
Stanford/CS336/Projects/Transformer_LM/
```

可以存在。

---

## 不可以进入仓库

如果我自己在 GitHub 找到：

```text
nano-vLLM
```

然后决定复现。

即使它和 CS336、LLM inference 等知识有关，也不能放入：

```text
Stanford/CS336/
```

因为它不是 CS336 的课程项目。

它应该属于我的独立项目 / AI Research & Engineering 工作区，而不是本仓库。

---

# 18. 不要混淆“课程阅读论文”和“论文复现项目”

如果某门课程要求阅读论文，可以记录：

```text
Course_Name/
└── Notes/
    └── Papers/
        └── Attention_Is_All_You_Need.md
```

这是课程阅读笔记，因此允许。

但如果我之后决定：

> 我要单独复现 Attention Is All You Need。

这个复现项目本身不应该放入本仓库。

---

# 19. 新课程加入流程

当我开始一门新的课程时，你应该：

1. 判断课程来源；
2. 确定应该位于哪个大学 / 类别目录；
3. 创建课程目录；
4. 根据实际需要创建子目录；
5. 创建课程 README；
6. 更新仓库根 README 中的课程列表；
7. 不要创建暂时没有用途的空目录。

例如我说：

> 我开始学习 Stanford CS336。

可以组织为：

```text
Stanford/
└── CS336/
    ├── README.md
    └── Notes/
```

随着课程推进再逐渐增加：

```text
Assignments/
Labs/
Projects/
Code/
```

---

# 20. 已有课程整理原则

如果目录中已经存在文件：

先理解当前结构，再进行整理。

优先：

> 保留已有内容 → 判断所属课程 → 分类 → 重命名 → 建立清晰结构。

不要为了符合模板而把已经合理的结构全部推翻。

一致性很重要，但数据安全和已有学习记录更重要。

---

# 21. 我给你新文件时的默认行为

以后如果我给你：

- 一份课程笔记
- 一段代码
- 一道题
- 一个 Assignment
- 一个实验
- 一个 Course Project

你首先判断：

> 它属于哪门课程？

然后判断：

> 它应该属于 Notes / Exercises / Assignments / Labs / Projects / Code / Resources 中哪一类？

然后放到合理位置。

如果课程归属已经明确，就不要重复询问。

---

# 22. 我的核心目标

这个仓库不是为了堆文件。

最终目标是形成一个能够长期维护的：

> Course Learning Archive

它应该让我几年以后依然可以快速找到：

- 我学过哪些课程；
- 每门课程学到了哪里；
- 我的课程笔记；
- 做过哪些题；
- 完成过哪些 Assignment；
- 做过哪些实验；
- 完成过哪些课程 Project；
- 当时写过什么代码；
- 对课程最终掌握到了什么程度。

同时它也是一个可以上传 GitHub 的、结构清晰的学习记录仓库。

---

# 23. 工作原则优先级

你进行任何文件操作时，遵循以下优先级：

1. **严格保证这里只保存课程相关内容**
2. **不丢失我的原始学习资料**
3. **保持清晰的课程归属**
4. **保持目录结构长期可维护**
5. **保持 GitHub 仓库整洁**
6. **改善命名和 Markdown 可读性**
7. **自动化重复整理工作**

如果某项操作违反更高优先级规则，则不要执行。

---

# 24. 最重要的永久规则

始终记住：

> `class_for_AI` = 课程学习仓库。

这里记录的是：

> Course → Notes / Exercises / Assignments / Labs / Course Projects / Code

这里**不是**：

> GitHub 项目收藏夹  
> 论文复现仓库  
> AI Research 仓库  
> 独立工程项目仓库  
> 临时文件仓库

即使某个项目、论文或代码对我的学习非常重要，只要它不是某门课程本身的一部分，就不要放入这里。