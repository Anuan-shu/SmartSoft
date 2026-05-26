# SWE-bench Homework Agent Framework

## Features

- 输入 `swebench_tasks.txt`，按顺序输出标准 `predictions.jsonl`
- 强制通过 SWE-bench 官方 Docker 环境与仓库交互
- 大家只需要实现 `src/agent.py` 中的 `solve_task(...)`
- `utils/models.py` 负责读取根目录 `.env`、调用模型、记录 token 用量
- `evaluate.py` 负责执行官方评估脚本 `swebench.harness.run_evaluation`

## Layout

```text
.
├── main.py
├── evaluate.py
├── scripts/
│   └── download_dataset.py
├── src/
│   └── agent.py
├── utils/
│   ├── docker_env.py
│   ├── models.py
│   ├── output.py
│   ├── patches.py
│   └── tasks.py
├── tests/
├── swebench_tasks.txt
└── .env.example
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure Model

在仓库根目录创建 `.env`：

```env
apikey=your-api-key
base=https://xiaoai.plus/v1
model=deepseek-v3.2
```

## Prepare Tasks

`swebench_tasks.txt` 每行一个 `instance_id`，支持空行和 `#` 注释：

```text
# one instance per line
sympy__sympy-20590
django__django-11099
```

## Download Dataset To Local Disk

先把 SWE-bench 数据集从 Hugging Face 拉到本地目录，之后主流程只读本地：

```bash
python scripts/download_dataset.py \
  --dataset princeton-nlp/SWE-bench_Lite \
  --output-dir data/princeton-nlp__SWE-bench_Lite
```

随后拉取所需的镜像（镜像占用空间较大，推荐同学们只拉取几个进行调试）

```bash
python -m scripts.pull_images \
  --tasks swebench_tasks.txt
```

## Generate Predictions

```bash
python main.py \
  --tasks swebench_tasks.txt \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --output predictions.jsonl
```

生成完成后会同时写出：

- `predictions.jsonl`
- `runs/<run_id>/usage.json`
- `runs/<run_id>/instances/<instance_id>/...` 日志

## Evaluate Predictions

```bash
python evaluate.py \
  --predictions predictions.jsonl \
  --dataset princeton-nlp/SWE-bench_Lite \
  --split test \
  --run-id demo-run \
  --max-workers 1
```

## TODOs

同学们需要实现：

```python
def solve_task(context: TaskContext, env: DockerEnv, model: ModelClient) -> str:
    ...
```

返回值必须是 unified diff patch 字符串，也就是最终写入 `model_patch` 的内容。

## Docker Note

框架默认通过 SWE-bench 官方 harness 构建并启动实例容器，工作目录固定是 `/testbed`

## Sources

- 官方 Harness Reference: [swebench.com/SWE-bench/reference/harness](https://www.swebench.com/SWE-bench/reference/harness/)
- 官方 Harness API: [swebench.com/SWE-bench/api/harness](https://www.swebench.com/SWE-bench/api/harness/)
