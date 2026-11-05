# PC、GitHub与学校服务器同步

GitHub `SWT110/eeg` 的 `main` 为代码和文档的交换中枢。PC与学校服务器各有工作副本，修改先经过Git提交，再推送/拉取。两个学校SSH入口使用同一份仓库；更新一次即可。

同步范围为代码、配置、测试、说明文档与模型结构设计文件。`analysis/`、`local_artifacts/`、原始/中间数据、模型权重、Python环境、日志、崩溃转储与运行归档保留在各机器本地，不进入GitHub。说明文档中引用的本机分析图表需要在本机打开，不会随代码自动下载。

## PC提交并更新服务器

在PC的eeg仓库中，先审阅并提交需要同步的文件：

```bash
git status --short
git add <本次审阅过的文件>
git commit -m "说明本次修改"
python3 scripts/sync_repo.py sync --host school-eeg-2
```

`sync` 先将PC已提交版本推送到GitHub，再让学校服务器快进拉取；也可指定`--host school-eeg`。当前学校仓库路径为`/public/home/gd_110413/SWT/work/eeg`。

首次部署脚本前，可在学校仓库直接使用`git pull --ff-only origin main`完成引导。

## 学校服务器修改后更新PC

服务器完成代码或文档修改后，审阅、提交并使用已有GitHub认证推送。仓库已有`push_to_github.sh`可通过加密的仓库部署密钥交互推送，私钥与口令均保留在仓库外；不需要把PC的GitHub令牌复制到学校服务器。

```bash
git add <本次审阅过的文件>
git commit -m "说明服务器修改"
./push_to_github.sh
```

随后在PC运行：

```bash
python3 scripts/sync_repo.py pull
```

## 查看与恢复

```bash
python3 scripts/sync_repo.py status --fetch
python3 scripts/sync_repo.py push
python3 scripts/sync_repo.py deploy --host school-eeg-2
```

脚本仅传输已提交内容；工作副本有未提交改动、处于其他分支或历史发生分叉时会停止，保留现场供处理。它不会自动暂存/提交、强推、重置、删除本地产物或启动GPU任务。

双端同时修改时，先分别保存提交，再合并并解决实际冲突。初次三端对齐的旧工作区补丁、服务器文档导出和合并记录保存在各自`local_artifacts/sync_backups/`，该目录不参与Git同步。

实验复现继续使用运行目录中的冻结代码快照及其指纹；后续Git更新不修改已有`local_artifacts/benchmark_sources/`或历史训练结果。
