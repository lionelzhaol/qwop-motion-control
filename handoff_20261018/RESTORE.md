# 完整工作区快照恢复

本次按用户要求保存整个工作区，包含所有模型、中间 checkpoint、日志、虚拟环境、依赖安装包、缓存、PPT 与验证资料。常用代码和精选模型可直接从仓库获取；其余内容并未删除，而是放在同一私有仓库的 Release 分卷中。

1. 从 `workspace-snapshot-20261018` Release 下载所有 `qwop-workspace-20261018.tar.zst.partNNN` 文件及 `SHA256SUMS.txt`，放在同一文件夹。
2. 在 Windows 11 上使用 Python 和系统自带的 tar，执行仓库中的恢复脚本：

```powershell
python handoff_20261018/tools/restore_snapshot.py --parts "E:/下载的分卷目录" --destination "E:/空的恢复目录"
```

脚本先逐卷校验 SHA256，再按编号拼接输入 tar。解压后的 `qwop-workspace/` 对应原 `E:/Work/强化学习小项目/`。`snapshot_manifest.json` 列出每个原文件的路径、大小与 SHA256。

快照使用标准 tar 硬链接表达内容完全相同的文件，避免重复保存安装包与相同 checkpoint；解压后的逻辑文件内容不变。Zstandard 窗口为 128 MiB。

三个指向工作区以外的软件目录联接仅保存在 `directory_links.json`，没有复制外部软件。跨电脑运行建议重建虚拟环境；已有 `.venv` 作为原始资料保留，不承诺可直接迁移使用。

Git 管理目录 `.git` 不作为普通资料再次上传；原工程完整提交历史保存在仓库 `training-history` 分支和 `handoff_20261018/training-history.bundle`。需要恢复原训练仓库历史时可使用：

```powershell
git clone handoff_20261018/training-history.bundle training-history
```

正在生成的归档自身及传输状态文件不递归包含进快照。原目录未进行删除、移动或清理。
