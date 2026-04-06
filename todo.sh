#!/bin/bash
#
# MoXing 开发助手脚本
#

echo "=============================================="
echo "  MoXing 开发进度"
echo "=============================================="
echo ""

# 显示 git status
echo "Git Status:"
git status --short
echo ""

# 检查未推送的 commits
UNPUSHED=$(git log origin/main..HEAD --oneline 2>/dev/null | wc -l)
if [ "$UNPUSHED" -gt 0 ]; then
    echo "⚠️  未推送的 commits: $UNPUSHED"
    git log origin/main..HEAD --oneline
    echo ""
fi

# 显示最新的 CI 运行
echo "最新 CI 运行:"
gh run list --limit 3
echo ""

# 如果有 tag，显示最新版本
LATEST_TAG=$(git tag -l 'v0.1.*' | sort -V | tail -1)
if [ -n "$LATEST_TAG" ]; then
    echo "最新版本: $LATEST_TAG"
    echo ""
fi

echo "=============================================="
echo "  快捷命令"
echo "=============================================="
echo ""
echo "推送代码:      git push"
echo "创建版本:      git tag v0.1.XX && git push origin v0.1.XX"
echo "查看 CI:       gh run watch"
echo "清理临时文件:  rm scripts/test_*.sh scripts/comprehensive_*"
echo ""