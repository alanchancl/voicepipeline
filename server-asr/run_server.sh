#!/bin/bash

# 使用用户指定的chenl环境路径
PYTHON_PATH="/data/envs/chenl/bin/python"

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 检查Python路径是否存在
if [ -f "$PYTHON_PATH" ]; then
    echo "使用环境: $PYTHON_PATH"
    # 运行main_server.py
    $PYTHON_PATH main_server.py
else
    echo "找不到Python路径: $PYTHON_PATH，尝试使用系统默认Python"
    python main_server.py
fi

echo "服务器已停止运行" 