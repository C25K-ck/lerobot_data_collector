
#!/bin/bash

# 查找包含"act_get", "robot_kinemic"关键字的进程
processes=$(ps aux | grep -E 'save_act|robot_con' | grep -v grep)

# 检查是否找到匹配的进程
if [ -z "$processes" ]; then
    echo "没有找到包含'act_get", "robot_kinemic'关键字的进程。"
else
    echo "找到以下进程："
    echo "$processes"

    # 询问用户是否要杀死这些进程
    read -p "是否要杀死这些进程？(y/n): " answer

    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        # 杀死每个匹配的进程
        while read -r line; do
            pid=$(echo "$line" | awk '{print $2}')
            kill -9 "$pid"
            echo "已杀死进程：$pid"
        done <<< "$processes"
    else
        echo "未杀死任何进程。"
    fi
fi
