eval "$(conda shell.bash hook)"

case "$2" in
    "manual_exit")
        if [ -f "log_$1.status" ]; then
            # 从status文件中提取PID
            pid=$(grep "Process" "log_$1.status" | cut -d' ' -f2)
            if [ ! -z "$pid" ]; then
                kill $pid
                echo "Process $pid manually terminated" > log_$1.status
                exit 0
            else
                echo "No valid PID found in status file" > log_$1.status
                exit 1
            fi
        else
            echo "Status file not found"
            exit 1
        fi
        ;;
    *)
        # 正常启动流程
        conda activate cs0 && python tune.py > log_$1 2>&1 & 
        pid=$!

        # 监控进程状态
        INTERVAL=10
        (while true; do
            if fuser -v /dev/nvidia* 2>/dev/null | grep -q "$pid"; then
                echo "Process $pid: Running (GPU)" > log_$1.status
                fuser -v /dev/nvidia* 2>/dev/null | grep "$pid" >> log_$1.status
            else
                echo "Process $pid: Finished" > log_$1.status
                break
            fi
            sleep $INTERVAL
        done) &
        ;;
esac