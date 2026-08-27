#!/usr/bin/env bash
# 低闪烁资源监控：关键 CPU/GPU 数值按秒刷新，进程排行按较慢间隔刷新。
# 用法：
#   ./monitor_resources.sh          # 指标每 1 秒、进程排行每 5 秒刷新
#   ./monitor_resources.sh 1 10     # 指标每 1 秒、进程排行每 10 秒刷新
#   TOP_N=20 ./monitor_resources.sh # 显示前 20 个进程

set -u

METRIC_INTERVAL="${1:-1}"
PROCESS_INTERVAL="${2:-5}"
TOP_N="${TOP_N:-12}"

for value in "$METRIC_INTERVAL" "$PROCESS_INTERVAL" "$TOP_N"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：刷新间隔和 TOP_N 必须是正整数。" >&2
    exit 2
  fi
done

if [[ ! -t 1 ]]; then
  echo "此脚本需要在交互式终端中运行。" >&2
  exit 1
fi

have_command() { command -v "$1" >/dev/null 2>&1; }
move_to() { printf '\033[%d;1H' "$1"; }
print_line() { printf '\033[2K%s\n' "$1"; }
trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

hardware_logical="?"
physical_cores="?"
if have_command lscpu; then
  hardware_logical=$(lscpu -p=CPU 2>/dev/null | awk -F, '!/^#/ {n++} END {print n+0}')
  physical_cores=$(lscpu -p=SOCKET,CORE 2>/dev/null | awk -F, '!/^#/ {seen[$1 FS $2]=1} END {print length(seen)+0}')
fi
available_logical=$(nproc 2>/dev/null || echo "?")
affinity="?"
if have_command taskset; then
  affinity=$(taskset -pc $$ 2>/dev/null | sed 's/.*: //')
fi

has_gpu=0
gpu_count=0
if have_command nvidia-smi; then
  has_gpu=1
  gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
fi
# 动态区高度：CPU 两行 + GPU 标签/表头两行 + 每张 GPU 一行。
gpu_display_lines=$gpu_count
(( gpu_display_lines < 1 )) && gpu_display_lines=1
dynamic_start=7
process_start=$((dynamic_start + 4 + gpu_display_lines + 1))

render_static_layout() {
  clear
  printf '========== 系统资源监控（Ctrl+C 退出） ==========\n'
  printf '硬件逻辑 CPU（线程）：%s    硬件物理核心：%s\n' "$hardware_logical" "$physical_cores"
  printf '当前环境可用逻辑 CPU：%s    当前 shell CPU 集：%s\n' "$available_logical" "$affinity"
  printf '说明：进程总线程数包含等待的 Python/PyTorch/CUDA 线程；%%CPU 才是实际占用的 CPU 核等价值。\n\n'
  printf '[实时关键指标：每 %s 秒刷新；不刷新进程列表]\n' "$METRIC_INTERVAL"
}

render_dynamic_metrics() {
  local line r b swpd free buff cache si so bi bo intr cs us sy idle wa st
  local mem_line load
  line=$(vmstat "$METRIC_INTERVAL" 2 | tail -n 1)
  read -r r b swpd free buff cache si so bi bo intr cs us sy idle wa st <<< "$line"
  load=$(cut -d' ' -f1-3 /proc/loadavg)
  mem_line=$(free -h | awk '/^Mem:/ {printf "RAM 已用 %s / 总计 %s，空闲 %s，可用 %s", $3, $2, $4, $7}')

  move_to "$dynamic_start"
  print_line "CPU：用户态 ${us}% | 系统态 ${sy}% | 空闲 ${idle}% | I/O 等待 ${wa}% | 可运行任务 ${r} | Load ${load}"
  print_line "${mem_line} | 当前系统进程线程数：$(ps -eLf | awk 'NR > 1 {n++} END {print n+0}')"

  if (( has_gpu == 0 )); then
    print_line "GPU：未找到 nvidia-smi。"
    print_line ""
    return
  fi

  print_line "GPU：${gpu_count} 张（SM(%) 是 GPU 实际计算利用率；已用/总显存是容量占用）"
  print_line "GPU  名称               SM(%)  显存控制器  已用/总显存(MiB)       功耗(W)  温度(°C)"

  local output index name sm memory_util memory_used memory_total power temperature
  output=$(nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -z "${output//[[:space:]]/}" ]]; then
    print_line "无法读取 GPU 数据。"
    return
  fi

  while IFS=',' read -r index name sm memory_util memory_used memory_total power temperature; do
    index=$(trim "$index"); name=$(trim "$name"); sm=$(trim "$sm"); memory_util=$(trim "$memory_util")
    memory_used=$(trim "$memory_used"); memory_total=$(trim "$memory_total"); power=$(trim "$power"); temperature=$(trim "$temperature")
    printf '\033[2K%-4s %-18s %5s %10s %9s / %-9s %8s %9s\n' \
      "$index" "$name" "$sm" "$memory_util" "$memory_used" "$memory_total" "$power" "$temperature"
  done <<< "$output"
}

render_processes() {
  local now cpu_processes gpu_processes count=0
  now=$(date '+%F %T')

  move_to "$process_start"
  print_line "[进程排行：每 ${PROCESS_INTERVAL} 秒刷新；上次刷新 ${now}]"
  print_line "CPU 占用最高前 ${TOP_N} 个："
  print_line "USER             PID   PPID  THR   %CPU   %MEM  RSS(MiB)  ELAPSED  COMMAND"

  cpu_processes=$(ps -eo user=,pid=,ppid=,nlwp=,pcpu=,pmem=,rss=,etime=,args= --sort=-pcpu | head -n "$TOP_N")
  while IFS= read -r row; do
    [[ -z "$row" ]] && continue
    printf '%s\n' "$row" | awk '{
      user=$1; pid=$2; ppid=$3; thr=$4; cpu=$5; mem=$6; rss=$7/1024; elapsed=$8;
      $1=$2=$3=$4=$5=$6=$7=$8=""; sub(/^[[:space:]]+/, ""); cmd=$0;
      if (length(cmd) > 58) cmd=substr(cmd, 1, 55) "...";
      printf "\033[2K%-12s %6s %6s %4s %6s %6s %8.1f %8s  %s\n", user, pid, ppid, thr, cpu, mem, rss, elapsed, cmd;
    }'
    ((count += 1))
  done <<< "$cpu_processes"
  while (( count < TOP_N )); do print_line ""; ((count += 1)); done

  print_line ""
  if (( has_gpu == 0 )); then
    print_line "GPU 进程：未找到 nvidia-smi。"
    return
  fi

  print_line "GPU 显存占用进程（按显存从高到低）："
  print_line "PID        PROCESS                              GPU_MEMORY(MiB)"
  gpu_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ -z "${gpu_processes//[[:space:]]/}" ]]; then
    print_line "无 GPU 计算进程。"
  else
    printf '%s\n' "$gpu_processes" | sort -t',' -k3,3nr | while IFS=',' read -r pid process memory; do
      printf '\033[2K%-10s %-36s %s\n' "$(trim "$pid")" "$(trim "$process")" "$(trim "$memory")"
    done
  fi
}

trap 'printf "\n"; exit 0' INT TERM
render_static_layout
last_process_refresh=0

while true; do
  render_dynamic_metrics
  now_epoch=$(date +%s)
  if (( now_epoch - last_process_refresh >= PROCESS_INTERVAL )); then
    render_processes
    last_process_refresh=$now_epoch
  fi
done
