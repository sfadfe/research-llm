#!/usr/bin/env bash
# Hang capture for a single-GPU run (adapted from llm-project Train/hang.sh). Usage: Train/hang1.sh <run1.sh pid> <gpu>
sup=$1
G=$2
w=$(pgrep -P "$sup" -f Train/train.py)
[ -n "$w" ] || exit 0
dir=hang/$(date +%F_%H%M%S)_gpu$G
mkdir -p "$dir"
echo "[run1.sh] $(date +%F_%T) hang log silent capture $dir kill $w"
kill -USR1 "$w"
{
    echo "=== worker $w"
    for t in /proc/$w/task/*; do
        echo "$(basename "$t") $(cat "$t/comm") $(awk '{print $3}' "$t/stat") $(cat "$t/wchan")"
    done
    echo "evicted_ms $(cat /sys/class/kfd/kfd/proc/$w/stats_*/evicted_ms | tr '\n' ' ')"
} >> "$dir/proc.txt" 2>&1
d=/sys/class/drm/card$G/device
{
    echo "=== card$G busy $(cat $d/gpu_busy_percent) mem_busy $(cat $d/mem_busy_percent)"
    cat $d/pp_dpm_sclk $d/pp_dpm_mclk
} >> "$dir/proc.txt" 2>&1
pci=$(basename "$(readlink -f $d)")
for f in kfd/mqds kfd/hqds dri/$pci/amdgpu_fence_info; do
    timeout -k 2 15 sudo -n cat "/sys/kernel/debug/$f" > "$dir/$(echo "$f" | tr '/:' '__')" 2>&1 &
done
sleep 18
timeout -k 2 15 sudo -n python3 Train/aql.py "$dir/kfd_hqds" $w > "$dir/aql.txt" 2>&1
kill -9 $w
