#!/usr/bin/env -S --default-signal=INT bash
# Single-GPU crash/hang auto-resume wrapper (adapted from llm-project Train/run.sh).
# Usage: Train/run1.sh <gpu 0|1> --config <toml>   |   Train/run1.sh <gpu> --resume model/<run dir>
# Logs to logs/gpu<gpu>.log (or $RUN1_LOG if set). Stop: kill -INT <pid of this script or of queue.sh>  (python saves latest.pt)
# Exit codes: 0 = finished, 130 = stopped by signal, 1 = gave up.
# The body is a function so that editing this file while it runs cannot corrupt a running wrapper.
main() {
    cd "$(dirname "$0")/.."
    G=$1
    shift
    LOG=${RUN1_LOG:-logs/gpu$G.log}
    mkdir -p logs
    exec >> "$LOG" 2>&1
    export HIPBLASLT_TENSILE_LIBPATH=$PWD/hipblaslt-lib/gfx1201
    export HIP_VISIBLE_DEVICES=$G
    echo $$ > logs/run1_gpu$G.pid
    stop=0
    child=
    trap 'stop=1; [ -n "$child" ] && kill -INT "$child" 2>/dev/null' INT TERM
    (
        while kill -0 $$ 2>/dev/null; do
            d=/sys/class/drm/card$G/device
            h=$(echo $d/hwmon/hwmon*)
            echo "$(date +%F_%T) | card$G $(cat $h/temp1_input $h/temp2_input $h/temp3_input $h/power1_average $h/fan1_input $d/gpu_busy_percent 2>/dev/null | tr '\n' ' ')"
            if pgrep -P $$ -f Train/train.py > /dev/null && [ $(($(date +%s) - $(stat -c %Y "$LOG"))) -gt 180 ]; then
                Train/hang1.sh $$ $G >> "$LOG" 2>&1
            fi
            sleep 30
        done
    ) >> logs/gpu${G}_stats.log 2>&1 &
    args=("$@")
    from=$(($(wc -l < "$LOG") + 1))
    run=
    [ "${args[0]}" = --resume ] && run=${args[1]}
    before=0
    fails=0
    while [ "$stop" = 0 ]; do
        echo "[run1.sh] $(date +%F_%T) start gpu$G ${args[*]}"
        started=$(date +%s)
        .venv/bin/python Train/train.py "${args[@]}" &
        child=$!
        wait "$child"
        rc=$?
        while kill -0 "$child" 2>/dev/null; do
            wait "$child"
            rc=$?
        done
        child=
        echo "[run1.sh] $(date +%F_%T) exit rc=$rc stop=$stop"
        [ "$stop" = 1 ] && exit 130
        [ "$rc" = 0 ] && exit 0
        [ -n "$run" ] && [ -f "$run/latest.pt" ] || run=$(tail -n +$from "$LOG" | grep '^run ' | tail -n 1 | cut -d' ' -f2)
        after=0
        if [ -f "$run/latest.pt" ]; then
            after=$(stat -c %Y "$run/latest.pt")
            args=(--resume "$run")
        fi
        if [ "$after" = "$before" ]; then
            fails=$((fails + 1))
        else
            fails=0
        fi
        before=$after
        echo "[run1.sh] $(date +%F_%T) crash rc=$rc no_progress=$fails next ${args[*]}"
        if [ "$fails" -ge 3 ]; then
            echo "[run1.sh] $(date +%F_%T) GIVE UP"
            exit 1
        fi
        sleep 30
        if journalctl -k --since "@$started" --no-pager 2>/dev/null | grep -qE 'GPU reset end with ret = -|resume of IP block .* failed|device lost from bus|PSP resume failed'; then
            echo "[run1.sh] $(date +%F_%T) GPU reset failed, reboot needed, GIVE UP"
            exit 1
        fi
    done
    exit 130
}
main "$@"
exit
