#!/usr/bin/env -S --default-signal=INT bash
# Run configs one after another on one GPU; after each finished run, call Eval/evaluate.py (if it exists).
# Usage: nohup Train/queue.sh <gpu> configs/X.toml configs/Y.toml ... > /dev/null 2>&1 &
# Per config (run dir = latest model/<run_name>_<timestamp>): eval_done marker -> skip; final step_*.pt -> eval only;
# latest.pt -> resume; else start new. Re-running the same command therefore continues an interrupted queue.
# Stop: kill -INT $(cat logs/queue_gpu<gpu>.pid)   (the current run saves latest.pt, the queue exits)
# Logs: logs/gpu<gpu>.log (training + queue), logs/eval_gpu<gpu>.log (Eval/evaluate.py output).
# The body is a function so that editing this file while it runs cannot corrupt a running queue.
main() {
    cd "$(dirname "$0")/.."
    G=$1
    shift
    LOG=logs/gpu$G.log
    mkdir -p logs
    exec >> "$LOG" 2>&1
    echo $$ > logs/queue_gpu$G.pid
    export HIPBLASLT_TENSILE_LIBPATH=$PWD/hipblaslt-lib/gfx1201
    stop=0
    child=
    trap 'stop=1; [ -n "$child" ] && kill -INT "$child" 2>/dev/null' INT TERM
    for cfg in "$@"; do
        [ "$stop" = 1 ] && break
        name=$(grep -m1 '^run_name' "$cfg" | cut -d'"' -f2)
        dir=$(ls -d model/${name}_2* 2>/dev/null | tail -n 1)
        rc=0
        if [ -n "$dir" ] && [ -f "$dir/eval_done" ]; then
            echo "[queue] $(date +%F_%T) skip $name (done: $dir)"
            continue
        elif [ -n "$dir" ] && ls "$dir"/step_*.pt > /dev/null 2>&1; then
            echo "[queue] $(date +%F_%T) $name trained, eval only"
        elif [ -n "$dir" ] && [ -f "$dir/latest.pt" ]; then
            echo "[queue] $(date +%F_%T) resume $dir"
            runit Train/run1.sh "$G" --resume "$dir"
            rc=$?
        else
            echo "[queue] $(date +%F_%T) start $cfg"
            runit Train/run1.sh "$G" --config "$cfg"
            rc=$?
            dir=$(ls -d model/${name}_2* 2>/dev/null | tail -n 1)
        fi
        [ "$stop" = 1 ] && break
        if ! ls "$dir"/step_*.pt > /dev/null 2>&1; then
            echo "[queue] $(date +%F_%T) $name not finished (rc=$rc), stop queue"
            exit 1
        fi
        if [ -f Eval/evaluate.py ]; then
            echo "[queue] $(date +%F_%T) eval $dir"
            echo "[queue] $(date +%F_%T) eval $dir" >> logs/eval_gpu$G.log
            runit .venv/bin/python Eval/evaluate.py --run "$dir" --device "cuda:$G" >> logs/eval_gpu$G.log 2>&1
            rc=$?
            echo "[queue] $(date +%F_%T) eval rc=$rc $dir"
        else
            echo "[queue] $(date +%F_%T) no Eval/evaluate.py, eval skipped for $dir"
        fi
        [ "$stop" = 1 ] && break
        echo "$(date +%F_%T) rc=$rc" > "$dir/eval_done"
    done
    echo "[queue] $(date +%F_%T) END stop=$stop"
}
runit() {
    "$@" &
    child=$!
    wait "$child"
    local r=$?
    while kill -0 "$child" 2>/dev/null; do
        wait "$child"
        r=$?
    done
    child=
    return $r
}
main "$@"
exit
