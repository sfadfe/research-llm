#!/usr/bin/env -S --default-signal=INT bash
# Overnight runs: two independent per-GPU queues, started together. Each queue runs its configs one after another
# and moves on whatever the exit code of the previous run (crash, give-up, NaN: the next run still starts).
#   GPU 0: plain-26m             -> okkie-transformer-60m
#   GPU 1: plain-60m             -> okkie-transformer-26m
# Each run goes through Train/run1.sh (HIP_VISIBLE_DEVICES=<gpu>, as the 117M runs; crash auto-resume, hang
# capture) with its own log logs/overnight/<run>.log. Top-level status (start/end, exit code, run dir):
# logs/overnight/status.log.
# Start (from research-llm):  nohup Train/run_overnight.sh > logs/overnight/nohup.log 2>&1 &
# Stop: kill -INT $(cat logs/overnight/overnight.pid)   (running train.py saves latest.pt; no further runs start)
main() {
    cd "$(dirname "$0")/.."
    D=logs/overnight
    mkdir -p "$D"
    ST=$D/status.log
    echo $$ > "$D/overnight.pid"
    say() { echo "$(date '+%F %T') $*" >> "$ST"; }
    say "overnight start pid $$"
    for G in 0 1; do
        p=/sys/class/drm/card$G/device/power
        if [ "$(cat $p/control 2>/dev/null)" != on ]; then
            if echo on | sudo -n tee $p/control > /dev/null 2>&1; then
                say "card$G power/control -> on (was runtime_status $(cat $p/runtime_status 2>/dev/null))"
            else
                say "card$G WARNING: could not set power/control=on (sudo -n failed); continuing"
            fi
        fi
    done
    stop=0
    # background subshells ignore SIGINT (no job control): queues are stopped with TERM
    trap 'stop=1; kill -TERM $q0 $q1 2>/dev/null' INT TERM
    queue 0 configs/plain-26m.toml configs/okkie-transformer-60m.toml &
    q0=$!
    queue 1 configs/plain-60m.toml configs/okkie-transformer-26m.toml &
    q1=$!
    for q in $q0 $q1; do
        while kill -0 $q 2>/dev/null; do
            wait $q
        done
    done
    say "overnight end stop=$stop"
}
queue() {
    G=$1
    shift
    qstop=0
    child=
    trap 'qstop=1; [ -n "$child" ] && kill -INT "$child" 2>/dev/null' TERM
    for cfg in "$@"; do
        [ "$qstop" = 1 ] && break
        name=$(grep -m1 '^run_name' "$cfg" | cut -d'"' -f2)
        log=$D/$name.log
        say "gpu$G start $name ($cfg) log $log"
        RUN1_LOG=$log Train/run1.sh "$G" --config "$cfg" &
        child=$!
        wait "$child"
        rc=$?
        while kill -0 "$child" 2>/dev/null; do
            wait "$child"
            rc=$?
        done
        child=
        dir=$(grep '^run ' "$log" 2>/dev/null | tail -n 1 | cut -d' ' -f2)
        last=$(grep -o '"step": [0-9]*' "$dir/log.jsonl" 2>/dev/null | tail -n 1 | cut -d' ' -f2)
        say "gpu$G end $name rc=$rc run_dir=${dir:-?} last_logged_step=${last:-?}"
    done
    say "gpu$G queue done stop=$qstop"
}
main "$@"
exit
