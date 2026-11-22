#!/bin/bash
# Stop every running drive or manual-control program and turn the motor off.
# Used by `aicar stop` and by Ctrl+C during autonomous driving.
cd ~/aicar || exit 1

find_pids() {
    # Real python processes only. `pgrep -f` also matches the shell that launched this script.
    ps -eo pid=,comm=,args= |
        awk '$2 ~ /^python/ && /(lane_follow|drive_keys|tune_center|tune_white|tune_light|tune_person|tune_landmark|camera_view|calibrate|gap_probe|car)\.py/ {print $1}'
}

pids=$(find_pids)
if [ -z "$pids" ]; then
    echo "No drive running"
else
    kill -TERM $pids 2>/dev/null
    for _ in $(seq 25); do      # Wait up to 5s for a clean exit (motor off, recording saved).
        [ -z "$(find_pids)" ] && break
        sleep 0.2
    done
    left=$(find_pids)
    if [ -n "$left" ]; then
        kill -KILL $left 2>/dev/null
        sleep 0.3
        echo "No response: killed"
    fi
    echo "Drive program stopped"
fi

# If the process was killed, grab the motor pins again and stop them.
# Retried: another stop (Ctrl+C on the Mac and aicar stop at once) can hold the pins for a moment.
for _ in 1 2 3 4 5; do
    if venv/bin/python car.py center >/dev/null 2>&1; then
        echo "Motor stopped, steering centered"
        exit 0
    fi
    sleep 0.5
done
echo "Could not confirm the motor stopped. Switch the car off by hand."
exit 1
