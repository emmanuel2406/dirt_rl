#!/bin/bash

# Run survival comparisons with clear delineations between experiments.
# For each experiment/growth/seed, print the first zero index (or length if none).

set -u

seeds=(1234 1235 1236 1237 1238)
growths=(1 4 16)

run_group() {
    local label="$1"
    local template="$2"

    echo "===== ${label} ====="
    for growth in "${growths[@]}"; do
        echo "Growth: $growth"
        local values=()
        for seed in "${seeds[@]}"; do
            json_path=$(printf "$template" "$growth" "$seed")

            if [ ! -f "$json_path" ]; then
                echo "Missing file: $json_path" >&2
                continue
            fi

            result=$(./scripts/average_json_field.sh "$json_path" active_players_per_step)
            echo "$result"
            values+=("$result")
        done

        # Calculate mean and sample standard deviation
        if [ ${#values[@]} -gt 0 ]; then
            stats=$(printf '%s\n' "${values[@]}" | awk '
            {
                values[NR] = $1
                sum += $1
            }
            END {
                n = NR
                if (n == 0) {
                    print "Error: No values" > "/dev/stderr"
                    exit 1
                }
                mean = sum / n
                
                if (n == 1) {
                    stddev = 0
                } else {
                    sum_sq_diff = 0
                    for (i = 1; i <= n; i++) {
                        diff = values[i] - mean
                        sum_sq_diff += diff * diff
                    }
                    stddev = sqrt(sum_sq_diff / (n - 1))
                }
                
                printf "Mean: %.2f, Std Dev: %.2f\n", mean, stddev
            }')
            echo "$stats"
        fi
    done
    echo ""
}

run_group "RL experiments (linear_model_rl)" "rl_experiments/linear_model_rl/mean_food_growth_%s_rl_seed_%s/players_time_series.json"
run_group "Baseline (linear_model)" "linear_model/mean_food_growth_%s_seed_%s/players_time_series.json"








