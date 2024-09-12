
# Generate processed datafiles (each file represents an image) of a range
# of delta C9 values. Each delta C9 value has multiple trials (datafiles).

# Assumes basf2 environment is already setup (including my library).
# Only works for ell = mu right now.


n_events=24000

output_dir="../datafiles"

values_dc9_real=("0.0" "-1.93" "-1.78" "-1.63" "-1.48" "-1.34" "-1.19" "-1.04" "-0.89" "-0.75" \
       "-0.6" "-0.45" "-0.3" "-0.15" "-0.01" "0.14" "0.29" "0.44" "0.58" \
       "0.73" "0.88" "1.03" \
       "-2.0" "-1.85" "-1.7" "-1.56" "-1.41" \
       "-1.26" "-1.11" "-0.97" "-0.82" "-0.67" "-0.52" "-0.38" \
       "-0.23" "-0.08" "0.07" "0.21" "0.36" "0.51" "0.66" "0.8" "0.95" "1.1")

trial_begin=36
trial_end=40

for dc9_real in ${values_dc9_real[@]}; do
    for ((trial=trial_begin; trial<=trial_end; trial++)); do
        bsub -q l "bash pipeline.sh mu ${dc9_real} ${trial} ${n_events} ${output_dir}"
        sleep 1
    done
done