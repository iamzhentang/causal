eval "$(conda shell.bash hook)"

conda activate cs0 && python tune.py $1> log_$1 2>&1 &