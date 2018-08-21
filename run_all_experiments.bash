#!/usr/bin/env bash

# Increase this to 50000 for the experiments in the paper
# Recommended to try with a low number first to see if all the experiments work
N_TRAINING_EXAMPLES=400
N_VALIDATION_EXAMPLES=400  # Increase to 10000 for full experiment
# If you hit "Matrix is singular!" errors, reducing N_MAX (and thus the memory the GPU uses) might help
N_MAX=100
N_GPUS=1  # Can be larger too. 0 GPUs isn't supported yet.
WORKDIR="/tmp/conv_kernels"
DFS="dataframes"

mkdir -p "$WORKDIR" "$WORKDIR/$DFS"

# We reported 27 in the paper because several of these fail due to the VALID
# padding eventually making the output image of negative size.
for SEED in $(seq 1 60) 1234; do
	if [ "$SEED" -lt 36 ]; then
		ALLOW_SKIP="False"
	else
		ALLOW_SKIP="True"
	fi

	python3 save_kernels.py --seed=$SEED --n_gpus=$N_GPUS --n_max=$N_MAX \
			--N_train=$N_TRAINING_EXAMPLES --path="$WORKDIR" \
			--N_vali=$N_VALIDATION_EXAMPLES  --allow_skip=$ALLOW_SKIP

	python3 classify_gp.py --seed=$SEED --csv_dir="$DFS" --path="$WORKDIR" \
			--N_train=$N_TRAINING_EXAMPLES --N_vali=$N_VALIDATION_EXAMPLES
done

# If you hit "Matrix is singular!" errors, reducing N_MAX (and thus the memory the GPU uses) might help
N_MAX=20
python3 save_kernels_resnet.py --seed=0 --n_gpus=$N_GPUS --n_max=$N_MAX \
	 --N_train=$N_TRAINING_EXAMPLES --path="$WORKDIR" \
	 --N_vali=$N_VALIDATION_EXAMPLES

python3 classify_gp.py --seed=0 --csv_dir="$DFS" --path="$WORKDIR" \
	 --N_train=$N_TRAINING_EXAMPLES --N_vali=$N_VALIDATION_EXAMPLES
