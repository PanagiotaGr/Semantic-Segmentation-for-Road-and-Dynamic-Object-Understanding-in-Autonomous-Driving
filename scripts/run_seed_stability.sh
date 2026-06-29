# Random seed stability runner.
# Adapt script names if your training files use different names.

python --version

for seed in 0 1 2 42 123
do
  echo "Running seed stability experiment with seed: $seed"
  if test -f train_unet.py
  then
    python train_unet.py --seed "$seed"
  else
    echo "train_unet.py not found. Use this file as an experiment template."
  fi
done
