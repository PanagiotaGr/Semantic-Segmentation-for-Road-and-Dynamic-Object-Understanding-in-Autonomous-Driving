# Resolution ablation runner.
# Adapt script names if your training files use different names.

python --version

for size in 128 256 384 512
do
  echo "Running resolution ablation: ${size}x${size}"
  if test -f train_unet.py
  then
    python train_unet.py --image-size "$size"
  else
    echo "train_unet.py not found. Use this file as an experiment template."
  fi
done
