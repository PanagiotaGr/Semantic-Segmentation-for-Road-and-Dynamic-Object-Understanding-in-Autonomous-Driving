# Loss ablation runner.
# Adapt script names if your training files use different names.

python --version

for loss_name in ce weighted_ce focal dice ce_dice
do
  echo "Running loss ablation: $loss_name"
  if test -f train_unet.py
  then
    python train_unet.py --loss "$loss_name"
  else
    echo "train_unet.py not found. Use this file as an experiment template."
  fi
done
