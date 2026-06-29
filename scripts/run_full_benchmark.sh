# Full benchmark runner for the unified training pipeline.
# Run from repository root after preparing data/CamVid.

python --version

for model in unet deeplabv3plus deeplabv3 pspnet fpn pan linknet manet
do
  echo "Training $model"
  python train.py \
    --model "$model" \
    --encoder resnet34 \
    --loss ce \
    --scheduler plateau \
    --image-size 256 \
    --batch-size 4 \
    --epochs 25 \
    --seed 42 \
    --experiment-name "${model}_resnet34_ce_256_seed42"
done
