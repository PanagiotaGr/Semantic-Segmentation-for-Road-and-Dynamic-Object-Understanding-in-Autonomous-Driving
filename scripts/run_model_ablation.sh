# Run model ablation experiments for all supported architectures.
# This script assumes that scripts/train_model_template.py has been connected
# to the project-specific DataLoader objects before full training.

python --version

for model in unet deeplabv3plus deeplabv3 pspnet fpn pan linknet manet
do
  echo "Running architecture ablation: $model"
  python scripts/train_model_template.py \
    --architecture "$model" \
    --encoder resnet34 \
    --loss ce \
    --scheduler plateau \
    --experiment-name "${model}_ce_ablation"
done
