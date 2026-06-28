# Run the main semantic segmentation experiments from the repository root.

python --version
pip install -r requirements.txt

for script in train_*.py
do
  if test -f "$script"
  then
    echo "Running $script"
    python "$script"
  fi
done

for script in predict_*.py
do
  if test -f "$script"
  then
    echo "Running $script"
    python "$script"
  fi
done

echo "Experiments completed."
