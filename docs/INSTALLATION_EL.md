# Οδηγίες Εγκατάστασης και Εκτέλεσης

## Clone repository

```bash
git clone https://github.com/PanagiotaGr/Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving.git
cd Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving
```

## Virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Εγκατάσταση βιβλιοθηκών

```bash
pip install -r requirements.txt
```

Το project χρησιμοποιεί PyTorch, torchvision, segmentation-models-pytorch, transformers, OpenCV, Pillow, Albumentations, NumPy, pandas, scikit-learn, matplotlib και tqdm.

## Dataset

Το CamVid dataset πρέπει να τοποθετηθεί τοπικά στον φάκελο:

```text
data/CamVid/
- train/
- train_labels/
- val/
- val_labels/
- test/
- test_labels/
```

Οι φάκελοι δεδομένων αγνοούνται από το `.gitignore`, ώστε να μη γίνονται push μεγάλα datasets στο GitHub.

## Εκπαίδευση μοντέλων

```bash
python train_camvid_reduced.py
python train_deeplab_camvid_reduced.py
python train_camvid_weighted.py
```

Τα checkpoints αποθηκεύονται στον φάκελο `models/`, ο οποίος επίσης αγνοείται από το GitHub.
