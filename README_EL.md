# Semantic Segmentation for Road and Dynamic Object Understanding in Autonomous Driving

## Ελληνική αναλυτική τεκμηρίωση

Το project μελετά το πρόβλημα του semantic segmentation σε αστικές σκηνές οδήγησης. Ο στόχος είναι η ταξινόμηση κάθε pixel μίας εικόνας σε σημασιολογική κατηγορία, ώστε ένα σύστημα αυτόνομης οδήγησης να μπορεί να κατανοεί τον δρόμο, το πεζοδρόμιο, τα οχήματα και τους πεζούς.

Σε αντίθεση με το object detection, το semantic segmentation δεν επιστρέφει μόνο bounding boxes. Παράγει πλήρη μάσκα ανά pixel, κάτι που είναι σημαντικό για εφαρμογές όπου απαιτείται ακριβής χωρική κατανόηση του περιβάλλοντος.

## Στόχος της εργασίας

Η εργασία συγκρίνει διαφορετικές αρχιτεκτονικές segmentation και διαφορετικές διαμορφώσεις του label space. Το κεντρικό ερευνητικό ερώτημα είναι πώς επηρεάζουν την απόδοση:

- η επιλογή μοντέλου,
- η μείωση των αρχικών CamVid κλάσεων,
- η χρήση weighted loss για class imbalance,
- η συμπεριφορά των μοντέλων σε static και dynamic αντικείμενα.

## Dataset

Τα πειράματα βασίζονται στο CamVid dataset, το οποίο χρησιμοποιείται συχνά για road-scene semantic segmentation.

Η αναμενόμενη δομή είναι:

```text
data/CamVid/
- train/
- train_labels/
- val/
- val_labels/
- test/
- test_labels/
```

Τα δεδομένα δεν ανεβαίνουν στο GitHub, επειδή είναι μεγάλα και πρέπει να αποθηκεύονται τοπικά.

## Μείωση κλάσεων

Η αρχική μορφή του CamVid περιέχει 12 semantic classes. Στην εργασία έγινε mapping σε 5 πιο πρακτικές και safety-critical κατηγορίες:

| Κλάση | Περιγραφή |
| --- | --- |
| background | μη κρίσιμες ή συγχωνευμένες περιοχές |
| road | οδηγούμενη επιφάνεια δρόμου |
| sidewalk | πεζοδρόμιο |
| vehicle | οχήματα |
| pedestrian | πεζοί |

Η μείωση κλάσεων βοηθά το μοντέλο να επικεντρωθεί στις πιο σημαντικές κατηγορίες για αυτόνομη οδήγηση.

## Μοντέλα που χρησιμοποιήθηκαν

### U-Net με ResNet34 encoder

Το U-Net είναι encoder-decoder αρχιτεκτονική. Ο encoder εξάγει χαρακτηριστικά από την εικόνα και ο decoder ανακατασκευάζει τη segmentation mask. Ο ResNet34 encoder επιτρέπει αξιοποίηση ισχυρών convolutional features.

### DeepLabV3+

Το DeepLabV3+ είναι ισχυρή semantic segmentation αρχιτεκτονική που μπορεί να αξιοποιήσει πληροφορία σε διαφορετικές κλίμακες. Είναι χρήσιμο σε σκηνές όπου αντικείμενα εμφανίζονται με διαφορετικά μεγέθη.

### SegFormer

Το SegFormer είναι transformer-based μοντέλο. Στο συγκεκριμένο πείραμα δεν ξεπέρασε τα CNN-based μοντέλα, πιθανώς επειδή χρειάζεται περισσότερο tuning, περισσότερα δεδομένα ή μεγαλύτερη ανάλυση.

## Προεπεξεργασία

Η βασική διαδικασία περιλαμβάνει:

1. φόρτωση εικόνας RGB,
2. resize σε 256 x 256,
3. μετατροπή σε tensor,
4. φόρτωση RGB mask,
5. resize με nearest-neighbor,
6. μετατροπή RGB χρωμάτων σε class ids,
7. εφαρμογή mapping από 12 σε 5 κλάσεις.

Το nearest-neighbor resize στις μάσκες είναι απαραίτητο για να μη δημιουργούνται ενδιάμεσα χρώματα που δεν αντιστοιχούν σε πραγματικές κλάσεις.

## Εκπαίδευση

Βασικές ρυθμίσεις:

| Παράμετρος | Τιμή |
| --- | --- |
| Input size | 256 x 256 |
| Batch size | 4 |
| Epochs | 10 |
| CNN optimizer | Adam |
| Transformer optimizer | AdamW |
| CNN learning rate | 1e-3 |
| Transformer learning rate | 5e-5 |
| Loss | Cross-Entropy και Weighted Cross-Entropy |
| Metric | Mean IoU και class-wise IoU |

## Αποτελέσματα

| Model | Label setup | Loss | mIoU |
| --- | --- | --- | ---: |
| U-Net | 12 classes | Cross-Entropy | 0.4797 |
| U-Net | 5 classes | Cross-Entropy | 0.7240 |
| DeepLabV3+ | 5 classes | Cross-Entropy | 0.7132 |
| U-Net | 5 classes | Weighted Cross-Entropy | 0.6823 |
| SegFormer | 5 classes | Cross-Entropy | 0.6135 |

Το καλύτερο αποτέλεσμα πέτυχε το U-Net 5 κλάσεων με Cross-Entropy loss.

## Class-wise IoU καλύτερου μοντέλου

| Class | IoU |
| --- | ---: |
| background | 0.9646 |
| road | 0.9591 |
| sidewalk | 0.8424 |
| vehicle | 0.5944 |
| pedestrian | 0.2596 |

Η κλάση pedestrian είναι η δυσκολότερη, επειδή οι πεζοί έχουν μικρό μέγεθος, διαφορετικές στάσεις και εμφανίζονται λιγότερο συχνά.

## Πώς εκτελείται

```bash
git clone https://github.com/PanagiotaGr/Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving.git
cd Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving
pip install -r requirements.txt
python train_camvid_reduced.py
python train_deeplab_camvid_reduced.py
python train_camvid_weighted.py
```

## Τι κάναμε αναλυτικά

Στο repository προστέθηκε ελληνική τεκμηρίωση για να παρουσιάζεται το project με τρόπο κατάλληλο για πανεπιστημιακή εργασία.

Συγκεκριμένα:

- αναλύθηκε ο σκοπός του semantic segmentation στην αυτόνομη οδήγηση,
- περιγράφηκε το CamVid dataset,
- τεκμηριώθηκε η μείωση των 12 κλάσεων σε 5 safety-critical κατηγορίες,
- εξηγήθηκαν τα μοντέλα U-Net, DeepLabV3+ και SegFormer,
- καταγράφηκαν οι βασικές ρυθμίσεις εκπαίδευσης,
- παρουσιάστηκαν τα αποτελέσματα mIoU και class-wise IoU,
- προστέθηκαν οδηγίες εγκατάστασης και εκτέλεσης,
- προστέθηκε αναλυτικό ελληνικό project report στο `docs/PROJECT_REPORT_EL.md`,
- προστέθηκε ελληνικός οδηγός εγκατάστασης στο `docs/INSTALLATION_EL.md`,
- δημιουργήθηκε και PDF report για χρήση σε παρουσίαση ή παράδοση.

## Συμπέρασμα

Το project δείχνει ότι η επιλογή label space είναι εξίσου σημαντική με την επιλογή μοντέλου. Η μείωση σε 5 κλάσεις βελτίωσε σημαντικά το mIoU και έκανε το πρόβλημα πιο σχετικό με την αυτόνομη οδήγηση. Το U-Net με ResNet34 encoder αποτέλεσε την καλύτερη λύση για το συγκεκριμένο experimental setup.
