# Αναλυτική Τεκμηρίωση Project

**Project:** Semantic Segmentation for Road and Dynamic Object Understanding in Autonomous Driving  
**Repository:** PanagiotaGr/Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving  
**Γλώσσα:** Ελληνικά

## 1. Σκοπός της εργασίας

Η εργασία υλοποιεί και αξιολογεί μοντέλα semantic segmentation για αστικές σκηνές οδήγησης. Στόχος είναι η ανάθεση μίας σημασιολογικής κατηγορίας σε κάθε pixel της εικόνας, ώστε το σύστημα αντίληψης ενός αυτόνομου οχήματος να μπορεί να αναγνωρίζει δρόμο, πεζοδρόμιο, οχήματα και πεζούς.

Το project δεν αντιμετωπίζει απλώς το πρόβλημα ως γενική ταξινόμηση εικόνας. Πρόκειται για πυκνή πρόβλεψη ανά pixel, επομένως η χωρική ακρίβεια είναι βασική. Για τον λόγο αυτό χρησιμοποιούνται segmentation architectures όπως U-Net, DeepLabV3+ και SegFormer.

## 2. Dataset

Τα πειράματα βασίζονται στο CamVid dataset. Η αναμενόμενη τοπική δομή είναι:

```text
data/CamVid/
├── train/
├── train_labels/
├── val/
├── val_labels/
├── test/
└── test_labels/
```

Τα δεδομένα δεν αποθηκεύονται στο GitHub repository, επειδή αποτελούν μεγάλο dataset και πρέπει να διατηρούνται τοπικά.

## 3. Μείωση κλάσεων

Η αρχική μορφή του CamVid περιέχει 12 κατηγορίες. Στο βασικό πείραμα οι κατηγορίες μειώθηκαν σε 5, ώστε το πρόβλημα να εστιάζει σε safety-critical perception:

| Νέα κλάση | Περιγραφή |
| --- | --- |
| background | Μη κρίσιμες ή συγχωνευμένες περιοχές |
| road | Οδηγήσιμη επιφάνεια δρόμου |
| sidewalk | Πεζοδρόμιο / περιοχή πεζών |
| vehicle | Οχήματα |
| pedestrian | Πεζοί |

Η μείωση κλάσεων βελτίωσε σημαντικά το mIoU, επειδή περιορίζει την αβεβαιότητα μεταξύ οπτικά παρόμοιων ή λιγότερο κρίσιμων κατηγοριών.

## 4. Προεπεξεργασία δεδομένων

Κάθε εικόνα ανοίγεται ως RGB, γίνεται resize σε 256 x 256, μετατρέπεται σε tensor και περνάει στο μοντέλο. Κάθε μάσκα ανοίγεται ως RGB, γίνεται resize με nearest-neighbor interpolation, μετατρέπεται από RGB χρώματα σε class ids και εφαρμόζεται το reduced mapping από 12 σε 5 κλάσεις.

Η χρήση nearest-neighbor στις μάσκες είναι απαραίτητη, γιατί διαφορετικά δημιουργούνται ενδιάμεσα χρώματα που δεν αντιστοιχούν σε πραγματικές κλάσεις.

## 5. Μοντέλα που αξιολογήθηκαν

### U-Net με ResNet34 encoder

Το U-Net χρησιμοποιείται ως encoder-decoder αρχιτεκτονική. Ο encoder εξάγει χαρακτηριστικά σε διαφορετικά επίπεδα αφαίρεσης, ενώ ο decoder ανακατασκευάζει την τελική μάσκα. Το ResNet34 encoder αξιοποιεί προεκπαιδευμένα βάρη ImageNet στο βασικό training script.

### DeepLabV3+

Το DeepLabV3+ αξιοποιεί μηχανισμούς πολυκλίμακης πληροφορίας και είναι κατάλληλο για semantic segmentation σε σκηνές όπου τα αντικείμενα εμφανίζονται σε διαφορετικά μεγέθη.

### SegFormer

Το SegFormer είναι transformer-based προσέγγιση. Στο συγκεκριμένο project αξιολογήθηκε συγκριτικά, αλλά δεν ξεπέρασε τα CNN-based μοντέλα, πιθανώς λόγω ανάγκης για περισσότερο tuning, περισσότερα δεδομένα ή υψηλότερη ανάλυση.

## 6. Εκπαίδευση

| Παράμετρος | Τιμή |
| --- | --- |
| Input size | 256 x 256 |
| Batch size | 4 |
| Epochs | 10 |
| Optimizer CNN | Adam |
| Learning rate CNN | 1e-3 |
| Loss | Cross-Entropy / Weighted Cross-Entropy |
| Metric | mIoU και class-wise IoU |

Η εκπαίδευση ακολουθεί το κλασικό σχήμα forward pass, loss calculation, backward pass και optimizer step. Μετά από κάθε epoch γίνεται validation και αποθηκεύεται το καλύτερο μοντέλο όταν το mIoU βελτιώνεται.

## 7. Weighted loss

Για την αντιμετώπιση class imbalance χρησιμοποιήθηκε weighted cross-entropy με μεγαλύτερα βάρη για vehicle και pedestrian:

```python
class_weights = torch.tensor([1.0, 1.0, 1.2, 2.0, 4.0])
```

Η λογική είναι ότι οι μικρές και σπάνιες κλάσεις πρέπει να έχουν μεγαλύτερη επίδραση στο loss. Ωστόσο, το weighted loss δεν βελτίωσε το συνολικό mIoU σε σχέση με το απλό U-Net 5 κλάσεων.

## 8. Αποτελέσματα

| Model | Label setup | Loss | mIoU |
| --- | --- | --- | ---: |
| U-Net | 12 classes | Cross-Entropy | 0.4797 |
| U-Net | 5 classes | Cross-Entropy | **0.7240** |
| DeepLabV3+ | 5 classes | Cross-Entropy | 0.7132 |
| U-Net | 5 classes | Weighted Cross-Entropy | 0.6823 |
| SegFormer | 5 classes | Cross-Entropy | 0.6135 |

### Class-wise IoU για το καλύτερο U-Net μοντέλο

| Class | IoU |
| --- | ---: |
| background | 0.9646 |
| road | 0.9591 |
| sidewalk | 0.8424 |
| vehicle | 0.5944 |
| pedestrian | 0.2596 |

## 9. Ερμηνεία αποτελεσμάτων

Το U-Net 5 κλάσεων πέτυχε την καλύτερη συνολική επίδοση. Η βελτίωση σε σχέση με τις 12 κλάσεις δείχνει ότι η διαμόρφωση του label space είναι κρίσιμη. Για autonomous driving, η πρακτική κατανόηση κρίσιμων περιοχών μπορεί να είναι πιο χρήσιμη από την εξαντλητική ταξινόμηση όλων των λεπτομερειών της σκηνής.

Οι κλάσεις road και background έχουν υψηλό IoU επειδή καταλαμβάνουν μεγάλες και οπτικά σταθερές περιοχές. Αντίθετα, η κλάση pedestrian έχει χαμηλό IoU επειδή οι πεζοί είναι μικρά αντικείμενα, εμφανίζονται σε διαφορετικές στάσεις και συχνά καλύπτονται μερικώς.

## 10. Οδηγίες εκτέλεσης

```bash
git clone https://github.com/PanagiotaGr/Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving.git
cd Semantic-Segmentation-for-Road-and-Dynamic-Object-Understanding-in-Autonomous-Driving
pip install -r requirements.txt
python train_camvid_reduced.py
python train_deeplab_camvid_reduced.py
python train_camvid_weighted.py
```

## 11. Περιορισμοί

- Η ανάλυση 256 x 256 περιορίζει την αναγνώριση μικρών αντικειμένων.
- Η κλάση pedestrian παραμένει δύσκολη λόγω class imbalance.
- Δεν αξιοποιείται χρονική πληροφορία από video.
- Δεν υπάρχει πλήρης benchmark χρόνου inference.
- Το SegFormer δεν φαίνεται να έχει υποστεί εκτεταμένο hyperparameter tuning.

## 12. Μελλοντικές βελτιώσεις

- Δοκιμή Dice Loss, Focal Loss ή συνδυαστικών loss functions.
- Εκπαίδευση σε μεγαλύτερη ανάλυση.
- Ισχυρότερο data augmentation.
- Περισσότερα epochs και systematic hyperparameter tuning.
- Benchmark σε χρόνο inference για real-time εφαρμογές.
- Χρήση temporal consistency σε video sequences.
- Δοκιμή πιο σύγχρονων transformer-based segmentation models.

## 13. Συμπέρασμα

Το project δείχνει ότι το semantic segmentation μπορεί να χρησιμοποιηθεί αποτελεσματικά για road-scene understanding στην αυτόνομη οδήγηση. Η σημαντικότερη παρατήρηση είναι ότι η απλοποίηση του label space σε 5 safety-critical κλάσεις βελτίωσε ουσιαστικά την απόδοση. Το καλύτερο αποτέλεσμα επιτεύχθηκε από U-Net με ResNet34 encoder και απλή Cross-Entropy loss.
