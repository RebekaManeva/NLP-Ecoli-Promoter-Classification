# NLP-Ecoli-Promoter-Classification
Promoter Prediction in E. coli using NLP Approaches
This repository contains code for the paper:
"Exploring NLP Approaches for Classifying Promoter Regions in E. coli DNA Sequences"

Rebeka Maneva, Konstantin Lozankoski  
Faculty of Computer Science and Engineering, Ss. Cyril and Methodius University in Skopje


Overview:
We compare traditional machine learning (k-mer features with XGBoost), neural architectures (CNN, BiLSTM, hybrid) and transformer models (DNABERT) for bacterial promoter prediction in Escherichia coli. CNNs achieve the strongest performance on RegulonDB (F1: 0.823), while BiLSTM and DNABERT show greater robustness on the UCI dataset. Genome-wide scanning reveals that DNABERT requires organism-specific fine-tuning for practical deployment.

Data
\- RegulonDB: Download from \[RegulonDB] (https://regulondb.ccg.unam.mx/datasets)

\- UCI: Download from \[UCI Repository] (https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Promoter+Gene+Sequences))

\- E. coli genome: NCBI U00096.3


Models
\- XGBoost baseline
\- CNN (3 conv layers, kernel size 12)
\- BiLSTM (2 layers, 128 units)
\- CNN+BiLSTM hybrid
\- DNABERT (zhihan1996/DNA\_bert\_6)

Contact
Rebeka Maneva - rebeka.maneva@students.finki.ukim.mk  
Konstantin Lozankoski - konstantin.lozankoski@students.finki.ukim.mk

Citation
If you use this code, please cite:
```bibtex
@inproceedings{maneva2025promoter,
&nbsp; title={Exploring NLP Approaches for Classifying Promoter Regions in E. coli DNA Sequences},
&nbsp; author={Maneva, Rebeka and Lozankoski, Konstantin},
&nbsp; year={2026}
}
