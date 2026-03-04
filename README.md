# CRC Gut Microbiome–Metabolome Analysis

<img width="1193" height="667" alt="image" src="https://github.com/user-attachments/assets/a4d14ef4-da9f-4f43-a6b7-0ed70b243200" />

## Overview

This computational biology research project investigates the relationship between the gut microbiome and fecal metabolome across multiple colorectal cancer (CRC) and gastrointestinal disease cohorts. The central objective is to identify which microbial species are the most likely biosynthetic sources of polyamines (putrescine, spermidine, spermine, N-acetylputrescine, N1,N12-diacetylspermine, cadaverine) and related metabolites in the context of colorectal cancer progression.

<img width="994" height="695" alt="image" src="https://github.com/user-attachments/assets/959f8f36-a9ac-46ec-8e6b-0cb6bef699b6" />


## Directory Structure

### Repository Contents

Code/: Analysis pipeline organized in four sequential stages (preprocessing → association mapping → validation → source attribution)
Data/: Raw taxonomic abundance tables (species.tsv), metabolite abundance tables (mtb.tsv), and metadata (metadata.tsv) for 4 cohorts
Results/: Publication-quality figures and summary tables organized by analysis step

```
Code/
├── 01_preprocessing/          (14 scripts) Raw TSV files → Preprocessed Data
├── 02_association_maps/       (8 scripts) Preprocessed Data → Association Maps
├── 03_validated_associations/ (8 scripts) Association Maps → Validated Associations
├── 04_source_attribution/     (9 scripts) Validated Associations → Outputs
├── utils.py
└── README.md

Data/                          (raw TSV files [with each containing species.tsv, mtb.tsv, metadata.tsv, mtb.map.tsv], files >90 MB gzip-compressed)
└──Erawijantari                (Gastric)
└──Yachida                     (Colorectal Carcinoma) 
└──Kim                         (16s metagenomic Colorectal Adenomas)
└──Sinha                       (16s metagenomic CRC)

Results/
├── 01_preprocessing/
├── 02_association_maps/
├── 03_validated_associations/
└── 04_source_attribution/
```

## Dependencies

```
pandas numpy scipy scikit-learn matplotlib seaborn
statsmodels pingouin networkx xgboost shap
```

## Data

Find all data files (Associated to CRC and otherwise) uploaded in the Data/ folder (with special thanks to the Borenstein Lab [https://github.com/borenstein-lab/microbiome-metabolome-curated-data] and to Erawijantari for speedy and prompt replies
All code dependancies can be found in the `utils.py`.
