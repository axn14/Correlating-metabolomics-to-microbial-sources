# CRC Gut Microbiome–Metabolome Analysis

## Directory Structure

```
01_preprocessing/          Raw TSV files → Preprocessed Data
02_association_maps/       Preprocessed Data → Association Maps
03_validated_associations/ Association Maps → Validated Associations
04_source_attribution/     Validated Associations → Outputs
utils.py                   Shared utility functions
```

## Dependencies

```
pandas numpy scipy scikit-learn matplotlib seaborn
statsmodels pingouin networkx xgboost shap
```

## Data

Find all data files (Associated to CRC and otherwise) uploaded in the Data/ folder (with special thanks to the Borenstein Lab [https://github.com/borenstein-lab/microbiome-metabolome-curated-data] and to Erawijantari for speedy and prompt replies
All code dependancies can be found in the `utils.p`.

## Overview

This computational biology research project investigates the relationship between the gut microbiome and fecal metabolome across multiple colorectal cancer (CRC) and gastrointestinal disease cohorts. The central objective is to identify which microbial species are the most likely biosynthetic sources of polyamines (putrescine, spermidine, spermine, N-acetylputrescine, N1,N12-diacetylspermine, cadaverine) and related metabolites in the context of colorectal cancer progression.Scientific RationalePolyamines are essential metabolites implicated in cell proliferation, differentiation, and tumor progression. While both host and microbial cells produce polyamines, the relative contribution of gut microbiota to the fecal polyamine pool—and how this shifts during colorectal carcinogenesis—remains poorly characterized. This project employs a multi-cohort, multi-omic approach combining:
1. Species-level metagenomic profiling (GTDB taxonomy)
2. Untargeted metabolomics (KEGG-annotated metabolite abundances)
3. Statistical association mapping (Spearman correlations, partial correlations, network analysis)
4. Machine learning classification and regression (Random Forest, XGBoost, SHAP feature importance)
5. Meta-analysis across cohorts (DerSimonian-Laird random-effects models)
6. Literature-based validation (known polyamine producers)

