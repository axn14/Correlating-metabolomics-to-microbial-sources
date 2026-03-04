

# --------------------------------------------------------------------------
"""
Step 1 — Environment setup: import libraries, configure plotting, create output directories.
"""


import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from utils import (
    DATA_DIR, RESULTS_DIR, POLYAMINE_KEGG,
    load_all_datasets,
    harmonize_metadata, find_polyamine_columns,
    reduce_species_names,
    compute_sample_qc, compute_feature_qc,
    filter_by_prevalence, filter_near_zero_variance,
    clr_transform, log_transform,
    validate_sample_alignment,
    plot_pca, plot_detection_histogram,
    differential_abundance, volcano_plot,
    compute_correlations,
    plot_correlation_heatmap,
)

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100

# Create results directory structure
CRC_RESULTS_DIR = RESULTS_DIR / 'crc_gut_analysis'
CRC_RESULTS_DIR.mkdir(exist_ok=True)
(CRC_RESULTS_DIR / 'figures').mkdir(exist_ok=True)
(CRC_RESULTS_DIR / 'figures' / 'qc').mkdir(exist_ok=True)
(CRC_RESULTS_DIR / 'figures' / 'pca').mkdir(exist_ok=True)
(CRC_RESULTS_DIR / 'figures' / 'da').mkdir(exist_ok=True)
(CRC_RESULTS_DIR / 'figures' / 'correlations').mkdir(exist_ok=True)
(CRC_RESULTS_DIR / 'tables').mkdir(exist_ok=True)
(CRC_RESULTS_DIR / 'intermediate').mkdir(exist_ok=True)

print('Libraries loaded successfully.')
print(f'Results will be saved to: {CRC_RESULTS_DIR}')
