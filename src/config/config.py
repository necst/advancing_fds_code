from pathlib import Path
import os

# ------------------------------------------------------------------------------
# GENERAL
# ------------------------------------------------------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = Path(dir_path).parent

# ------------------------------------------------------------------------------
# REPORTS AND RESULTS
# ------------------------------------------------------------------------------
RESULTS_PATH = PROJECT_PATH / "results"
MODEL_SELECTION_PATH = RESULTS_PATH / "model_selection"

EXPERIMENT_1_RESULT_PATH = RESULTS_PATH / "experiment_1"
EXPERIMENT_2_RESULT_PATH = RESULTS_PATH / "experiment_2"

SYNTH_FRAUDS_PATH = RESULTS_PATH / "dataset_synth_fraud"
SYNTH_FRAUDS_REPORT = "synth_fraud_gen_report.txt"


# ------------------------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------------------------
DATA_PATH = PROJECT_PATH  / "dataset"

# ORIGINAL DATASET PATHS
ORIGINAL_DATA_PATH = DATA_PATH / "original_datasets"
TRAINING_DATASET_PATH = DATA_PATH / "original_datasets"
AGGREGATED_DATASET_PATH = DATA_PATH / "original_datasets"


# SYNTHETIC FRAUD DATASETS
FRAUD_GENERATED_PATH = DATA_PATH / "fraud_generated"
STANDARDIZED_DATA_PATH = DATA_PATH / "standardized_datasets"
CUSTOM_CAMPAIGN_PATH = DATA_PATH / "custom_campaigns"

# PREFIXES
AGGREGATED_DATA_PREFIX = "aggregated_"
AUGMENTED_DATA_PREFIX = "augmented_"
LEGIT_REMOVED_DATA_PREFIX = "removed_"
SYNTHETIC_FRAUDS_DATA_PREFIX = "frauds_"

# ------------------------------------------------------------------------------
# CONFIGURATION FILES
# ------------------------------------------------------------------------------

# ALL CONFIG FILES PATH
CONFIG_FILES_PATH = PROJECT_PATH / "config"
SYNTH_FRAUDS_CONFIG = CONFIG_FILES_PATH / "dataset_synth_fraud"

# SYNTHETIC FRAUD CRAFTING
VICTIM_PROFILES_INFO_FILE = SYNTH_FRAUDS_CONFIG / "victims_profiles.json"
CANDIDATE_VICTIMS_LIST = SYNTH_FRAUDS_CONFIG / "all_candidate_victims.json"
SYNTH_FRAUDS_PARAMS = SYNTH_FRAUDS_CONFIG / "default_poisoning_attack.json"

# MODEL SELECTION (FEATURES, HYPERPARAMETERS, THRESHOLDS)
MODEL_CONF_DIR = CONFIG_FILES_PATH / "models"
FEATURES_DIR = MODEL_CONF_DIR / "features"
HPARAMS_DIR = MODEL_CONF_DIR / "hparams"
THRESHOLDS_DIR = MODEL_CONF_DIR / "thresholds"

# ------------------------------------------------------------------------------
#  CACHE FILES
# ------------------------------------------------------------------------------

SAVE_PATH = PROJECT_PATH / "cache"
STD_SCALER_DIR = SAVE_PATH / "std_scalers"
MODEL_CACHE_DIR = SAVE_PATH / "models"
