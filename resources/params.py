import numpy as np

#-------------------------------------------Raw sequences-----------------------------------------
# -----Public Databases-----
VDJ_DATA_PATH = './datasets/filtered_vdj.csv'
MCPAS_DATA_PATH = './datasets/filtered_mcpas.csv'
IEDB_DATA_PATH = './datasets/filtered_iedb.csv'
MIRA_DATA_PATH = './datasets/mira.csv'

# -----Data Inputs-----
ATCHLEY_DATA_PATH = './datasets/atchley.txt'
RAW_TRAIN_DATA_PATH = './datasets/raw_train_data.csv'
TRAIN_DATA_PATH = './datasets/train_data.csv'
RAW_ID_DATA_PATH = './datasets/raw_id_data.csv'
ID_DATA_PATH = './datasets/id_data.csv'
OOD_DATA_PATH = './datasets/ood_data.csv'
RAW_TRAINVAL_SET_PATH = './datasets/trainval_set.csv'
TRAINVAL_SET_PATH = './datasets/filtered_trainval_set.csv'
RAW_TEST_SET_PATH = './datasets/test_set.csv'
TEST_SET_PATH = './datasets/filtered_test_set.csv'
TEST_DATA_PATH = './datasets/test_data.csv'

RAW_MIRA_TRAIN_DATA_PATH = './datasets/raw_mira_train_data.csv'
MIRA_TRAIN_DATA_PATH = './datasets/mira_train_data.csv'
MIRA_TRAIN_AND_EURO_DATA_PATH = './datasets/mira_train_and_euro_data.csv'
RAW_EURO_COVID_ASSOC_DATA_PATH = './datasets/euro_covid_assoc_data.csv'
RAW_EURO_COVID_ONLY_DATA_PATH = './datasets/euro_covid_only_data.csv'
RAW_EURO_PUBLIC_CLONE_DATA_PATH = './datasets/euro_public_clones_data.csv'
EURO_DATA_PATH = './datasets/euro_data.csv'
EURO_HLA_MAP_DATA_PATH = './datasets/euro-hla-map.csv'


# -----Network Data-----
TRAIN_MATRIX_PATH = './datasets/train_matrix_lv_one.mtx'
ID_MATRIX_PATH = './datasets/id_matrix_lv_one.mtx'
OOD_MATRIX_PATH = './datasets/ood_matrix_lv_one.mtx'
MIRA_TRAIN_MATRIX_PATH = './datasets/network_mira_lv_0/MIRA_Train_Network_AdjacencyMatrix.mtx'
EURO_MATRIX_PATH = './datasets/network_euro_lv_0/European_Network_AdjacencyMatrix.mtx'

# -----Model Params-----
AE_CHECKPOINT_PATH = './checkpoints/ae_best_tcr_train_model.h5'
ED_CHECKPOINT_PATH = './checkpoints/ed_best_train_model.h5'
NE_CHECKPOINT_PATH = './checkpoints/ne_best_train_model.h5'

#-------------------------------------------Embeddings-----------------------------------------
# ----- TEST Data Embs-----
TESSA_TEST_EMB_PATH = './datasets/test-tessa-emb.csv'
PE_TEST_EMB_PATH = './datasets/test-pe-emb.csv'
ED_TEST_EMB_PATH = './datasets/test-ed-emb.csv'
NE_TEST_EMB_PATH = './datasets/test-node-emb.csv'
HLA_TEST_EMB_PATH = './datasets/test-hla-emb.csv'
VJ_TEST_EMB_PATH = './datasets/test-vj-emb.csv'

# ----- OOD Data Embs-----
TESSA_OOD_EMB_PATH = './datasets/ood-tessa-emb.csv'
PE_OOD_EMB_PATH = './datasets/ood-pe-emb.csv'
ED_OOD_EMB_PATH = './datasets/ood-ed-emb.csv'
NE_OOD_EMB_PATH = './datasets/ood-node-emb.csv'
HLA_OOD_EMB_PATH = './datasets/ood-hla-emb.csv'
VJ_OOD_EMB_PATH = './datasets/ood-vj-emb.csv'

# ----- MIRA Train Data Embs-----
TESSA_MIRA_TRAIN_EMB_PATH = './datasets/mira-train-tessa-emb.csv'
PE_MIRA_TRAIN_EMB_PATH = './datasets/mira-train-pe-emb.csv'
ED_MIRA_TRAIN_EMB_PATH = './datasets/mira-train-ed-emb.csv'
NE_MIRA_TRAIN_EMB_PATH = './datasets/mira-train-node-emb.csv'
HLA_MIRA_TRAIN_EMB_PATH = './datasets/mira-train-hla-emb.csv'
VJ_MIRA_TRAIN_EMB_PATH = './datasets/mira-train-vj-emb.csv'

# ----- MIRA Train and Euro Case Data Embs-----
HLA_OOD_EURO_EMB_PATH = './datasets/mira-euro-hla-emb.csv'
VJ_OOD_EURO_EMB_PATH = './datasets/mira-euro-vj-emb.csv'

# ----- TEST European Case Data Embs-----
TESSA_TEST_EURO_EMB_PATH = './datasets/test-euro-tessa-emb.csv'
PE_TEST_EURO_EMB_PATH = './datasets/test-euro-pe-emb.csv'
ED_TEST_EURO_EMB_PATH = './datasets/test-euro-ed-emb.csv'
NE_TEST_EURO_EMB_PATH = './datasets/test-euro-node-emb.csv'
HLA_TEST_EURO_EMB_PATH = './datasets/test-euro-hla-emb.csv'
VJ_TEST_EURO_EMB_PATH = './datasets/test-euro-vj-emb.csv'

# ----- ID Data Embs-----
TESSA_ID_EMB_PATH = './datasets/id-tessa-emb.csv'
PE_ID_EMB_PATH = './datasets/id-pe-emb.csv'
ED_ID_EMB_PATH = './datasets/id-ed-emb.csv'
NE_ID_EMB_PATH = './datasets/id-node-emb.csv'
HLA_ID_EMB_PATH = './datasets/id-hla-emb.csv'
VJ_ID_EMB_PATH = './datasets/id-vj-emb.csv'

# -----Train/Val/Test sets---
SAMPLE_PATH = './outputs/samples/'

# -----Training Params-----
CASES = np.arange(1, 32, 1).tolist() # A list of case number index of the corresponding feature combination 
NUM_EPOCHS = 150
NUM_PEPTIDES = np.arange(5,21,5).tolist() # A list of top number of peptides for ID data
VAL_FRACTION = 0.1 # Fraction of val in the train+val set
TEST_FRACTION = 0.2 # Fraction of test in total set
VAL_RANDOM_SEED = 21 # Set seed for val set
TEST_RANDOM_SEED = 69 # Set seed for test set
LABEL_COLUMN = 'Peptide' # Column name of ground truth label

# -----Tuning Params-----
LEARNING_RATES = [0.005, 0.001, 0.0005, 0.0001] # learning rate parameters
BATCH_SIZES = [64, 128] # batch size parameters
## ID data
ID_MINOR_WEIGHTS = np.arange(0,4,1).tolist() # sample weight parameters for minor class
ID_MAJOR_WEIGHTS = np.arange(0.25, 1.6, 0.25).tolist() # sample weight parameters for major class
## OOD data
OOD_MINOR_WEIGHTS = np.arange(0,0.6,0.25).tolist() # sample weight parameters for minor class
OOD_MAJOR_WEIGHTS = np.arange(-0.25,0.3,0.25).tolist()  # sample weight parameters for major class

# -----Outputs-----
FIGURE_PATH = './outputs/figures/'
EVAL_METRICS_PATH = './outputs/metrics/'
BAYE_PROB_PATH = './outputs/bayesian/'

# -----Model Params-----
CLF_CHECKPOINT_PATH = './checkpoints/'
