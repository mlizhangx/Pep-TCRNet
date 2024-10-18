import pandas as pd 
import numpy as np
import os
from sklearn import model_selection, preprocessing

#--------------------------------------------Preprocess Raw Sequences----------------------------------------
def read_data(fpath_data: str):
    """Read raw csv and return a dataframe"""

    df = pd.read_csv(fpath_data)
    
    return df

def get_global_max_sequence_length(
        input_data: any,
        seq_type: str,
   ):
   """Get the global max sequence length
   """

   max_seq_lens = input_data[seq_type].str.len()
   max_seq_len = max(max_seq_lens)
   
   return max_seq_len

def read_atchley(fpath_atchley: str):
    """Read raw atchley factor txt file and return a dataframe"""

    atchley_table = pd.read_csv(fpath_atchley, sep='\t')
    atchley_table = atchley_table.rename(columns={'amino.acid': 'token_idx'}) 

    return atchley_table

def get_token_table(atchley_table):
    """Generate a dataframe with 2 columns:
     1. Token indexes as single amino acids
     2. Tokens as integers 0-20
    """
    
    token_indexes = atchley_table['token_idx'] # token indexes as single amino acids
    tokens = (range(20)) # tokens as integers 0-20

    token_table = pd.DataFrame({
        "token_idx": token_indexes,
        "token": tokens
    }).set_index(['token_idx']) 
    token_table.index.name = None

    return token_table

def get_atchley_vectors(atchley_table):
    """Create an embedding matrix to map to the raw sequences"""

    emb_matrix = atchley_table.drop(columns=['token_idx'])

    # Convert dtypes object to float
    for col in emb_matrix.columns:
        converted_column = pd.to_numeric(emb_matrix[col].str.replace('−','-'), errors='coerce')
        emb_matrix[col] = converted_column

    # Create a new column that is the average of the first 5 columns
    emb_matrix['avg'] = emb_matrix.sum(axis=1)
    num_cols = emb_matrix.shape[1]

    # Pad 0's to the embedding matrix
    zeros= [[0] * num_cols]
    zero = pd.DataFrame(zeros, columns=emb_matrix.columns)
    
    # Add the zero vector to the embedding matrix
    emb_matrix = pd.concat([zero, emb_matrix], axis=0)
    emb_matrix = np.array(emb_matrix)

    return emb_matrix

def map_idx_to_token(
        token_table,
        input_sequence: str,
        max_seq_length: int,
):
    """Map each amino acid in the sequence to its corresponding Atchley Factor vector and 
    return token indexes mapped by the single amino acids in the input sequence"""

    input_seq_length = len(input_sequence)
    amino_acids = token_table.index

    mapped_tokens = np.zeros((max_seq_length), dtype='int') # initialize an embedding matrix
    mapped_seq_length = min(max_seq_length, input_seq_length) # get the actual sequence length

    for i in range(mapped_seq_length):
        amino_acid = input_sequence[i]
        token_index = token_table.loc[amino_acid].values
        if amino_acid in amino_acids:
            mapped_tokens[i] = 1 + token_index
        else:
            mapped_tokens[i] = np.random.randint(0, mapped_seq_length)

    return mapped_tokens


#-------------------------------------------Combine the Embeddings-----------------------------------------
def load_data(
    fpath: str, # input data
):
    """Load raw data csv file as pandas DataFrame"""
    input_df = pd.read_csv(fpath)
    
    return input_df

def get_top_peptides(
    input_df: any,
    num_peptide: int,
):
    """Retrieve the peptides that have the most frequency"""

    top_peptides = input_df.Peptide.value_counts().head(num_peptide).index.tolist()

    return top_peptides

def slice_data(
    input_df: any,
    top_peptides: any,
):
    """Slice dataset with data only contain the peptides in the list"""
    subset_df = input_df[input_df.Peptide.isin(top_peptides)]

    return subset_df

def remap_labels(
    input_df: any
):
    """Remap peptide sequence labels into integers"""
    # Map labels to 0:K definition, where K is the number of classes
    labels_raw = input_df['Peptide'].unique().tolist()
    # Create a map from raw labels to redefined labels
    class2idx = {labels_raw[k]:k for k in range(len(labels_raw))}
    # Create a reverse map from redefined labels back to raw labels
    idx2class = {v:k for k,v in class2idx.items()}
    # Get data and labels
    data = input_df
    labels = data['Peptide']
    # Apply new redefined labels
    labels = labels.replace(class2idx)

    return data, labels, class2idx, idx2class

def get_number_classes(class2idx):
    """Get the total number of classes"""

    num_class = len(class2idx)

    return num_class

def get_sampled_data(
    data: any,
    labels: any,
    val_frac: float = 0.1, # Fraction of val in the train+val set
    test_frac: float = 0.2, # Fraction of test in total set
    val_rand_seed: int = 21,
    test_rand_seed: int = 69,
):
    """Split the total set into train/val/test sets"""

    # Split total dataset into train+val and test
    trainval_subjects, test_subjects, trainval_labels, test_labels = model_selection.train_test_split(
        data,
        labels,
        test_size=test_frac,
        stratify=labels,
        random_state=test_rand_seed,
    )

    # Split train+val into train and val
    train_subjects, val_subjects, train_labels, val_labels = model_selection.train_test_split(
        trainval_subjects,
        trainval_labels,
        test_size=val_frac,
        stratify=trainval_labels,
        random_state=val_rand_seed,
    )

    # Concatenate val and test
    valtest_subjects = pd.concat([val_subjects, test_subjects])
    valtest_labels = pd.concat([val_labels, test_labels])

    return trainval_subjects, valtest_subjects, train_subjects, val_subjects, test_subjects, trainval_labels, valtest_labels, train_labels, val_labels, test_labels

#-------------------------------------------Preprocess embeddings-----------------------------------------

def load_embeddings(
    fpath_tessa: str, # Tessa embeddings for TCR sequences
    fpath_pe: str, # PE embeddings for TCR sequences
    fpath_ed: str, # ED embeddings for TCR sequences
    fpath_ne: str, # Node embeddings for TCR sequences
    fpath_hla: str, # HLA embeddings
    fpath_vj: str, # VJ embeddings
):
    """Load raw data csv files as pandas DataFrame"""
    
    tessa_df= pd.read_csv(fpath_tessa).iloc[:, 1:]
    pe_df= pd.read_csv(fpath_pe)
    ed_df = pd.read_csv(fpath_ed)
    ne_df= pd.read_csv(fpath_ne)
    hla_df = pd.read_csv(fpath_hla)
    vj_df = pd.read_csv(fpath_vj)

    return tessa_df, pe_df, ed_df, ne_df, hla_df, vj_df

def concat_id_hla_data(
    case: int, # case number index for feature combination input
    hla_df: any, 
    vj_df: any, 
    train_subjects: any,
    valtest_subjects: any,
    train_labels: any,
    valtest_labels: any,
):
    """For cases of ID data that contain HLA and VJ: 
    Load and drop duplicates for train/val/test subjects, 
    map train/val/test set indexes to the feature set, 
    and concatenate feature columns"""

    # ----HLA embeddings----
    if (case == 4):
        # Drop duplicates
        train_subjects = train_subjects[['MHC', 'Peptide']].drop_duplicates()
        valtest_subjects = valtest_subjects[['MHC', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index] 
        X_test = hla_df.add_prefix('HLA_', axis=1).loc[valtest_subjects.index]

    # -----VJ embeddings-----
    elif (case == 5):
        # Drop duplicates
        train_subjects = train_subjects[['V', 'J', 'Peptide']].drop_duplicates()
        valtest_subjects = valtest_subjects[['V', 'J', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index] 
        X_test = vj_df.add_prefix('VJ_', axis=1).loc[valtest_subjects.index]
    
    # ------HLA + VJ------
    elif (case == 28):
        # Drop duplicates
        train_subjects = train_subjects[['MHC','V','J','Peptide']].drop_duplicates()
        valtest_subjects = valtest_subjects[['MHC','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index],
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[valtest_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[valtest_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # Remove common data between train set and test set
    train_test = pd.merge(X_train.reset_index(), 
                         X_test.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_test = X_test.drop(train_test.index_y) # drop duplicates of test set that exist in train set

    # Get the labels
    y_train = train_labels.loc[X_train.index]
    y_test = valtest_labels.loc[X_test.index]

    return X_train, X_test, y_train, y_test

def concat_id_data(
        case: int, # case number index for feature combination input
        tessa_df: any, 
        pe_df: any, 
        ed_df: any, 
        ne_df: any, 
        hla_df: any, 
        vj_df: any, 
        train_subjects: any,
        val_subjects: any,
        test_subjects: any,
        train_labels: any,
        val_labels: any,
        test_labels: any,
):
    """For ID data: Load and drop duplicates for train/val/test subjects, 
    map train/val/test set indexes to the feature set, 
    and concatenate feature columns"""

    # Concatenate feature columns to create 31 feature combinations for ID data
    # -----------------------------------BLOCK 1 (1 input)-----------------------------------
    # -----ED embeddings-----
    if (case==1):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index]
        X_val = ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index]
        X_test = ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index]

    # -----PE embeddings-----
    elif (case==2): 
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index]
        X_val = pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index]
        X_test = pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index]

    # ----Tessa embeddings----
    elif (case == 3):      
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index]
        X_val = tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index]
        X_test = tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index]

    
    # ----Node embeddings----
    elif (case == 6):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]
        X_val = ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]
        X_test = ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]
    
    # -----------------------------------BLOCK 2 (2 inputs)-----------------------------------
    # --------ED + HLA--------
    elif (case == 7):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + HLA--------        
    elif (case == 8):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA-------        
    elif (case == 9):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------ED + VJ---------
    elif (case == 10):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ---------PE + VJ---------
    elif (case == 11):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------Tessa + VJ--------
    elif (case == 12):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V' , 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V' , 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------ED + NE----------
    elif (case == 13):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------PE + NE----------
    elif (case == 14):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------Tessa + NE---------
    elif (case == 15):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 3 (3 inputs)-----------------------------------
    # --------ED + HLA + VJ--------
    elif (case == 16):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------PE + HLA + VJ--------
    elif (case == 17):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA + VJ-------
    elif (case == 18):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------ED + HLA + NE--------
    elif (case == 19):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'MHC','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'MHC','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # --------PE + HLA + NE--------
    elif (case == 20):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'MHC','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'MHC','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # -------Tessa + HLA + NE-------
    elif (case == 21):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'MHC','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'MHC','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)   

    # --------ED + VJ + NE--------
    elif (case == 22):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------PE + VJ + NE--------
    elif (case == 23):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + VJ + NE-------
    elif (case == 24):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 4 (4 inputs)-----------------------------------
    # -------ED + NE + HLA + VJ-------
    elif (case == 25):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------PE + NE + HLA + VJ-------
    elif (case == 26):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------Tessa + NE + HLA + VJ------
    elif (case == 27):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'MHC','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
    
    # -----------------------------------BLOCK 5 (inputs w/o TCR)-----------------------------------
    # ------HLA + NE------
    elif (case == 29):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','MHC','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------VJ + NE------
    elif (case == 30):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------HLA + VJ + NE------
    elif (case == 31):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','MHC', 'V', 'J', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','MHC','V', 'J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','MHC','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index],
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index],
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # Remove common data between train set and val set
    train_val = pd.merge(X_train.reset_index(), 
                         X_val.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_val = X_val.drop(train_val.index_y) # drop duplicates of val set that exist in train set
    
    # Remove common data between train set and test set
    train_test = pd.merge(X_train.reset_index(), 
                         X_test.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_test = X_test.drop(train_test.index_y) # drop duplicates of test set that exist in train set

    # Get the labels
    y_train = train_labels.loc[X_train.index]
    y_val = val_labels.loc[X_val.index] 
    y_test = test_labels.loc[X_test.index]

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_id_samples(
        case: int, # case number index for feature combination input
        tessa_df: any, 
        pe_df: any, 
        ed_df: any, 
        ne_df: any, 
        hla_df: any, 
        vj_df: any, 
        train_idx: any,
        val_idx: any,
        test_idx: any,
        train_labels: any,
        val_labels: any,
        test_labels: any,
):
    """For ID data: Load and drop duplicates for train/val/test subjects, 
    map train/val/test set indexes to the feature set, 
    and concatenate feature columns"""

    # Concatenate feature columns to create 31 feature combinations for ID data
    # -----------------------------------BLOCK 1 (1 input)-----------------------------------
    # -----ED embeddings-----
    if (case==1):
        # Map the set index
        X_train = ed_df.add_prefix('ED_', axis=1).loc[train_idx]
        X_val = ed_df.add_prefix('ED_', axis=1).loc[val_idx]
        X_test = ed_df.add_prefix('ED_', axis=1).loc[test_idx]

    # -----PE embeddings-----
    elif (case==2): 
        # Map the set index
        X_train = pe_df.add_prefix('PE_', axis=1).loc[train_idx]
        X_val = pe_df.add_prefix('PE_', axis=1).loc[val_idx]
        X_test = pe_df.add_prefix('PE_', axis=1).loc[test_idx]

    # ----Tessa embeddings----
    elif (case == 3):      
        # Map the set index
        X_train = tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx]
        X_val = tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx]
        X_test = tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx]

    
    # ----Node embeddings----
    elif (case == 6):
        # Map the set index
        X_train = ne_df.add_prefix('NE_', axis=1).loc[train_idx]
        X_val = ne_df.add_prefix('NE_', axis=1).loc[val_idx]
        X_test = ne_df.add_prefix('NE_', axis=1).loc[test_idx]
    
    # -----------------------------------BLOCK 2 (2 inputs)-----------------------------------
    # --------ED + HLA--------
    elif (case == 7):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + HLA--------        
    elif (case == 8):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA-------        
    elif (case == 9):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------ED + VJ---------
    elif (case == 10):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ---------PE + VJ---------
    elif (case == 11):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------Tessa + VJ--------
    elif (case == 12):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------ED + NE----------
    elif (case == 13):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------PE + NE----------
    elif (case == 14):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------Tessa + NE---------
    elif (case == 15):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 3 (3 inputs)-----------------------------------
    # --------ED + HLA + VJ--------
    elif (case == 16):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------PE + HLA + VJ--------
    elif (case == 17):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA + VJ-------
    elif (case == 18):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------ED + HLA + NE--------
    elif (case == 19):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # --------PE + HLA + NE--------
    elif (case == 20):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # -------Tessa + HLA + NE-------
    elif (case == 21):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)   

    # --------ED + VJ + NE--------
    elif (case == 22):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + VJ + NE--------
    elif (case == 23):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + VJ + NE-------
    elif (case == 24):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 4 (4 inputs)-----------------------------------
    # -------ED + NE + HLA + VJ-------
    elif (case == 25):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------PE + NE + HLA + VJ-------
    elif (case == 26):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------Tessa + NE + HLA + VJ------
    elif (case == 27):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
    
    # -----------------------------------BLOCK 5 (inputs w/o TCR)-----------------------------------
    # ------HLA + NE------
    elif (case == 29):
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------VJ + NE------
    elif (case == 30):
        # Map the set index and get the embedding features 
        train_features = [vj_df.add_prefix('VJ_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [vj_df.add_prefix('VJ_', axis=1).loc[val_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [vj_df.add_prefix('VJ_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------HLA + VJ + NE------
    elif (case == 31):
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_idx],
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_idx],
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # Get the labels
    y_train = train_labels.loc[X_train.index]
    y_val = val_labels.loc[X_val.index] 
    y_test = test_labels.loc[X_test.index]

    return X_train, X_val, X_test, y_train, y_val, y_test

def concat_ood_hla_data(
    case: int, # case number index for feature combination input
    hla_df: any, 
    vj_df: any, 
    train_subjects: any,
    valtest_subjects: any,
    train_labels: any,
    valtest_labels: any,
):
    """For cases of OOD that contain HLA & VJ: 
    Load and drop duplicates for train/val/test subjects, 
    map train/val/test set indexes to the feature set, 
    and concatenate feature columns"""

    # ----HLA embeddings----
    if (case == 4):
        # Drop duplicates
        train_subjects = train_subjects[['HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        valtest_subjects = valtest_subjects[['HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index] 
        X_test = hla_df.add_prefix('HLA_', axis=1).loc[valtest_subjects.index]

    # -----VJ embeddings-----
    elif (case == 5):
        # Drop duplicates
        train_subjects = train_subjects[['V', 'J', 'Peptide']].drop_duplicates()
        valtest_subjects = valtest_subjects[['V', 'J', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index] 
        X_test = vj_df.add_prefix('VJ_', axis=1).loc[valtest_subjects.index]

    # ------HLA + VJ------
    elif (case == 28):
        # Drop duplicates
        train_subjects = train_subjects[['HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        valtest_subjects = valtest_subjects[['HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[valtest_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[valtest_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # Remove common data between train set and val set/test set
    train_test = pd.merge(X_train.reset_index(), 
                         X_test.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_test = X_test.drop(train_test.index_y) # drop duplicates of test set that exist in train set

    # Get the labels
    y_train = train_labels.loc[X_train.index]
    y_test = valtest_labels.loc[X_test.index]

    return X_train, X_test, y_train, y_test
        
def concat_ood_data(
    case: int, # case number index for feature combination input
    tessa_df: any, 
    pe_df: any, 
    ed_df: any, 
    ne_df: any, 
    hla_df: any, 
    vj_df: any, 
    train_subjects: any,
    val_subjects: any,
    test_subjects: any,
    train_labels: any,
    val_labels: any,
    test_labels: any,
):
    """For OOD data: Load and drop duplicates for train/val/test subjects, 
    map train/val/test set indexes to the feature set, 
    and concatenate feature columns"""

    # Concatenate feature columns to create feature combinations for OOD data
    # -----------------------------------BLOCK 1 (1 input)-----------------------------------
    # -----ED embeddings-----
    if (case==1):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index]
        X_val = ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index]
        X_test = ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index]

    # -----PE embeddings-----
    elif (case==2): 
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index]
        X_val = pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index]
        X_test = pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index]

    # ----Tessa embeddings----
    elif (case == 3):      
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates() 
        # Map the set index
        X_train = tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index]
        X_val = tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index]
        X_test = tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index]
    
    # ----Node embeddings----
    elif (case == 6):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()   
        # Map the set index
        X_train = ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]
        X_val = ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]
        X_test = ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]
    
    # -----------------------------------BLOCK 2 (2 inputs)-----------------------------------
    # --------ED + HLA--------
    elif (case == 7):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates() 
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + HLA--------        
    elif (case == 8):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA-------        
    elif (case == 9):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()  
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------ED + VJ---------
    elif (case == 10):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ---------PE + VJ---------
    elif (case == 11):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------Tessa + VJ--------
    elif (case == 12):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V' , 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V' , 'J','Peptide']].drop_duplicates()       
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------ED + NE----------
    elif (case == 13):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()      
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------PE + NE----------
    elif (case == 14):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()      
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------Tessa + NE---------
    elif (case == 15):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 3 (3 inputs)-----------------------------------
    # --------ED + HLA + VJ--------
    elif (case == 16):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------PE + HLA + VJ--------
    elif (case == 17):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA + VJ-------
    elif (case == 18):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------ED + HLA + NE--------
    elif (case == 19):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # --------PE + HLA + NE--------
    elif (case == 20):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # -------Tessa + HLA + NE-------
    elif (case == 21):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)   

    # --------ED + VJ + NE--------
    elif (case == 22):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + VJ + NE--------
    elif (case == 23):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()      
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + VJ + NE-------
    elif (case == 24):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 4 (4 inputs)-----------------------------------
    # -------ED + NE + HLA + VJ-------
    elif (case == 25):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------PE + NE + HLA + VJ-------
    elif (case == 26):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------Tessa + NE + HLA + VJ------
    elif (case == 27):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)  

    # -----------------------------------BLOCK 5 (cases w/o TCRs)-----------------------------------
    # ------HLA + NE------
    elif (case == 29):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------VJ + NE------
    elif (case == 30):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------HLA + VJ + NE------
    elif (case == 31):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1', 'V', 'J', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA.A','HLA.A.1','HLA.B','HLA.B.1','HLA.C','HLA.C.1','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index],
                         vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index],
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # Remove common data between train set and val set/test set
    train_val = pd.merge(X_train.reset_index(), 
                         X_val.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_val = X_val.drop(train_val.index_y) # drop duplicates of val set that exist in train set
    train_test = pd.merge(X_train.reset_index(), 
                         X_test.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_test = X_test.drop(train_test.index_y) # drop duplicates of test set that exist in train set

    # Get the labels
    y_train = train_labels.loc[X_train.index]
    y_val = val_labels.loc[X_val.index] 
    y_test = test_labels.loc[X_test.index]

    return X_train, X_val, X_test, y_train, y_val, y_test

def concat_mira_train_data(
    case: int, # case number index for feature combination input
    tessa_df: any, 
    pe_df: any, 
    ed_df: any, 
    ne_df: any, 
    hla_df: any, 
    vj_df: any, 
    train_subjects: any,
    val_subjects: any,
    test_subjects: any,
    train_labels: any,
    val_labels: any,
    test_labels: any,
):
    """For OOD data: Load and drop duplicates for train/val/test subjects, 
    map train/val/test set indexes to the feature set, 
    and concatenate feature columns"""

    # Concatenate feature columns to create feature combinations for OOD data
    # -----------------------------------BLOCK 1 (1 input)-----------------------------------
    # -----ED embeddings-----
    if (case==1):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index]
        X_val = ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index]
        X_test = ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index]

    # -----PE embeddings-----
    elif (case==2): 
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index]
        X_val = pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index]
        X_test = pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index]

    # ----Tessa embeddings----
    elif (case == 3):      
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates() 
        # Map the set index
        X_train = tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index]
        X_val = tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index]
        X_test = tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index]
    
    # ----Node embeddings----
    elif (case == 6):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()   
        # Map the set index
        X_train = ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]
        X_val = ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]
        X_test = ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]
    
    # -----------------------------------BLOCK 2 (2 inputs)-----------------------------------
    # --------ED + HLA--------
    elif (case == 7):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates() 
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + HLA--------        
    elif (case == 8):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA-------        
    elif (case == 9):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()  
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------ED + VJ---------
    elif (case == 10):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ---------PE + VJ---------
    elif (case == 11):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------Tessa + VJ--------
    elif (case == 12):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V' , 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V' , 'J','Peptide']].drop_duplicates()       
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------ED + NE----------
    elif (case == 13):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()      
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------PE + NE----------
    elif (case == 14):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()      
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------Tessa + NE---------
    elif (case == 15):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 3 (3 inputs)-----------------------------------
    # --------ED + HLA + VJ--------
    elif (case == 16):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------PE + HLA + VJ--------
    elif (case == 17):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA + VJ-------
    elif (case == 18):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------ED + HLA + NE--------
    elif (case == 19):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # --------PE + HLA + NE--------
    elif (case == 20):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # -------Tessa + HLA + NE-------
    elif (case == 21):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)   

    # --------ED + VJ + NE--------
    elif (case == 22):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + VJ + NE--------
    elif (case == 23):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()      
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + VJ + NE-------
    elif (case == 24):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 4 (4 inputs)-----------------------------------
    # -------ED + NE + HLA + VJ-------
    elif (case == 25):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------PE + NE + HLA + VJ-------
    elif (case == 26):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------Tessa + NE + HLA + VJ------
    elif (case == 27):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A','HLA-B','HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)  

    # -----------------------------------BLOCK 5 (cases w/o TCRs)-----------------------------------
    # ------HLA + NE------
    elif (case == 29):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------VJ + NE------
    elif (case == 30):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------HLA + VJ + NE------
    elif (case == 31):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A','HLA-B','HLA-C', 'V', 'J', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A','HLA-B','HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A','HLA-B','HLA-C','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index],
                         vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index],
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # Remove common data between train set and val set/test set
    train_val = pd.merge(X_train.reset_index(), 
                         X_val.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_val = X_val.drop(train_val.index_y) # drop duplicates of val set that exist in train set
    train_test = pd.merge(X_train.reset_index(), 
                         X_test.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_test = X_test.drop(train_test.index_y) # drop duplicates of test set that exist in train set

    # Get the labels
    y_train = train_labels.loc[X_train.index]
    y_val = val_labels.loc[X_val.index] 
    y_test = test_labels.loc[X_test.index]

    return X_train, X_val, X_test, y_train, y_val, y_test

def concat_euro_data(
    case: int, # case number index for feature combination input
    tessa_df: any, 
    pe_df: any, 
    ed_df: any, 
    ne_df: any, 
    hla_df: any, 
    vj_df: any, 
    train_subjects: any,
    val_subjects: any,
    test_subjects: any,
    train_labels: any,
    val_labels: any,
    test_labels: any,
):
    """For EURO data: Load and drop duplicates for train/val/test subjects, 
    map train/val/test set indexes to the feature set, 
    and concatenate feature columns"""

    # Concatenate feature columns to create feature combinations for OOD data
    # -----------------------------------BLOCK 1 (1 input)-----------------------------------
    # -----ED embeddings-----
    if (case==1):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index]
        X_val = ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index]
        X_test = ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index]

    # -----PE embeddings-----
    elif (case==2): 
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index
        X_train = pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index]
        X_val = pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index]
        X_test = pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index]

    # ----Tessa embeddings----
    elif (case == 3):      
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates() 
        # Map the set index
        X_train = tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index]
        X_val = tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index]
        X_test = tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index]
    
    # ----Node embeddings----
    elif (case == 6):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()   
        # Map the set index
        X_train = ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]
        X_val = ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]
        X_test = ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]
    
    # -----------------------------------BLOCK 2 (2 inputs)-----------------------------------
    # --------ED + HLA--------
    elif (case == 7):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates() 
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + HLA--------        
    elif (case == 8):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA-------        
    elif (case == 9):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()  
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------ED + VJ---------
    elif (case == 10):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ---------PE + VJ---------
    elif (case == 11):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------Tessa + VJ--------
    elif (case == 12):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V' ,'J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V' , 'J', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V' , 'J','Peptide']].drop_duplicates()       
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------ED + NE----------
    elif (case == 13):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()      
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------PE + NE----------
    elif (case == 14):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()      
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------Tessa + NE---------
    elif (case == 15):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 3 (3 inputs)-----------------------------------
    # --------ED + HLA + VJ--------
    elif (case == 16):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------PE + HLA + VJ--------
    elif (case == 17):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA + VJ-------
    elif (case == 18):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------ED + HLA + NE--------
    elif (case == 19):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # --------PE + HLA + NE--------
    elif (case == 20):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # -------Tessa + HLA + NE-------
    elif (case == 21):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)   

    # --------ED + VJ + NE--------
    elif (case == 22):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + VJ + NE--------
    elif (case == 23):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()      
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + VJ + NE-------
    elif (case == 24):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 4 (4 inputs)-----------------------------------
    # -------ED + NE + HLA + VJ-------
    elif (case == 25):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------PE + NE + HLA + VJ-------
    elif (case == 26):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------Tessa + NE + HLA + VJ------
    elif (case == 27):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3', 'HLA-A', 'HLA-B', 'HLA-C','V','J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_subjects.index], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)  

    # -----------------------------------BLOCK 5 (cases w/o TCRs)-----------------------------------
    # ------HLA + NE------
    elif (case == 29):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------VJ + NE------
    elif (case == 30):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','V', 'J', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------HLA + VJ + NE------
    elif (case == 31):
        # Drop duplicates
        train_subjects = train_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C', 'V', 'J', 'Peptide']].drop_duplicates()
        val_subjects = val_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','Peptide']].drop_duplicates()
        test_subjects = test_subjects[['CDR3','HLA-A', 'HLA-B', 'HLA-C','V', 'J','Peptide']].drop_duplicates()
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_subjects.index], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_subjects.index], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_subjects.index]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_subjects.index],
                         vj_df.add_prefix('VJ_', axis=1).loc[val_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_subjects.index]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_subjects.index],
                         vj_df.add_prefix('VJ_', axis=1).loc[test_subjects.index], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_subjects.index]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # Remove common data between train set and val set/test set
    train_val = pd.merge(X_train.reset_index(), 
                         X_val.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_val = X_val.drop(train_val.index_y) # drop duplicates of val set that exist in train set
    train_test = pd.merge(X_train.reset_index(), 
                         X_test.reset_index(), 
                         how='inner', 
                         on=X_train.columns.tolist()) # get common rows between train set and val set
    X_test = X_test.drop(train_test.index_y) # drop duplicates of test set that exist in train set

    # Get the labels
    y_train = train_labels.loc[X_train.index]
    y_val = val_labels.loc[X_val.index] 
    y_test = test_labels.loc[X_test.index]

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_ood_samples(
    case: int, # case number index for feature combination input
    tessa_df: any, 
    pe_df: any, 
    ed_df: any, 
    ne_df: any, 
    hla_df: any, 
    vj_df: any, 
    train_idx: any,
    val_idx: any,
    test_idx: any,
    train_labels: any,
    val_labels: any,
    test_labels: any,
):
    """For OOD data: Load and drop duplicates for train/val/test subjects, 
    map train/val/test set indexes to the feature set, 
    and concatenate feature columns"""

    # Concatenate feature columns to create feature combinations for OOD data
    # -----------------------------------BLOCK 1 (1 input)-----------------------------------
    # -----ED embeddings-----
    if (case==1):
        # Map the set index
        X_train = ed_df.add_prefix('ED_', axis=1).loc[train_idx]
        X_val = ed_df.add_prefix('ED_', axis=1).loc[val_idx]
        X_test = ed_df.add_prefix('ED_', axis=1).loc[test_idx]

    # -----PE embeddings-----
    elif (case==2): 
        # Map the set index
        X_train = pe_df.add_prefix('PE_', axis=1).loc[train_idx]
        X_val = pe_df.add_prefix('PE_', axis=1).loc[val_idx]
        X_test = pe_df.add_prefix('PE_', axis=1).loc[test_idx]

    # ----Tessa embeddings----
    elif (case == 3):      
        # Map the set index
        X_train = tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx]
        X_val = tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx]
        X_test = tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx]
    
    # ----Node embeddings----
    elif (case == 6): 
        # Map the set index
        X_train = ne_df.add_prefix('NE_', axis=1).loc[train_idx]
        X_val = ne_df.add_prefix('NE_', axis=1).loc[val_idx]
        X_test = ne_df.add_prefix('NE_', axis=1).loc[test_idx]
    
    # -----------------------------------BLOCK 2 (2 inputs)-----------------------------------
    # --------ED + HLA--------
    elif (case == 7):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + HLA--------        
    elif (case == 8):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA-------        
    elif (case == 9):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------ED + VJ---------
    elif (case == 10):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ---------PE + VJ---------
    elif (case == 11):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------Tessa + VJ--------
    elif (case == 12):     
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------ED + NE----------
    elif (case == 13):   
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ----------PE + NE----------
    elif (case == 14):    
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)
        
    # ---------Tessa + NE---------
    elif (case == 15):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 3 (3 inputs)-----------------------------------
    # --------ED + HLA + VJ--------
    elif (case == 16):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------PE + HLA + VJ--------
    elif (case == 17):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + HLA + VJ-------
    elif (case == 18):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # --------ED + HLA + NE--------
    elif (case == 19):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # --------PE + HLA + NE--------
    elif (case == 20):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1) 

    # -------Tessa + HLA + NE-------
    elif (case == 21):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)   

    # --------ED + VJ + NE--------
    elif (case == 22):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)


    # --------PE + VJ + NE--------
    elif (case == 23):    
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------Tessa + VJ + NE-------
    elif (case == 24):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -----------------------------------BLOCK 4 (4 inputs)-----------------------------------
    # -------ED + NE + HLA + VJ-------
    elif (case == 25):
        # Map the set index and get the embedding features 
        train_features = [ed_df.add_prefix('ED_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [ed_df.add_prefix('ED_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [ed_df.add_prefix('ED_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # -------PE + NE + HLA + VJ-------
    elif (case == 26):
        # Map the set index and get the embedding features 
        train_features = [pe_df.add_prefix('PE_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [pe_df.add_prefix('PE_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [pe_df.add_prefix('PE_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------Tessa + NE + HLA + VJ------
    elif (case == 27):
        # Map the set index and get the embedding features 
        train_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx], 
                          hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx]]
        val_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[val_idx], 
                        ne_df.add_prefix('NE_', axis=1).loc[val_idx],
                        hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                        vj_df.add_prefix('VJ_', axis=1).loc[val_idx]]
        test_features = [tessa_df.add_prefix('TESSA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx],
                         hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)  

    # -----------------------------------BLOCK 5 (cases w/o TCRs)-----------------------------------
    # ------HLA + NE------
    elif (case == 29):
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------VJ + NE------
    elif (case == 30):
        # Map the set index and get the embedding features 
        train_features = [vj_df.add_prefix('VJ_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [vj_df.add_prefix('VJ_', axis=1).loc[val_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [vj_df.add_prefix('VJ_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # ------HLA + VJ + NE------
    elif (case == 31):
        # Map the set index and get the embedding features 
        train_features = [hla_df.add_prefix('HLA_', axis=1).loc[train_idx], 
                          vj_df.add_prefix('VJ_', axis=1).loc[train_idx], 
                          ne_df.add_prefix('NE_', axis=1).loc[train_idx]]
        val_features = [hla_df.add_prefix('HLA_', axis=1).loc[val_idx],
                         vj_df.add_prefix('VJ_', axis=1).loc[val_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[val_idx]]
        test_features = [hla_df.add_prefix('HLA_', axis=1).loc[test_idx],
                         vj_df.add_prefix('VJ_', axis=1).loc[test_idx], 
                         ne_df.add_prefix('NE_', axis=1).loc[test_idx]]
        # Concatenate the embedding features
        X_train = pd.concat(train_features, axis=1)
        X_val = pd.concat(val_features, axis=1)
        X_test = pd.concat(test_features, axis=1)

    # Get the labels
    y_train = train_labels.loc[X_train.index]
    y_val = val_labels.loc[X_val.index] 
    y_test = test_labels.loc[X_test.index]

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_sample_index(
    X_train: any,
    X_val: any, 
    X_test: any, 
):
    """Save the sample index"""
    train_idx = X_train.index
    val_idx = X_val.index
    test_idx = X_test.index

    return train_idx, val_idx, test_idx    

def get_id_sample_index(
    case: int, 
    num_peptide: int,

):
    """Get the sample index"""
    train_file = './outputs/samples/id_train_top_'+str(num_peptide)+'_case_'+str(case)+'.csv'
    train_sample = pd.read_csv(train_file).iloc[:,1:]
    train_idx = list(train_sample['0'])
    val_file = './outputs/samples/id_val_top_'+str(num_peptide)+'_case_'+str(case)+'.csv'
    val_sample = pd.read_csv(val_file).iloc[:,1:]
    val_idx = list(val_sample['0'])
    test_file = './outputs/samples/id_test_top_'+str(num_peptide)+'_case_'+str(case)+'.csv'
    test_sample= pd.read_csv(test_file).iloc[:,1:]
    test_idx = list(test_sample['0'])

    return train_idx, val_idx, test_idx    

def get_ood_sample_index(
    case: int, 
):
    """Get the sample index"""
    train_file = './outputs/samples/ood_train_case_'+str(case)+'.csv'
    train_sample = pd.read_csv(train_file).iloc[:,1:]
    train_idx = list(train_sample['0'])
    val_file = './outputs/samples/ood_val_case_'+str(case)+'.csv'
    val_sample = pd.read_csv(val_file).iloc[:,1:]
    val_idx = list(val_sample['0'])
    test_file = './outputs/samples/ood_test_case_'+str(case)+'.csv'
    test_sample= pd.read_csv(test_file).iloc[:,1:]
    test_idx = list(test_sample['0'])

    return train_idx, val_idx, test_idx  

def save_ood_sample_index(
    case: int,
    outdir: str,
    train_idx: any,
    val_idx: any,
    test_idx: any
    
):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    train_fname='ood_train_'+'case_'+str(case)+'.csv'
    train_fpath=os.path.join(outdir, train_fname)
    pd.DataFrame(train_idx).to_csv(train_fpath)
    val_fname='ood_val_'+'case_'+str(case)+'.csv'
    val_fpath=os.path.join(outdir, val_fname)
    pd.DataFrame(val_idx).to_csv(val_fpath)
    test_fname='ood_test_'+'case_'+str(case)+'.csv'
    test_fpath=os.path.join(outdir, test_fname)
    pd.DataFrame(test_idx).to_csv(test_fpath)

def save_id_sample_index(
    case: int,
    num_peptide: int,
    outdir: str,
    train_idx: any,
    val_idx: any,
    test_idx: any
):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    train_fname='id_train_top_'+str(num_peptide)+'_case_'+str(case)+'.csv' # train set
    train_fpath=os.path.join(outdir, train_fname)
    pd.DataFrame(train_idx).to_csv(train_fpath)
    val_fname='id_val_top_'+str(num_peptide)+'_case_'+str(case)+'.csv' # val set
    val_fpath=os.path.join(outdir, val_fname)
    pd.DataFrame(val_idx).to_csv(val_fpath)
    test_fname='id_test_top_'+str(num_peptide)+'_case_'+str(case)+'.csv' # test set
    test_fpath=os.path.join(outdir, test_fname)
    pd.DataFrame(test_idx).to_csv(test_fpath)

def get_major_class_ratios(
    y_array: any,
):
    """Calculate the ratio between the majority class and the remaining classes."""
    class_counts = pd.DataFrame(pd.DataFrame(y_array).value_counts())
    major_class_ratios = class_counts.iloc[:,0].max()/class_counts

    return major_class_ratios

# def save_sampled_data(
#     case: int,
#     fpath: str,
#     input_df: any,
#     X_train: any, 
#     X_val: any, 
#     X_test: any
# ):
#     """Save the train/val/test sets as csv's"""
    
#     data = {
#     'train': X_train, 
#     'val': X_val, 
#     'test': X_test,
#     }
    
#     for i in range(3):
#         # Remap train, val and test sets' indexes with the index of the input dataframe 
#         df = input_df.iloc[list(list(data.values())[i].index)] # slice rows based on the set index
#         df.insert(loc=0, column='set_idx', value = df.index) # keep the original set index
#         # Save the train, val and test data frames to csv files
#         filename = list(data.keys())[i] + '_case-' + str(case) + '.csv'
#         filepath = os.path.join(fpath, filename)
#         df.reset_index(drop=True).to_csv(filepath)


def binarize_labels(
    y_train: any,
    y_val: any,
    y_test: any
):
    """ Binarize labels in one-vs-all scheme to deal with the multi-class classification case"""

    label_encoding = preprocessing.LabelBinarizer()
    train_binary_labels = label_encoding.fit_transform(y_train)
    val_binary_labels = label_encoding.fit_transform(y_val)
    test_binary_labels = label_encoding.fit_transform(y_test)

    return train_binary_labels, val_binary_labels, test_binary_labels

def get_label_array(
    y_train: any, 
    y_val: any, 
    y_test: any
):
    """Convert label dataframes to numpy arrays"""

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    return y_train, y_val, y_test

def scale_and_reshape_data(
    X_train: any, 
    X_val: any, 
    X_test: any
):
    """Estimate the scale from training data, rescale and reshape all input embedding features"""

    scaler = preprocessing.StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)

    train_input = X_train.reshape(-1, X_train.shape[1], 1)
    val_input = X_val.reshape(-1, X_val.shape[1], 1)
    test_input = X_test.reshape(-1, X_test.shape[1], 1)

    return train_input, val_input, test_input

