library(NAIR)

# set working directory
setwd('/Users/lung/Documents/Projects/deep_learning/PepTCR-Net/')

# directory of input data
dir <- './datasets' 
# your data file path
data_dir <- file.path('./datasets/mira_train_data.csv')
# read data file path
data <- read.csv(data_dir)
# output data file path
distance <- 0
output_file_name <- paste0('network_mira_lv_', distance)
output_dir <- file.path(dir, output_file_name)
# build the sequence network
network <- buildRepSeqNetwork(data = data,
                              seq_col = c("CDR3"),
                              color_nodes_by = c("Peptide"),
                              plot_title = 'MIRA Train Data',
                              dist_cutoff = distance,
                              drop_isolated_nodes = FALSE,
                              output_dir = output_dir,
                              output_name = 'MIRA Train Network',
                              dist_type = "levenshtein",
                              stats_to_include = "all",
                              output_type = "individual",
                              node_stats =TRUE,
                              cluster_stats = TRUE
)
