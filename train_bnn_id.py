import sys
sys.path.insert(1, '/wynton/home/zhang/leah/PepTCR-Net') # <---insert your directory path
import resources.params as p 
from resources.data_loader import *
from resources.models import *
from resources.utils import *


for num_peptide in p.NUM_PEPTIDES:
    cases = np.delete(p.CASES, [3,4,27]) # Delete cases #4,#5,#28
    for case in cases:
        try:
            # Load the dataset
            id_df = load_data(p.ID_DATA_PATH)
            # Get the top peptides
            top_peptides = get_top_peptides(input_df=id_df, num_peptide=num_peptide)
            # Slice data based on top peptides
            subset_df = slice_data(id_df, top_peptides)     
            # Remap labels
            data, labels, class2idx, idx2class = remap_labels(input_df=subset_df)
            # Get the number of classes
            num_class=get_number_classes(class2idx)
            # Split the data
            trainval_subjects, valtest_subjects, train_subjects, val_subjects, test_subjects, trainval_labels, valtest_labels, train_labels, val_labels, test_labels = get_sampled_data(data,labels)
            # Load the embeddings
            tessa_df, pe_df, ed_df, ne_df, hla_df, vj_df = load_embeddings(
                fpath_tessa=p.TESSA_ID_EMB_PATH,
                fpath_pe=p.PE_ID_EMB_PATH,
                fpath_ed=p.ED_ID_EMB_PATH,
                fpath_ne=p.NE_ID_EMB_PATH,
                fpath_hla=p.HLA_ID_EMB_PATH,
                fpath_vj=p.VJ_ID_EMB_PATH,
            )
            # Concatenate the data based on case number
            X_train, X_val, X_test, y_train, y_val, y_test = concat_id_data(
                case=case, 
                tessa_df=tessa_df, 
                pe_df=pe_df, 
                ed_df=ed_df, 
                ne_df=ne_df, 
                hla_df=hla_df, 
                vj_df=vj_df, 
                train_subjects=train_subjects, 
                val_subjects=val_subjects, 
                test_subjects=test_subjects, 
                train_labels=train_labels,
                val_labels=val_labels,
                test_labels=test_labels, 
            )    
            # Get the ratio between the major class and the remaining classes
            major_class_ratios = get_major_class_ratios(y_train)
            # Binarize label df
            train_binary_labels, val_binary_labels, test_binary_labels = binarize_labels(y_train, y_val, y_test)
            # Change label df to array 
            y_train, y_val, y_test = get_label_array(y_train, y_val, y_test)
            # Get the sample index 
            train_idx, val_idx, test_idx = get_sample_index(X_train, X_test, X_val)
            # Save the sample index
            train_fname='id_train_top_'+str(num_peptide)+'_case_'+str(case)+'.csv'
            train_fpath=os.path.join(p.SAMPLE_PATH, train_fname)
            pd.DataFrame(train_idx).to_csv(train_fpath) # train set indices
            val_fname='id_val_top_'+str(num_peptide)+'_case_'+str(case)+'.csv'
            val_fpath=os.path.join(p.SAMPLE_PATH, val_fname)
            pd.DataFrame(val_idx).to_csv(val_fpath) # val set indices
            test_fname='id_test_top_'+str(num_peptide)+'_case_'+str(case)+'.csv'
            test_fpath=os.path.join(p.SAMPLE_PATH, test_fname)
            pd.DataFrame(test_idx).to_csv(test_fpath) # test set indices
            # Scale and reshape the data input
            train_input, val_input, test_input = scale_and_reshape_data(X_train, X_val, X_test)
            # Create a Bayesian classifier
            model = create_bayesian_classifier(train_input=train_input, num_class=num_class)
            # Create a checkpoint path
            filename = 'id_best_model_params_top-'+ str(num_peptide) + '_case-' + str(case) + '.h5'
            filepath = os.path.join(p.CLF_CHECKPOINT_PATH, filename)
            # Tune the model
            best_loaded_model = tune_id_model(
                train_input=train_input,
                val_input=val_input,  
                y_train=y_train, 
                train_binary_labels=train_binary_labels,
                val_binary_labels=val_binary_labels,
                model=model, 
                epochs=p.NUM_EPOCHS,
                learning_rate=0.001,
                batch_size=64, 
                minor_weights=p.ID_MINOR_WEIGHTS, 
                major_weights=p.ID_MAJOR_WEIGHTS, 
                major_class_ratios=major_class_ratios,
                fpath_ckp=filepath,
            )
            # Final evaluation on the test set
            y_score, y_pred = evaluate_model(test_input=test_input, model=best_loaded_model)
            # Get the performance statistics
            performance_stats = get_performance_stats(
                y_score=y_score,
                y_pred=y_pred,
                y_test=y_test,
            )
            # Save the performance table as a csv 
            filename = 'id_metrics_top_'+str(num_peptide)+'_case_'+str(case)+'.csv'
            filepath = os.path.join(p.EVAL_METRICS_PATH, filename)
            pd.DataFrame.from_dict(performance_stats, orient='index', columns=[case]).to_csv(filepath)
        except (Exception, ValueError, TypeError, NameError, OSError, RuntimeError, SystemError) as e:
            print(e)
            continue
        finally:
            print("Analysis completed!")