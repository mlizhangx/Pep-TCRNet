import sys
sys.path.insert(1, '/wynton/home/zhang/leah/PepTCR-Net') # <---insert your directory path
import resources.params as p 
from resources.data_loader import *
from resources.models import *
from resources.utils import *

cases = np.delete(p.CASES, [3,4,27]) # Delete cases #4,#5,#28
for case in cases:
    try:
        # Load the dataset
        ood_df = load_data(p.OOD_DATA_PATH)
        # Remap labels
        data, labels, class2idx, idx2class = remap_labels(ood_df)
        # Split the data
        trainval_subjects, valtest_subjects, train_subjects, val_subjects, test_subjects, trainval_labels, valtest_labels, train_labels, val_labels, test_labels = get_sampled_data(data,labels)
        # Get the number of classes
        num_class=get_number_classes(class2idx) 
        # Load the embeddings
        tessa_df, pe_df, ed_df, ne_df, hla_df, vj_df = load_embeddings(
            fpath_tessa=p.TESSA_OOD_EMB_PATH,
            fpath_pe=p.PE_OOD_EMB_PATH,
            fpath_ed=p.ED_OOD_EMB_PATH,
            fpath_ne=p.NE_OOD_EMB_PATH,
            fpath_hla=p.HLA_OOD_EMB_PATH,
            fpath_vj=p.VJ_OOD_EMB_PATH,
        )
        # Concatenate the data based on case number
        X_train, X_val, X_test, y_train, y_val, y_test = concat_ood_data(
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
            train_labels = train_labels,
            val_labels=val_labels,
            test_labels=test_labels, 
        )
        # Binarize label df
        train_binary_labels, val_binary_labels, test_binary_labels = binarize_labels(y_train, y_val, y_test)
        # Change label df to array 
        y_train, y_val, y_test = get_label_array(y_train, y_val, y_test)
        # Get the sample index
        train_idx, val_idx, test_idx = get_sample_index(X_train, X_val, X_test)
        # Save the sample index
        train_fname='ood_train_'+'case_'+str(case)+'.csv'
        train_fpath=os.path.join(p.SAMPLE_PATH, train_fname)
        pd.DataFrame(train_idx).to_csv(train_fpath)
        val_fname='ood_val_'+'case_'+str(case)+'.csv'
        val_fpath=os.path.join(p.SAMPLE_PATH, val_fname)
        pd.DataFrame(val_idx).to_csv(val_fpath)
        test_fname='ood_test_'+'case_'+str(case)+'.csv'
        test_fpath=os.path.join(p.SAMPLE_PATH, test_fname)
        pd.DataFrame(test_idx).to_csv(test_fpath)
        # Scale and reshape data
        train_input, val_input, test_input = scale_and_reshape_data(X_train, X_val, X_test)
        # Create the Bayesian classifier
        model = create_bayesian_classifier(train_input=train_input, num_class=num_class)
        # Create a checkpoint path
        filename = 'ood_best_model_params'+'_case-' + str(case) + '.h5'
        filepath = os.path.join(p.CLF_CHECKPOINT_PATH, filename)
        # Tune the model
        best_loaded_model = tune_ood_model(
            train_input=train_input,
            val_input=val_input,  
            y_train=y_train, 
            train_binary_labels=train_binary_labels,
            val_binary_labels=val_binary_labels,
            model=model, 
            epochs=p.NUM_EPOCHS,
            learning_rates=p.LEARNING_RATES, 
            batch_sizes=p.BATCH_SIZES, 
            minor_weights=p.OOD_MINOR_WEIGHTS, 
            major_weights=p.OOD_MAJOR_WEIGHTS, 
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
        filename = 'ood_metrics_'+'case_'+str(case)+'.csv'
        filepath = os.path.join(p.EVAL_METRICS_PATH, filename)
        pd.DataFrame.from_dict(performance_stats, orient='index', columns=[case]).to_csv(filepath)
    except (Exception, ValueError, TypeError, NameError, OSError, RuntimeError, SystemError) as e:
        print(e)
        continue
    finally:
        print("Analysis completed!")