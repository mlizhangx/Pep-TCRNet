import os
import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt 
import hdbscan
import umap.umap_ as umap
from sklearn import metrics
from pathlib import Path
from natsort import index_natsorted
from mpl_toolkits.axes_grid1 import make_axes_locatable 

def evaluate_model(test_input, model):
    """Generate the predictions on the loaded best model"""
    
    print('Making prediction on the test set:')
    y_score = model.predict(test_input)
    y_pred = y_score.argmax(axis=1)

    return y_score, y_pred

def get_performance_stats(y_score, y_pred, y_test):
    """Save the model performance metrics and output a csv"""  

    performance_stats = {
        'weighted_auc': [],
        'macro_auc': [],
        'test_accuracy': [],
        'f1_score': [],
        'precision': [],
        'recall': []
    }

    # Append performance metrics to the summary table
    performance_stats['weighted_auc'].append(round(metrics.roc_auc_score(y_test, y_score, multi_class='ovr', average='weighted'), 4))
    performance_stats['macro_auc'].append(round(metrics.roc_auc_score(y_test, y_score, multi_class='ovr', average='macro'), 4))
    performance_stats['test_accuracy'].append(round(metrics.accuracy_score(y_test, y_pred), 4))
    performance_stats['f1_score'].append(round(metrics.f1_score(y_test, y_pred, average='weighted'), 4))
    performance_stats['precision'].append(round(metrics.precision_score(y_test, y_pred, average='weighted'), 4))
    performance_stats['recall'].append(round(metrics.recall_score(y_test, y_pred, average='weighted'), 4))

    return performance_stats

def get_class_metrics(
    idx2class: any,
    y_test: any,
    y_score: any,
    df: any,
):
    # Create a class metrics dataframe 
    class_metrics_stats = {
        'Peptide': list(idx2class.values()),
        'OvR AUC': list(metrics.roc_auc_score(y_test, y_score, multi_class='ovr', average=None)),
    }
    class_df = pd.DataFrame.from_dict(class_metrics_stats, orient='columns')
    # Create a peptide count dataframe
    peptide_index = list(df.Peptide.value_counts().index)
    peptide_values = list(df.Peptide.value_counts().values)
    peptide_percentage = list(df.Peptide.value_counts().values/df.shape[0]*100)
    peptide_df = pd.DataFrame({
        'Peptide': peptide_index,
        'Count': peptide_values,
        'Percentage': peptide_percentage
    })
    # Merge peptide metrics data and peptide count data into one dataframe
    metrics_df = pd.merge(class_df, peptide_df, on='Peptide').sort_values(by=['Count'], ascending=False)   
    
    return metrics_df

def plot_confusion_matrix(
    case, 
    peptide_classes, 
    size, 
    y_pred, 
    y_test, 
    class2idx, 
    idx2class, 
    fpath_figure, 
    figsize, 
    analysis_key
    ):
    """Plot a confusion matrix"""

    # Create a confusion matrix
    cm = pd.DataFrame(metrics.confusion_matrix(y_test, y_pred, normalize = 'true'))
    cm = cm.rename(columns=idx2class, index=idx2class)

    # Create the plot
    threshold = 0.5
    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout()
    im = ax.imshow(cm, cmap='afmhot_r', vmax=1, vmin=0)

    # Change label font size
    plt.xlabel('Predicted', fontsize=18)
    plt.ylabel('True', fontsize=18)

    # Plot colorbar
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=0.5)
    fig.colorbar(im, cax=cax)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(peptide_classes)), labels = class2idx, va='center', fontsize=18)
    ax.set_yticks(np.arange(len(peptide_classes)), labels = class2idx, va='center', fontsize=18)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                    labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Loop over data dimensions and create text annotations.
    for i in range(len(peptide_classes)):
        for j in range(len(peptide_classes)):
            value = round(cm.iloc[i, j], 2)
            if value > threshold:
                text = ax.text(j, i, value,
                        ha="center", va="center", color="white", fontsize=14)
            else: 
                text = ax.text(j, i, value,
                        ha="center", va="center", color="black", fontsize=14)
    
    # Output a pdf
    filename = 'cm_' + str(analysis_key)+ '_case-' +str(case)+'.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, bbox_inches = 'tight', format='pdf') 
    plt.show()

def plot_roc_curves(
    case, 
    fpath_figure, 
    analysis_key,
    figsize, 
    test_binary_labels, 
    y_score, 
    num_class, 
    idx2class,
    bbox_to_anchor,
    ncol
):
    """Plot OvR ROC Curves"""

    # Create a plot
    _, ax = plt.subplots(figsize=figsize)

    # Define color schemes
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_class)]

    # Plot the curves
    for class_id, color in zip(range(num_class), colors):
        metrics.RocCurveDisplay.from_predictions(
        test_binary_labels[:, class_id],
        y_score[:, class_id],
        name = idx2class[class_id],
        color = color,
        ax=ax,
    )

    # Plot the chance level
    plt.plot([0, 1], [0, 1], 'k--', label='chance level (AUC = 0.5)')

    # Adjust plot properties
    plt.axis('square') # make square axes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=18) # change x-axis label font size
    plt.ylabel('True Positive Rate', fontsize=18) # change y-axis label font size
    plt.legend(prop={'size': 18}, 
               bbox_to_anchor=bbox_to_anchor, loc='lower center',
               fancybox=True,ncol=ncol) # change legend font size

    # Output a pdf
    filename = 'roc_' + str(analysis_key)+ '_case-' + str(case)  +'.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, format='pdf', bbox_inches='tight') # save the figure as pdf
    plt.show()

def get_first_peptide(
    test_subjects,
):
    peptide_group = test_subjects.reset_index(drop=False).groupby(['Peptide'])
    peptide_idx_pairs = peptide_group.first()['index']
    first_peptide_class = peptide_idx_pairs.index.to_list()
    first_peptide_idx = list(peptide_idx_pairs.values)

    return peptide_idx_pairs, first_peptide_idx, first_peptide_class

def get_peptide_classes(
    idx2class
):
    peptide_classes = list(idx2class.values())

    return peptide_classes

def get_tcr_peptide_samples(
    test_subjects,
    peptide_classes
):
    """Randomly sample 5 rows from each peptide"""
    samples = pd.DataFrame()
    for i in range(len(peptide_classes)):
        sample = test_subjects[(test_subjects.Peptide == peptide_classes[i])].sample(5)
        samples = pd.concat([samples, sample])
    samples = samples.iloc[:, 0:2]

    return samples

def get_tcr_peptide_one_samples(
    test_subjects,
    peptide_classes
):
    """Randomly sample 5 rows from each peptide"""
    samples = pd.DataFrame()
    for i in range(len(peptide_classes)):
        sample = test_subjects[(test_subjects.Peptide == peptide_classes[i])].sample(1)
        samples = pd.concat([samples, sample])
    samples = samples[["CDR3", "Peptide"]]

    return samples

def get_test_bayesian_dist(
    model,
    data,
    test_input,
    test_idx,
    idx2class,
    peptide_classes,
    analysis_key,
    fpath_baye,
    case,
):
    """"""
    # Run the prediction 200 times
    baye_dist = dict() # initial list of 200 predictions
    for i in range(200):
        y_score = model.predict(test_input)
        baye_dist[i] = y_score

    # Stack 200 prediction scores
    baye_pred_dist = np.stack([baye_dist[key] for key in baye_dist.keys()])
    # Get the median, min and max predictions
    baye_score_med = pd.DataFrame(np.mean(baye_pred_dist, axis = 0))
    baye_score_min = pd.DataFrame(np.min(baye_pred_dist, axis = 0))
    baye_score_max = pd.DataFrame(np.max(baye_pred_dist, axis = 0))
    baye_score_med = baye_score_med.rename(columns=idx2class)
    baye_score_min = baye_score_min.rename(columns=idx2class)
    baye_score_max = baye_score_max.rename(columns=idx2class)
    baye_score_med.columns = 'Mean_' + baye_score_med.columns
    baye_score_min.columns = 'Min_' + baye_score_min.columns
    baye_score_max.columns = 'Max_' + baye_score_max.columns
    baye_score_df = pd.concat([baye_score_med, baye_score_min, baye_score_max], axis=1)
    baye_score_df.index = test_idx
    # Add additional values to finalize the pred prob
    baye_score_df['FirstMaxOfMedProb'] = baye_score_med.max(axis=1)
    baye_score_df['SecMaxOfMedProb'] = baye_score_med.apply(lambda row: row.nlargest(2).values[-1],axis=1)
    baye_score_df['DifOfFirstSecMax'] = baye_score_df['FirstMaxOfMedProb'] - baye_score_df['SecMaxOfMedProb']
    baye_score_df['MaxOfMinProb'] = baye_score_min.max(axis=1)
    baye_score_df['MinOfMaxProb'] = baye_score_max.min(axis=1)
    baye_score_df['PredProb'] = baye_score_df['FirstMaxOfMedProb']
    baye_score_df['PredPeptide'] = baye_score_med.idxmax(axis=1).str.replace('Mean_','')
    baye_score_df['Confidence'] = "High"

    threshold = ((baye_score_df['FirstMaxOfMedProb'] <=0.8) | (baye_score_df['DifOfFirstSecMax'] <= 0.5))
    baye_score_df.loc[threshold, 'Confidence'] = "Low"

    baye_score_df['ExactPeptide'] = data['Peptide']
    baye_score_df['NewExactPeptide'] = baye_score_df['ExactPeptide']
    baye_score_df.loc[~baye_score_df['ExactPeptide'].isin(peptide_classes), 'NewExactPeptide'] = 'Others'
    baye_score_df['TCR'] = data['CDR3']
    extra_data = data[['V','J','HLA-A','HLA-B','HLA-C']]
    baye_score_df = pd.concat([baye_score_df, extra_data], axis=1)

    # Save the baye_score distribution
    filename = 'baye_mean_'+ str(analysis_key) +'_case-' +str(case) +'.csv'
    filepath = os.path.join(fpath_baye, filename)
    baye_score_df.to_csv(filepath)

    return baye_dist, baye_score_df

def get_bayesian_dist(
    model,
    test_input,
    test_idx,
    idx2class,
    analysis_key,
    fpath_baye,
    case
):
    """"""

    # Run the prediction 200 times
    baye_dist = dict() # initial list of 200 predictions
    for i in range(200):
        y_score = model.predict(test_input)
        baye_dist[i] = y_score
        
    # Find the mean of probabilities
    baye_score = baye_dist[0] # initial baye_score value df
    for j in range(1,200):
        baye_score = baye_score + baye_dist[j]
    baye_score=baye_score/200 # mean
    # Stack 200 prediction scores
    baye_pred_dist = np.stack([baye_dist[key] for key in baye_dist.keys()])
    # Get the median, min and max predictions
    baye_score_df = pd.DataFrame(np.median(baye_pred_dist, axis = 0))
    baye_score_df = baye_score_df.rename(columns=idx2class)
    baye_score_df.index = test_idx

    # Save the baye_score distribution
    filename = 'baye_median_'+ str(analysis_key) +'_case-' +str(case) +'.csv'
    filepath = os.path.join(fpath_baye, filename)
    baye_score_df.to_csv(filepath)

    return baye_dist, baye_score, baye_score_df

def get_violin_data(
    peptide_idx: any,
    test_idx:any,
    idx2class: any,
    baye_dist: any,
):
    violin_data = {}
    # Per peptide index, generate prediction for this same data point 200 times
    for idx in peptide_idx:
        peptide_dist = pd.DataFrame()
        for i in range(200):
            # Get the i-th prediction of the test set  
            baye_pred = pd.DataFrame(baye_dist[i]) 
            # Map test set index to the prediction
            baye_pred.index = test_idx
            # Insert the i-th prediction of a single TCR at idx to the peptide distribution
            peptide_dist = pd.concat([peptide_dist, baye_pred.loc[[idx]]], ignore_index=True)
        peptide_dist.rename(columns=idx2class, inplace=True) # get columns as peptides
        violin_data[idx] = peptide_dist

    return violin_data

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def plot_full_violin_plots(
    case: int,
    analysis_key: str,
    fpath_figure: str,
    num_class: any,
    peptide_classes: any,
    samples: any,
    baye_score_df: any,
    peptide_idx: any,
    violin_data: any,
):
    """Plot multiple violin plots in one panel"""
    fig, axs = plt.subplots(nrows=num_class, ncols=5, figsize=(24, 16), sharex=True, sharey=True)
    fig.tight_layout(h_pad=6)
    # Set params for violin plots
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(num_class)]
    cols = [0,1,2,3,4]*num_class
    num_plot = 5*num_class
    # Set params for ood data and id data
    if analysis_key=='ood': # OOD data
        rows = [0]*5 +[1]*5 + [2]*5 + [3]*5 
        ticks = [1,2,3,4]
        last_ax = axs[3,0]
    else: # ID data
        rows = [0]*5 +[1]*5 + [2]*5 + [3]*5 + [4]*5
        ticks = [1,2,3,4,5]
        last_ax = axs[4,0]
    # Plot the violin plots
    for i,j,k in zip(range(num_plot), rows, cols):
        # Create data to plot
        data = [violin_data[peptide_idx[i]][peptide_classes[col]].values for col in range(num_class)]
        # Define a violin plot
        plot = axs[j, k].violinplot(
            dataset=data,
            widths=0.3,
            showmeans=False, 
            showextrema=False, 
            showmedians=False
        )
        # Fill violin plots with colors
        for pc, color in zip(plot['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.7)
            pc.set_linewidth(3)
        # Plot whiskers
        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        # Plot scatter for mean
        inds = np.arange(1, len(medians) + 1)
        axs[j, k].scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
        axs[j, k].vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=1)
        axs[j, k].vlines(inds, whiskers_min, whiskers_max, colors='k', linestyle='-', lw=1)
        # Set titles for subplots
        tcr = samples.loc[peptide_idx[i]].CDR3
        true_label = samples.loc[peptide_idx[i]].Peptide
        pred_label = baye_score_df.idxmax(axis=1).loc[peptide_idx[i]]
        axs[j, k].set_title('TCR: ' + tcr + '\n' + 'True label: ' + true_label + '\n' + 'Predicted label: ' + pred_label, fontsize=14)
        axs[j, 0].set_ylabel('Posterior Predicted Probability', fontsize=13)
    # Set ticks as peptides    
    last_ax.set_xticks(ticks=ticks, labels=peptide_classes)

    # Rotate the x tick labels and set their alignment
    if analysis_key=='ood':
        plt.setp(axs[3,0].get_xticklabels(), 
                rotation=90, 
                ha='right',
                rotation_mode='anchor') 
        plt.setp(axs[3,1].get_xticklabels(), 
                rotation=90, 
                ha='right',
                rotation_mode='anchor')
        plt.setp(axs[3,2].get_xticklabels(), 
                rotation=90, 
                ha='right',
                rotation_mode='anchor')
        plt.setp(axs[3,3].get_xticklabels(), 
                rotation=90, 
                ha='right',
                rotation_mode='anchor')
        plt.setp(axs[3,4].get_xticklabels(), 
                rotation=90, 
                ha='right',
                rotation_mode='anchor')

    # Output a figure pdf  
    filename = 'violin_' + str(analysis_key) +'_'+ 'case-'+ str(case) +  '.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, bbox_inches = 'tight', format='pdf') 
    

def plot_compact_violin_plots(
    case: int,
    analysis_key: str,
    fpath_figure: str,
    num_class: any,
    peptide_classes: any,
    samples: any,
    baye_score_df: any,
    peptide_idx: any,
    violin_data: any,
    figsize: any,
    rotation_angle: int,
):
    """Plot multiple violin plots in one panel"""
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True, sharey=True)
    fig.tight_layout(h_pad=6)
    # Set params for violin plots
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(num_class)]
    cols = [0,1,0,1]
    num_plot = 4
    # Set params for ood data and id data
    rows = [0]*2 +[1]*2 
    ticks = [1,2,3,4]
    last_ax = axs[1,0]

    # Plot the violin plots
    for i,j,k in zip(range(num_plot), rows, cols):
        # Create data to plot
        data = [violin_data[peptide_idx[i]][peptide_classes[col]].values for col in range(num_class)]
        # Define a violin plot
        plot = axs[j, k].violinplot(
            dataset=data,
            widths=0.6,
            showmeans=False, 
            showextrema=False, 
            showmedians=False
        )
        # Fill violin plots with colors
        for pc, color in zip(plot['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(1)
            pc.set_linewidth(3)
        # Plot whiskers
        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        # Plot scatter for mean
        inds = np.arange(1, len(medians) + 1)
        axs[j, k].scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
        axs[j, k].vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=1)
        axs[j, k].vlines(inds, whiskers_min, whiskers_max, colors='k', linestyle='-', lw=1)
        # Set titles for subplots
        tcr = samples.loc[peptide_idx[i]].CDR3
        true_label = samples.loc[peptide_idx[i]].Peptide
        pred_label = baye_score_df.idxmax(axis=1).loc[peptide_idx[i]]
        axs[j, k].set_title('TCR: ' + tcr + '\n' + 'True label: ' + true_label + '\n' + 'Predicted label: ' + pred_label, fontsize=14)
        axs[j, 0].set_ylabel('Posterior Predicted Probability', fontsize=13)
    # Set ticks as peptides    
    last_ax.set_xticks(ticks=ticks, labels=peptide_classes)

    # Rotate the x tick labels and set their alignment
    if analysis_key=='ood':
        plt.setp(axs[1,0].get_xticklabels(), 
                rotation=rotation_angle, 
                ha='right',
                rotation_mode='anchor') 
        plt.setp(axs[1,1].get_xticklabels(), 
                rotation=rotation_angle, 
                ha='right',
                rotation_mode='anchor')
    
    # Output a pdf  
    filename = 'violin_four_' + str(analysis_key) +'_'+ 'case-'+ str(case) +  '.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, bbox_inches = 'tight', format='pdf') 
    plt.show()

def plot_euro_violin_plots(
    case: int,
    analysis_key: str,
    fpath_figure: str,
    num_class: any,
    peptide_classes: any,
    samples: any,
    baye_score_df: any,
    peptide_idx: any,
    violin_data: any,
    figsize: any,
    rotation_angle: int,
):
    """Plot multiple violin plots in one panel"""
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=figsize, sharex=True, sharey=True)
    fig.tight_layout(h_pad=4)
    # Set params for violin plots
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(num_class)]
    # cols = [0,1,0,1,0]
    num_plot = 5
    # Set params for ood data and id data
    # rows = [0]*2 +[1]*2 +[0]
    ticks = [1,2,3,4,5]
    # last_ax = axs[2,0]

    # Plot the violin plots
    for i in range(num_plot):
        # Create data to plot
        data = [violin_data[peptide_idx[i]][peptide_classes[col]].values for col in range(num_class)]
        # Define a violin plot
        plot = axs[i].violinplot(
            dataset=data,
            widths=0.6,
            showmeans=False, 
            showextrema=False, 
            showmedians=False
        )
        # Fill violin plots with colors
        for pc, color in zip(plot['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(1)
            pc.set_linewidth(3)
        # Plot whiskers
        quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        # Plot scatter for mean
        inds = np.arange(1, len(medians) + 1)
        axs[i].scatter(inds, medians, marker='o', color='white', s=10, zorder=3)
        axs[i].vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=1)
        axs[i].vlines(inds, whiskers_min, whiskers_max, colors='k', linestyle='-', lw=1)
        # Set titles for subplots
        tcr = samples.loc[peptide_idx[i]].CDR3
        true_label = samples.loc[peptide_idx[i]].Peptide
        pred_label = baye_score_df.drop(columns=['Confidence']).idxmax(axis=1).loc[peptide_idx[i]]
        confidence = baye_score_df.Confidence.loc[peptide_idx[i]]
        axs[i].set_title('TCR: ' + tcr + '\n' +
                        'Exact-matched label: ' + true_label + '\n' + 
                        'Predicted label: ' + pred_label + '\n' +
                        'Confidence: ' + confidence,
                        fontsize=14)
        axs[i].set_xticks(ticks=ticks, labels=peptide_classes)
        # Rotate the x tick labels and set their alignment
        plt.setp(axs[i].get_xticklabels(), 
            rotation=rotation_angle, 
            ha='right',
            rotation_mode='anchor')
    axs[0].set_ylabel('Posterior Predicted Probability', fontsize=13)

    # Output a pdf  
    filename = 'violin_five_' + str(analysis_key) +'_'+ 'case-'+ str(case) +  '.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, bbox_inches = 'tight', format='pdf') 
    plt.show()

def get_auc_line_data(
    fpath_metrics: str,
    cases: any
):    
    line_data = {}
    for case in cases:
        # Read multiple files
        metric_files = Path(fpath_metrics).glob('id_class_metrics_top-*_case-'+str(case)+'.csv')
        dfs = []
        for f in metric_files:
            df = pd.read_csv(f, index_col=0)
            dfs.append(df['OvR AUC'])
        # Concatenate multiple files to one dataframe
        data = pd.concat(dfs, axis=1).iloc[:5,:]
        data.columns =  ['5 classes', '10 classes', '15 classes', '20 classes']
        data = data.reset_index()
        data = data.T
        data.columns = data.iloc[0]
        data = data[1:]
        data = data.reset_index()
        data = data.rename(columns={'index':'Class'})
        data = data.sort_values(
            by="Class",
            key=lambda x: np.argsort(index_natsorted(data["Class"]))
        )
        # Append the data 
        line_data[case] = data

    return line_data

def plot_auc_line_graph(
    cases: any,
    titles: any,
    peptide_classes: any,
    line_data: any,
    fpath_figure: str
):

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 7), sharex=True, sharey=True)
    fig.tight_layout(h_pad=4, w_pad=6)
    rows = [0]*2+[1]*2
    cols = [0,1]*2
    xticks = np.arange(0,4,1).tolist()
    yticks = np.arange(0.5,1.1, 0.05)
    for case, row, col in zip(cases, rows, cols):
        for peptide in peptide_classes:
            axs[row, col].plot('Class', peptide, data=line_data[case], linestyle='-', marker='o')
        axs[row, col].set_title(titles[case], fontsize=18)
        # axs[1, col].set_xlabel('Number of Classes', fontsize=18)
        axs[row, 0].set_ylabel('OvR AUC', fontsize=18)
        axs[1, col].set_xticks(
            ticks = xticks,
            labels = line_data[case].Class.tolist(), 
            fontsize=18)
        axs[row, 0].set_yticks(
            ticks = yticks, 
            labels = ['%1.2f' % val for val in yticks],
            fontsize=18)
    plt.legend(peptide_classes, bbox_to_anchor=(-0.08, -0.35), loc='center', fancybox=True, ncol=5, prop={'size':18})
    # Output a pdf  
    filename = 'linescatter_auc.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, bbox_inches = 'tight', format='pdf') 
    plt.show()

# def get_ovr_auc_line_data(
#     fpath_metrics: str,
#     cases: any
# ):    
    

#     return line_data

# def plot_ovr_auc_line_graph(
#     cases: any,
#     titles: any,
#     peptide_classes: any,
#     line_data: any,
#     fpath_figure: str
# ):

#     fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(14, 7), sharex=True, sharey=True)
#     fig.tight_layout(h_pad=4, w_pad=6)
#     rows = [0]*3+[1]*3
#     cols = [0,1,2]*2
#     xticks = np.arange(0,4,1).tolist()
#     yticks = np.arange(0.5,1.1, 0.05)
#     for case, row, col in zip(cases, rows, cols):
#         for peptide in peptide_classes:
#             axs[row, col].plot('Class', peptide, data=line_data[case], linestyle='-', marker='o')
#         axs[row, col].set_title(titles[case], fontsize=18)
#         # axs[1, col].set_xlabel('Number of Classes', fontsize=18)
#         axs[row, 0].set_ylabel('OvR AUC', fontsize=18)
#         axs[1, col].set_xticks(
#             ticks = xticks,
#             labels = line_data[case].Class.tolist(), 
#             fontsize=18)
#         axs[row, 0].set_yticks(
#             ticks = yticks, 
#             labels = ['%1.2f' % val for val in yticks],
#             fontsize=18)
#     plt.legend(peptide_classes, bbox_to_anchor=(-0.08, -0.35), loc='center', fancybox=True, ncol=5, prop={'size':18})
#     # Output a pdf  
#     filename = 'linescatter_auc.pdf'
#     filepath = os.path.join(fpath_figure, filename)
#     plt.savefig(filepath, bbox_inches = 'tight', format='pdf') 
#     plt.show()

def get_acc_line_data(
    num_peptides:any,
    fpath_metrics: str,
    y_labels: any,
    cases: any
):
    data = pd.DataFrame()
    for i in range(len(num_peptides)):
        files = Path(fpath_metrics).glob('id_metrics_top-'+str(num_peptides[i])+'_case_*.csv')
        dfs = list()
        for f in files:
            df = pd.read_csv(f, index_col=0)
            dfs.append(df.iloc[0])
        col = pd.concat(dfs, axis=0)
        data.insert(i, str(num_peptides[i]) + ' classes', col, True)
    data = data.loc[[str(case) for case in cases]].T
    data['idx'] = data.index
    data = data.rename(y_labels, axis='columns')
    
    return data

def get_wauc_line_data(
    num_peptides:any,
    fpath_metrics: str,
    y_labels: any,
    cases: any
):
    data = pd.DataFrame()
    for i in range(len(num_peptides)):
        files = Path(fpath_metrics).glob('id_metrics_top-'+str(num_peptides[i])+'_case_*.csv')
        dfs = list()
        for f in files:
            df = pd.read_csv(f, index_col=0)
            dfs.append(df.iloc[1])
        col = pd.concat(dfs, axis=0)
        data.insert(i, str(num_peptides[i]) + ' classes', col, True)
    data = data.loc[[str(case) for case in cases]].T
    data['idx'] = data.index
    data = data.rename(y_labels, axis='columns')
    
    return data

def plot_auc_acc_line_graph(
    cases: any,
    y_labels: any,
    acc_line_data: any,
    auc_line_data: any,
    figsize: any,
    fpath_figure: str, 
    ncol:int,
    bbox_to_anchor: any
):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.tight_layout(h_pad=4, w_pad=6)
    yticks = {
        0: np.arange(0.0,1.05, 0.1),
        1: np.arange(0.5,1.05, 0.1)
    }

    # Left panel - Accuracy
    for case in cases:
        axs[0].plot('idx', y_labels[str(case)], data=acc_line_data, linestyle='-', marker='o')
    axs[0].set_ylabel('Accuracy', fontsize=18)
    axs[0].set_yticks(
        ticks = yticks[0], 
        labels = ['%1.1f' % val for val in yticks[0]],
        fontsize=18)
    axs[0].set_xticks(
        ticks = np.arange(0,4,1).tolist(),
        labels = auc_line_data.index.tolist(), 
        fontsize=18)
    # Right panel - Weighted AUC
    for case in cases:
        axs[1].plot('idx', y_labels[str(case)], data=auc_line_data, linestyle='-', marker='o')
    axs[1].set_ylabel('Weighted OvR AUC', fontsize=18)
    axs[1].set_yticks(
        ticks = yticks[1], 
        labels = ['%1.1f' % val for val in yticks[1]],
        fontsize=18)
    axs[1].set_xticks(
        ticks = np.arange(0,4,1).tolist(),
        labels = auc_line_data.index.tolist(), 
        fontsize=18)
    
    plt.legend(list(y_labels.values()), bbox_to_anchor=bbox_to_anchor, loc='center',
            fancybox=True,ncol=ncol, prop={'size':18})

    # Output a pdf  
    filename = 'linescatter_auc_acc.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, bbox_inches = 'tight', format='pdf') 
    plt.show()

def plot_acc_line_graph(
    acc_line_data: any,
    y_labels: any,
    fpath_figure: str,
    ncol: int,
    figsize: any
):
    _, ax = plt.subplots(figsize=figsize)
    plt.plot('idx', list(y_labels.values())[0], data=acc_line_data, linestyle='-', marker='o')
    plt.plot('idx', list(y_labels.values())[1], data=acc_line_data, linestyle='-', marker='o')
    plt.plot('idx', list(y_labels.values())[2], data=acc_line_data, linestyle='-', marker='o')
    plt.plot('idx', list(y_labels.values())[3], data=acc_line_data, linestyle='-', marker='o')
    plt.plot('idx', list(y_labels.values())[4], data=acc_line_data, linestyle='-', marker='o')
    plt.plot('idx', list(y_labels.values())[5], data=acc_line_data, linestyle='-', marker='o')
    plt.plot('idx', list(y_labels.values())[6], data=acc_line_data, linestyle='-', marker='o')
    plt.plot('idx', list(y_labels.values())[7], data=acc_line_data, linestyle='-', marker='o')


    plt.legend(list(y_labels.values()), bbox_to_anchor=(0.5, -0.25), loc='center',
            fancybox=True,ncol=ncol, prop={'size':18})
    plt.xlabel('Number of Classes', fontsize=18)
    plt.ylabel('Accuracy', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    # Output a pdf  
    filename = 'linescatter_acc.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, bbox_inches = 'tight', format='pdf') 
    plt.show()

def plot_heatmap(data, vmin, vmax, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", shrink=1, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw ={
        "shrink": shrink
        }

    # Plot the heatmap
    im = ax.imshow(data, **kwargs, vmin=vmin, vmax=vmax)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True, labelsize=14)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", va='center',
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5,minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5,minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), fontsize=15.5, **kw)
            texts.append(text)

    return texts

def plot_single_emb_umap(
    figsize: any,
    analysis_key:str,
    fpath_figure:str,
    X_test: any,
    y_test: any,
    y_pred: any,
    idx2class: any,
    peptide_classes:any,
    size: float,
    markersize: float,
    alpha: float,
    n_neighbors: int,
    min_dist: float,
    ncol: int,
    fontsize: int,
    left: float,
    right: float,
    bbox_to_anchor: any,
    loc: str
):
    """Plot a single UMAP"""
    fig, axs = plt.subplots(figsize =figsize, sharex=True, sharey=True)
    fig.tight_layout(h_pad=2)
    test_embs = umap.UMAP(
            random_state = 42, 
            n_neighbors=n_neighbors, 
            # min_dist=min_dist).fit_transform(X_test)    
            min_dist=min_dist).fit_transform(X_test, y_pred)
        
    color_label_set = set(zip(list(idx2class.keys()), list(idx2class.values())))
    
    scatter_plot = axs.scatter(
        *test_embs.T,
        c=y_test,
        cmap='Spectral',
        s=size,
        alpha=alpha
    )
    
    # calculate cluster accuracy
    labels = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=500,
        ).fit_predict(test_embs)                        
    cluster_score = metrics.cluster.adjusted_rand_score(y_test, labels)
    textstr = 'Rand Index = ' + str(round(cluster_score, 2))
    props = dict(boxstyle='round', facecolor = 'w', alpha=0.5)
    axs.text(left, right, textstr,horizontalalignment='right',
    verticalalignment='bottom', transform=axs.transAxes, fontsize=fontsize, bbox=props)

    for item in (axs.get_xticklabels() + axs.get_yticklabels()):
        item.set_fontsize(fontsize)

    # axs.set_title('ED + NE + HLA + VJ', fontsize=fontsize)
    
    axs.legend(
        handles=[plt.plot([],color=scatter_plot.get_cmap()(scatter_plot.norm(c)),ls="", marker="o", markersize=markersize)[0] for c,_ in color_label_set],
        labels=peptide_classes, 
        bbox_to_anchor=bbox_to_anchor, 
        loc=loc,
        fancybox=True,
        ncol=ncol, 
        prop={'size':40}
    )
    # Output a pdf  
    filename = 'umap_single_spv_all_' + str(analysis_key)+'.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, dpi=300, bbox_inches = 'tight', format='pdf') 
    plt.show()


def plot_emb_umap(
    cases: any,
    combos: any,
    figsize: any,
    analysis_key:str,
    fpath_figure:str,
    umap_data: any,
    idx2class: any,
    peptide_classes:any,
    size: float,
    markersize: float,
    alpha: float,
    n_neighbors: int,
    min_dist: float,
    ncol: int,
    fontsize: int,
    kw: str,
    left: float,
    right: float
):
    """"""
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize =figsize, sharex=True, sharey=True)
    fig.tight_layout(h_pad=6)
    rows = [0]*2 + [1]*2 + [2]*2
    cols = [0,1] * 3
    for case,combo,j,k in zip(cases, combos, rows, cols):

        X_test = umap_data[case]['X_test']
        y_test = umap_data[case]['y_test']
        y_pred = umap_data[case]['y_pred']


        if kw == 'unspv':
            test_embs = umap.UMAP(
                random_state = 42, 
                n_neighbors=n_neighbors, 
                min_dist=min_dist).fit_transform(X_test)  
        elif kw == 'spv':
            test_embs = umap.UMAP(
                random_state = 42, 
                n_neighbors=n_neighbors, 
                min_dist=min_dist).fit_transform(X_test, y=y_pred)    
        else:
            raise ValueError
        
        color_label_set = set(zip(list(idx2class.keys()), list(idx2class.values())))
        
        scatter_plot = axs[j,k].scatter(
            *test_embs.T,
            c=y_test,
            cmap='Spectral',
            s=size,
            alpha=alpha
        )
        
        # calculate cluster accuracy
        labels = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=500,
            ).fit_predict(test_embs)                        
        cluster_score = metrics.cluster.adjusted_rand_score(y_test, labels)
        textstr = 'Rand Index = ' + str(round(cluster_score, 2))
        props = dict(boxstyle='round', facecolor = 'w', alpha=0.5)
        axs[j,k].text(left, right, textstr,horizontalalignment='left',
        verticalalignment='bottom', transform=axs[j,k].transAxes, fontsize=20, bbox=props)

        for item in (axs[j,k].get_xticklabels() + axs[j,k].get_yticklabels()):
            item.set_fontsize(fontsize)

        axs[j, k].set_title(combo, fontsize=fontsize)
        
        
    fig.legend(
        handles=[plt.plot([],color=scatter_plot.get_cmap()(scatter_plot.norm(c)),ls="", marker="o", markersize=markersize)[0] for c,_ in color_label_set],
        labels=peptide_classes, 
        bbox_to_anchor=(0.5, -0.05), 
        loc='center',
        fancybox=True,
        ncol=ncol, 
        prop={'size':26}
    )
    # Output a pdf  
    filename = 'umap_' + str(kw) +'_' + str(analysis_key)+'.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, dpi=300, bbox_inches = 'tight', format='pdf') 
    plt.show()



def plot_all_emb_umap(
    cases: any,
    combos: any,
    figsize: any,
    analysis_key:str,
    fpath_figure:str,
    umap_data: any,
    idx2class: any,
    peptide_classes:any,
    size: float,
    alpha: float,
    n_neighbors: int,
    min_dist: float,
    ncol: int,
    fontsize: int
):
    """"""
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize =figsize, sharex=True, sharey=True)
    fig.tight_layout(h_pad=6)
    rows = [0]*2 + [1]*2 + [2]*2
    cols = [0,1] * 3
    for case,combo,j,k in zip(cases, combos, rows, cols):

        X_test = umap_data[case]['X_test']
        y_test = umap_data[case]['y_test']
        y_pred = umap_data[case]['y_pred']


        if k == 0:
            test_embs = umap.UMAP(
            random_state = 42, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist).fit_transform(X_test, 
                                             y=y_pred
                                             )
        else:
            test_embs = umap.UMAP(
                random_state = 42, 
                n_neighbors=n_neighbors, 
                min_dist=min_dist).fit_transform(X_test)
        
        color_label_set = set(zip(list(idx2class.keys()), list(idx2class.values())))
        
        scatter_plot = axs[j,k].scatter(
            *test_embs.T,
            c=y_test,
            cmap='Spectral',
            s=size,
            alpha=alpha
        )

        # calculate cluster accuracy
        labels = hdbscan.HDBSCAN(
            min_samples=10,
            min_cluster_size=500,
            ).fit_predict(test_embs)                        
        cluster_score = metrics.cluster.adjusted_rand_score(y_test, labels)
        textstr = 'Rand Index = ' + str(round(cluster_score, 2))
        props = dict(boxstyle='round', facecolor = 'w', alpha=0.5)
        axs[j,k].text(0.68, 0.95, textstr, transform=axs[j,k].transAxes, fontsize=20,
        verticalalignment='top', bbox=props)

        axs[j, k].set_title(combo, fontsize=fontsize)

        for item in (axs[j,k].get_xticklabels() + axs[j,k].get_yticklabels()):
            item.set_fontsize(fontsize)
        
    fig.legend(
        handles=[plt.plot([],color=scatter_plot.get_cmap()(scatter_plot.norm(c)),ls="", marker="o", markersize=size/12)[0] for c,_ in color_label_set],
        labels=peptide_classes, 
        bbox_to_anchor=(0.5, -0.05), 
        loc='center',
        fancybox=True,
        ncol=ncol, 
        prop={'size':26}
    )

    # Output a pdf  
    filename = 'umap_all_' + str(analysis_key)+'.pdf'
    filepath = os.path.join(fpath_figure, filename)
    plt.savefig(filepath, dpi=300, bbox_inches = 'tight', format='pdf') 
    plt.show()