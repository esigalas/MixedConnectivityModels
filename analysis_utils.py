from collections import defaultdict 
import numpy as np
from itertools import combinations,product
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt

def get_trialwise_mean(dataset, labels):
    # calculate mean across trials for each stimulus location for all neurons
    # resulting in a train set of size (number_of_target_locations X bin_size X Neurons)
    target_locs = len(np.unique(labels))
    if len(dataset.shape) == 3:
        trialwise_average = np.zeros((target_locs, dataset.shape[1], dataset.shape[2]))
        for i, stim_loc in enumerate(np.unique(labels)):
            trialwise_average[i] = np.mean(dataset[labels==stim_loc, :, :], axis=0)
        return trialwise_average
    elif len(dataset.shape) == 2:
        trialwise_average = np.zeros((target_locs, dataset.shape[1]))
        for i, stim_loc in enumerate(np.unique(labels)):
            trialwise_average[i] = np.mean(dataset[labels==stim_loc, :], axis=0)
        return trialwise_average

def get_group_pci(group):
    # expect a numpy array as input with shape (number of neurons in the group X neural activities)
    # will calculate the Pearson’s correlation-coefficient for each pair of neural activities
    combinations_list = list(combinations(group,2))
    
    for pair_idx, neuron_pair in enumerate(combinations_list):
        pair_pci = stats.pearsonr(neuron_pair[0], neuron_pair[1])

        yield pair_pci[0], pair_pci[1]

def get_group_pci_product(group1, group2, names=None):
    # expect a numpy array as input with shape (number of neurons in the group X neural activities)
    # will calculate the Pearson’s correlation-coefficient for each pair of neural activities between the two groups
    product_list = list(product(group1,group2))
    if names is not None:
        name_list = list(product(names[0], names[1]))
        print(len(name_list))
        print(len(product_list))

    for ii, neuron in enumerate(product_list):
        pair_pci = stats.pearsonr(neuron[0], neuron[1])
#         if np.isnan(pair_pci[0]):
#             print(neuron_pair[0])
#             print(neuron_pair[1])
        if names is None:
            yield pair_pci[0], pair_pci[1]
        else:
            yield pair_pci[0], pair_pci[1], name_list[ii]
def find_selective_locations(neuron1, neuron2, selective_dict):
    for loc in selective_dict.keys():
        if neuron1 in selective_dict[loc] and neuron2 in selective_dict[loc]:
            yield loc

def find_non_selective_locations(neuron1, neuron2, selective_dict):
    for loc in selective_dict.keys():
        if neuron1 not in selective_dict[loc] and neuron2 not in selective_dict[loc]:
            yield loc

def check_cond(W):
    W_diag_only = np.diag(np.diag(W))
    W_diag_pos_only = W_diag_only.copy()
    W_diag_pos_only[W_diag_pos_only < 0] = 0.0
    W_abs_cond = np.abs(W - W_diag_only) + W_diag_pos_only
    max_eig_abs_cond = np.max(np.real(np.linalg.eigvals(W_abs_cond)))
    print(max_eig_abs_cond)
    if max_eig_abs_cond < 1:
        return True
    else:
        return False
    
def performance_measure(trial_batch, trial_y, output_mask, output, epoch, losses, verbosity):
    """
    Function that calculates the performance of the model on the given trial batch and the output of the model
    *task should be initialized beforehand
    """
    return drt.accuracy_function(trial_y, output, output_mask)

def zscore_dataset(dataset, dataset_labels):
    # expect a 3D array (trials X timebins X neurons)
    # return the zscored array for each neuron per location
    zscored_dataset = np.zeros(dataset.shape)
    uniq_labels = np.unique(dataset_labels)
    for neuron in np.arange(zscored_dataset.shape[-1]):
#         print(neuron)
        for label in uniq_labels:
            location_trials = np.where(dataset_labels==label)[0]
            zscored_dataset[location_trials, :, neuron] = stats.zscore(dataset[location_trials, :, neuron], axis=0)

    return zscored_dataset    

def get_neuron_selectivity(dataset, labels, method='mean-threshold', anova_significance_p=0.01, ttest_significance_p=0.01, prestim_dataset=None):
    # expect a numpy array as input with shape (trials X timebins X Neurons)
    # return dictionary with a list of the selective neurons for each stimulus
    
    selective_neurons = defaultdict(list)
    # set a dictionary for label mapping in case labels are not sequential
    label_mapping = {}
    for k_label, uniq_label in enumerate(np.unique(labels)):
        label_mapping[k_label] = uniq_label

    # perform oneway anova to select neurons with significant variance between their locations activities
    # get the f oneway anova stats (f statistic, p value) for all neurons
    anova_stats = np.array([anova_stats for anova_stats in oneway_anova(dataset, labels)])
    # pick the indices of the neurons that have anova p value less than a significance level
    anova_indices = np.where(anova_stats[:,1] < anova_significance_p)[0]
    # pick only these indices from the whole dataset of neurons
    anova_dataset = dataset[:,:,anova_indices]
#     print(anova_dataset.shape)
    
    # calculate mean across trials based on stimulus
    avg_trials = get_trialwise_mean(anova_dataset, labels)
    # calculate the mean activity of each neuron
    neurons_stim_activities = np.mean(avg_trials, axis=1).T
    
    # if prestim data were provided filter them based on the anova test as well and
    # calculate the mean activitiy of each neuron during prestim
    if prestim_dataset is not None: 
        anova_prestim_dataset = prestim_dataset[:,:,anova_indices]
#         print(anova_prestim_dataset.shape)
        
        avg_trials_prestim = get_trialwise_mean(anova_prestim_dataset, labels)
        neurons_stim_activities_prestim = np.mean(avg_trials_prestim, axis=1).T

    # check if the method given is in the list of methods
    assert(method in ['mean-threshold', 't-test-locations'])
    if method == 'mean-threshold':
        for n, neuron_activities in enumerate(neurons_stim_activities):
            # select the stimuli where the neuron activity is higher than the average activity between stimulus for that neuron
            neuron_stim_selectivity = np.where(neuron_activities > np.mean(neuron_activities))[0]
            for stim in neuron_stim_selectivity:
                selective_neurons[stim].append(anova_indices[n])
            # select the stimuli with the highest activity
    #         neuron_stim_selectivity = np.argmax(neuron_activities)
    #         selective_neurons[neuron_stim_selectivity].append(anova_indices[n])
    if method == 't-test-locations':
        # perform a t-test between all location activities and the lowest activity and pick only those with high significance
        for n in np.arange(anova_dataset.shape[-1]):
            # find the location with the minimum average activity for that neuron
            min_loc = np.argmin(neurons_stim_activities[n,:])
#             print(neurons_stim_activities[n,:])
            t_test_p_values = t_test(anova_dataset[:,:,n], labels, min_loc)
            t_test_stim_selectivity = np.where(t_test_p_values < ttest_significance_p)[0]
#             print(t_test_stim_selectivity)
            for stim in t_test_stim_selectivity:
#                 print(neurons_stim_activities[n,stim])
                if prestim_dataset is not None:
                    if neurons_stim_activities[n,stim] > neurons_stim_activities_prestim[n, stim]:
                        selective_neurons[stim].append(anova_indices[n])
                else:
                    selective_neurons[stim].append(anova_indices[n])

    final_selective_neurons = {}
    # return all the selective neurons for each mapped stimulus
    for key in selective_neurons.keys():
        final_selective_neurons[label_mapping[key]] = np.array(selective_neurons[key])
    return final_selective_neurons

def t_test(data, data_labels, min_loc):
    # data (numoy array): numpy array of shape (trials X Neural activity) expecting data for only one neuron
    # perform t-test between every location activities and its lowest activity for a neuron
    # get the mean activity for all trials of the neuron
    mean_activities = np.mean(data, axis=1)
    uniq_labels = np.unique(data_labels)
    p_values = []
    for stim in np.arange(len(uniq_labels)):
        if stim != min_loc:
            t_stat, p_value = ttest_ind(mean_activities[data_labels==uniq_labels[stim]],
                                        mean_activities[data_labels==uniq_labels[min_loc]])
            p_values.append(p_value)
        else:
            p_values.append(1)
    return np.array(p_values)

def oneway_anova(data, data_labels):
    # data (numpy array): numpay array of shape (trials X neural activity X neurons)
    # Perform one way anova for all the neurons in the data
#     f_stats = []
    uniq_labels = np.unique(data_labels)
    assert(len(uniq_labels)==8 or len(uniq_labels)==7 or len(uniq_labels)==4 or len(uniq_labels)==3 or len(uniq_labels)==2 or len(uniq_labels)==6)
    for neuron in np.arange(data.shape[-1]):
        # get the mean activity of each neuron for all its trials
        mean_activities = np.mean(data[:, :, neuron], axis=1)
        # apply oneway anova with groups the locations of the stimulus (mean activities of the neuron in each location)
        if len(uniq_labels)==8:
            f_stat, p_value = f_oneway(mean_activities[data_labels==uniq_labels[0]],
                                       mean_activities[data_labels==uniq_labels[1]],
                                       mean_activities[data_labels==uniq_labels[2]],
                                       mean_activities[data_labels==uniq_labels[3]],
                                       mean_activities[data_labels==uniq_labels[4]],
                                       mean_activities[data_labels==uniq_labels[5]],
                                       mean_activities[data_labels==uniq_labels[6]],
                                       mean_activities[data_labels==uniq_labels[7]])
        if len(uniq_labels)==7:
            f_stat, p_value = f_oneway(mean_activities[data_labels==uniq_labels[0]],
                                       mean_activities[data_labels==uniq_labels[1]],
                                       mean_activities[data_labels==uniq_labels[2]],
                                       mean_activities[data_labels==uniq_labels[3]],
                                       mean_activities[data_labels==uniq_labels[4]],
                                       mean_activities[data_labels==uniq_labels[5]],
                                       mean_activities[data_labels==uniq_labels[6]])
        if len(uniq_labels)==6:
            f_stat, p_value = f_oneway(mean_activities[data_labels==uniq_labels[0]],
                                       mean_activities[data_labels==uniq_labels[1]],
                                       mean_activities[data_labels==uniq_labels[2]],
                                       mean_activities[data_labels==uniq_labels[3]],
                                       mean_activities[data_labels==uniq_labels[4]],
                                       mean_activities[data_labels==uniq_labels[5]])
        if len(uniq_labels)==4:
            f_stat, p_value = f_oneway(mean_activities[data_labels==uniq_labels[0]],
                                       mean_activities[data_labels==uniq_labels[1]],
                                       mean_activities[data_labels==uniq_labels[2]],
                                       mean_activities[data_labels==uniq_labels[3]])
        if len(uniq_labels)==3:
            f_stat, p_value = f_oneway(mean_activities[data_labels==uniq_labels[0]],
                                       mean_activities[data_labels==uniq_labels[1]],
                                       mean_activities[data_labels==uniq_labels[2]])
        if len(uniq_labels)==2:
            f_stat, p_value = f_oneway(mean_activities[data_labels==uniq_labels[0]],
                                       mean_activities[data_labels==uniq_labels[1]])
        yield f_stat, p_value

def pick_strongest_selectivity(neuron_activities, selectivity_dict, anova_indices, x_neurons=9):
    strong_selective_neurons = {}
#     print(neuron_activities.shape)
#     print(selectivity_dict.keys())
    # iterate all stimulus
    for stim_key in selectivity_dict.keys():
#         print(selectivity_dict[stim_key])
        back_mapping = [np.where(anova_indices==neuron)[0][0] for neuron in selectivity_dict[stim_key]]
#         print(back_mapping)
        # select the X neurons, out of the selective neurons for that stimulus, with
        # the highest mean activity
        strong_idxs = neuron_activities[back_mapping, stim_key].argsort()[-x_neurons:]
        # pick the indices of the strongest selective neurons and update the list of selective neurons for
        # that stimulus
        strong_selective_neurons[stim_key] = np.array(selectivity_dict[stim_key])[strong_idxs]
    return strong_selective_neurons

def get_percentile(values, percentil=90):
    # Get the Nth percentile of a list of data values.
    # values (numpy array): numpy array of values
    return np.where(values > np.percentile(values, q=percentil))[0]

def get_pair_mapping(selective_stim_neurons, N):
    return [1 if (npair[0]>=N and npair[1]>=N) else (2 if (npair[0]<N and npair[1]<N) else 3) for npair in list(combinations(selective_stim_neurons,2))]

def compute_performance_LDA_with_PCAprefit(train_set, train_labels, test_set, test_labels, pca_prefit):
    # use prefitted model(pca_prefit) to transform the train and test datasets
    transformed_train_set_components = pca_prefit.transform(train_set)
    transformed_test_set_components = pca_prefit.transform(test_set)
    
    score = compute_performance_LDA(transformed_train_set_components, train_labels,
                                    transformed_test_set_components, test_labels)
    return score

def compute_performance_LDA_with_PCAcomponents(train_set, train_labels, test_set, test_labels, components=0.9):
    # Fit pca on train dataset and pass onto LDA the transformed train and test datasets
    pca = PCA(components)

    transformed_train_set_components = pca.fit_transform(train_set)
    transformed_test_set_components = pca.transform(test_set)
    
    score = compute_performance_LDA(transformed_train_set_components, train_labels,
                                    transformed_test_set_components, test_labels)
    
    return score

def compute_performance_LDA(train_set, train_labels, test_set, test_labels, components=3):
    # Perform LDA on train and test datasets
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_set, train_labels)
    score = clf.score(test_set, test_labels)

    return score

def plot_cross_temporal_decoding(data_array, title='Cross Temporal Decoding', xylines=[], min_val=0, max_val=1, cmap_name='jet', save=False, size=(6,6)):
    plt.figure(figsize=size, dpi=60)
    plt.imshow(data_array, origin='lower', vmin=min_val, vmax=max_val, cmap=cmap_name)

    for xyline in xylines:
        plt.axvline(x=xyline, color='white')
        plt.axhline(y=xyline, color='white')
    
    plt.tight_layout()
#     plt.xlabel("Test Timebins")
#     plt.ylabel("Train Timebins")
    if save:
        plt.xticks([],[])
        plt.yticks([],[])
#         plt.colorbar(shrink=0.75, ticks=[])
        plt.savefig('eps_images-new/'+title+'.pdf',format='pdf',bbox_inches='tight')
    cb = plt.colorbar(shrink=0.7)
    cb.ax.tick_params(direction="in")
    plt.title(title)
    plt.show()

def split_train_test(dataset, dataset_labels, train_size=None, test_size=None, neurons=None):
    """
    dataset(numpy array): the dataset to split in train and test of shape (trials X timebins X neurons)
    dataset_labels(numpy array): labels to split
    train_size(float): size of the trainset
    neurons(numpy array): which neurons to pick for the train and test sets if None it is set to all available neurons
    """
    split_dataset = {}
    if neurons is None:
        neurons = np.arange(dataset.shape[2])
    if train_size is None:
        train_size = int(dataset.shape[0]*2/3)
    if test_size is None:
        test_size = int(dataset.shape[0]*1/3)

    trial_indices = np.arange(dataset.shape[0])
    np.random.shuffle(trial_indices)
    dataset = dataset[trial_indices,:,:]
    dataset_labels = dataset_labels[trial_indices]

    split_dataset['train'] = dataset[:train_size, :, neurons]
    split_dataset['test'] = dataset[-test_size:, :, neurons]
    split_dataset['train_labels'] = dataset_labels[:train_size]
    split_dataset['test_labels'] = dataset_labels[-test_size:]
    
    return split_dataset

def get_code_stability(matrix):
    square_matrix = matrix.copy()
    square_matrix[np.where(square_matrix<0)]=0
    diagonal_mean = np.nanmean(square_matrix.diagonal())
    np.fill_diagonal(square_matrix, np.NaN)
    square_mean = np.nanmean(square_matrix)
    
    if diagonal_mean !=0:
        return square_mean/diagonal_mean
    else:
        return 0

def get_code_stability_off_diagonal(matrix):
    square_matrix = matrix.copy()
    square_matrix[np.where(square_matrix<0)]=0
    
    
    diagonal_mean = np.nanmean(np.flip(square_matrix,0).diagonal())
    np.fill_diagonal(square_matrix, np.NaN)
    square_mean = np.nanmean(square_matrix)
    
    if diagonal_mean !=0:
        return square_mean/diagonal_mean
    else:
        return 0

def get_information_quantity(matrix):
    square_matrix = matrix.copy()
    square_matrix[np.where(square_matrix<0)]=0
    diagonal_mean = np.mean(square_matrix.diagonal())
    return diagonal_mean