import itertools
import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer

from SysEvalOffTarget_src import general_utilies

import random
random.seed(10)

def order_targets(data_type='CHANGE'):
    positive_df = pd.read_csv(
        general_utilies.datatsets_PATH+data_type+'_seq_positive.csv', index_col=0)

    targets = positive_df["target"].unique()
    random.shuffle(targets)

    return targets


def create_nucleotides_to_position_mapping():
    # matrix positions for ('A','A'), ('A','C'),...
    # tuples of ('A','A'), ('A','C'),...
    nucleotides_product = list(itertools.product(*(['ACGT'] * 2)))
    # tuple of (0,0), (0,1), ...
    position_product = [(int(x[0]), int(x[1]))
                        for x in list(itertools.product(*(['0123'] * 2)))]
    nucleotides_to_position_mapping = dict(
        zip(nucleotides_product, position_product))

    # tuples of ('N','A'), ('N','C'),...
    N_mapping_nucleotides_list = [('N', char) for char in ['A', 'C', 'G', 'T']]
    N_mapping_position_list = [([nucleotides_to_position_mapping[(N, char)][0] for N in ['A', 'C', 'G', 'T']], [nucleotides_to_position_mapping[(
        N, char)][1] for N in ['A', 'C', 'G', 'T']]) for char in ['A', 'C', 'G', 'T']]  # list of position tuples corresponding to ('N','A'), ('N','C'),...

    nucleotides_to_position_mapping.update(
        dict(zip(N_mapping_nucleotides_list, N_mapping_position_list)))

    # tuples of ('A','N'), ('C','N'),...
    N_mapping_nucleotides_list = [(char, 'N') for char in ['A', 'C', 'G', 'T']]
    N_mapping_position_list = [([nucleotides_to_position_mapping[(char, N)][0] for N in ['A', 'C', 'G', 'T']], [nucleotides_to_position_mapping[(
        char, N)][1] for N in ['A', 'C', 'G', 'T']]) for char in ['A', 'C', 'G', 'T']]  # list of tuplus positions corresponding to ('A','N'), ('C','N'),...
    nucleotides_to_position_mapping.update(
        dict(zip(N_mapping_nucleotides_list, N_mapping_position_list)))

    return nucleotides_to_position_mapping


def build_sequence_features(dataset_df, nucleotides_to_position_mapping, include_distance_feature=False, include_sequence_features=True):
    if((not include_distance_feature) and (not include_sequence_features)):
        raise ValueError(
            'include_distance_feature and include_sequence_features can not be both False')
    if(include_sequence_features):
        final_result = np.zeros((len(dataset_df), (23*16)+1),
                                dtype=np.int8) if include_distance_feature else np.zeros((len(dataset_df), 23*16), dtype=np.int8)
    else:
        final_result = np.zeros((len(dataset_df), 1), dtype=np.int8)
    for i, (seq1, seq2) in enumerate(zip(dataset_df["target"], dataset_df["offtarget_sequence"])):
        if(include_sequence_features):
            intersection_matrices = np.zeros((23, 4, 4), dtype=np.int8)
            for j in range(23):
                matrix_positions = nucleotides_to_position_mapping[(
                    seq1[j], seq2[j])]
                intersection_matrices[j, matrix_positions[0],
                                      matrix_positions[1]] = 1

        if(include_distance_feature):
            if(include_sequence_features):
                final_result[i, :-1] = intersection_matrices.flatten()
            final_result[i, -1] = dataset_df["distance"].iloc[i]
        else:
            final_result[i, :] = intersection_matrices.flatten()

    return final_result

##########################################################################


def create_fold_sets(target_fold, targets, positive_df, negative_df, balanced=True):
    test_targets = target_fold
    train_targets = [x for x in targets if x not in target_fold]
    positive_df_test = positive_df[positive_df['target'].isin(test_targets)]
    positive_df_train = positive_df[positive_df['target'].isin(train_targets)]

    if(balanced):
        # obtain the negative samples for train (for each target the positive and negtive samples numbers is equal)
        negative_indices_train = []
        for target in targets:
            if (target in test_targets):
                continue
            negative_indices_train = negative_indices_train + list(negative_df[(negative_df['target'] == target)].sample(
                n=len(positive_df_train[(positive_df_train['target'] == target)])).index)
        negative_df_train = negative_df.loc[negative_indices_train]

        # obtain the negative samples for test (for test take all negatives not in the trains set)
        negative_df_test = negative_df[negative_df['target'].isin(test_targets)]
        # remove from negatives all the samples found in the train negative set
        #TODO:remove this filter, I'm sure that it doesn't do anything since negative_indices_train does not contain any values from the test_targets
        negative_df_test = negative_df.drop(
            negative_indices_train, axis='index')

        # negative_indices_test = []
        # for target in target_fold:
        #     negative_indices_test = negative_indices_test + list(negative_df[negative_df['target']==target].sample(n=len(positive_df_test[positive_df_test['target']==target])).index)
        # negative_df_test = negative_df.loc[negative_indices_test]
    else:
        negative_df_test = negative_df[negative_df['target'].isin(test_targets)]
        negative_df_train = negative_df[negative_df['target'].isin(
            train_targets)]

    return negative_df_train, positive_df_train, negative_df_test, positive_df_test


##########################################################################
def build_sampleweight(y):

    vec = np.zeros((len(y)))
    for l in np.unique(y):
        vec[y == l] = np.sum(y != l) / len(y)
    return vec

##########################################################################


def create_transformer(positive_df, negative_df, trans_type, model_type, data_type='CHANGEseq'):
    if(model_type == "regression_with_negatives"):
        negative_labels = np.zeros(
            (len(negative_df),))
        positive_labels = positive_df[data_type+"_reads"].values
        sequence_labels = np.concatenate(
            (negative_labels, positive_labels))
    elif(model_type == "regression_without_negatives"):
        sequence_labels = positive_df[data_type+"_reads"].values
    else:
        return None

    if(trans_type == "box_cox"):
        transformer = PowerTransformer(method='box-cox')
        # we perform box cox on sequence_labels+1
        transformer.fit((sequence_labels+1).reshape(-1, 1))
    elif(trans_type == "yeo_johnson"):
        transformer = PowerTransformer(method='yeo-johnson')
        # we perform yeo-johnson on sequence_labels
        transformer.fit((sequence_labels).reshape(-1, 1))
    else:
        transformer = None

    return transformer


def data_trans(data, trans_type="ln_x_plus_one_trans", transformer=None, inverse=False):
    if (trans_type == "no_trans"):
        return data
    elif (trans_type == "ln_x_plus_one_trans"):
        # log(x+1) trans
        if inverse:
            return np.exp(data)-1
        else:
            return np.log(data+1)
    elif (trans_type == "box_cox"):
        # box_cox(x+1) trans
        if (transformer is None):
            raise ValueError('transformer should not be None.')
        if inverse:
            return np.squeeze(transformer.inverse_transform(data.reshape(-1, 1))-1)
        else:
            return np.squeeze(transformer.transform((data+1).reshape(-1, 1)))
    elif(trans_type == "yeo_johnson"):
        # yeo_johnson(x) trans
        if (transformer is None):
            raise ValueError('transformer should not be None.')
        if inverse:
            return np.squeeze(transformer.inverse_transform(data.reshape(-1, 1)))
        else:
            return np.squeeze(transformer.transform((data).reshape(-1, 1)))
    else:
        raise ValueError('trans_type is invalid.')