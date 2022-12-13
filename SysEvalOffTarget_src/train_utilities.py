"""
    This module contains the function for training all the xgboost model variants
"""

import random
import time

from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as pd
import pickle as pkl
import tqdm
import os

from sklearn.utils import shuffle

from SysEvalOffTarget_src.utilities import create_fold_sets, extract_model_path,\
    build_sequence_features, build_sampleweight, transformer_generator, transform
from SysEvalOffTarget_src import general_utilities

random.seed(general_utilities.SEED)


def data_preprocessing(positive_df, negative_df, trans_type, data_type, trans_all_fold, trans_only_positive):
    """
    data_preprocessing
    """
    data_type = "" if data_type is None else data_type + "_"
    reads_col = "{}reads".format(data_type)
    # it might include, but just confirm:
    positive_df["label"] = 1
    negative_df["label"] = 0
    negative_df[reads_col] = 0

    positive_labels_df = positive_df[["target", "offtarget_sequence", "label", reads_col]]
    if trans_only_positive:
        labels_df = positive_labels_df
    else:
        negative_labels_df = negative_df[["target", "offtarget_sequence", "label", reads_col]]
        labels_df = pd.concat([positive_labels_df, negative_labels_df])

    if trans_all_fold:
        labels = labels_df[reads_col].values
        transformer = transformer_generator(labels, trans_type)
        labels_df[reads_col] = transform(labels, transformer)
    else:
        # preform the preprocessing on each sgRNA data individually
        for target in labels_df["target"].unique():
            target_df = labels_df[labels_df["target"] == target]
            target_labels = target_df[reads_col].values
            transformer = transformer_generator(target_labels, trans_type)
            labels_df.loc[labels_df["target"] == target, reads_col] = transform(target_labels, transformer)

    if trans_only_positive:
        positive_df[reads_col] = labels_df[reads_col]
    else:
        positive_labels_df = labels_df[labels_df["label"] == 1]
        negative_labels_df = labels_df[labels_df["label"] == 0]
        positive_df[reads_col] = positive_labels_df[reads_col]
        negative_df[reads_col] = negative_labels_df[reads_col]

    return positive_df, negative_df


def create_input_data(positive_df, negative_df, targets, nucleotides_to_position_mapping,
                      data_type, out_format, out_dir):
    """
    The train function
    """
    trans_type = "ln_x_plus_one_trans"
    trans_all_fold = False
    trans_only_positive = False
    negative_sequence_features_train, sequence_labels_train = None, None

    # n_folds = 10
    # target_folds_list = np.array_split(targets, n_folds)

    # target_folds_proc = [(target_folds_list[idx], [l for l in targets if l not in target_folds_list[idx]]) for idx in range(len(target_folds_list))]
    #
    # added_test_folds = set()

    # for i, (test_fold, train_fold) in tqdm.tqdm(enumerate(target_folds_proc), total=len(target_folds_proc)):

    for target in tqdm.tqdm(targets, desc=f'Creating {data_type} data'):
        # assert set(test_fold).intersection(added_test_folds) == set()
        # added_test_folds = added_test_folds.union(test_fold)
        # assert set(test_fold).intersection(set(train_fold)) == set()
        # assert set(test_fold).union(set(train_fold)) == set(targets)

        positive_df_target = positive_df[positive_df['target'] == target]
        negative_df_target = negative_df[negative_df['target'] == target]
        positive_sequence_features, positive_sequence_dist = build_sequence_features(
            positive_df_target, nucleotides_to_position_mapping, out_format=out_format)
        negative_sequence_features, negative_sequence_dist = build_sequence_features(
            negative_df_target, nucleotides_to_position_mapping, out_format=out_format)
        sequence_features_target = np.concatenate((negative_sequence_features, positive_sequence_features))
        sequence_dist_target = np.concatenate((negative_sequence_dist, positive_sequence_dist))
        # obtain classes
        negative_class_target = negative_df_target["label"].values
        positive_class_target = positive_df_target["label"].values
        sequence_class_target = np.concatenate((negative_class_target, positive_class_target))

        # obtain regression labels
        positive_df_target, negative_df_target = \
            data_preprocessing(positive_df_target, negative_df_target, trans_type=trans_type, data_type=data_type,
                               trans_all_fold=trans_all_fold, trans_only_positive=trans_only_positive)
        negative_labels_target = negative_df_target[data_type +
                                                  "_reads"].values
        positive_labels_target = positive_df_target[data_type +
                                                  "_reads"].values
        sequence_labels_target = np.concatenate(
            (negative_labels_target, positive_labels_target))

        sequence_class_target, sequence_features_target, sequence_labels_target = shuffle(
            sequence_class_target, sequence_features_target, sequence_labels_target,
            random_state=general_utilities.SEED)

        dtype_out_path = os.path.join(out_dir, data_type, out_format)
        os.makedirs(dtype_out_path, exist_ok=True)
        with open(os.path.join(dtype_out_path, f'{target}.pkl'), 'wb') as out_f:
            pkl.dump({'seq_features': sequence_features_target,
                      'class': sequence_class_target,
                      'label': sequence_labels_target,
                      'distance': sequence_dist_target}, out_f)
