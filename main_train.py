"""
Contains function for training the models. Note this code show partial examples of the test options.
 You can see the options in the function's documentation.
"""
import os
import pickle as pkl
import random

import numpy as np
import pandas as pd

from SysEvalOffTarget_src import general_utilities
from SysEvalOffTarget_src.train_utilities import create_input_data
from SysEvalOffTarget_src.utilities import create_nucleotides_to_position_mapping
from SysEvalOffTarget_src.utilities import order_sg_rnas, load_order_sg_rnas

random.seed(general_utilities.SEED)


def load_train_datasets(data_type):
    """
    Load datasets for the regular training.
    :param union_model: bool. train the CS-GS-union model if True. Default: False
    :param data_type: str. The data type on which the models are trained. "CHANGEseq" or "GUIDEseq".
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True.
    :return: (targets, positive_df, negative_df)
        targets is a list of the sgRNAs which we will train on.
        positive_df, negative_df are Pandas dataframes used for training the models.
    """
    exclude_on_targets = False
    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'

    try:
        targets = load_order_sg_rnas(data_type)
    except FileNotFoundError:
        targets = order_sg_rnas(data_type)

    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format(data_type), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format(data_type), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    return targets, positive_df, negative_df


def prepare_data(out_dirpath, exclude_targets_without_positives=False):
    """
    Function for training the models. This performs k-fold training.
    :param models_options: tuple. A tuple with the model types to train. support these options:
        ("regression_with_negatives", "classifier", "regression_without_negatives").
        Default: ("regression_with_negatives", "classifier", "regression_without_negatives")
    :param union_model: bool. train the CS-GS-union model if True. Default: False
    :param include_distance_feature_options:  tuple. A tuple that determinate whether to add the distance feature.
        The tuple can contain both True and False. Default: (True, False)
    :param include_sequence_features_options:  tuple. A tuple that determinate whether to add the sequence
        feature. The tuple can contain both True and False. In case False is included in the tuple, False cannot be
        included in include_distance_feature_options. Default: (True, False)
    :param n_trees: int. number XGBoost trees estimators. Default: 1000
    :param trans_type: str. define which transformation is applied on the read counts.
        Options: "no_trans" - no transformation; "ln_x_plus_one_trans" - log(x+1) where x the read count;
        "ln_x_plus_one_and_max_trans" - log(x+1)/log(MAX_READ_COUNT_IN_TRAIN_SET) where x the read count;
        "standard_trans" - Z-score; "max_trans" - x/MAX_READ_COUNT_IN_TRAIN_SET;
        "box_cox_trans" - Box-Cox; or "yeo_johnson_trans" - Yeo-Johnson. In Box-Cox and Yeo-Johnson, the transformation
        is learned on a balanced set of active and inactive sites (if trans_only_positive=False).
        Default: "ln_x_plus_one_trans"
    :param trans_all_fold: bool. apply the data transformation on each sgRNA dataset if False,
        else apply on the entire train fold. Default: False
    :param trans_only_positive: bool. apply the data transformation on only on active sites if True. Default: False
    :param exclude_targets_without_positives: bool. exclude sgRNAs data without positives if True. Default: False
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True. Default: False
    :param k_fold_number: int. number of k-folds. Default: 10
    :param data_type: str. The data type on which the models are trained. "CHANGEseq" or "GUIDEseq".
        Default: "CHANGEseq"
    :param xgb_model: str or None. Path of the pretrained model (used for transfer learning.
        Assuming that (models_options + include_distance_feature_options + include_sequence_features_options)
        has one option that corresponds to the pre-trained model.
        Default: None
    :param transfer_learning_type: str. Transfer learning type. can be "add" or "update".
        Relevant if xgb_model is not None. Default: "add"
    :param exclude_targets_with_rhampseq_exp: bool. exclude the targets that appear in the rhAmpSeq experiment if True.
        Default: False
    :param save_model: bool. Save the models if True
    :return: None
    """

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()

    for data_type in ("CHANGEseq", "GUIDEseq"):
        for out_format in ('4x4', '16x1'):
            targets, positive_df, negative_df = load_train_datasets(data_type)
            test_splits = np.array_split(targets, 10)
            split_dict = {}
            for i in range(10):
                split_dict[i] = {'test': test_splits[i], 'train': [s for s in targets if s not in test_splits[i]]}
            with open(os.path.join(out_dirpath, 'sequence_splits.pkl'), 'wb') as out_f:
                pkl.dump(split_dict, out_f)

            create_input_data(positive_df, negative_df, targets, nucleotides_to_position_mapping,
                              data_type=data_type, out_format=out_format, out_dir=out_dirpath)


def main():
    """
    main function
    """
    out_dirpath = '/Users/aboud/Desktop/cmpt983_data/'
    prepare_data(out_dirpath)


if __name__ == '__main__':
    main()
