"""
 Note this code show partial examples of the training options.
 A code with full documentation and with more training options will be upload.
"""

import random
import pandas as pd
from SysEvalOffTarget_src.train_utilities import train
from SysEvalOffTarget_src.test_utilities import test

from SysEvalOffTarget_src.utilities import create_nucleotides_to_position_mapping, order_targets
from SysEvalOffTarget_src import general_utilities

random.seed(general_utilities.SEED)


def train_example():
    """
    Example function for training the models
    It contains only partial examples
    """
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()

    # Train CHANGE-seq model - exclude the sgRNA contained in GUIDE-seq
    targets_change_seq = order_targets()
    targets_guide_seq = order_targets("GUIDE")
    targets_change_seq_filtered = [
        target for target in targets_change_seq if target not in targets_guide_seq]
    print("targets_change_seq:", len(targets_change_seq))
    print("targets_guide_seq:", len(targets_guide_seq))
    print("targets_change_seq_filtered:", len(targets_change_seq_filtered))

    positive_df = pd.read_csv(
        general_utilities.DATASETS_PATH + 'CHANGE_seq_positive.csv', index_col=0)
    negative_df = pd.read_csv(
        general_utilities.DATASETS_PATH + 'CHANGE_seq_negative.csv', index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    positive_df = positive_df[positive_df['target'].isin(
        targets_change_seq_filtered)]
    negative_df = negative_df[negative_df['target'].isin(
        targets_change_seq_filtered)]

    train(positive_df, negative_df, targets_change_seq_filtered, nucleotides_to_position_mapping,
          data_type='CHANGEseq', model_type="regression_with_negatives", k_fold_number=1,
          include_distance_feature=True, include_sequence_features=True, balanced=False,
          path_prefix="train_CHANGE_seq_on_non_overlapping_targets_with_GUIDE_seq/")

    # GUIDE-seq - transfer learning
    targets = order_targets("GUIDE")
    positive_df = pd.read_csv(
        general_utilities.DATASETS_PATH + 'GUIDE_seq_positive.csv', index_col=0)
    negative_df = pd.read_csv(
        general_utilities.DATASETS_PATH + 'GUIDE_seq_negative.csv', index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    train_targets = targets[0:len(targets) - 10]
    random.shuffle(train_targets)
    print("Targets in train:")
    print(train_targets)
    # we save 10 sgRNA as test set
    print("Targets in test:")
    print(targets[-10:])

    # train transfer learning model with update option
    train(positive_df, negative_df, train_targets, nucleotides_to_position_mapping,
          data_type='GUIDEseq', model_type="classifier", k_fold_number=1,
          include_distance_feature=True, include_sequence_features=True, balanced=False,
          path_prefix="transfer_learning_process_type_update_updater_refresh_" /
                      "train_on_GUIDEseq_seed_" +
                      str(general_utilities.SEED) + "/trained_with_" + str(len(train_targets)) + "_guides_",
          xgb_model=general_utilities.FILES_DIR +
                    "models_1fold/train_CHANGE_seq_" /
                    "on_non_overlapping_targets_with_GUIDE_seq/" /
                    "classifier_xgb_model_fold_0_with_distance_without_Kfold_imbalanced.xgb",
          transfer_learning_type="update")

    # train transfer learning model with add option
    train(positive_df, negative_df, train_targets, nucleotides_to_position_mapping,
          data_type='GUIDEseq', model_type="classifier", k_fold_number=1,
          include_distance_feature=True, include_sequence_features=True, balanced=False,
          path_prefix="transfer_learning_add_type_train_on_GUIDEseq_seed_" +
                      str(general_utilities.SEED) + "/trained_with_" + str(len(train_targets)) + "_guides_",
          xgb_model=general_utilities.FILES_DIR + "models_1fold/train_CHANGE_seq_on_non_overlapping" /
                    "_targets_with_GUIDE_seq/classifier_xgb_model_fold_0_with_distance_" /
                    "without_Kfold_imbalanced.xgb", transfer_learning_type="add")


def test_example():
    """
    Example function for testing the models
    It contains only partial examples
    """
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    targets = order_targets('GUIDE')
    test_targets = targets[-10:]
    train_targets = targets[:-10]
    positive_df = pd.read_csv(
        general_utilities.DATASETS_PATH + 'GUIDE_seq_positive.csv', index_col=0)
    negative_df = pd.read_csv(
        general_utilities.DATASETS_PATH + 'GUIDE_seq_negative.csv', index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    positive_df = positive_df[positive_df['target'].isin(test_targets)]
    negative_df = negative_df[negative_df['target'].isin(test_targets)]

    for model_path_prefix in \
            ["transfer_learning_process_type_update_updater_refresh_train_on_GUIDEseq_seed_",
             "transfer_learning_train_on_GUIDEseq_seed_"]:
        for i in range(len(train_targets)):
            print('sgRNA:', i)
            test(positive_df, negative_df, test_targets, nucleotides_to_position_mapping,
                 dataset_type='GUIDEseq', model_type="classifier", k_fold_number=1,
                 include_distance_feature=True, include_sequence_features=True, balanced=False,
                 path_prefix=model_path_prefix + str(general_utilities.SEED) + "/trained_with_" + str(i + 1) +
                 "_guides_")


if __name__ == '__main__':
    train_example()
    test_example()
