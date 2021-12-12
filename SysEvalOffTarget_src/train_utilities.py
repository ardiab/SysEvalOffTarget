"""
    This module contains the function for training all the xgboost model variants
"""

import random
from pathlib import Path
import xgboost as xgb
import numpy as np

from sklearn.utils import shuffle

from SysEvalOffTarget_src.utilities import create_fold_sets, \
    create_transformer, build_sequence_features, data_trans, build_sampleweight
from SysEvalOffTarget_src import general_utilities

random.seed(general_utilities.SEED)


def train(positive_df, negative_df, targets, nucleotides_to_position_mapping,
          data_type='CHANGEseq', model_type="classifier", k_fold_number=10,
          include_distance_feature=False, include_sequence_features=True,
          balanced=True, trans_type="ln_x_plus_one_trans", skip_num_folds=0,
          path_prefix="", xgb_model=None, transfer_learning_type="add"):
    """
    The train function
    """
    # To avoid a situation in which they are not defined (not really needed)
    negative_sequence_features_train, sequence_labels_train = None, None

    # set transfer_learning setting if needed
    if xgb_model is not None:
        # update the trees or train additional trees
        transfer_learning_args = {'process_type': 'update', 'updater': 'refresh'} \
                                 if transfer_learning_type == 'update' \
                                 else {'tree_method': 'gpu_hist'}
    else:
        transfer_learning_args = {'tree_method': 'gpu_hist'}

    # model_type can get: 'classifier, regression_with_negatives, regression_without_negatives
    # in case we don't have k_fold, we train all the dataset with test set.
    target_folds_list = np.array_split(
        targets, k_fold_number) if k_fold_number > 1 else [[]]

    for i, target_fold in enumerate(target_folds_list[skip_num_folds:]):
        negative_df_train, positive_df_train, _, _ = create_fold_sets(
            target_fold, targets, positive_df, negative_df, balanced)
        transformer = create_transformer(
            positive_df=positive_df_train, negative_df=negative_df_train,
            trans_type=trans_type, model_type=model_type, data_type=data_type)
        if model_type in ("classifier", "regression_with_negatives"):
            negative_sequence_features_train = build_sequence_features(
                negative_df_train, nucleotides_to_position_mapping,
                include_distance_feature=include_distance_feature,
                include_sequence_features=include_sequence_features)
            positive_sequence_features_train = build_sequence_features(
                positive_df_train, nucleotides_to_position_mapping,
                include_distance_feature=include_distance_feature,
                include_sequence_features=include_sequence_features)
            sequence_features_train = np.concatenate(
                (negative_sequence_features_train, positive_sequence_features_train))
        elif model_type == 'regression_without_negatives':
            positive_sequence_features_train = build_sequence_features(
                positive_df_train, nucleotides_to_position_mapping,
                include_distance_feature=include_distance_feature,
                include_sequence_features=include_sequence_features)
            sequence_features_train = positive_sequence_features_train
        else:
            raise ValueError('model_type is invalid.')

        negative_class_train = negative_df_train["label"].values
        positive_class_train = positive_df_train["label"].values
        sequence_class_train = \
            np.concatenate((negative_class_train, positive_class_train)) if \
            model_type != "regression_without_negatives" else positive_class_train
        if model_type == "regression_with_negatives":
            negative_labels_train = np.zeros(
                (len(negative_sequence_features_train),))
            positive_labels_train = positive_df_train[data_type +
                                                      "_reads"].values
            sequence_labels_train = np.concatenate(
                (negative_labels_train, positive_labels_train))
            # perform transform
            sequence_labels_train = data_trans(
                data=sequence_labels_train, trans_type=trans_type, transformer=transformer)
        elif model_type == "regression_without_negatives":
            # regression_without_negatives
            sequence_labels_train = positive_df_train[data_type +
                                                      "_reads"].values
            # perform transform
            sequence_labels_train = data_trans(
                data=sequence_labels_train, trans_type=trans_type, transformer=transformer)

        if model_type == "classifier":
            sequence_class_train, sequence_features_train = shuffle(
                sequence_class_train, sequence_features_train)
        else:
            sequence_class_train, sequence_features_train, sequence_labels_train = shuffle(
                sequence_class_train, sequence_features_train, sequence_labels_train)

        negative_num = 0 if model_type == "regression_without_negatives" else len(
            negative_sequence_features_train)
        print("train fold ", i+skip_num_folds, " positive:",
              len(positive_sequence_features_train), ", negative:", negative_num)
        if model_type == "classifier":
            model = xgb.XGBClassifier(max_depth=10,
                                      learning_rate=0.1,
                                      n_estimators=1000,
                                      nthread=100,
                                      **transfer_learning_args)  # tree_method='gpu_hist'

            print(sequence_features_train.shape)
            model.fit(sequence_features_train, sequence_class_train,
                      sample_weight=build_sampleweight(sequence_class_train), xgb_model=xgb_model)
        else:
            model = xgb.XGBRegressor(max_depth=10,
                                     learning_rate=0.1,
                                     n_estimators=1000,
                                     nthread=100,
                                     **transfer_learning_args)  # tree_method='gpu_hist'

            if model_type == "regression_with_negatives":
                model.fit(sequence_features_train, sequence_labels_train,
                          sample_weight=build_sampleweight(sequence_class_train),
                          xgb_model=xgb_model)
            else:
                model.fit(sequence_features_train,
                          sequence_labels_train, xgb_model=xgb_model)

        suffix = "_with_distance" if include_distance_feature else ""
        suffix = suffix + "" if include_sequence_features else "_without_sequence_features"
        suffix = suffix + ("_without_Kfold" if k_fold_number == 1 else "")
        suffix = suffix + ("" if balanced == 1 else "_imbalanced")
        if trans_type != "ln_x_plus_one_trans" and model_type != "classifier":
            suffix = suffix + "_"+trans_type
        dir_path = general_utilities.FILES_DIR + "models_" + \
            str(k_fold_number) + "fold/" + path_prefix + model_type + \
            "_xgb_model_fold_" + str(i+skip_num_folds) + suffix + ".xgb"
        Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(dir_path)
