"""
Contains function for training the models. Note this code show partial examples of the test options.
 You can see the options in the functions documentation.
"""

import random
import pandas as pd
from SysEvalOffTarget_src.train_utilities import train

from SysEvalOffTarget_src.utilities import create_nucleotides_to_position_mapping
from SysEvalOffTarget_src.utilities import order_sg_rnas, load_order_sg_rnas
from SysEvalOffTarget_src import general_utilities
random.seed(general_utilities.SEED)


def regular_train_models(
        models_options=("regression_with_negatives", "classifier", "regression_without_negatives"),
        include_distance_feature_options=(True, False), include_sequence_features_options=(True, False),
        test_with_no_trans=False, exclude_on_targets=False, k_fold_number=10, data_type="CHANGEseq",
        xgb_model=None, transfer_learning_type="add", exclude_targets_with_rhampseq_exp=False, save_model=True):
    """
    Function for training the models. This perform k-fold training.
    :param models_options: tuple. A tuple with the model types to train. support these options:
        ("regression_with_negatives", "classifier", "regression_without_negatives").
        Default: ("regression_with_negatives", "classifier", "regression_without_negatives")
    :param include_distance_feature_options:  tuple. A tuple that determinate whether to add the distance feature.
        The tuple can contain both True and False. Default: (True, False)
    :param include_sequence_features_options:  tuple. A tuple that determinate whether to add the sequence
        feature. The tuple can contain both True and False. in can the tuple include False, False can not be included in
        include_distance_feature_options. Default: (True, False)
    :param test_with_no_trans: bool. in addition to training the models with the log transformation on the read counts,
        decide whether to include a training of the regression models without any transformation. True or False.
        Default: False
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True. Default: False
    :param k_fold_number: int. number of k-folds. Default: 10
    :param data_type: str. The data type on which the models are trained. "CHANGEseq" or "GUIDEseq".
        Default: "CHANGEseq"
    :param xgb_model: str or None. Path of the pretrained model (used for transfer learning.
        Assuming that (models_options + include_distance_feature_options + include_sequence_features_options)
        has one option that corresponds to the pre-trained model. In addition, assume test_with_no_trans=False.
        Default: None
    :param transfer_learning_type: str. Transfer learning type. can be "add" or "update".
        Relevant if xgb_model is not None. Default: "add"
    :param exclude_targets_with_rhampseq_exp: bool. exclude the targets that appear in the rhAmpSeq experiment if True.
        Default: False
    :param save_model: bool. Save the models if True
    :return: None
    """
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    # Train CHANGE-seq/GUIDE-seq model
    try:
        targets_change_seq = load_order_sg_rnas(data_type)
    except FileNotFoundError:
        targets_change_seq = order_sg_rnas(data_type)

    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format(data_type), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format(data_type), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    if exclude_targets_with_rhampseq_exp:
        targets_list = ["GTCAGGGTTCTGGATATCTGNGG",  # TRAC_site_1
                        "GCTGGTACACGGCAGGGTCANGG",  # TRAC_site_2
                        "GAGAATCAAAATCGGTGAATNGG"  # TRAC_site_3,
                        "GAAGGCTGAGATCCTGGAGGNGG",  # LAG3_site_9
                        "GGACTGAGGGCCATGGACACNGG"  # CTLA4_site_9
                        "GTCCCTAGTGGCCCCACTGTNGG"  # AAVS1_site_2
                        ]
        positive_df = positive_df[~positive_df["target"].isin(targets_list)]
        negative_df = negative_df[~negative_df["target"].isin(targets_list)]

    save_model_dir_path_prefix = 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    save_model_dir_path_prefix = "trained_without_rhampseq_exp_targets/" + save_model_dir_path_prefix \
        if exclude_targets_with_rhampseq_exp else save_model_dir_path_prefix
    save_model_dir_path_prefix = data_type + '/' + save_model_dir_path_prefix \
        if data_type != "CHANGEseq" else save_model_dir_path_prefix
    save_model_dir_path_prefix += "TL_" + transfer_learning_type + '/' if xgb_model is not None else ""
    for model_type in models_options:
        path_prefix = save_model_dir_path_prefix + model_type + "/"
        for include_distance_feature in include_distance_feature_options:
            for include_sequence_features in include_sequence_features_options:
                if (not include_distance_feature) and (not include_sequence_features):
                    continue
                train(positive_df, negative_df, targets_change_seq, nucleotides_to_position_mapping,
                      data_type=data_type, model_type=model_type, k_fold_number=k_fold_number,
                      include_distance_feature=include_distance_feature,
                      include_sequence_features=include_sequence_features, balanced=False,
                      path_prefix=path_prefix, xgb_model=xgb_model, transfer_learning_type=transfer_learning_type,
                      save_model=save_model)
                if (model_type in ("regression_with_negatives", "regression_without_negatives")) and test_with_no_trans:
                    train(positive_df, negative_df, targets_change_seq, nucleotides_to_position_mapping,
                          data_type=data_type, model_type=model_type, k_fold_number=k_fold_number,
                          include_distance_feature=include_distance_feature,
                          include_sequence_features=include_sequence_features, balanced=False, trans_type="no_trans",
                          path_prefix=path_prefix, save_model=save_model)


def incremental_pretrain_base_models(models_options=("regression_with_negatives", "classifier"),
                                     exclude_on_targets=False):
    """
    pre-train the base models for the TL incremental training. save the model in files.
    :param models_options: tuple. A tuple with the model types to train. support these options:
        ("regression_with_negatives", "classifier", "regression_without_negatives").
        Default: ("regression_with_negatives", "classifier")
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True. Default: False
    :return: None
    """
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    try:
        targets_change_seq = load_order_sg_rnas("CHANGE")
    except FileNotFoundError:
        targets_change_seq = order_sg_rnas("CHANGE")

    try:
        targets_guide_seq = load_order_sg_rnas("GUIDE")
    except FileNotFoundError:
        targets_guide_seq = order_sg_rnas("GUIDE")

    targets_change_seq_filtered = [target for target in targets_change_seq if target not in targets_guide_seq]
    print("targets_change_seq:", len(targets_change_seq))
    print("targets_guide_seq:", len(targets_guide_seq))
    print("targets_change_seq_filtered:", len(targets_change_seq_filtered))

    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format("CHANGEseq"), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format("CHANGEseq"), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    positive_df = positive_df[positive_df['target'].isin(targets_change_seq_filtered)]
    negative_df = negative_df[negative_df['target'].isin(targets_change_seq_filtered)]

    save_model_dir_path_prefix = \
        'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    save_model_dir_path_prefix = \
        "train_CHANGE_seq_on_non_overlapping_targets_with_GUIDE_seq/" + save_model_dir_path_prefix
    for model_type in models_options:
        path_prefix = save_model_dir_path_prefix + model_type + "/"
        for include_distance_feature in (True, False):
            train(positive_df, negative_df, targets_change_seq, nucleotides_to_position_mapping,
                  data_type="CHANGEseq", model_type=model_type, k_fold_number=1,
                  include_distance_feature=include_distance_feature,
                  include_sequence_features=True, balanced=False,
                  path_prefix=path_prefix)


def incremental_train_models(models_options=("regression_with_negatives", "classifier"), exclude_on_targets=False,
                             transfer_learning_types=(None, "add", "update"), pretrain=True,
                             seeds=(i for i in range(1, 11))):
    """
    training the TL models from CHANGE-seq to GUIDE-seq in incremental way. For each model type, we training n
    (where n is the number sgRNA we dedicated for training the TL models) models. we start with one sgRNA in a train,
    and then increase the number until we train the final model with n sgRNAs.
    :param models_options: models_options: tuple. A tuple with the model types to train. support these options:
        ("regression_with_negatives", "classifier").
        Default: ("regression_with_negatives", "classifier")
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True. Default: False
    :param transfer_learning_types: bool. Transfer learning types. support these options: (None, "add", "update").
        When None is included, then models are trained with TL. Default: (None, "add", "update")
    :param pretrain: bool. pretrain the model if True. If False, we assume the the pretrained models are exists.
        Default: True
    :param seeds: tuple. tuple of seeds numbers (n numbers). we are training n time the models, each time we souffle the
        all training set randomly according to the seeds.
    :return: None
    """
    # CHANGE-seq
    if pretrain:
        incremental_pretrain_base_models(models_options=models_options,
                                         exclude_on_targets=exclude_on_targets)

    # GUIDE-seq
    random.seed(general_utilities.SEED)
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    try:
        targets = load_order_sg_rnas("GUIDE")
    except FileNotFoundError:
        targets = order_sg_rnas("GUIDE")

    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format("GUIDEseq"), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format("GUIDEseq"), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them
    for seed in seeds:
        print("seed:", seed)
        random.seed(seed)
        # save 10 targets as test set
        train_targets = targets[0:len(targets)-10]
        random.shuffle(train_targets)
        print("Targets in train:")
        print(train_targets)
        print("Targets in test:")
        print(targets[-10:])

        random.seed(general_utilities.SEED)
        save_model_dir_path_prefix = 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
        save_model_dir_path_prefix = "GUIDEseq/incremental_training/" + save_model_dir_path_prefix
        for transfer_learning_type in transfer_learning_types:
            type_save_model_dir_path_prefix = save_model_dir_path_prefix + \
                "GS_TL_" + transfer_learning_type + '/' if transfer_learning_type is not None else "GS/"
            xgb_model_path = general_utilities.FILES_DIR + "models_1fold/" + \
                "train_CHANGE_seq_on_non_overlapping_targets_with_GUIDE_seq/"
            xgb_model_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
            for i in range(len(train_targets)):
                path_prefix = type_save_model_dir_path_prefix + "seed_" + str(seed) + \
                              "/trained_with_"+str(i+1)+"_guides_"
                if "classifier" in models_options:
                    train(positive_df, negative_df, train_targets[0:i+1], nucleotides_to_position_mapping,
                          data_type='GUIDEseq', model_type="classifier", k_fold_number=1,
                          include_distance_feature=False, include_sequence_features=True, balanced=False,
                          path_prefix=path_prefix,
                          xgb_model=xgb_model_path +
                                    "classifier/classifier_xgb_model_fold_0_without_Kfold_imbalanced.xgb"
                                    if transfer_learning_type is not None else None,
                          transfer_learning_type=transfer_learning_type)
                    train(positive_df, negative_df, train_targets[0:i+1], nucleotides_to_position_mapping,
                          data_type='GUIDEseq', model_type="classifier", k_fold_number=1, include_distance_feature=True,
                          include_sequence_features=True, balanced=False, path_prefix=path_prefix,
                          xgb_model=xgb_model_path +
                                    "classifier/classifier_xgb_model_fold_0_with_distance_without_Kfold_imbalanced.xgb"
                                    if transfer_learning_type is not None else None,
                          transfer_learning_type=transfer_learning_type)
                if "regression_with_negatives" in models_options:
                    train(positive_df, negative_df, train_targets[0:i+1], nucleotides_to_position_mapping,
                          data_type='GUIDEseq', model_type="regression_with_negatives", k_fold_number=1,
                          include_distance_feature=False, include_sequence_features=True, balanced=False,
                          path_prefix=path_prefix, xgb_model=xgb_model_path + "regression_with_negatives/"
                          "regression_with_negatives_xgb_model_fold_0_without_Kfold_imbalanced.xgb"
                          if transfer_learning_type is not None else None,
                          transfer_learning_type=transfer_learning_type)
                    train(positive_df, negative_df, train_targets[0:i+1], nucleotides_to_position_mapping,
                          data_type='GUIDEseq', model_type="regression_with_negatives", k_fold_number=1,
                          include_distance_feature=True,
                          include_sequence_features=True, balanced=False, path_prefix=path_prefix,
                          xgb_model=xgb_model_path + "regression_with_negatives/"
                          "regression_with_negatives_xgb_model_fold_0_with_distance_without_Kfold_imbalanced.xgb"
                          if transfer_learning_type is not None else None,
                          transfer_learning_type=transfer_learning_type)


def main():
    """
    main function
    """
    # some training examples. You need to run prepre_data.py before trying to train
    # See regular_train_models and incremental_train_models to train other options
    regular_train_models(
        models_options=tuple(("regression_without_negatives",)),
        include_distance_feature_options=(True,),
        include_sequence_features_options=(True,),
        test_with_no_trans=False, k_fold_number=10,
        data_type="CHANGEseq")
    regular_train_models(
        models_options=tuple(("classifier", "regression_with_negatives")),
        include_distance_feature_options=(True, False),
        include_sequence_features_options=tuple((True,)),
        test_with_no_trans=False, k_fold_number=10,
        data_type="CHANGEseq")


if __name__ == '__main__':
    main()
