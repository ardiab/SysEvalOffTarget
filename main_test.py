"""
 Contains function for testing the models. Note this code show partial examples of the test options.
 You can see the options in the functions documentation.
"""

import random
import pandas as pd
from SysEvalOffTarget_src.test_utilities import test, model_folds_predictions

from SysEvalOffTarget_src.utilities import create_nucleotides_to_position_mapping, order_sg_rnas, load_order_sg_rnas
from SysEvalOffTarget_src import general_utilities

random.seed(general_utilities.SEED)


def regular_test_models(models_options=("regression_with_negatives", "classifier", "regression_without_negatives"),
                        include_distance_feature_options=(True, False), include_sequence_features_options=(True, False),
                        test_with_no_trans=False, train_exclude_on_targets=False, test_exclude_on_targets=False,
                        k_fold_number=10, task="evaluation", data_types=('CHANGEseq', 'GUIDEseq'), intersection=None):
    """
    Function for testing the models. The corresponding function to regular_train_models.
    Save the results in the files directory according to the type if the model.
    The parameters help locate the models, and define on which dataset the test will be applied.
    intersection: can be "CHANGE_GUIDE_intersection_by_both" or "CHANGE_GUIDE_intersection_by_GUIDE"
    :param models_options: models_options: tuple. A tuple with the model types to test. support these options:
        ("regression_with_negatives", "classifier", "regression_without_negatives").
        Default: ("regression_with_negatives", "classifier", "regression_without_negatives")
    :param include_distance_feature_options: tuple. A tuple that determinate whether to add the sequence
        feature. The tuple can contain both True and False. in can the tuple include False, False can not be included in
        include_distance_feature_options. Default: (True, False)
    :param include_sequence_features_options: tuple. A tuple that determinate whether to add the sequence
        feature. The tuple can contain both True and False. in can the tuple include False, False can not be included in
        include_distance_feature_options. Default: (True, False)
    :param test_with_no_trans: bool. in addition to testing the models with the log transformation on the read counts,
        decide whether to include a training of the regression models without any transformation. True or False.
        Default: False
    :param train_exclude_on_targets: test the models trained without on-targets in case of True. Default: False
    :param test_exclude_on_targets: test on the dataset without the on-targets in case of True. Default: False
    :param k_fold_number: int. number of k-folds of the tested models. Default: 10
    :param task: str. options: "evaluation" or "prediction". If task="evaluation", we evaluate the model performance
        and save the results. If task="prediction", prediction of the models on the dataset samples is made.
        The predictions are save into a table. Default: "evaluation"
    :param data_types: tuple. The dataset on which we perform the evaluation/predictions.
        Options: ('CHANGEseq', 'GUIDEseq'). Default: ('CHANGEseq', 'GUIDEseq')
    :param intersection: str or None. In case it is not none, ignore data_types, and perform the test over an
        intersection dataset of GUIDE-seq and CHANGE-seq. works only with task="evaluation". Options:
        "CHANGE_GUIDE_intersection_by_both" or "CHANGE_GUIDE_intersection_by_GUIDE"
        Default: None
    :return: None
    """
    if intersection is not None and task == "prediction":
        raise ValueError("prediction task does not support prediction on the intersection")

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    # test CHANGE-seq model
    try:
        targets_change_seq = load_order_sg_rnas()
    except FileNotFoundError:
        targets_change_seq = order_sg_rnas()

    for data_type in data_types:
        datasets_dir_path = general_utilities.DATASETS_PATH
        datasets_dir_path += 'exclude_on_targets/' if test_exclude_on_targets else 'include_on_targets/'
        dataset_type = data_type if intersection is None else intersection
        positive_df = pd.read_csv(
            datasets_dir_path + '{}_positive.csv'.format(dataset_type), index_col=0)
        negative_df = pd.read_csv(
            datasets_dir_path + '{}_negative.csv'.format(dataset_type), index_col=0)
        negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
            'N') == -1]  # some off_targets contains N's. we drop them

        save_model_dir_path_prefix = 'exclude_on_targets/' if train_exclude_on_targets else 'include_on_targets/'
        # predictions path
        prediction_results_path = general_utilities.FILES_DIR + "models_{}_fold/".format(k_fold_number) + \
            save_model_dir_path_prefix
        prediction_results_path += \
            'predictions_exclude_on_targets/' if test_exclude_on_targets else 'predictions_include_on_targets/'
        prediction_results_path += "{0}_results_all_{1}_folds.csv".format(data_type, k_fold_number)

        for model_type in models_options:
            models_path_prefix = save_model_dir_path_prefix + model_type + "/"
            # evaluation path
            evaluation_results_path_prefix = save_model_dir_path_prefix + model_type + "/"
            evaluation_results_path_prefix += \
                'test_results_exclude_on_targets/' if test_exclude_on_targets else 'test_results_include_on_targets/'
            evaluation_results_path_prefix += intersection + "_" if intersection is not None else ""
            # make the evaluations/predictions
            for include_distance_feature in include_distance_feature_options:
                for include_sequence_features in include_sequence_features_options:
                    call_args = (positive_df, negative_df, targets_change_seq, nucleotides_to_position_mapping)
                    call_kwargs = {"dataset_type": data_type, "model_type": model_type, "k_fold_number": k_fold_number,
                                   "include_distance_feature": include_distance_feature,
                                   "include_sequence_features": include_sequence_features, "balanced": False}
                    if task == "evaluation":
                        call_kwargs.update({"models_path_prefix": models_path_prefix,
                                            "results_path_prefix": evaluation_results_path_prefix})
                    elif task == "prediction":
                        call_kwargs.update({"add_to_results_table": True,
                                            "results_table_path": prediction_results_path,
                                            "save_results": True, "path_prefix": models_path_prefix})
                    else:
                        raise ValueError("Invalid task argument value")
                    if (not include_distance_feature) and (not include_sequence_features):
                        continue
                    # update the call kwargs according to task
                    if task == "evaluation":
                        call_kwargs.update({"models_path_prefix": models_path_prefix,
                                            "results_path_prefix": evaluation_results_path_prefix})
                        test(*call_args, **call_kwargs)
                    elif task == "prediction":
                        call_kwargs.update({"add_to_results_table": True,
                                            "results_table_path": prediction_results_path,
                                            "save_results": True, "path_prefix": models_path_prefix})
                        model_folds_predictions(*call_args, **call_kwargs)
                    else:
                        raise ValueError("Invalid task argument value")
                    # add no-transformation results if requested
                    if (model_type in ("regression_with_negatives", "regression_without_negatives")) \
                            and test_with_no_trans:
                        call_kwargs.update({"trans_type": "no_trans"})
                        if task == "evaluation":
                            test(*call_args, **call_kwargs)
                        else:
                            model_folds_predictions(*call_args, **call_kwargs)


def incremental_test_models(models_options=("regression_with_negatives", "classifier"),
                            train_exclude_on_targets=False, test_exclude_on_targets=False,
                            transfer_learning_types=(None, "add", "update"), seeds=(i for i in range(1, 11))):
    """
    Function for testing the models. The corresponding function to incremental_train_models.
    Save the results in the files directory according to the type if the model.
    :param models_options: tuple. A tuple with the model types to test. support these options:
        ("regression_with_negatives", "classifier").
        Default: ("regression_with_negatives", "classifier")

    :param train_exclude_on_targets: test the models trained without on-targets in case of True. Default: False
    :param test_exclude_on_targets: test on the dataset without the on-targets in case of True. Default: False
    :param transfer_learning_types: bool. Transfer learning types. support these options: (None, "add", "update").
        When None is included, then models are trained with TL. Default: (None, "add", "update")
    :param seeds: tuple. tuple of seeds numbers (n numbers). we are training n time the models, each time we souffle the
        all training set randomly according to the seeds.
    :return: None
    """
    # GUIDE-seq
    random.seed(general_utilities.SEED)
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    try:
        targets = load_order_sg_rnas("GUIDE")
    except FileNotFoundError:
        targets = order_sg_rnas("GUIDE")

    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'exclude_on_targets/' if test_exclude_on_targets else 'include_on_targets/'
    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format("GUIDEseq"), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format("GUIDEseq"), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    for seed in seeds:
        print("seed:", seed)
        # save 10 targets as test set
        test_targets = targets[-10:]
        print("Targets in test:")
        print(test_targets)
        save_model_dir_path_prefix = 'exclude_on_targets/' if train_exclude_on_targets else 'include_on_targets/'
        save_model_dir_path_prefix = "GUIDEseq/incremental_training/" + save_model_dir_path_prefix
        for transfer_learning_type in transfer_learning_types:
            type_save_model_dir_path_prefix = save_model_dir_path_prefix + \
                "GS_TL_" + transfer_learning_type + '/' if transfer_learning_type is not None else \
                save_model_dir_path_prefix + "GS/"
            xgb_model_path = general_utilities.FILES_DIR + "models_1fold/" + \
                "train_CHANGE_seq_on_non_overlapping_targets_with_GUIDE_seq/"
            xgb_model_path += 'exclude_on_targets/' if train_exclude_on_targets else 'include_on_targets/'
            for i in range(len(targets)-10):
                path_prefix = type_save_model_dir_path_prefix + "seed_" + str(seed) + \
                              "/trained_with_" + str(i+1) + "_guides_"

                evaluation_results_path_prefix = path_prefix + "/"
                evaluation_results_path_prefix += 'test_results_exclude_on_targets/' \
                    if test_exclude_on_targets else 'test_results_include_on_targets/'
                if "classifier" in models_options:
                    try:
                        test(positive_df, negative_df, test_targets, nucleotides_to_position_mapping,
                             dataset_type='GUIDEseq', model_type="classifier", k_fold_number=1,
                             include_distance_feature=False, include_sequence_features=True, balanced=False,
                             models_path_prefix=path_prefix, results_path_prefix=evaluation_results_path_prefix)
                    except ValueError as error:
                        print("got exception:", error)
                        print("in seed", seed, "i:", i, "model_type: classifier without distance")
                        print("TL type:", transfer_learning_type)

                    try:
                        test(positive_df, negative_df, test_targets, nucleotides_to_position_mapping,
                             dataset_type='GUIDEseq', model_type="classifier", k_fold_number=1,
                             include_distance_feature=True, include_sequence_features=True, balanced=False,
                             models_path_prefix=path_prefix, results_path_prefix=evaluation_results_path_prefix)
                    except ValueError as error:
                        print("got exception:", error)
                        print("in seed", seed, "i:", i, "model_type: classifier with distance")
                        print("TL type:", transfer_learning_type)

                if "regression_with_negatives" in models_options:
                    try:
                        test(positive_df, negative_df, test_targets, nucleotides_to_position_mapping,
                             dataset_type='GUIDEseq', model_type="regression_with_negatives", k_fold_number=1,
                             include_distance_feature=False, include_sequence_features=True, balanced=False,
                             models_path_prefix=path_prefix, results_path_prefix=evaluation_results_path_prefix)
                    except ValueError as error:
                        print("got exception:", error)
                        print("in seed", seed, "i:", i, "model_type: regression without distance")
                        print("TL type:", transfer_learning_type)

                    try:
                        test(positive_df, negative_df, test_targets, nucleotides_to_position_mapping,
                             dataset_type='GUIDEseq', model_type="regression_with_negatives", k_fold_number=1,
                             include_distance_feature=True, include_sequence_features=True, balanced=False,
                             models_path_prefix=path_prefix, results_path_prefix=evaluation_results_path_prefix)
                    except ValueError as error:
                        print("got exception:", error)
                        print("in seed", seed, "i:", i, "model_type: regression with distance")
                        print("TL type:", transfer_learning_type)


def main():
    """
    main function
    """
    # some testing examples. The modles you are tesing must be exsit before
    # See regular_test_models and incremental_test_models to test other options
    regular_test_models(
        models_options=tuple(("regression_without_negatives",)),
        include_distance_feature_options=(True,),
        include_sequence_features_options=(True,),
        test_with_no_trans=False, k_fold_number=10, task="evaluation",
        data_types=('CHANGEseq', 'GUIDEseq'))
    regular_test_models(
        models_options=tuple(("classifier", "regression_with_negatives")),
        include_distance_feature_options=(True, False),
        include_sequence_features_options=(True,),
        test_with_no_trans=False, k_fold_number=10, task="evaluation",
        data_types=('CHANGEseq', 'GUIDEseq'))

    #to obtain the predictions we change the task into prediction
    regular_test_models(
        models_options=tuple(("regression_without_negatives",)),
        include_distance_feature_options=(True,),
        include_sequence_features_options=(True,),
        test_with_no_trans=False, k_fold_number=10, task="prediction",
        data_types=('CHANGEseq', 'GUIDEseq'))
    regular_test_models(
        models_options=tuple(("classifier", "regression_with_negatives")),
        include_distance_feature_options=(True, False),
        include_sequence_features_options=(True,),
        test_with_no_trans=False, k_fold_number=10, task="prediction",
        data_types=('CHANGEseq', 'GUIDEseq'))


if __name__ == '__main__':
    main()
