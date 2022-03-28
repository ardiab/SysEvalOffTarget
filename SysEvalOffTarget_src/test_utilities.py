"""
test utilities
"""
import random
from pathlib import Path
import xgboost as xgb
import pandas as pd
import numpy as np

from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from SysEvalOffTarget_src import utilities

from SysEvalOffTarget_src.utilities import create_fold_sets, build_sequence_features, extract_model_path, \
    extract_model_results_path, transformer_generator, transform
from SysEvalOffTarget_src import general_utilities

random.seed(general_utilities.SEED)


def score_function_classifier(y_test, y_pred, y_proba):
    """
    compute scores for classification model
    """
    pearson = pearsonr(y_test, y_proba)[0]
    spearman = spearmanr(y_test, y_proba)[0]
    auc = roc_auc_score(y_test, y_proba)
    aupr = average_precision_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = (np.sum(y_test == y_pred) * 1.0 / len(y_test))

    return {"accuracy": accuracy, "auc": auc, "aupr": aupr,
            "precision": precision, "recall": recall,
            "pearson": pearson, "spearman": spearman}


def score_function_reg_classifier(y_test, y_proba):
    """
    compute scores for regression model
    """
    pearson = pearsonr(y_test, y_proba)[0]
    spearman = spearmanr(y_test, y_proba)[0]
    auc = roc_auc_score(y_test, y_proba)
    aupr = average_precision_score(y_test, y_proba)

    return {"reg_to_class_auc": auc, "reg_to_class_aupr": aupr,
            "reg_to_class_pearson": pearson, "reg_to_class_spearman": spearman}


############################################################################


# def load_test_target_data(model_type, data_type, target, negative_df_test, positive_df_test,
#                           nucleotides_to_position_mapping, include_distance_feature=False,
#                           include_sequence_features=True, trans_type="ln_x_plus_one_trans",
#                           transformer=None):
#     """
#     load the test dataset according to the target sequence
#     """
#     if isinstance(target, str):
#         targets = tuple((target,))
#     elif isinstance(target, tuple):
#         targets = target
#     else:
#         raise ValueError('Invalid target type')

#     target_negative_df_test, target_positive_df_test = \
#         negative_df_test[negative_df_test["target"].isin(targets)], \
#         positive_df_test[positive_df_test["target"].isin(targets)]
#     # sequence features
#     negative_sequence_features_test, positive_sequence_features_test = \
#         build_sequence_features(target_negative_df_test, nucleotides_to_position_mapping,
#                                 include_distance_feature=include_distance_feature,
#                                 include_sequence_features=include_sequence_features), \
#         build_sequence_features(target_positive_df_test, nucleotides_to_position_mapping,
#                                 include_distance_feature=include_distance_feature,
#                                 include_sequence_features=include_sequence_features)
#     sequence_features_test = np.concatenate(
#         (negative_sequence_features_test, positive_sequence_features_test))
#     # labels
#     negative_labels_test, positive_labels_test = target_negative_df_test[
#                                                      "label"].values, target_positive_df_test["label"].values
#     sequence_labels_test = np.concatenate(
#         (negative_labels_test, positive_labels_test))
#     negative_reads_test, positive_reads_test = np.zeros(
#         (len(target_negative_df_test),)), target_positive_df_test[data_type + "_reads"].values
#     sequence_reads_test = np.concatenate(
#         (negative_reads_test, positive_reads_test))
#     # perform transform in case of regression
#     sequence_reads_test = data_trans(data=sequence_reads_test, trans_type=trans_type,
#                                      transformer=transformer) if model_type != "classifier" else sequence_reads_test
#     return (negative_sequence_features_test, positive_sequence_features_test, sequence_features_test,
#             negative_labels_test, positive_labels_test, sequence_labels_test, negative_reads_test,
#             positive_reads_test, sequence_reads_test)


##########################################################################
def create_scores_dataframe(model_type):
    """
    create the dataframe that will contain the performance scores
    """
    if model_type == "classifier":
        results_df = pd.DataFrame(columns=["target", "positives", "negatives", "accuracy", "auc",
                                           "aupr", "precision", "recall", "pearson",
                                           "pearson_reads_to_proba_for_positive_set",
                                           "spearman", "spearman_reads_to_proba_for_positive_set"])
    elif model_type == "regression_with_negatives" or model_type == 'regression_without_negatives':
        results_df = pd.DataFrame(
            columns=["target", "positives", "negatives", "pearson", "pearson_after_inv_trans", "pearson_only_positives",
                     "pearson_only_positives_after_inv_trans", "spearman", "spearman_after_inv_trans",
                     "spearman_only_positives", "spearman_only_positives_after_inv_trans",
                     "reg_to_class_auc", "reg_to_class_aupr",
                     "reg_to_class_pearson", "reg_to_class_spearman"])
    else:
        raise ValueError('model_type is invalid.')

    return results_df


def load_fold_dataset(data_type, target_fold, targets, positive_df, negative_df, balanced, evaluate_only_distance,
                      exclude_targets_without_positives):
    """
    load fold dataset
    """
    if data_type in ("CHANGEseq", "GUIDEseq"):
        _, _, negative_df_test, positive_df_test = \
            create_fold_sets(target_fold, targets, positive_df, negative_df, balanced,
                             exclude_targets_without_positives)
    else:
        raise ValueError('data_type is invalid.')
    if evaluate_only_distance is not None:
        negative_df_test, positive_df_test = \
            negative_df_test[negative_df_test["distance"] == evaluate_only_distance], \
            positive_df_test[positive_df_test["distance"] == evaluate_only_distance]

    return negative_df_test, positive_df_test


def load_model(model_type, k_fold_number, fold_index, gpu, trans_type, balanced,
               include_distance_feature, include_sequence_features, path_prefix,
               trans_all_fold, trans_only_positive, exclude_targets_without_positives):
    """
    load model
    """
    if model_type == "classifier":
        model = xgb.XGBClassifier()
    else:
        model = xgb.XGBRegressor()

    # speedup prediction using GPU
    if gpu:
        model.set_params(**{'tree_method': 'gpu_hist'})

    dir_path = extract_model_path(model_type, k_fold_number, include_distance_feature,
                                  include_sequence_features, balanced, trans_type, trans_all_fold,
                                  trans_only_positive, exclude_targets_without_positives,
                                  fold_index, path_prefix)
    model.load_model(dir_path)

    return model


def model_folds_predictions(positive_df, negative_df, targets, nucleotides_to_position_mapping,
                            data_type="CHANGEseq", model_type="classifier", k_fold_number=10,
                            include_distance_feature=False, include_sequence_features=True,
                            balanced=True, trans_type="ln_x_plus_one_trans", trans_all_fold=False,
                            trans_only_positive=False, exclude_targets_without_positives=False,
                            evaluate_only_distance=None, add_to_results_table=False,
                            results_table_path=None, gpu=True, suffix_add="", path_prefix="", save_results=False):
    """
    split targets to fold (if needed) and make predictions
    assumption: if results_table_path is not None, then it has the same format and order as
    positive and negative datasets
    """
    # load the results' table if exist
    try:
        results_df = pd.read_csv(results_table_path) if \
            (add_to_results_table and results_table_path is not None) else None
        dir_path = results_table_path
    except FileNotFoundError:
        results_df = None
        dir_path = results_table_path

    # load the model name
    model_name = utilities.extract_model_name(model_type, include_distance_feature, include_sequence_features,
                                              balanced, trans_type, trans_all_fold, trans_only_positive,
                                              exclude_targets_without_positives)

    # create the predictions df and inset the predictions of the fold models
    predictions_dfs = [pd.DataFrame(), pd.DataFrame()]
    target_folds_list = np.array_split(targets, k_fold_number)
    for i, target_fold in enumerate(target_folds_list):
        # we don't exclude the targets without positives from the prediction stage.
        # if required, it is done in the evaluation stage
        negative_df_test, positive_df_test = load_fold_dataset(data_type, target_fold, targets, positive_df,
                                                               negative_df, balanced=False,
                                                               evaluate_only_distance=evaluate_only_distance,
                                                               exclude_targets_without_positives=False)
        model = load_model(model_type, k_fold_number, i, gpu, trans_type, balanced,
                           include_distance_feature, include_sequence_features, path_prefix,
                           trans_all_fold, trans_only_positive, exclude_targets_without_positives)
        # predict and insert the predictions into the predictions dfs

        for j, dataset_df in enumerate((positive_df_test, negative_df_test)):

            sequence_features_test = build_sequence_features(dataset_df, nucleotides_to_position_mapping,
                                                             include_distance_feature=include_distance_feature,
                                                             include_sequence_features=include_sequence_features)
            if model_type == "classifier":
                predictions = model.predict_proba(sequence_features_test)[:, 1]
            else:
                predictions = model.predict(sequence_features_test)

            dataset_df[model_name] = predictions
            predictions_dfs[j] = predictions_dfs[j].append(dataset_df.copy())

    if add_to_results_table:
        predictions_neg_pos_df = \
            pd.concat(predictions_dfs, axis=0, ignore_index=True)
        if results_df is None:
            results_df = predictions_neg_pos_df
        else:
            results_df[model_name] = predictions_neg_pos_df[model_name]

    if save_results:
        if add_to_results_table and results_table_path is None:
            dir_path = general_utilities.FILES_DIR + "models_" + str(k_fold_number) + \
                "fold/" + path_prefix + data_type + "_results_all_" + \
                str(k_fold_number) + "_folds" + suffix_add + ".csv"
            Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(dir_path, index=False)
        else:
            Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(dir_path, index=False)

    return predictions_dfs[0], predictions_dfs[1], results_df


def data_for_evaluation(target_positive_df, target_negative_df, entire_df, model_name, trans_type, data_type,
                        trans_only_positive, trans_by_all_targets, exclude_targets_without_positives):
    """
    transform the labels and predictions data for evaluation
    """
    data_type = "" if data_type is None else data_type + "_"
    reads_col = "{}reads".format(data_type)
    # it might include, but just confirm:
    target_negative_df[reads_col] = 0
    entire_df.loc[entire_df["label"] == 0, reads_col] = 0

    # if required exclude targets without positives when transformer is learned by all data
    if trans_by_all_targets and exclude_targets_without_positives:
        targets_with_positives = entire_df[entire_df["label"] == 1]["target"].unique()
        entire_df = entire_df[entire_df["target"].isin(targets_with_positives)]

    # in case the target df contain more than one sgRNA,
    # exclude targets without positives when transformer is learned by all data if required
    targets = target_negative_df["target"].unique()
    if len(targets) > 1 and exclude_targets_without_positives:
        targets_with_positives = target_positive_df["target"].unique()
        target_negative_df = target_negative_df[target_negative_df["target"].isin(targets_with_positives)]

    positive_target_preds_and_labels_df = \
        target_positive_df[["target", "offtarget_sequence", "label", reads_col, model_name]]
    negative_target_preds_and_labels_df = \
        target_negative_df[["target", "offtarget_sequence", "label", reads_col, model_name]]
    target_preds_and_labels_df = \
        pd.concat([positive_target_preds_and_labels_df, negative_target_preds_and_labels_df])

    positive_entire_preds_and_labels_df = \
        entire_df[entire_df["label"] == 1][["target", "offtarget_sequence", "label", reads_col, model_name]]
    negative_entire_preds_and_labels_df = \
        entire_df[entire_df["label"] == 0][["target", "offtarget_sequence", "label", reads_col, model_name]]
    entire_preds_and_labels_df = \
        pd.concat([positive_entire_preds_and_labels_df, negative_entire_preds_and_labels_df])

    if trans_by_all_targets:
        # transformer train based on the entire data
        transform_fit_df = entire_preds_and_labels_df
    else:
        transform_fit_df = target_preds_and_labels_df

    for target in target_preds_and_labels_df["target"].unique():
        target_transform_fit_df = transform_fit_df[transform_fit_df["target"] == target]
        if trans_only_positive:
            labels = target_transform_fit_df[target_transform_fit_df["label"] == 1][reads_col].values
        else:
            labels = target_transform_fit_df[reads_col].values
        transformer = transformer_generator(labels, trans_type)

        # apply transformation on both negative and positive samples
        target_preds_and_labels_df.loc[target_preds_and_labels_df["target"] == target, reads_col + "_trans"] = \
            transform(target_preds_and_labels_df[target_preds_and_labels_df["target"] == target][reads_col].values,
                      transformer)
        target_preds_and_labels_df.loc[target_preds_and_labels_df["target"] == target, model_name + "_inverse"] = \
            transform(target_preds_and_labels_df[target_preds_and_labels_df["target"] == target][model_name].values,
                      transformer, inverse=True)

    if trans_only_positive:
        # in case trans only positives, then  undo the transformation for the negative samples
        target_preds_and_labels_df.loc[target_preds_and_labels_df["label"] == 0, reads_col + "_trans"] = \
            target_preds_and_labels_df[target_preds_and_labels_df["label"] == 0][reads_col]
        target_preds_and_labels_df.loc[target_preds_and_labels_df["label"] == 0, model_name + "_inverse"] = \
            target_preds_and_labels_df[target_preds_and_labels_df["label"] == 0][model_name]

    class_labels = target_preds_and_labels_df["label"].values
    positive_read_labels = target_preds_and_labels_df[target_preds_and_labels_df["label"] == 1][reads_col].values
    negative_read_labels = target_preds_and_labels_df[target_preds_and_labels_df["label"] == 0][reads_col].values
    read_labels = target_preds_and_labels_df[reads_col].values
    positive_read_labels_trans = \
        target_preds_and_labels_df[target_preds_and_labels_df["label"] == 1][reads_col + "_trans"].values
    negative_read_labels_trans = \
        target_preds_and_labels_df[target_preds_and_labels_df["label"] == 0][reads_col + "_trans"].values
    read_labels_trans = target_preds_and_labels_df[reads_col + "_trans"].values
    positive_predictions = target_preds_and_labels_df[target_preds_and_labels_df["label"] == 1][model_name].values
    negative_predictions = target_preds_and_labels_df[target_preds_and_labels_df["label"] == 0][model_name].values
    predictions = target_preds_and_labels_df[model_name].values
    positive_predictions_inverse = \
        target_preds_and_labels_df[target_preds_and_labels_df["label"] == 1][model_name + "_inverse"].values
    negative_predictions_inverse = \
        target_preds_and_labels_df[target_preds_and_labels_df["label"] == 0][model_name + "_inverse"].values
    predictions_inverse = target_preds_and_labels_df[model_name + "_inverse"].values

    return class_labels, positive_read_labels, negative_read_labels, read_labels, \
        positive_read_labels_trans, negative_read_labels_trans, read_labels_trans, \
        positive_predictions, negative_predictions, predictions, \
        positive_predictions_inverse, negative_predictions_inverse, predictions_inverse


def classification_evaluation(class_labels, predictions, positive_read_labels, positive_predictions):
    target_scores = score_function_classifier(class_labels, np.where(predictions > 0.5, 1, 0), predictions)
    # test if classifier can predict reads
    if len(positive_predictions) > 1:
        # pearson
        target_scores.update({"pearson_reads_to_proba_for_positive_set": pearsonr(
            positive_read_labels, positive_predictions)[0]})
        # spearman
        target_scores.update({"spearman_reads_to_proba_for_positive_set": spearmanr(
            positive_read_labels, positive_predictions)[0]})
    else:
        # pearson
        target_scores.update({"pearson_reads_to_proba_for_positive_set": np.nan})
        # spearman
        target_scores.update({"spearman_reads_to_proba_for_positive_set": np.nan})

    return target_scores


def regression_evaluation(class_labels, predictions, predictions_inverse, read_labels,
                          read_labels_trans, positive_read_labels, positive_read_labels_trans,
                          positive_predictions, positive_predictions_inverse):
    if len(predictions) > 1:
        # pearson
        target_scores = {"pearson": pearsonr(read_labels_trans, predictions)[0]}
        try:
            target_scores.update({"pearson_after_inv_trans": pearsonr(read_labels, predictions_inverse)[0]})
        except ValueError as e:
            print("Got:", str(e))
            print("Setting pearson_after_inv_trans for this target as nan")
            target_scores.update({"pearson_after_inv_trans": np.nan})
        # spearman
        target_scores.update({"spearman": spearmanr(read_labels_trans, predictions)[0]})
        try:
            target_scores.update({"spearman_after_inv_trans": spearmanr(read_labels, predictions_inverse)[0]})
        except ValueError as e:
            print("Got:", str(e))
            print("Setting spearman_after_inv_trans for this target as nan")
            target_scores.update({"spearman_after_inv_trans": np.nan})
    else:
        # pearson
        target_scores = {"pearson": np.nan}
        target_scores.update({"pearson_after_inv_trans": np.nan})
        # spearman
        target_scores.update({"spearman": np.nan})
        target_scores.update({"spearman_after_inv_trans": np.nan})

    # test corr only on the positive set
    if len(positive_predictions) > 1:
        # positive_reads_test is before the transformation
        # pearson
        target_scores.update({"pearson_only_positives": pearsonr(
            positive_read_labels_trans, positive_predictions)[0]})
        try:
            target_scores.update(
                {"pearson_only_positives_after_inv_trans": pearsonr(
                    positive_read_labels, positive_predictions_inverse)[0]})
        except ValueError as e:
            print("Got:", str(e))
            print("Setting pearson_only_positives_after_inv_trans for this target as nan")
            target_scores.update({"pearson_only_positives_after_inv_trans": np.nan})
        # spearman
        target_scores.update({"spearman_only_positives": spearmanr(
            positive_read_labels_trans, positive_predictions)[0]})
        try:
            target_scores.update(
                {"spearman_only_positives_after_inv_trans": spearmanr(
                    positive_read_labels, positive_predictions_inverse)[0]})
        except ValueError as e:
            print("Got:", str(e))
            print("Setting spearman_only_positives_after_inv_trans for this target as nan")
            target_scores.update({"spearman_only_positives_after_inv_trans": np.nan})
    else:
        # pearson
        target_scores.update(
            {"pearson_only_positives": np.nan})
        target_scores.update(
            {"pearson_only_positives_after_inv_trans": np.nan})
        # spearman
        target_scores.update(
            {"spearman_only_positives": np.nan})
        target_scores.update(
            {"spearman_only_positives_after_inv_trans": np.nan})

    # test if regressor can perform off-target classification
    # normalized_sequence_reads_predicted = \
    #     (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))
    target_scores.update(score_function_reg_classifier(
        class_labels, predictions))

    return target_scores


def compute_evaluation_scores(target, model_type, model_name, target_predictions_positive_df,
                              target_predictions_negative_df, predictions_df, trans_type, data_type,
                              trans_only_positive, trans_by_all_targets, exclude_targets_without_positives):
    class_labels, positive_read_labels, _negative_read_labels, read_labels, \
        positive_read_labels_trans, _negative_read_labels_trans, read_labels_trans, \
        positive_predictions, _negative_predictions, predictions, \
        positive_predictions_inverse, _negative_predictions_inverse, predictions_inverse = \
        data_for_evaluation(target_predictions_positive_df, target_predictions_negative_df, predictions_df,
                            model_name, trans_type=trans_type, data_type=data_type,
                            trans_only_positive=trans_only_positive, trans_by_all_targets=trans_by_all_targets,
                            exclude_targets_without_positives=exclude_targets_without_positives)

    if model_type == "classifier":
        target_scores = classification_evaluation(class_labels, predictions,
                                                  positive_read_labels, positive_predictions)
    else:
        # regression model
        target_scores = regression_evaluation(class_labels, predictions, predictions_inverse, read_labels,
                                              read_labels_trans, positive_read_labels, positive_read_labels_trans,
                                              positive_predictions, positive_predictions_inverse)

    target_scores.update({"target": target})
    target_scores.update(
        {"positives": len(target_predictions_positive_df)})
    target_scores.update(
        {"negatives": len(target_predictions_negative_df)})

    return target_scores


def evaluation(positive_df, negative_df, targets, nucleotides_to_position_mapping, data_type="CHANGEseq",
               model_type="classifier", k_fold_number=10, include_distance_feature=False,
               include_sequence_features=True, balanced=True, trans_type="ln_x_plus_one_trans",
               trans_all_fold=False, trans_only_positive=False, exclude_targets_without_positives=False,
               evaluate_only_distance=None, gpu=True, suffix_add="", models_path_prefix="",
               results_path_prefix=""):
    """
    the test function
    """
    # load the model name
    model_name = utilities.extract_model_name(model_type, include_distance_feature, include_sequence_features,
                                              balanced, trans_type, trans_all_fold, trans_only_positive,
                                              exclude_targets_without_positives)
    # create the scores results dataframe
    results_df = create_scores_dataframe(model_type)

    predictions_positive_df, predictions_negative_df, predictions_df = \
        model_folds_predictions(positive_df, negative_df, targets, nucleotides_to_position_mapping,
                                data_type=data_type, model_type=model_type, k_fold_number=k_fold_number,
                                include_distance_feature=include_distance_feature,
                                include_sequence_features=include_sequence_features, balanced=balanced,
                                trans_type=trans_type, trans_all_fold=trans_all_fold,
                                trans_only_positive=trans_only_positive,
                                exclude_targets_without_positives=exclude_targets_without_positives,
                                evaluate_only_distance=evaluate_only_distance,
                                add_to_results_table=True, results_table_path=None, gpu=gpu, save_results=False,
                                path_prefix=models_path_prefix)

    for target in targets:
        if target not in positive_df["target"].unique():
            # if target is not in the test set
            target_scores = {"target": target}
        else:
            # if target is in the test set
            target_predictions_positive_df = predictions_positive_df[predictions_positive_df["target"] == target]
            target_predictions_negative_df = predictions_negative_df[predictions_negative_df["target"] == target]
            print("target set:", target, ", negatives:", len(target_predictions_negative_df),
                  ", positives:", len(target_predictions_positive_df))
            target_scores = compute_evaluation_scores(target, model_type, model_name, target_predictions_positive_df,
                                                      target_predictions_negative_df, predictions_df, trans_type,
                                                      data_type, trans_only_positive, trans_all_fold,
                                                      exclude_targets_without_positives)
        # write to result dataframe
        results_df = results_df.append(target_scores, ignore_index=True)

    # evaluation of all data
    target_predictions_positive_df = predictions_positive_df[predictions_positive_df["target"].isin(targets)]
    target_predictions_negative_df = predictions_negative_df[predictions_negative_df["target"].isin(targets)]
    print("target set: All Targets", ", negatives:", len(target_predictions_negative_df),
          ", positives:", len(target_predictions_positive_df))
    all_targets_scores = compute_evaluation_scores(
                              "All Targets", model_type, model_name, target_predictions_positive_df,
                              target_predictions_negative_df, predictions_df, trans_type, data_type,
                              trans_only_positive, trans_all_fold, exclude_targets_without_positives)
    results_df = results_df.append(all_targets_scores, ignore_index=True)

    dir_path = extract_model_results_path(model_type, data_type, k_fold_number, include_distance_feature,
                                          include_sequence_features, balanced, trans_type, trans_all_fold,
                                          trans_only_positive, exclude_targets_without_positives,
                                          evaluate_only_distance, suffix_add, results_path_prefix)
    Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(dir_path)

    return results_df


# def test_old(positive_df, negative_df, targets, nucleotides_to_position_mapping, data_type="CHANGEseq",
#              model_type="classifier", k_fold_number=10, include_distance_feature=False, include_sequence_features=True,
#              balanced=True, trans_type="ln_x_plus_one_trans", evaluate_only_distance=None, gpu=True, suffix_add="",
#              transformer=None, models_path_prefix="", results_path_prefix=""):
#     """
#     the test function
#     """
#     results_df = create_scores_dataframe(model_type)

#     target_folds_list = np.array_split(targets, k_fold_number)

#     total_positives, total_negatives = 0, 0
#     predicted_labels_list, predicted_labels_proba_list, true_labels_list, positive_true_reads_list, \
#         positive_predicted_reads_list, positive_prediction_proba_list, predicted_reads_list, true_reads_list = \
#         [], [], [], [], [], [], [], []
#     for i, target_fold in enumerate(target_folds_list):
#         negative_df_test, positive_df_test = load_fold_dataset(data_type, target_fold, targets, positive_df,
#                                                                negative_df, balanced=False,
#                                                                evaluate_only_distance=evaluate_only_distance)
#         model = load_model(model_type, k_fold_number, i, gpu, trans_type, balanced,
#                            include_distance_feature, include_sequence_features, models_path_prefix)
#         for target in target_fold:
#             if target in positive_df["target"].unique():
#                 # #############load target data####################
#                 negative_sequence_features_test, positive_sequence_features_test, \
#                     sequence_features_test, _negative_labels_test, _positive_labels_test, \
#                     sequence_labels_test, _negative_reads_test, positive_reads_test, sequence_reads_test = \
#                     load_test_target_data(model_type, data_type, target, negative_df_test, positive_df_test,
#                                           nucleotides_to_position_mapping,
#                                           include_distance_feature=include_distance_feature,
#                                           include_sequence_features=include_sequence_features, trans_type=trans_type,
#                                           transformer=transformer)
#                 print("target set:", target, ", negatives:", len(
#                     negative_sequence_features_test), ", positives:", len(positive_sequence_features_test))
#                 total_positives += len(positive_sequence_features_test)
#                 total_negatives += len(negative_sequence_features_test)

#                 # #############target prediction and evaluation########
#                 if model_type == "classifier":
#                     sequence_labels_predicted = model.predict(
#                         sequence_features_test)
#                     sequence_labels_predicted_proba = model.predict_proba(sequence_features_test)[:, 1]
#                     target_scores = score_function_classifier(
#                         sequence_labels_test, sequence_labels_predicted, sequence_labels_predicted_proba)
#                     # test if classifier can predict reads
#                     positive_sequence_labels_predicted_proba = model.predict_proba(
#                         positive_sequence_features_test)[:, 1]
#                     if len(positive_sequence_labels_predicted_proba) > 1:
#                         # pearson
#                         target_scores.update({"pearson_reads_to_proba_for_positive_set": pearsonr(
#                             positive_reads_test, positive_sequence_labels_predicted_proba)[0]})
#                         # spearman
#                         target_scores.update({"spearman_reads_to_proba_for_positive_set": spearmanr(
#                             positive_reads_test, positive_sequence_labels_predicted_proba)[0]})
#                     else:
#                         # pearson
#                         target_scores.update({"pearson_reads_to_proba_for_positive_set": np.nan})
#                         # spearman
#                         target_scores.update({"spearman_reads_to_proba_for_positive_set": np.nan})

#                     true_labels_list.append(sequence_labels_test)
#                     predicted_labels_list.append(sequence_labels_predicted)
#                     predicted_labels_proba_list.append(sequence_labels_predicted_proba)
#                     positive_true_reads_list.append(positive_reads_test)
#                     positive_prediction_proba_list.append(positive_sequence_labels_predicted_proba)
#                 else:
#                     sequence_reads_predicted = model.predict(
#                         sequence_features_test)
#                     if len(sequence_reads_predicted) > 1:
#                         # pearson
#                         target_scores = {"pearson": pearsonr(
#                             sequence_reads_test, sequence_reads_predicted)[0]}
#                         target_scores.update({"pearson_after_inv_trans": pearsonr(
#                             data_trans(data=sequence_reads_test, trans_type=trans_type, transformer=transformer,
#                                        inverse=True), data_trans(
#                                 data=sequence_reads_predicted, trans_type=trans_type, transformer=transformer,
#                                 inverse=True))[0]})
#                         # spearman
#                         target_scores.update({"spearman": spearmanr(
#                             sequence_reads_test, sequence_reads_predicted)[0]})
#                         target_scores.update({"spearman_after_inv_trans": spearmanr(
#                             data_trans(data=sequence_reads_test, trans_type=trans_type, transformer=transformer,
#                                        inverse=True), data_trans(
#                                 data=sequence_reads_predicted, trans_type=trans_type, transformer=transformer,
#                                 inverse=True))[0]})
#                     else:
#                         # pearson
#                         target_scores = {"pearson": np.nan}
#                         target_scores.update(
#                             {"pearson_after_inv_trans": np.nan})
#                         # spearman
#                         target_scores.update({"spearman": np.nan})
#                         target_scores.update(
#                             {"spearman_after_inv_trans": np.nan})
#                     # test corr only on the positive set
#                     positive_sequence_reads_predicted = model.predict(
#                         positive_sequence_features_test)
#                     if len(positive_sequence_reads_predicted) > 1:
#                         # positive_reads_test is before the transformation
#                         # pearson
#                         target_scores.update({"pearson_only_positives": pearsonr(data_trans(data=positive_reads_test,
#                                                                                             trans_type=trans_type,
#                                                                                             transformer=transformer),
#                                                                                  positive_sequence_reads_predicted)[0]})
#                         target_scores.update(
#                             {"pearson_only_positives_after_inv_trans": pearsonr(positive_reads_test, data_trans(
#                                 data=positive_sequence_reads_predicted, trans_type=trans_type, transformer=transformer,
#                                 inverse=True))[0]})
#                         # spearman
#                         target_scores.update({"spearman_only_positives": spearmanr(data_trans(data=positive_reads_test,
#                                                                                             trans_type=trans_type,
#                                                                                             transformer=transformer),
#                                                                                  positive_sequence_reads_predicted)[0]})
#                         target_scores.update(
#                             {"spearman_only_positives_after_inv_trans": spearmanr(positive_reads_test, data_trans(
#                                 data=positive_sequence_reads_predicted, trans_type=trans_type, transformer=transformer,
#                                 inverse=True))[0]})
#                     else:
#                         # pearson
#                         target_scores.update(
#                             {"pearson_only_positives": np.nan})
#                         target_scores.update(
#                             {"pearson_only_positives_after_inv_trans": np.nan})
#                         # spearman
#                         target_scores.update(
#                             {"spearman_only_positives": np.nan})
#                         target_scores.update(
#                             {"spearman_only_positives_after_inv_trans": np.nan})
#                     # test if regressor can perform off-target classification
#                     normalized_sequence_reads_predicted = (sequence_reads_predicted - np.min(
#                         sequence_reads_predicted)) / (np.max(sequence_reads_predicted) -
#                                                       np.min(sequence_reads_predicted))
#                     target_scores.update(score_function_reg_classifier(
#                         sequence_labels_test, normalized_sequence_reads_predicted))

#                     predicted_reads_list.append(sequence_reads_predicted)
#                     positive_predicted_reads_list.append(positive_sequence_reads_predicted)
#                     true_reads_list.append(sequence_reads_test)
#                     positive_true_reads_list.append(positive_reads_test)
#                     true_labels_list.append(sequence_labels_test)

#                 target_scores.update({"target": target})
#                 target_scores.update(
#                     {"positives": len(positive_sequence_features_test)})
#                 target_scores.update(
#                     {"negatives": len(negative_sequence_features_test)})
#             else:  # if target is not in the test set
#                 target_scores = {"target": target}
#             # write to results dataframe
#             results_df = results_df.append(target_scores, ignore_index=True)

#     # ########evluation of all data#################
#     if model_type == "classifier":
#         true_labels_all_targets = np.concatenate(true_labels_list)
#         predicted_labels_all_targets = np.concatenate(predicted_labels_list)
#         predicted_labels_proba_all_targets = np.concatenate(
#             predicted_labels_proba_list)
#         positive_true_reads_all_targets = np.concatenate(
#             positive_true_reads_list)
#         positive_prediction_proba_all_targets = np.concatenate(
#             positive_prediction_proba_list)

#         all_targets_scores = score_function_classifier(
#             true_labels_all_targets, predicted_labels_all_targets, predicted_labels_proba_all_targets)
#         # pearson
#         all_targets_scores.update({"pearson_reads_to_proba_for_positive_set": pearsonr(
#             positive_true_reads_all_targets, positive_prediction_proba_all_targets)[0]})
#         # spearman
#         all_targets_scores.update({"spearman_reads_to_proba_for_positive_set": spearmanr(
#             positive_true_reads_all_targets, positive_prediction_proba_all_targets)[0]})
#     else:
#         predicted_reads_all_targets = np.concatenate(predicted_reads_list)
#         positive_predicted_reads_all_targets = np.concatenate(
#             positive_predicted_reads_list)
#         true_reads_all_targets = np.concatenate(true_reads_list)
#         positive_true_reads_all_targets = np.concatenate(
#             positive_true_reads_list)
#         true_labels_all_targets = np.concatenate(true_labels_list)

#         # pearson
#         all_targets_scores = {"pearson": pearsonr(
#             true_reads_all_targets, predicted_reads_all_targets)[0]}
#         all_targets_scores.update({"pearson_after_inv_trans": pearsonr(
#             data_trans(data=true_reads_all_targets, trans_type=trans_type, transformer=transformer,
#                        inverse=True),
#             data_trans(data=predicted_reads_all_targets, trans_type=trans_type, transformer=transformer, inverse=True))[
#             0]})
#         all_targets_scores.update(
#             {"pearson_only_positives": pearsonr(data_trans(data=positive_true_reads_all_targets, trans_type=trans_type,
#                                                            transformer=transformer),
#                                                 positive_predicted_reads_all_targets)[
#                 0]})  # positive_true_reads_all_targets is before the transformation
#         all_targets_scores.update(
#             {"pearson_only_positives_after_inv_trans": pearsonr(positive_true_reads_all_targets, data_trans(
#                 data=positive_predicted_reads_all_targets, trans_type=trans_type, transformer=transformer,
#                 inverse=True))[0]})
#         # spearman
#         all_targets_scores.update({"spearman": spearmanr(
#             true_reads_all_targets, predicted_reads_all_targets)[0]})
#         all_targets_scores.update({"spearman_after_inv_trans": spearmanr(
#             data_trans(data=true_reads_all_targets, trans_type=trans_type, transformer=transformer,
#                        inverse=True),
#             data_trans(data=predicted_reads_all_targets, trans_type=trans_type, transformer=transformer, inverse=True))[
#             0]})
#         all_targets_scores.update(
#             {"spearman_only_positives": spearmanr(data_trans(data=positive_true_reads_all_targets, trans_type=trans_type,
#                                                            transformer=transformer),
#                                                 positive_predicted_reads_all_targets)[
#                 0]})  # positive_true_reads_all_targets is before the transformation
#         all_targets_scores.update(
#             {"spearman_only_positives_after_inv_trans": spearmanr(positive_true_reads_all_targets, data_trans(
#                 data=positive_predicted_reads_all_targets, trans_type=trans_type, transformer=transformer,
#                 inverse=True))[0]})

#         normalized_predicted_reads_all_targets = (predicted_reads_all_targets - np.min(
#             predicted_reads_all_targets)) / (np.max(predicted_reads_all_targets) - np.min(predicted_reads_all_targets))
#         all_targets_scores.update(score_function_reg_classifier(
#             true_labels_all_targets, normalized_predicted_reads_all_targets))

#     all_targets_scores.update({"target": "All Targets"})
#     all_targets_scores.update({"positives": total_positives})
#     all_targets_scores.update({"negatives": total_negatives})
#     results_df = results_df.append(all_targets_scores, ignore_index=True)

#     suffix = "_with_distance" if include_distance_feature else ""
#     suffix = suffix + "" if include_sequence_features else "_without_sequence_features"
#     suffix = suffix + ("" if balanced == 1 else "_imbalanced")
#     suffix = suffix + ("" if (trans_type == "ln_x_plus_one_trans" or model_type ==
#                               "classifier") else ("_" + trans_type))
#     suffix = suffix + ("" if evaluate_only_distance is None else "_distance_" + str(evaluate_only_distance))
#     suffix = suffix + suffix_add
#     dir_path = general_utilities.FILES_DIR + "models_" + str(k_fold_number) +\
#         "fold/" + results_path_prefix + data_type + "_" + model_type +\
#         "_results_xgb_model_all_" + str(k_fold_number) + "_folds" + suffix + ".csv"
#     Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
#     results_df.to_csv(dir_path)

#     return results_df
