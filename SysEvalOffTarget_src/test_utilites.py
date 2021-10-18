from SysEvalOffTarget_src.utilies import create_fold_sets, build_sequence_features, data_trans
from SysEvalOffTarget_src import general_utilies
from scipy.stats.stats import pearsonr
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path


import random
random.seed(10)

##########################################################################


def score_function_classifier(y_test, y_pred, y_proba):
    pearson = pearsonr(y_test, y_proba)[0]
    auc = roc_auc_score(y_test, y_proba)
    aupr = average_precision_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = (np.sum(y_test == y_pred)*1.0 / len(y_test))

    return {"accuracy": accuracy, "auc": auc, "aupr": aupr, "precision": precision, "recall": recall, "pearson": pearson}


def score_function_reg_classifier(y_test, y_proba):
    pearson = pearsonr(y_test, y_proba)[0]
    auc = roc_auc_score(y_test, y_proba)
    aupr = average_precision_score(y_test, y_proba)

    return {"reg_to_class_auc": auc, "reg_to_class_aupr": aupr, "reg_to_class_pearson": pearson}

############################################################################


def load_test_target_data(type, dataset_type, target, negative_df_test, positive_df_test, nucleotides_to_position_mapping, include_distance_feature=False, include_sequence_features=True, trans_type="ln_x_plus_one_trans", transformer=None):
    target_negative_df_test, target_positive_df_test = negative_df_test[negative_df_test[
        "target"] == target], positive_df_test[positive_df_test["target"] == target]
    # sequence features
    negative_sequence_features_test, positive_sequence_features_test = build_sequence_features(target_negative_df_test, nucleotides_to_position_mapping, include_distance_feature=include_distance_feature, include_sequence_features=include_sequence_features), build_sequence_features(
        target_positive_df_test, nucleotides_to_position_mapping, include_distance_feature=include_distance_feature, include_sequence_features=include_sequence_features)
    sequence_features_test = np.concatenate(
        (negative_sequence_features_test, positive_sequence_features_test))
    # labels
    negative_labels_test, positive_labels_test = target_negative_df_test[
        "label"].values, target_positive_df_test["label"].values
    sequence_labels_test = np.concatenate(
        (negative_labels_test, positive_labels_test))
    negative_reads_test, positive_reads_test = np.zeros(
        (len(target_negative_df_test),)), target_positive_df_test[dataset_type+"_reads"].values
    sequence_reads_test = np.concatenate(
        (negative_reads_test, positive_reads_test))
    sequence_reads_test = data_trans(data=sequence_reads_test, trans_type=trans_type,
                                     transformer=transformer) if type != "classifier" else sequence_reads_test  # perform transforme in case of regression
    return negative_sequence_features_test, positive_sequence_features_test, sequence_features_test, negative_labels_test, positive_labels_test, sequence_labels_test, negative_reads_test, positive_reads_test, sequence_reads_test

##########################################################################


def test(positive_df, negative_df, targets, nucleotides_to_position_mapping, dataset_type="CHANGEseq", type="classifier", K_fold_number=10, include_distance_feature=False, include_sequence_features=True, balanced=True, trans_type="ln_x_plus_one_trans", evalutate_only_distance=None, GPU=True, suffix_add="", transformer=None, path_prefix=""):
    if(type == "classifier"):
        results_df = pd.DataFrame(columns=["target", "positives", "negatives", "accuracy", "auc",
                                           "aupr", "precision", "recall", "pearson", "pearson_reads_to_proba_for_positive_set"])
    elif(type == "regression_with_negatives" or type == 'regression_without_negatives'):
        results_df = pd.DataFrame(columns=["target", "positives", "negatives", "pearson", "pearson_after_inv_trans", "pearson_only_positives",
                                           "pearson_only_positives_after_inv_trans", "reg_to_class_auc", "reg_to_class_aupr", "reg_to_class_pearson"])
    else:
        raise ValueError('type is invalid.')

    target_folds_list = np.array_split(targets, K_fold_number)

    total_positives, total_negatives = 0, 0
    predicted_labels_list, predicted_labels_proba_list, true_labels_list, positive_true_reads_list, positive_predicted_reads_list, positive_prediction_proba_list, predicted_reads_list, true_reads_list = [], [], [], [], [], [], [], []
    for i, target_fold in enumerate(target_folds_list):
        if(dataset_type == "CHANGEseq"):
            _, _, negative_df_test, positive_df_test = create_fold_sets(
                target_fold, targets, positive_df, negative_df, balanced)
        elif(dataset_type == "GUIDEseq"):
            negative_df_test, positive_df_test = negative_df, positive_df
        else:
            raise ValueError('dataset_type is invalid.')
        # load model
        if(evalutate_only_distance is not None):
            negative_df_test, positive_df_test = negative_df_test[negative_df_test["distance"] ==
                                                                  evalutate_only_distance], positive_df_test[positive_df_test["distance"] == evalutate_only_distance]

        if(type == "classifier"):
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor()

        # speedup prediction using GPU
        if(GPU):
            model.set_params(tree_method='gpu_hist')

        suffix = "_with_distance" if include_distance_feature else ""
        suffix = suffix + "" if include_sequence_features else "_without_sequence_features"
        suffix = suffix + ("_without_Kfold" if K_fold_number == 1 else "")
        suffix = suffix + ("" if balanced == 1 else "_imbalanced")
        if (trans_type != "ln_x_plus_one_trans" and type != "classifier"):
            suffix = suffix + "_"+trans_type
        model.load_model(general_utilies.files_dir+"models_"+str(K_fold_number) +
                         "fold/"+path_prefix+type+"_xgb_model_fold_"+str(i)+suffix+".xgb")

        for target in target_fold:
            if (target in positive_df["target"].unique()):
                ##############load target data####################
                negative_sequence_features_test, positive_sequence_features_test, sequence_features_test, negative_labels_test, positive_labels_test, sequence_labels_test, negative_reads_test, positive_reads_test, sequence_reads_test = \
                    load_test_target_data(type, dataset_type, target, negative_df_test, positive_df_test, nucleotides_to_position_mapping,
                                          include_distance_feature=include_distance_feature, include_sequence_features=include_sequence_features, trans_type=trans_type, transformer=transformer)
                print("target set:", target, ", negatives:", len(
                    negative_sequence_features_test), ", positives:", len(positive_sequence_features_test))
                total_positives += len(positive_sequence_features_test)
                total_negatives += len(negative_sequence_features_test)

                ##############target prediction and evluation########
                if(type == "classifier"):
                    sequence_labels_predicted = model.predict(
                        sequence_features_test)
                    sequence_labels_predicted_proba = model.predict_proba(sequence_features_test)[
                        :, 1]
                    target_scores = score_function_classifier(
                        sequence_labels_test, sequence_labels_predicted, sequence_labels_predicted_proba)
                    # test if classifier can predict reads
                    positive_sequence_labels_predicted_proba = model.predict_proba(
                        positive_sequence_features_test)[:, 1]
                    if (len(positive_sequence_labels_predicted_proba) > 1):
                        target_scores.update({"pearson_reads_to_proba_for_positive_set": pearsonr(
                            positive_reads_test, positive_sequence_labels_predicted_proba)[0]})
                    else:
                        target_scores.update(
                            {"pearson_reads_to_proba_for_positive_set": np.nan})

                    true_labels_list.append(sequence_labels_test)
                    predicted_labels_list.append(sequence_labels_predicted)
                    predicted_labels_proba_list.append(
                        sequence_labels_predicted_proba)
                    positive_true_reads_list.append(positive_reads_test)
                    positive_prediction_proba_list.append(
                        positive_sequence_labels_predicted_proba)
                else:
                    sequence_reads_predicted = model.predict(
                        sequence_features_test)
                    if (len(sequence_reads_predicted) > 1):
                        target_scores = {"pearson": pearsonr(
                            sequence_reads_test, sequence_reads_predicted)[0]}
                        target_scores.update({"pearson_after_inv_trans": pearsonr(data_trans(data=sequence_reads_test, trans_type=trans_type, transformer=transformer, inverse=True), data_trans(
                            data=sequence_reads_predicted, trans_type=trans_type, transformer=transformer, inverse=True))[0]})
                    else:
                        target_scores = {"pearson": np.nan}
                        target_scores.update(
                            {"pearson_after_inv_trans": np.nan})
                    # test corr only on the positive set
                    positive_sequence_reads_predicted = model.predict(
                        positive_sequence_features_test)
                    if (len(positive_sequence_reads_predicted) > 1):
                        target_scores.update({"pearson_only_positives": pearsonr(data_trans(data=positive_reads_test, trans_type=trans_type,
                                                                                            transformer=transformer), positive_sequence_reads_predicted)[0]})  # positive_reads_test is before the transformation
                        target_scores.update({"pearson_only_positives_after_inv_trans": pearsonr(positive_reads_test, data_trans(
                            data=positive_sequence_reads_predicted, trans_type=trans_type, transformer=transformer, inverse=True))[0]})
                    else:
                        target_scores.update(
                            {"pearson_only_positives": np.nan})
                        target_scores.update(
                            {"pearson_only_positives_after_inv_trans": np.nan})
                    # test if regressor can perform off-target classification
                    normalized_sequence_reads_predicted = (sequence_reads_predicted-np.min(
                        sequence_reads_predicted))/(np.max(sequence_reads_predicted)-np.min(sequence_reads_predicted))
                    target_scores.update(score_function_reg_classifier(
                        sequence_labels_test, normalized_sequence_reads_predicted))

                    predicted_reads_list.append(sequence_reads_predicted)
                    positive_predicted_reads_list.append(
                        positive_sequence_reads_predicted)
                    true_reads_list.append(sequence_reads_test)
                    positive_true_reads_list.append(positive_reads_test)
                    true_labels_list.append(sequence_labels_test)

                target_scores.update({"target": target})
                target_scores.update(
                    {"positives": len(positive_sequence_features_test)})
                target_scores.update(
                    {"negatives": len(negative_sequence_features_test)})
            else:  # if target is not in the test set
                target_scores = {"target": target}
            # write to restults dataframe
            results_df = results_df.append(target_scores, ignore_index=True)

    #########evluation of all data#################
    if(type == "classifier"):
        true_labels_all_targets = np.concatenate(true_labels_list)
        predicted_labels_all_targets = np.concatenate(predicted_labels_list)
        predicted_labels_proba_all_targets = np.concatenate(
            predicted_labels_proba_list)
        positive_true_reads_all_targets = np.concatenate(
            positive_true_reads_list)
        positive_prediction_proba_all_targets = np.concatenate(
            positive_prediction_proba_list)

        all_targets_scores = score_function_classifier(
            true_labels_all_targets, predicted_labels_all_targets, predicted_labels_proba_all_targets)

        all_targets_scores.update({"pearson_reads_to_proba_for_positive_set": pearsonr(
            positive_true_reads_all_targets, positive_prediction_proba_all_targets)[0]})
    else:
        predicted_reads_all_targets = np.concatenate(predicted_reads_list)
        positive_predicted_reads_all_targets = np.concatenate(
            positive_predicted_reads_list)
        true_reads_all_targets = np.concatenate(true_reads_list)
        positive_true_reads_all_targets = np.concatenate(
            positive_true_reads_list)
        true_labels_all_targets = np.concatenate(true_labels_list)

        all_targets_scores = {"pearson": pearsonr(
            true_reads_all_targets, predicted_reads_all_targets)[0]}
        all_targets_scores.update({"pearson_after_inv_trans": pearsonr(data_trans(data=true_reads_all_targets, trans_type=trans_type, transformer=transformer,
                                                                                  inverse=True), data_trans(data=predicted_reads_all_targets, trans_type=trans_type, transformer=transformer, inverse=True))[0]})
        all_targets_scores.update({"pearson_only_positives": pearsonr(data_trans(data=positive_true_reads_all_targets, trans_type=trans_type,
                                                                                 transformer=transformer), positive_predicted_reads_all_targets)[0]})  # positive_true_reads_all_targets is before the transformation
        all_targets_scores.update({"pearson_only_positives_after_inv_trans": pearsonr(positive_true_reads_all_targets, data_trans(
            data=positive_predicted_reads_all_targets, trans_type=trans_type, transformer=transformer, inverse=True))[0]})

        normalized_predicted_reads_all_targets = (predicted_reads_all_targets-np.min(
            predicted_reads_all_targets))/(np.max(predicted_reads_all_targets)-np.min(predicted_reads_all_targets))
        all_targets_scores.update(score_function_reg_classifier(
            true_labels_all_targets, normalized_predicted_reads_all_targets))

    all_targets_scores.update({"target": "All Targets"})
    all_targets_scores.update({"positives": total_positives})
    all_targets_scores.update({"negatives": total_negatives})
    results_df = results_df.append(all_targets_scores, ignore_index=True)

    suffix = "_with_distance" if include_distance_feature else ""
    suffix = suffix + "" if include_sequence_features else "_without_sequence_features"
    suffix = suffix + ("" if balanced == 1 else "_imbalanced")
    suffix = suffix + ("" if (trans_type == "ln_x_plus_one_trans" or type ==
                                            "classifier") else ("_" + trans_type))
    suffix = suffix + \
        ("" if evalutate_only_distance is None else "_distance_" +
         str(evalutate_only_distance))
    suffix = suffix + suffix_add
    dir_path = general_utilies.files_dir + "models_" + \
        str(K_fold_number) + "fold/" + path_prefix + dataset_type+"_"+type + \
        "_results_xgb_model_all_"+str(K_fold_number)+"_folds"+suffix+".csv"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(dir_path)

    return results_df
