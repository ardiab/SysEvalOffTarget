"""
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
"""

# %%
from pathlib import Path
import pandas as pd
from Bio.Seq import Seq
from SysEvalOffTarget_src import general_utilities


# %%
def create_positives(dataset_excel_path=general_utilities.CHANGE_SEQ_PATH, data_type="CHANGEseq",
                     read_threshold=None, exclude_on_targets=True, save_sets=False):
    """
    create the positive set
    """
    dataset_df = pd.read_excel(dataset_excel_path)
    # exclude bulges
    # drop off targets with len not equal to 23 and with '-'
    dataset_df = dataset_df[dataset_df["offtarget_sequence"].str.len() == 23]
    dataset_df = dataset_df[dataset_df["offtarget_sequence"].str.find('-') == -1]
    # set the condition for splitting to positive and undefined sets
    if read_threshold is not None:
        read_threshold_conds = dataset_df["{}_reads".format(data_type)] <= read_threshold
        undefined_mask = (read_threshold_conds | (dataset_df['distance'] == 0)) if exclude_on_targets \
            else read_threshold_conds
    elif exclude_on_targets:
        undefined_mask = dataset_df['distance'] == 0
    else:
        undefined_mask = None
    # splitting to positive and undefined sets
    if undefined_mask is not None:
        # take the undefined set
        dataset_undefined_df = dataset_df[undefined_mask]
        dataset_undefined_df['label'] = -1
        # take the positive set
        dataset_positive_df = dataset_df[~ undefined_mask]
        dataset_positive_df['label'] = 1
    else:
        dataset_undefined_df = None
        dataset_positive_df = dataset_df
        dataset_positive_df['label'] = 1
    # save the sets
    if save_sets:
        dir_path = general_utilities.DATASETS_PATH
        dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        dataset_positive_df.to_csv(dir_path + '{}_positive.csv'.format(data_type))
        if dataset_undefined_df is not None:
            dataset_undefined_df.to_csv(dir_path + '{}_undefined.csv'.format(data_type))

    return dataset_positive_df, dataset_undefined_df


def create_negatives(experiment_df, cas_offinder_optional_offtargets_path=general_utilities.DATASETS_PATH +
                     "output_file_pam_change.txt", data_type="CHANGEseq", save_sets=False,
                     exclude_on_targets=True):
    """
    create negative set
    """
    negative_df = pd.read_table(cas_offinder_optional_offtargets_path)
    negative_df['label'] = 0
    negative_df = negative_df[['chrom', 'chromStart', 'strand',
                               'offtarget_sequence', 'distance', 'target', 'label']]
    negative_df['offtarget_sequence'] = negative_df['offtarget_sequence'].str.upper()

    print("number of optional off targets before filtering: ", len(negative_df))

    print("Dropping chroms which do not appear in the experiment")
    chroms = experiment_df["chrom"].unique()
    negative_df = negative_df.drop(
        negative_df[~negative_df['chrom'].isin(chroms)].index)
    print("number of optional off targets after this stage: ", len(negative_df))

    print("Dropping off-targets which their target doesn't appear in the experiment_df")
    targets = experiment_df["target"].unique()
    negative_df = negative_df[negative_df["target"].isin(targets)]
    print("number of optional off targets after this stage: ", len(negative_df))

    print(
        "Dropping for each Target the optional off-targets which their sequences\
         (or their reverse complement) appear in the experiment (without connection to chromStart)")
    # The filter on the reverse complement is for very rare
    # situations (probably do not exist due to the number of mismatch allowed)
    for target in targets:
        df_target = experiment_df[experiment_df['target'] == target]
        target_change_seq_off_targets = df_target["offtarget_sequence"]
        target_change_seq_reverse_off_targets = df_target["offtarget_sequence"].apply(
            lambda seq: str(Seq(seq).reverse_complement()))
        negative_df = negative_df.drop(
            negative_df[(negative_df['target'] == target) & ((negative_df['offtarget_sequence'].isin(
                target_change_seq_off_targets)) | (negative_df['offtarget_sequence'].isin(
                    target_change_seq_reverse_off_targets)))].index)
    print("number of optional off targets after this stage: ", len(negative_df))

    # This is done since some CHANGE-seq (or other experiment) off-target
    # share same chromStart with optional off-target from cas-offinder but their sequences does not agree.
    print("Dropping for each chrom the optional off-targets which their chromStart appear in the experiment")
    chroms = experiment_df["chrom"].unique()
    for chrom in chroms:
        negative_df = negative_df.drop(negative_df[(negative_df['chrom'] == chrom) & (
            negative_df['chromStart'].isin(experiment_df["chromStart"]))].index)
    print("number of optional off targets after this stage: ", len(negative_df))

    # The machine learning can not different between same
    # optional off-target with different position for the same target
    print("Dropping for each Target duplicates of optional off-targets")
    targets = experiment_df["target"].unique()
    for target in targets:
        negative_df = negative_df[~(
                (experiment_df['target'] == target) & negative_df['offtarget_sequence'].duplicated())]
    print("number of optional off targets after this stage: ", len(negative_df))

    print("dropping on-targets if exists (at all and after all the previous stages(")
    negative_df = negative_df[negative_df["distance"] != 0]
    print("number of optional off targets after this stage: ", len(negative_df))

    if save_sets:
        dir_path = general_utilities.DATASETS_PATH
        dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        negative_df.to_csv(dir_path + '{}_negative.csv'.format(data_type))

    return negative_df


if __name__ == '__main__':
    # print("create CHANGE-seq dataset")
    # create_positives(dataset_excel_path=general_utilities.CHANGE_SEQ_PATH, data_type="CHANGEseq",
    #                  read_threshold=100, exclude_on_targets=False, save_sets=True)
    # change_seq_df = pd.read_excel(general_utilities.CHANGE_SEQ_PATH)
    # #drop off targets that contains '-'
    # change_seq_df = change_seq_df[change_seq_df["offtarget_sequence"].str.len()==23]
    # change_seq_df = change_seq_df[change_seq_df["offtarget_sequence"].str.find('-')==-1]
    # create_negatives(change_seq_df, cas_offinder_optional_offtargets_path=general_utilities.DATASETS_PATH +
    #                  "output_file_pam_change.txt", data_type="CHANGEseq", save_sets=True, exclude_on_targets=False)

    print("create GUIDE-seq dataset")
    create_positives(dataset_excel_path=general_utilities.GUIDE_SEQ_PATH, data_type="GUIDEseq",
                     read_threshold=None, exclude_on_targets=False, save_sets=True)
    guide_seq_df = pd.read_excel(general_utilities.GUIDE_SEQ_PATH)
    #drop off targets that contains '-' and ith len not equal to 23
    guide_seq_df = guide_seq_df[guide_seq_df["offtarget_sequence"].str.len()==23]
    guide_seq_df = guide_seq_df[guide_seq_df["offtarget_sequence"].str.find('-')==-1]
    create_negatives(guide_seq_df, cas_offinder_optional_offtargets_path=general_utilities.DATASETS_PATH +
                     "output_file_pam_change.txt",  data_type="GUIDEseq", save_sets=True, exclude_on_targets=False)
