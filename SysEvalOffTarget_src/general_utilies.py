import pathlib

###############paths######################
home_dir = str(pathlib.Path(__file__).parent.parent.absolute())+"/" #project path
files_dir = home_dir+"files/"

# change_seq_PATH = files_dir+"datasets/CHANGE-seq.xlsx"
# guide_seq_PATH = files_dir+"datasets/GUIDE-seq.xlsx"
datatsets_PATH = files_dir+"datasets/"