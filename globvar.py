import os

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, "data")
CLONEVOL_DIR = os.path.join(DATA_DIR, "clonevol")
TREES_DIR = os.path.join(DATA_DIR, "trees")
PICKLE_DIR = os.path.join(ROOT_DIR, "pickles")
DATASET_DIR = os.path.join(PICKLE_DIR, "dataset")
ENC_DIR = os.path.join(PICKLE_DIR, "ontologies")
CKPT_DIR = os.path.join(ROOT_DIR, 'ckpts/')
