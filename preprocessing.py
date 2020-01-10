import sys
import os
import numpy as np
import argparse
from ast import literal_eval
from itertools import permutations
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

import globvar
from utils.tree import Node, Tree
from utils.pickling import load_pkl, save_pkl
from utils.readers import *


def build_trees():
    cols = ["lab", "parent", "is.term", "events"]
    for name in os.listdir(globvar.CLONEVOL_DIR):
        df = read_tsv(os.path.join(globvar.CLONEVOL_DIR, name))[cols]
        events = df.events.copy()

        def fun(x):
            return [y.split("_")[0] for y in x.split(",")]
        df.events = events.apply(fun)

        def fun(x):
            return x - 1
        df.lab = df.lab.apply(fun)

        def fun(x):
            return x - 1 if x >= 0 else x
        df.parent = df.parent.apply(fun)

        new_cols = {"lab": "id", "parent": "parent_id",
                    "is.term": "is_leaf", "events": "genes"}

        df = df.rename(mapper=new_cols, axis="columns")
        write_csv(df, os.path.join(globvar.TREES_DIR, name))


def augment_dataset(tree, enc_type, max_iter=10):
    nodelist = [[] for _ in range(max_iter)]
    nodes = tree._nodelist[:]
    for node in nodes:
        for i in range(max_iter):
            idx = range(len(node.genes))
            perm = np.random.permutation(idx)
            node.encoding = node.encoding[idx, :]
            node.genes = [node.genes[j] for j in idx]
            nodelist[i].append(node)
    assert len(tree) == min([len(nl) for nl in nodelist])
    return [Tree(nl, tree.label) for nl in nodelist]


def parse_tree(path, target, ontology, enc_type):
    df = read_csv(path)

    nodes = []
    for (_, row) in df.iterrows():
        genes = literal_eval(row.genes)
        actual_genes = [g for g in genes if g in ontology.index]
        enc = ontology.loc[actual_genes, :].astype(float).values
        if enc.sum() == 0:
            return None

        if enc_type == "mean":
            mean = enc.mean(axis=0)
            if np.isnan(mean).sum() > 0:
                return None
            enc = mean.reshape(1, -1)

        nodes.append(Node(
            id=row.id,
            parent_id=row.parent_id,
            genes=actual_genes,
            encoding=enc,
            is_leaf=row.is_leaf
        ))
    if nodes != []:
        return Tree(nodes, target)
    else:
        return None


def get_trees(pairs, ontology, enc_type, augment, augment_max_iter):
    trees = []
    for (filename, target) in pairs:
        path = os.path.join(globvar.TREES_DIR, filename)
        tree = parse_tree(path, target, ontology, enc_type)
        if tree is not None:
            if augment:
                treelist = augment_dataset(
                    tree, enc_type, max_iter=augment_max_iter)
                trees += treelist
            else:
                trees.append(tree)
    return trees


SUBTYPES = ["LumA", "LumB", "claudin-low", "Basal", "Normal", "Her2"]


def clean_dataset(clinical, target_col):
    clinical["outcome"] = (clinical["Overall_Survival_(Months)"] * 30) > 2000
    clinical.outcome = clinical.outcome.astype(int)
    clinical[clinical.Tumor_Stage == 0] = 1
    clinical[clinical.Tumor_Stage == 4] = 3
    clinical.Tumor_Stage -= clinical.Tumor_Stage.min()
    if target_col == "Pam50_+_Claudin-low_subtype":
        clinical = clinical[clinical[target_col].isin(SUBTYPES)]
    clinical = clinical[~clinical[target_col].isnull()]
    return clinical


def parse_dataset(tune_subset, enc_type, augment, augment_max_iter,
                  gene_ontology, target_col, regression):
    clinical_path = os.path.join(globvar.DATA_DIR, "patientDataAll.txt")
    tumoridmap_path = os.path.join(globvar.DATA_DIR, "tumorIdMap.txt")
    filenames = os.listdir(globvar.TREES_DIR)
    train_file = os.path.join(globvar.DATASET_DIR, "train.pkl")
    dev_file = os.path.join(globvar.DATASET_DIR, "dev.pkl")
    ontology_file = os.path.join(globvar.ENC_DIR, gene_ontology)

    clinical = clean_dataset(read_tsv(clinical_path), target_col)
    idmap = read_tsv(tumoridmap_path)
    ontology = read_pickle(ontology_file)

    if not regression:
        print("num_classes", clinical[target_col].unique().shape[0])
        clinical[target_col] = LE().fit_transform(clinical[target_col])
        print(clinical[target_col].value_counts())

    data = []
    for filename in filenames:
        sample = filename.split(".")[0]
        mb_id = idmap[idmap["sample"] == sample].metabricId.squeeze()
        if len(mb_id) > 0:
            target = clinical[clinical.Patient_ID == mb_id][target_col]
            if not target.empty:
                data.append((filename, target.squeeze()))

    targets = [d[1] for d in data]
    if regression:
        sss = ShuffleSplit(n_splits=1, test_size=tune_subset)
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=tune_subset)
    train_idx, dev_idx = next(sss.split(targets, targets))
    train_pairs = [data[i] for i in train_idx]
    dev_pairs = [data[i] for i in dev_idx]

    train_trees = get_trees(train_pairs, ontology,
                            enc_type, augment, augment_max_iter)
    dev_trees = get_trees(dev_pairs, ontology, enc_type,
                          augment, augment_max_iter)
    save_pkl(train_trees, train_file)
    save_pkl(dev_trees, dev_file)
    print(len(train_trees), len(dev_trees))
    return train_trees, dev_trees


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_subset',
                        type=float,
                        help='Size of subset to select from tuning data.',
                        default=0.1)
    parser.add_argument('--enc_type',
                        type=str,
                        help='Type of gene encoding.',
                        default="rnn",
                        choices=["rnn", "mean"])
    parser.add_argument('--augment',
                        help='Whether to augment the dataset.',
                        action="store_true")
    parser.add_argument('--augment_max_iter',
                        help='How much to augment the dataset.',
                        type=int,
                        default=10)
    parser.add_argument('--gene_ontology',
                        type=str,
                        help='Gene ontology pickle filename.',
                        default='GO_Biological_Process.pkl')
    parser.add_argument('--target_col',
                        type=str,
                        help='Label in the clinical dataset to be used as target.',
                        default='outcome')
    parser.add_argument('--regression',
                        action='store_true',
                        help='Whether this is a regression task or not.')
    parser.add_argument('--rebuild_trees',
                        action='store_true',
                        help='Ignore rest of commands and rebuild tree data.')

    args = parser.parse_args()
    print(args)
    if args.rebuild_trees:
        build_trees()
    else:
        parse_dataset(tune_subset=args.tune_subset,
                      enc_type=args.enc_type,
                      augment=args.augment,
                      gene_ontology=args.gene_ontology,
                      target_col=args.target_col,
                      augment_max_iter=args.augment_max_iter,
                      regression=args.regression)
