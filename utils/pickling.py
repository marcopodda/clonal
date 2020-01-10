import os
import pickle


def load_pkl(path):
    return pkl.load(open(path), "rb")


def save_pkl(obj, path):
    pickle.dump(obj, open(path, "wb"))


def load(pkl_dir, pkl_name):
    file_path = os.path.join(pkl_dir, pkl_name)
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
            return obj
    except FileNotFoundError:
        raise Exception('Pickle not found: %s' % file_path)


def save(obj, pkl_dir, pkl_name):
    file_path = os.path.join(pkl_dir, pkl_name)
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)
