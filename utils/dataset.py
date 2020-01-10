from torch.utils.data import dataset as ds, dataloader as dl
from .tree import Forest
from . import pickling
import globvar


class ClonalDataset(ds.Dataset):

    def __init__(self, data):
        super(ClonalDataset, self).__init__()
        self.data = list(data)  # [:16]
        self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.length

    def collate(self, batch_data):
        return Forest(batch_data)


def get_data():
    train = pickling.load(globvar.PICKLE_DIR, 'dataset/train.pkl')
    dev = pickling.load(globvar.PICKLE_DIR, 'dataset/dev.pkl')
    return ClonalDataset(train), ClonalDataset(dev)


def get_data_loader(data_set, batch_size):
    return dl.DataLoader(
        data_set,
        batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=data_set.collate)
