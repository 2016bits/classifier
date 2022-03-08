import numpy as np


class RawData:
    def __init__(self, text, label):
        self.text = text
        self.label = label


class Sample:
    def __init__(self, id, mask, label):
        self.id = id
        self.mask = mask
        self.label = label
    

class BatchedData:
    def __init__(self, dataset):
        self._id = dataset.id
        self._mask = dataset.mask
        self._label = dataset.label
    
    def get_batch_num(self, batch_size):
        return len(self._id) // batch_size
    
    def __len__(self):
        return len(self._id)
    
    def __getitem__(self, item):
        id = self._id[item]
        mask = self._mask[item]
        label = self._label[item]

        return {
            'id': np.array(id),
            'mask': np.array(mask),
            'label': np.array(label)
        }