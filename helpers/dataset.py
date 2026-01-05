import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch
from skimage.io import imread

def flatten(xss):
    return [x for xs in xss for x in xs]

class DataReader():
    
    def __init__(self, data_path, eval_transform=None, train_transform=None, target_transform=None, seed=0, use_ms=False):
        self.data_path = data_path
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.seed = seed
        
        files, labels = self.__load_data(data_path)

        unique_labels = sorted(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = [self.label_to_idx[label] for label in labels]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(files, encoded, test_size=0.20, random_state=seed)
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.20, random_state=seed)

        assert set(self.x_train).isdisjoint(set(self.x_test)), "Train and test sets are not disjoint!"
        assert set(self.x_train).isdisjoint(set(self.x_val)), "Train and validation sets are not disjoint!"
        assert set(self.x_val).isdisjoint(set(self.x_test)), "Validation and test sets are not disjoint!"
        
        self.training_set = Data(self.x_train, self.y_train, transform=self.train_transform, target_transform=target_transform, use_ms=use_ms)
        self.validation_set = Data(self.x_val, self.y_val, transform=self.eval_transform, target_transform=target_transform, use_ms=use_ms)
        self.test_set = Data(self.x_test, self.y_test, transform=self.eval_transform, target_transform=target_transform, use_ms=use_ms)

    def __load_data(self, data_path) -> tuple[list[str], list[str]]:
        files = []
        labels = []
        for root_path, _, filenames in os.walk(data_path):
            if root_path == data_path:
                continue
            labels.append(os.path.basename(root_path))
            files.append([root_path + '/' + file for file in filenames])

        flat_files = []
        flat_labels = []

        for label, file_list in zip(labels, files):
            for f in file_list:
                flat_files.append(f)
                flat_labels.append(label)

        return flat_files, flat_labels



class Data(Dataset):
    
    def __init__(self, files, labels, transform=None, target_transform=None, use_ms=False):
        self.files = files
        self.labels = torch.tensor(labels, dtype=torch.int)
        self.transform = transform
        self.target_transform = target_transform
        self.use_ms = use_ms

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
            img_path = self.files[idx]
            if self.use_ms:
                image = imread(img_path)
                image = torch.from_numpy(image)
                image = image.permute(2, 0, 1)
                image = image / 65535.0
            else:
                image = decode_image(img_path)
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image.float(), label