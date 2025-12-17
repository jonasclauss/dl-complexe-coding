import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import decode_image
import torch

def flatten(xss):
    return [x for xs in xss for x in xs]

class DataReader():
    
    def __init__(self, data_path, transform=None, target_transform=None, seed=0):
        self.data_path = data_path
        self.transform = transform
        self.seed = seed
        
        files, labels = self.__load_data(data_path)
        print("Data loaded")

        unique_labels = list(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        encoded = [label_to_idx[label] for label in labels]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(files, encoded, test_size=0.25, random_state=seed)

        assert set(self.x_train).isdisjoint(set(self.x_test)), "Train and test sets are not disjoint!"
        
        self.training_set = Data(self.x_train, self.y_train)
        self.test_set = Data(self.x_test, self.y_test)

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
    
    def __init__(self, files, labels, transform=None, target_transform=None):
        self.files = files
        self.labels = torch.tensor(labels, dtype=torch.int)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
            img_path = self.files[idx]
            image = decode_image(img_path)
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image.float(), label
