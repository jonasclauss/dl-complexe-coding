import os
from sklearn.model_selection import train_test_split

def flatten(xss):
    return [x for xs in xss for x in xs]

class DataReader:
    
    def __init__(self, data_path, seed=0):
        self.data_path = data_path
        self.seed = seed

        files, labels = self.load_data(data_path)
        print("Data loaded")

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(files, labels, test_size=0.25, random_state=seed)

        assert set(self.x_train).isdisjoint(set(self.x_test)), "Train and test sets are not disjoint!"


    def load_data(self, data_path):
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