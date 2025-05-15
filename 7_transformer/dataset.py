from torch.utils.data import DataLoader

from helpers import ak, os, get_normalized_data, collate_fn_transformer

class IceCubeDataset():
    def __init__(self, data_path, batch_size):

        # Load the dataset
        train_dataset = ak.from_parquet(os.path.join(data_path, "train.pq"))
        val_dataset = ak.from_parquet(os.path.join(data_path, "val.pq"))
        test_dataset = ak.from_parquet(os.path.join(data_path, "test.pq"))

        # Normalize data and labels
        self.features_mean, self.features_std, self.labels_mean, self.labels_std = get_normalized_data(train_dataset, val_dataset, test_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_transformer)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_transformer)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_transformer)
