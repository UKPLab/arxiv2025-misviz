import torch
from torch.utils.data import Dataset


class PrecompMisvizSynthDatasetWithAxis(Dataset):
    def __init__(
        self,
        metadata,
        precomp_img_encodings,
        precomp_axis_encodings,
        label_mapping,
        table_generations=None,
    ):
        self.metadata = metadata
        self.image_encodings = precomp_img_encodings
        self.axis_encodings = precomp_axis_encodings
        self.label_to_idx = label_mapping
        self.table_generations = table_generations

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metadata = self.metadata[idx]
        label_name = metadata["label"]

        img_encoding = self.image_encodings[idx]
        axis_encoding = self.axis_encodings[idx]
        if self.table_generations is not None:
            table_generation = self.table_generations[idx]
            metadata["table_generation"] = table_generation

        concat_encodings = torch.cat([img_encoding, axis_encoding])

        label = self.label_to_idx[label_name]

        label = torch.nn.functional.one_hot(
            torch.tensor(label, dtype=torch.long), num_classes=len(self.label_to_idx)
        ).float()

        return concat_encodings, label, metadata

    def __len__(self):
        return len(self.metadata)

    def input_length(self):
        return self.image_encodings[0].shape[0] + self.axis_encodings[0].shape[0]


class PrecompMisvizSynthDatasetWithTableData(Dataset):
    def __init__(
        self,
        metadata,
        precomp_img_encodings,
        precomp_data_table_encodings,
        label_mapping,
    ):
        self.metadata = metadata
        self.image_encodings = precomp_img_encodings
        self.data_table_encodings = precomp_data_table_encodings
        self.label_to_idx = label_mapping

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metadata = self.metadata[idx]
        label_name = metadata["label"]

        img_encoding = self.image_encodings[idx]
        data_table_encoding = self.data_table_encodings[idx]

        concat_encodings = torch.cat([img_encoding, data_table_encoding])

        label = self.label_to_idx[label_name]

        label = torch.nn.functional.one_hot(
            torch.tensor(label, dtype=torch.long), num_classes=len(self.label_to_idx)
        ).float()

        return concat_encodings, label, metadata

    def __len__(self):
        return len(self.metadata)

    def input_length(self):
        return self.image_encodings[0].shape[0] + self.data_table_encodings[0].shape[0]


class PrecompMisvizSynthDataset(Dataset):
    def __init__(
        self,
        metadata,
        precomp_img_encodings,
        label_mapping,
    ):
        self.metadata = metadata
        self.image_encodings = precomp_img_encodings
        self.label_to_idx = label_mapping

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metadata = self.metadata[idx]
        label_name = metadata["label"]

        img_encoding = self.image_encodings[idx]

        label = self.label_to_idx[label_name]

        label = torch.nn.functional.one_hot(
            torch.tensor(label, dtype=torch.long), num_classes=len(self.label_to_idx)
        ).float()

        return img_encoding, label, metadata

    def __len__(self):
        return len(self.metadata)

    def input_length(self):
        return self.image_encodings[0].shape[0]


class PrecompMisvizSynthDatasetWithAxisAndData(Dataset):
    def __init__(
        self,
        metadata,
        precomp_img_encodings,
        precomp_axis_encodings,
        precomp_data_encodings,
        label_mapping,
    ):
        self.metadata = metadata
        self.image_encodings = precomp_img_encodings
        self.axis_encodings = precomp_axis_encodings
        self.precomp_data_encodings = precomp_data_encodings
        self.label_to_idx = label_mapping

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metadata = self.metadata[idx]
        label_name = metadata["label"]

        img_encoding = self.image_encodings[idx]
        axis_encoding = self.axis_encodings[idx]
        data_encoding = self.precomp_data_encodings[idx]

        concat_encodings = torch.cat([img_encoding, axis_encoding, data_encoding])

        label = self.label_to_idx[label_name]

        label = torch.nn.functional.one_hot(
            torch.tensor(label, dtype=torch.long), num_classes=len(self.label_to_idx)
        ).float()

        return concat_encodings, label, metadata

    def __len__(self):
        return len(self.metadata)

    def input_length(self):
        return (
            self.image_encodings[0].shape[0]
            + self.axis_encodings[0].shape[0]
            + self.precomp_data_encodings[0].shape[0]
        )


class PrecompMisvizDatasetWithAxis(Dataset):
    def __init__(
        self,
        metadata,
        precomp_img_encodings,
        precomp_axis_encodings,
        label_mapping,
        table_generations=None,
    ):
        self.metadata = metadata
        self.image_encodings = precomp_img_encodings
        self.axis_encodings = precomp_axis_encodings
        self.label_to_idx = label_mapping
        self.table_generations = table_generations

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metadata = self.metadata[idx]

        labels_mapped = metadata["misleader"]

        if labels_mapped == []:
            labels_mapped = ["no misleader"]

        img_encoding = self.image_encodings[idx]
        axis_encoding = self.axis_encodings[idx]
        if self.table_generations is not None:
            table_generation = self.table_generations[idx]
            metadata["table_generation"] = table_generation

        concat_encodings = torch.cat([img_encoding, axis_encoding])

        labels = [self.label_to_idx[label_name] for label_name in labels_mapped]

        labels = [
            torch.nn.functional.one_hot(
                torch.tensor(label, dtype=torch.long),
                num_classes=len(self.label_to_idx),
            ).float()
            for label in labels
        ]

        return concat_encodings, labels, metadata

    def __len__(self):
        return len(self.metadata)

    def input_length(self):
        return self.image_encodings[0].shape[0] + self.axis_encodings[0].shape[0]


class PrecompMisvizDataset(Dataset):
    def __init__(
        self,
        metadata,
        precomp_img_encodings,
        label_mapping,
    ):
        self.metadata = metadata
        self.image_encodings = precomp_img_encodings
        self.label_to_idx = label_mapping

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metadata = self.metadata[idx]
        labels_mapped = metadata["misleader"]

        if labels_mapped == []:
            labels_mapped = ["no misleader"]

        img_encoding = self.image_encodings[idx]

        labels = [self.label_to_idx[label_name] for label_name in labels_mapped]

        labels = [
            torch.nn.functional.one_hot(
                torch.tensor(label, dtype=torch.long),
                num_classes=len(self.label_to_idx),
            ).float()
            for label in labels
        ]

        return img_encoding, labels, metadata

    def __len__(self):
        return len(self.metadata)

    def input_length(self):
        return self.image_encodings[0].shape[0]
