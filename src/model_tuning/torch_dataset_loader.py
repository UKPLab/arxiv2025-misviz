import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ChartMisleaderDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        metadata_list=None,
        label_to_idx=None,
        processor=None,
    ):
        self.dataset_path = dataset_path
        if metadata_list is None:
            metadata_file_path = dataset_path + "/misviz_synth.json"
            with open(metadata_file_path, "r") as metadata_file:
                self.metadata_list = json.load(metadata_file)
        else:
            self.metadata_list = metadata_list
        if label_to_idx is None:
            labels = [misleader["label"] for misleader in self.metadata_list]
            unique_labels = np.unique(labels)
            self.label_to_idx = {
                misleader_name: i for i, misleader_name in enumerate(unique_labels)
            }
        else:
            self.label_to_idx = label_to_idx

        if processor:
            self.processor = processor

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.dataset_path, "vis_output", self.metadata_list[idx]["image_path"]
        )
        image = Image.open(img_path).convert("RGB")
        metadata = self.metadata_list[idx]
        label_name = self.metadata_list[idx]['misleader'][0] if len(self.metadata_list[idx]['misleader']) > 0 else "no misleader"
        label = self.label_to_idx[label_name]

        label = torch.nn.functional.one_hot(
            torch.tensor(label, dtype=torch.long), num_classes=len(self.label_to_idx)
        ).float()

        image = self.processor.process_images(image).squeeze(0)

        return image, label, metadata

    def __len__(self):
        return len(self.metadata_list)

    def get_label_to_idx_str(self):
        return self.label_to_idx


class MisvizSynthRawChartMisleaderDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        partition,
        test_run=False,
    ):
        self.dataset_path = dataset_path
        metadata_file_path = dataset_path + "/misviz_synth.json"
        with open(metadata_file_path, "r") as metadata_file:
            self.metadata_list = json.load(metadata_file)
        self.metadata_list = [
            entry for entry in self.metadata_list if partition in entry["split"]
        ]
        if test_run:
            self.metadata_list = self.metadata_list[:5]
        print(f"Loaded {len(self.metadata_list)} samples from partition {partition}")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.dataset_path,
            "vis_output",
            self.metadata_list[idx]["image_path"],
        )
        metadata = self.metadata_list[idx]
        label_name = self.metadata_list[idx]['misleader'][0] if len(self.metadata_list[idx]['misleader']) > 0 else "no misleader"

        return img_path, label_name, metadata

    def __len__(self):
        return len(self.metadata_list)

    def get_label_to_idx_str(self):
        return self.label_to_idx

    def get_name(self):
        return "misviz_synth"

    def get_all_metadata_ids(self):
        return [metadata_entry["id"] for metadata_entry in self.metadata_list]


class MisvizSynthRawChartMisleaderDatasetWithStratifiedFraction(Dataset):
    def __init__(
        self,
        dataset_path,
        partition,
    ):
        self.dataset_path = dataset_path
        metadata_file_path = dataset_path + "/misivz_synth.json"
        with open(metadata_file_path, "r") as metadata_file:
            self.metadata_list = json.load(metadata_file)

        self.metadata_list = [
            entry for entry in self.metadata_list if partition in entry["split"]
        ]
        print(f"Loaded {len(self.metadata_list)} samples from partition {partition}")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.dataset_path,
            "vis_output",
            self.metadata_list[idx]["image_path"],
        )
        metadata = self.metadata_list[idx]
        label_name = self.metadata_list[idx]['misleader'][0] if len(self.metadata_list[idx]['misleader']) > 0 else "no misleader"

        return img_path, label_name, metadata

    def __len__(self):
        return len(self.metadata_list)

    def get_label_to_idx_str(self):
        return self.label_to_idx

    def get_name(self):
        return "misviz_synth"


class MisvizRawChartMisleaderDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        partition,
        test_run=False
    ):
        self.dataset_path = dataset_path
        metadata_file_path = dataset_path + "/misviz.json"
        with open(metadata_file_path, "r") as metadata_file:
            self.metadata_list = json.load(metadata_file)

        # Filter out discretized continuous variables
        self.metadata_list = [
            metadata_entry
            for metadata_entry in self.metadata_list
            if metadata_entry["split"] == partition
        ]
        if test_run:
            self.metadata_list = self.metadata_list[:5]
        print(f"Loaded {len(self.metadata_list)} samples from partition {partition}")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.dataset_path,
            self.metadata_list[idx]["image_path"],
        )
        metadata = self.metadata_list[idx]
        label_name = metadata["misleader"]

        return img_path, label_name, metadata

    def __len__(self):
        return len(self.metadata_list)

    def get_label_to_idx_str(self):
        return self.label_to_idx

    def get_name(self):
        return "misviz"
