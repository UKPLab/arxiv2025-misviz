import json
import os


class DatasetType:
    def __init__(self, datset_path):
        self.dataset_path = datset_path

    def check_partition_in_available_partitions(self, partition):
        if partition not in self.get_available_partitions():
            raise ValueError(
                f"Partition {partition} is not available in the dataset {self.get_dataset_name()}"
            )

    def get_all_file_paths_for_partition(self, partition=None):
        pass

    def get_dataset_name(self):
        pass

    def get_available_partitions(self):
        pass


class MisvizSynthDataset(DatasetType):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        all_metadata_file_path = os.path.join(self.dataset_path, "misviz_synth.json")

        with open(all_metadata_file_path, "r") as existing_metadata_file:
            self.metadata = json.load(existing_metadata_file)

    def get_all_file_paths_for_partition(self, partition):
        self.check_partition_in_available_partitions(partition)

        all_file_paths_for_partition = []
        for entry in self.metadata:
            if partition in entry["split"]:
                filepath_in_partition = os.path.join(
                    self.dataset_path,
                    "vis_output",
                    entry["image_path"],
                )
                all_file_paths_for_partition.append(filepath_in_partition)
        return all_file_paths_for_partition

    def get_dataset_name(self):
        return "misviz_synth"

    def get_available_partitions(self):
        # return ["train", "val", "test", "train small"]
        return ["val", "test", "train small"]


class MisvizDataset(DatasetType):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        metadata_path = os.path.join(self.dataset_path, "misviz.json")
        with open(metadata_path, "r") as existing_metadata_file:
            self.metadata = json.load(existing_metadata_file)

    def get_all_file_paths_for_partition(self, partition):
        self.check_partition_in_available_partitions(partition)
        all_file_paths_for_partition = []
        for metadata_entry in self.metadata:
            if partition == metadata_entry["split"]:
                filepath_in_partition = os.path.join(
                    self.dataset_path, metadata_entry["image_path"]
                )
                all_file_paths_for_partition.append(filepath_in_partition)

        return all_file_paths_for_partition

    def get_dataset_name(self):
        return "misviz"

    def get_available_partitions(self):
        return ["val", "test"]
