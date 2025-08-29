import os
import sys
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision.transforms import v2 as transforms

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
import utils


def deplot_style_preprocessing(text):
    preprocessed = text
    preprocessed = preprocessed.replace("\n", "<0x0A>")
    preprocessed = preprocessed.replace("\r", "<0x0D>")
    preprocessed = preprocessed.replace("\t", "<0x09>")
    return preprocessed


def deplot_style_postprocessing(text):
    result = text
    result = result.replace("<0x0A>", "\n")
    result = result.replace("<0x0D>", "\r")
    result = result.replace("<0x09>", "\t")
    return result


def build_serialized_table_string(records):
    table_header = "Seq | Axis | Label | Relative Position"
    serialized_table = table_header + " \n "
    current_axis = records["axis"][0]
    axis_elem_counter = 0
    for i, (axis, label, pos) in enumerate(
        zip(records["axis"], records["label"], records["relative_position"])
    ):
        if axis != current_axis:
            axis_elem_counter = 1
            current_axis = axis
        else:
            axis_elem_counter += 1
        serialized_table += f"{axis_elem_counter} | {axis} | {label} | {pos}"
        if i < len(records["axis"]) - 1:
            serialized_table += " \n "
    return serialized_table


class ImageAxisDeplotDataset(Dataset):
    def __init__(
        self,
        dataset_folder_path,
        additional_axis_data_folder_path,
        metadata,
        processor,
        augmentations,
        max_output_length=512,
    ):
        self.metadata = metadata
        self.dataset_folder_path = dataset_folder_path
        self.additional_axis_data_path = additional_axis_data_folder_path
        self.dataset_visualization_folder = os.path.join(
            dataset_folder_path, "vis_output"
        )
        self.max_output_length = max_output_length
        if augmentations:
            self.transforms = transforms.Compose(
                utils.build_transform(augmentations),
            )
        else:
            self.transforms = None
        self.processor = processor

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        metadata = self.metadata[idx]
        relative_axis_info_path = metadata["axis_data_path"]
        relative_image_path = metadata["image_path"]

        data_folder = ""
        if "chosen_y_axis_scale" in metadata.keys():
            data_folder = self.additional_axis_data_path
        else:
            data_folder = self.dataset_visualization_folder

        axis_file_path = (
            os.path.join(data_folder, relative_axis_info_path)
            if relative_axis_info_path
            else None
        )
        image_path = os.path.join(data_folder, relative_image_path)

        image = Image.open(image_path)
        if self.transforms:
            image = self.transforms(image)
        # transformed_image.show()
        image_inputs = self.processor(
            images=image,
            text="Extract the axis information table of the figure below:",
            return_tensors="pt",
            add_special_tokens=True,
            legacy=False,
        )
        model_inputs = {k: v.squeeze() for k, v in image_inputs.items()}
        if axis_file_path:
            axis_data = json.load(open(axis_file_path, "r"))
            deplot_table_axis = build_serialized_table_string(
                axis_data,
            )
        else:
            deplot_table_axis = "NA"

        deplot_table_axis_to_encode = deplot_style_preprocessing(deplot_table_axis)
        target_seq_encoded = self.processor.tokenizer(
            deplot_table_axis_to_encode,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids

        model_inputs = {k: v.squeeze() for k, v in model_inputs.items()}
        model_inputs["labels"] = target_seq_encoded.squeeze()

        return model_inputs, deplot_table_axis

    def __len__(self):
        return len(self.metadata)
