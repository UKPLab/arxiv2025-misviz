import os
import sys
import json
import torch

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
import precomp_dataset

with open(os.path.join("data", "misviz_synth/label_mapping.json"), "r") as f:
    label_mapping = json.load(f)

label_mapping_revert = {v: k for k, v in label_mapping.items()}


def prepare_datasets_misviz_synth(
    model_name,
    precomp_path,
    misviz_synth_path,
    output_prev_steps_path,
    dataset_type,
    label_to_idx,
    device,
):
    metadata_path = os.path.join(misviz_synth_path, "misviz_synth.json")
    with open(metadata_path, "r") as metadata_file:
        all_metadata = json.load(metadata_file)

    val_metadata = [entry for entry in all_metadata if "val" == entry["split"]]
    train_metadata = [entry for entry in all_metadata if "train" in entry["split"]]
    test_metadata = [entry for entry in all_metadata if "test" == entry["split"]]

    indices_small_train_in_big_train_path = [
        index
        for index in range(len(train_metadata))
        if "small" in train_metadata[index]["split"]
    ]
    misviz_synth_precomp_path = precomp_path + "misviz_synth/"

    # Load the indices of the small train set in the big train set
    # Used to extract the correct indices from the precomputed embeddings for the original train set
    with open(
        indices_small_train_in_big_train_path, "r"
    ) as indices_small_train_in_big_train_file:
        indices_small_train_in_big_train = json.load(
            indices_small_train_in_big_train_file
        )

    train_data = torch.load(
        misviz_synth_precomp_path
        + f"train_misviz_synth_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )[indices_small_train_in_big_train]
    val_data = torch.load(
        misviz_synth_precomp_path + f"val_misviz_synth_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )
    test_data = torch.load(
        misviz_synth_precomp_path
        + f"test_misviz_synth_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )

    if dataset_type == "encoder_only":
        train_dataset = precomp_dataset.PrecompMisvizSynthDataset(
            train_metadata, train_data, label_to_idx
        )
        val_dataset = precomp_dataset.PrecompMisvizSynthDataset(
            val_metadata, val_data, label_to_idx
        )
        test_dataset = precomp_dataset.PrecompMisvizSynthDataset(
            test_metadata, test_data, label_to_idx
        )
    elif dataset_type == "with_axis":
        encoded_generated_tables_path = (
            output_prev_steps_path + "encoded_axis_extractions/"
        )

        misviz_synth_test_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_synth_test_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )
        misviz_synth_val_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_synth_val_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )
        misviz_synth_train_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_synth_train_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )

        raw_axis_generations_path = (
            output_prev_steps_path + "raw_axis_deplot_axis_extraction/"
        )

        with open(
            raw_axis_generations_path
            + "misviz_synth_test_axis_deplot_axis_generations.json",
            "r",
        ) as test_raw_axis_data_file:
            test_raw_generations = json.load(test_raw_axis_data_file)

        with open(
            raw_axis_generations_path
            + "misviz_synth_train_small_axis_deplot_axis_generations.json",
            "r",
        ) as train_raw_axis_data_file:
            train_raw_generations = json.load(train_raw_axis_data_file)

        with open(
            raw_axis_generations_path
            + "misviz_synth_val_axis_deplot_axis_generations.json",
            "r",
        ) as val_raw_axis_data_file:
            val_raw_generations = json.load(val_raw_axis_data_file)

        train_dataset = precomp_dataset.PrecompMisvizSynthDatasetWithAxis(
            train_metadata,
            train_data,
            misviz_synth_train_axis_encodings,
            label_to_idx,
            train_raw_generations,
        )
        val_dataset = precomp_dataset.PrecompMisvizSynthDatasetWithAxis(
            val_metadata,
            val_data,
            misviz_synth_val_axis_encodings,
            label_to_idx,
            val_raw_generations,
        )
        test_dataset = precomp_dataset.PrecompMisvizSynthDatasetWithAxis(
            test_metadata,
            test_data,
            misviz_synth_test_axis_encodings,
            label_to_idx,
            test_raw_generations,
        )

    return train_dataset, val_dataset, test_dataset


def prepare_datasets_misviz(
    model_name,
    precomp_path,
    output_prev_steps_path,
    misviz_path,
    dataset_type,
    label_to_idx,
    device,
):
    metadata_path = os.path.join(misviz_path, "misviz.json")
    with open(metadata_path, "r") as metadata_file:
        all_metadata = json.load(metadata_file)

    test_metadata = [
        metadata_entry
        for metadata_entry in all_metadata
        if metadata_entry["split"] == "test"
    ]
    val_metadata = [
        metadata_entry
        for metadata_entry in all_metadata
        if metadata_entry["split"] == "val"
    ]

    misviz_precomp_path = precomp_path + "misviz/"

    val_data = torch.load(
        misviz_precomp_path + f"val_misviz_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )
    test_data = torch.load(
        misviz_precomp_path + f"test_misviz_{model_name}_embedded_images.pt",
        map_location=torch.device(device),
    )

    # Need to filter out the discretized continuous variables from metadata and encoded images
    # They are not present in the misviz_synth dataset, so the model can't be trained to detect them
    test_indeces_to_keep = [
        index
        for index in range(len(test_metadata))
        if "discretized continuous variable" not in test_metadata[index]["misleader"]
        and "map" not in test_metadata[index]["chart_type"]
        and "scatter plot" not in test_metadata[index]["chart_type"]
        and "other" not in test_metadata[index]["chart_type"]
    ]
    val_indeces_to_keep = [
        index
        for index in range(len(val_metadata))
        if "discretized continuous variable" not in val_metadata[index]["misleader"]
        and "map" not in val_metadata[index]["chart_type"]
        and "scatter plot" not in val_metadata[index]["chart_type"]
        and "other" not in val_metadata[index]["chart_type"]
    ]
    test_metadata = [test_metadata[index] for index in test_indeces_to_keep]
    val_metadata = [val_metadata[index] for index in val_indeces_to_keep]
    val_data = val_data[val_indeces_to_keep]
    test_data = test_data[test_indeces_to_keep]

    if dataset_type == "encoder_only":

        val_dataset = precomp_dataset.PrecompMisvizDataset(
            val_metadata, val_data, label_to_idx
        )
        test_dataset = precomp_dataset.PrecompMisvizDataset(
            test_metadata, test_data, label_to_idx
        )
    elif dataset_type == "with_axis":
        encoded_generated_tables_path = (
            output_prev_steps_path + "encoded_axis_extractions/"
        )

        misviz_test_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_test_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )
        misviz_val_axis_encodings = torch.load(
            encoded_generated_tables_path
            + "misviz_val_data_tapas_table_encodings_cls.pt",
            map_location=torch.device(device),
        )

        raw_axis_generations_path = (
            output_prev_steps_path + "raw_axis_deplot_axis_extraction/"
        )

        with open(
            raw_axis_generations_path + "misviz_test_axis_deplot_axis_generations.json",
            "r",
        ) as test_raw_axis_data_file:
            test_raw_generations = json.load(test_raw_axis_data_file)

        with open(
            raw_axis_generations_path + "misviz_val_axis_deplot_axis_generations.json",
            "r",
        ) as val_raw_axis_data_file:
            val_raw_generations = json.load(val_raw_axis_data_file)

        val_dataset = precomp_dataset.PrecompMisvizDatasetWithAxis(
            val_metadata,
            val_data,
            misviz_val_axis_encodings,
            label_to_idx,
            val_raw_generations,
        )
        test_dataset = precomp_dataset.PrecompMisvizDatasetWithAxis(
            test_metadata,
            test_data,
            misviz_test_axis_encodings,
            label_to_idx,
            test_raw_generations,
        )

    return val_dataset, test_dataset
