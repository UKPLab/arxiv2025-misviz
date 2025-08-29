import os
import sys
import torch
import argparse
from PIL import Image
from tqdm import tqdm

import encoding_strategies
import misviz_datasets

script_dir = os.path.dirname(__file__)

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(parent_dir)
import utils


def compute_image_embeddings_for_partition(
    dataset_name,
    partition,
    img_encoding_strategy,
    img_paths,
    output_path,
    batch_size,
    backend,
):
    all_embedded_images = torch.tensor([]).to(backend)

    img_encoding_strategy.prepare_encoder(backend)

    for i in tqdm(range(0, len(img_paths), batch_size), desc="Processing batches"):
        img_paths_batch = img_paths[i : min(i + batch_size, len(img_paths))]
        image_batch = torch.tensor([]).to(backend)
        for img_path in img_paths_batch:
            image = Image.open(img_path).convert("RGB")
            preprocessed_image = img_encoding_strategy.preprocess_images(image).to(
                backend
            )
            image_batch = torch.cat([image_batch, preprocessed_image])

        with torch.no_grad():
            encoder_output = img_encoding_strategy.encode(image_batch)

        encoding = img_encoding_strategy.get_embedding_from_output(encoder_output)
        all_embedded_images = torch.cat([all_embedded_images, encoding])

    os.makedirs(output_path, exist_ok=True)

    model_name = img_encoding_strategy.get_name()
    with open(
        os.path.join(
            output_path, f"{partition}_{dataset_name}_{model_name}_embedded_images.pt"
        ),
        "wb",
    ) as output_file:
        torch.save(all_embedded_images, output_file)


def compute_image_embeddings(model, dataset_strategy, output_path, batch_size):
    img_encoding_strategy = derive_encoding_strategy(model)
    backend = utils.get_available_device()
    dataset_name = dataset_strategy.get_dataset_name()
    output_path = output_path + dataset_name
    os.makedirs(output_path, exist_ok=True)

    for dataset_partition in dataset_strategy.get_available_partitions():
        print(f"-- Computing embeddings for {dataset_partition} partition ...")

        if dataset_partition == "train":
            img_encoding_strategy.set_processor_augmentations(
                ["rotation", "perspective"]
            )
        elif dataset_partition in ["val", "test"]:
            img_encoding_strategy.set_processor_augmentations([])

        img_paths = dataset_strategy.get_all_file_paths_for_partition(dataset_partition)
        compute_image_embeddings_for_partition(
            dataset_name,
            dataset_partition,
            img_encoding_strategy,
            img_paths,
            output_path,
            batch_size,
            backend,
        )

    del img_encoding_strategy
    torch.cuda.empty_cache()


def derive_dataset_strategy(dataset_name, base_dataset_path):
    if dataset_name == "misviz_synth":
        return misviz_datasets.MisvizSynthDataset(base_dataset_path)
    elif dataset_name == "misviz":
        return misviz_datasets.MisvizDataset(base_dataset_path)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def derive_encoding_strategy(encoding_model):

    if encoding_model == "tinychart":
        return encoding_strategies.TinyChartOneEncoderEncodingStrategy()
    else:
        raise ValueError(f"Unknown encoding model {encoding_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate the embeddings for the images and tables for one of the datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="misviz_synth",
        choices=["misviz_synth", "misviz"],
        help="The dataset to use for the embeddings. Can be 'misviz_synth', 'misviz'.",
    )
    parser.add_argument(
        "--datasetpath",
        type=str,
        default="data/misviz/",
        help="Path at which the misleader dataset is located at.",
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        default="data/precomp/",
        help="Path at which the data embeddings will be saved to.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=2,
        help="The batch size to use for the image embeddings.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tinychart",
        help="The model to use for the image embeddings"
    )
    args = parser.parse_args()

    # Change to misviz_synth if wanting to compute embeddings for it
    dataset_strategy = derive_dataset_strategy("misviz", args.datasetpath)
    compute_image_embeddings(
        args.model,
        dataset_strategy,
        args.outputpath,
        args.batchsize,
    )
