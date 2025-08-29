import argparse
import precompute_img_embeddings as precompute_img_embeddings


def compute_embeddings_with_models_and_datasets(
    models,
    datasets,
    datasetpath,
    outputpath,
    batchsize,
):
    for dataset, datasetpath in zip(datasets, datasetpath):
        dataset_strategy = precompute_img_embeddings.derive_dataset_strategy(
            dataset, datasetpath
        )
        print(f"Computing embeddings for dataset {dataset}")
        for model in models:
            print(f"- Computing embeddings for model {model}")
            precompute_img_embeddings.compute_image_embeddings(
                model, dataset_strategy, outputpath, batchsize
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate all the embeddings for the images in the misviz_synth dataset."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tinychart"],
        help="List of models for which the image encodings should be created for.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["misviz_synth", "misviz"],
        help="List of datasets to use: 'misviz_synth', 'misviz'.",
    )
    parser.add_argument(
        "--datasetpaths",
        type=str,
        nargs="+",
        required=True,
        help="List of paths at which the misleader datasets are located at. Needs to match the length of the datasets argument.",
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        required=True,
        help="Path at which the data encodings will be saved to.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=16,
        help="Batch size to use for computing encodings.",
    )

    args = parser.parse_args()
    if len(args.datasetpaths) != len(args.datasets):
        raise ValueError("The length of datasetpath and datasets must be the same.")

    compute_embeddings_with_models_and_datasets(
        args.models,
        args.datasets,
        args.datasetpaths,
        args.outputpath,
        args.batchsize,
    )
