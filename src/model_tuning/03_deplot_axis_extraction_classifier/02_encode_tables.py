import os
import sys
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from io import StringIO
from transformers import (
    TapasTokenizer,
    TapasModel,
)

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
import utils


def convert_string_to_dataframe(string):
    cleaned_string = replace_hex_notation(string)
    expected_columns = ["Seq", "Axis", "Label", "Relative Position"]
    df = pd.read_csv(
        StringIO(cleaned_string),
        sep=r"\s*\|\s*",
        engine="python",
        names=expected_columns,
        usecols=[0, 1, 2, 3],
        skiprows=1,
        on_bad_lines="warn",
    )
    df.columns = [col.strip() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    return df


# No batch processing as input id length varies with input size of table
def compute_table_embeddings(encoding_name, all_generations, output_path, backend):
    # Create embeddings with average pooling and [CLS] token
    all_embedded_tables_cls = torch.tensor([]).to(backend)
    all_embedded_tables_mean = torch.tensor([]).to(backend)

    table_tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")
    table_model = TapasModel.from_pretrained("google/tapas-large-finetuned-wtq")
    table_model.to(backend)

    for generation in tqdm(all_generations):
        try:
            table_df = convert_string_to_dataframe(generation)
        except Exception as e:
            print(f"Error converting string to dataframe: {e}")
            all_embedded_tables_cls = torch.cat(
                [all_embedded_tables_cls, torch.zeros((1, 1024)).to(backend)]
            )
            all_embedded_tables_mean = torch.cat(
                [all_embedded_tables_mean, torch.zeros((1, 1024)).to(backend)]
            )
            continue

        table_df = table_df.astype(str)
        encoded_table = table_tokenizer(
            table=table_df, padding=True, return_tensors="pt", truncation=True
        )

        with torch.no_grad():
            input_data = {k: v.to(backend) for k, v in encoded_table.items()}
            table_embedding = table_model(**input_data)
            table_batch_embeddings_cls = table_embedding.last_hidden_state[:, 0, :]
            table_batch_embeddings_mean = table_embedding.last_hidden_state.mean(dim=1)

        all_embedded_tables_cls = torch.cat(
            [all_embedded_tables_cls, table_batch_embeddings_cls]
        )
        all_embedded_tables_mean = torch.cat(
            [all_embedded_tables_mean, table_batch_embeddings_mean]
        )

    os.makedirs(output_path, exist_ok=True)
    with open(
        os.path.join(output_path, f"{encoding_name}_tapas_table_encodings_cls.pt"),
        "wb",
    ) as output_file:
        torch.save(all_embedded_tables_cls, output_file)

    with open(
        os.path.join(output_path, f"{encoding_name}_tapas_table_encodings_mean.pt"),
        "wb",
    ) as output_file:
        torch.save(all_embedded_tables_mean, output_file)


def replace_hex_notation(text_to_convert):
    result = text_to_convert
    result = result.replace("<0x0A>", "\n")
    result = result.replace("<0x0D>", "\r")
    result = result.replace("<0x09>", "\t")

    return result


def compute_all_table_embeddings(data_path, output_path):
    device = utils.get_available_device()
    misviz_synth_train_small_generations_path = os.path.join(
        data_path, "misviz_synth_train_small_axis_deplot_axis_generations.json"
    )
    misviz_synth_test_generations_path = os.path.join(
        data_path, "misviz_synth_test_axis_deplot_axis_generations.json"
    )
    misviz_synth_val_generations_path = os.path.join(
        data_path, "misviz_synth_val_axis_deplot_axis_generations.json"
    )
    misivz_test_generations_path = os.path.join(
        data_path, "misviz_test_axis_deplot_axis_generations.json"
    )
    misviz_val_generations_path = os.path.join(
        data_path, "misviz_val_axis_deplot_axis_generations.json"
    )

    with open(
        misviz_synth_train_small_generations_path, "r"
    ) as misviz_synth_train_generations_file:
        misviz_synth_train_small_generations = json.load(
            misviz_synth_train_generations_file
        )

    with open(
        misviz_synth_val_generations_path, "r"
    ) as misviz_synth_val_generations_file:
        misviz_synth_val_generations = json.load(misviz_synth_val_generations_file)

    with open(
        misviz_synth_test_generations_path, "r"
    ) as misviz_synth_test_generations_file:
        misviz_synth_test_generations = json.load(misviz_synth_test_generations_file)

    with open(misivz_test_generations_path, "r") as misviz_test_generations_file:
        misviz_test_generations = json.load(misviz_test_generations_file)

    with open(misviz_val_generations_path, "r") as misviz_val_generations_file:
        misviz_val_generations = json.load(misviz_val_generations_file)

    # Computation for misviz
    compute_table_embeddings(
        "misviz_val_data",
        misviz_val_generations,
        output_path,
        device,
    )
    compute_table_embeddings(
        "misviz_test_data",
        misviz_test_generations,
        output_path,
        device,
    )

    # Computation for misviz synth
    compute_table_embeddings(
        "misviz_synth_val_data", misviz_synth_val_generations, output_path, device
    )
    compute_table_embeddings(
        "misviz_synth_test_data", misviz_synth_test_generations, output_path, device
    )
    compute_table_embeddings(
        "misviz_synth_train_data",
        misviz_synth_train_small_generations,
        output_path,
        device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate the embeddings for the tables in the misviz_synth dataset."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="src/model_tuning/03_deplot_axis_extraction_classifier/output/raw_axis_deplot_axis_extraction",
        help="Path at which the axis metadata generations from the previous step are located at. Should be in the folder seen above.",
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        default="src/model_tuning/03_deplot_axis_extraction_classifier/output/encoded_axis_extractions",
        help="Path at which the table encodings will be saved to.",
    )

    args = parser.parse_args()

    compute_all_table_embeddings(args.datapath, args.outputpath)
