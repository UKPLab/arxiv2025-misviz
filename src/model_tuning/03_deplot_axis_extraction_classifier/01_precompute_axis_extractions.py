import os
import sys
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel

from torch.utils.data import DataLoader
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
import torch_dataset_loader
import utils


def iterate_and_generate(
    model, processor, dataset, output_folder, partition, device, batch_size=4
):
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    results = []
    for batch in tqdm(dataloader):
        images = [Image.open(img_path) for img_path, _, _ in batch]

        inputs = processor(
            images=images,
            text=["Extract the axis information table of the figure below:"]
            * len(images),
            return_tensors="pt",
            padding=True,
        ).to(device)

        # inputs = {k: v.to(model.dtype) for k, v in inputs.items()}

        predictions = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        decoded = processor.batch_decode(predictions, skip_special_tokens=True)
        results.extend(decoded)

    output_file = os.path.join(
        output_folder, f"{partition}_axis_deplot_axis_generations.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


def main(model_path, misviz_synth_path, misviz_path, outputpath, split, device="cuda"):
    output_folder = os.path.join(outputpath, "axis_deplot_axis_extraction")
    os.makedirs(output_folder, exist_ok=True)

    base_model = Pix2StructForConditionalGeneration.from_pretrained(
        "google/deplot",
        # torch_dtype=torch.bfloat16
    )
    processor = Pix2StructProcessor.from_pretrained("google/deplot")

    # Load the LoRA model
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        # torch_dtype=torch.bfloat16
    )

    test_dataset_misviz_synth = (
        torch_dataset_loader.MisvizSynthRawChartMisleaderDataset(
            dataset_path=misviz_synth_path,
            partition="test",  #test_run=True
        )
    )
    train_dataset_misviz_synth = (
        torch_dataset_loader.MisvizSynthRawChartMisleaderDataset(
            dataset_path=misviz_synth_path,
            partition="train small", # test_run=True
        )
    )
    val_dataset_misviz_synth = torch_dataset_loader.MisvizSynthRawChartMisleaderDataset(
        dataset_path=misviz_synth_path,
        partition="val",  #test_run=True
    )

    val_dataset_misviz = torch_dataset_loader.MisvizRawChartMisleaderDataset(
        dataset_path=misviz_path,
        partition="val",   #test_run=True
    )

    test_dataset_misviz = torch_dataset_loader.MisvizRawChartMisleaderDataset(
        dataset_path=misviz_path,
        partition="test",   #test_run=True
    )

    model = model.merge_and_unload()
    model = torch.compile(model)
    model.to(device)
    model.eval()
    with torch.no_grad():

        if split == 0:
            iterate_and_generate(
                model,
                processor,
                train_dataset_misviz_synth,
                output_folder,
                "misviz_synth_train_small",
                device,
            )
        else:

            iterate_and_generate(
                model,
                processor,
                val_dataset_misviz_synth,
                output_folder,
                "misviz_synth_val",
                device,
            )
            iterate_and_generate(
                model,
                processor,
                test_dataset_misviz_synth,
                output_folder,
                "misviz_synth_test",
                device,
            )
            iterate_and_generate(
                model,
                processor,
                val_dataset_misviz,
                output_folder,
                "misviz_val",
                device,
            )
            iterate_and_generate(
                model,
                processor,
                test_dataset_misviz,
                output_folder,
                "misviz_test",
                device,
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate the embeddings for the images and tables in the misviz_synth dataset."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the lora adapter of the fine-tuned Axis-DePlot model. Should be in the folder seen above.",
    )
    parser.add_argument(
        "--datasetpath_misviz_synth",
        type=str,
        help="Path at which the Misviz Synth dataset is located at.",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="Split to use for the generation. 0 = train_small misviz_synth, 1 = val/test misviz and misviz_synth. Mainly used to speed up generation by parallelizing it.",
    )
    parser.add_argument(
        "--datasetpath_misviz",
        type=str,
        help="Path at which the misleader dataset is located at.",
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        help="Path at which the generated sequences will be saved to.",
    )
    args = parser.parse_args()

    main(
        args.model_path,
        args.datasetpath_misviz_synth,
        args.datasetpath_misviz,
        args.outputpath,
        args.split,
        utils.get_available_device(),
    )
