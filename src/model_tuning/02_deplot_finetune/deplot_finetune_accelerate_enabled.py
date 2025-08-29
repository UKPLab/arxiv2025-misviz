import os
import sys
import json
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from axis_dataset import ImageAxisDeplotDataset
from torch.utils.data import DataLoader
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

from peft import LoraConfig, get_peft_model
from accelerate.logging import get_logger

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
import utils

from accelerate import Accelerator
from accelerate.utils import set_seed


def save_checkpoint(
    accelerator,
    experiment_name,
    model,
    optimizer,
    epoch,
    steps_completed,
    path,
    lora_on,
):
    file_name = path + f"/{experiment_name}_epoch_{epoch}.pt"
    data_to_save = {
        "epoch": epoch,
        "steps_completed": steps_completed,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if lora_on:
        torch.save(
            data_to_save,
            file_name,
        )
        output_lora = path + f"/lora_axis_adapter_{epoch}"
        os.makedirs(output_lora, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_lora)
    else:
        data_to_save["model_state_dict"] = model.state_dict()
        torch.save(
            data_to_save,
            file_name,
        )


def collate_fn(batch):
    flattened_patches = []
    attention_masks = []
    labels = []
    ground_truths = []

    for item in batch:
        flattened_patches.append(item[0]["flattened_patches"])
        attention_masks.append(item[0]["attention_mask"])
        labels.append(item[0]["labels"])
        ground_truths.append(item[1])

    model_input = {
        "flattened_patches": torch.stack(flattened_patches),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
    }
    return model_input, ground_truths


def main(
    experiment_name,
    outputpath,
    datasetpath,
    axis_data_path,
    epochs,
    batch_size,
    checkpoint_path,
    test_mode,
    lora_on,
    seq_length,
    mixed_precision="fp32",
):
    accelerator = Accelerator(mixed_precision=mixed_precision)

    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    # Setup experiment directories
    experiment_path = os.path.join(outputpath, experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(os.path.join(experiment_path, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(experiment_path, "weights"), exist_ok=True)

    logger = get_logger(__name__)
    logger.setLevel(logging.INFO)
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"-> GPU {i}: {torch.cuda.get_device_name(i)}")

    config_params = {
        "experiment_name": experiment_name,
        "dataset_path": datasetpath,
        "epochs": epochs,
        "batch_size": batch_size,
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "mixed_precision": mixed_precision,
        "lora_on": lora_on,
        "max_seq_length": seq_length,
    }
    config_path = os.path.join(experiment_path, "config_params.json")

    # Load the dataset
    metadata_path = os.path.join(datasetpath, "misviz_synth.json")
    with open(metadata_path, "r") as metadata_file:
        all_metadata = json.load(metadata_file)

    additional_axis_metadata_path = os.path.join(axis_data_path, "metadata.json")
    #The additional instances for axis extraction training
    with open(additional_axis_metadata_path, "r") as axis_metadata_file:
        axis_metadata = json.load(axis_metadata_file)

    # Load split index
    #split index for training val and test instances
    split_index = utils.extract_split_and_indices(all_metadata)
    # axis_split_index = json.load(axis_metadata)

    #the main data from msiviz synth, only if axis data path is not empty  (it is not a pie chart)
    train_metadata = [
        all_metadata[i]
        for i in split_index["train"]
        if all_metadata[i]["axis_data_path"] != ""
    ]
    val_metadata = [
        all_metadata[i]
        for i in split_index["val"]
        if all_metadata[i]["axis_data_path"] != ""
    ]

    train_val_ratio = len(train_metadata)/(len(train_metadata)+len(val_metadata))
    cutoff = round(len(axis_metadata) * train_val_ratio)
    print(f"Cutoff point between train and val for the additional axis metadata: {cutoff}")
    axis_train_metadata = [axis_metadata[i] for i in range(0,cutoff)]
    axis_val_metadata = [axis_metadata[i] for i in range(cutoff,len(axis_metadata))]

    # merge both datasets
    train_metadata.extend(axis_train_metadata)
    np.random.shuffle(train_metadata)
    val_metadata.extend(axis_val_metadata)
    np.random.shuffle(val_metadata)
    print('Training data size %s'%(len(train_metadata)))
    print('Validation data size %s'%(len(val_metadata)))

    # Test Purpose
    if test_mode:
        train_metadata = train_metadata[:100]
        val_metadata = val_metadata[:100]
        epochs = 2

    print(f"Starting training with the following params \n {config_params}")

    with open(config_path, "w") as config_file:
        json.dump(config_params, config_file, indent=4)

    # Load model and processor
    model = Pix2StructForConditionalGeneration.from_pretrained(
        "google/deplot", max_length=seq_length
    )
    model.config.max_length = seq_length
    processor = Pix2StructProcessor.from_pretrained("google/deplot")

    if lora_on:
        # Init model with LoRA
        lora_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

    start_epoch = 0
    if checkpoint_path:
        checkpoint_state = accelerator.load(checkpoint_path)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint_state["model_state_dict"])
        start_epoch = int(checkpoint_path.split("/")[-1].split(".")[0].split("_")[-1])
        print(
            f"Loaded model from checkpoint {checkpoint_path}. Resuming training from epoch {start_epoch}"
        )

    # Prepare datasets and dataloaders
    train_dataset = ImageAxisDeplotDataset(
        dataset_folder_path=datasetpath,
        additional_axis_data_folder_path=axis_data_path,
        metadata=train_metadata,
        processor=processor,
        augmentations=["rotation", "perspective"],
        max_output_length=seq_length,
    )

    val_dataset = ImageAxisDeplotDataset(
        dataset_folder_path=datasetpath,
        additional_axis_data_folder_path=axis_data_path,
        metadata=val_metadata,
        processor=processor,
        augmentations=[],
        max_output_length=seq_length,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    checkpoint_path = os.path.join(
        experiment_path,
        "weights",
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
    )

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    evaluation_metrics = defaultdict(list)

    print(f"Finetuning of model {experiment_name} is starting.")
    early_stopping = utils.ValidationLossEarlyStopping(patience=3, min_delta=0.2)

    # Training loop
    for epoch in range(start_epoch, epochs):
        print(f"Training epoch: {epoch + 1} / {epochs}")
        model.train()

        training_losses = []
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}",
        )

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            model_input, _ = batch

            flattened_patches = model_input.pop("flattened_patches")
            attention_mask = model_input.pop("attention_mask")
            labels = model_input.pop("labels")

            with accelerator.accumulate(model):
                outputs = model(
                    flattened_patches=flattened_patches,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                accelerator.backward(loss)

                optimizer.step()

            loss_value = loss.detach().float()
            training_losses.append(float(accelerator.gather(loss_value).mean().item()))

            progress_bar.update(1)

            if accelerator.is_main_process and len(training_losses) % 100 == 0:
                last_losses = (
                    training_losses[-100:]
                    if len(training_losses) >= 100
                    else training_losses
                )
                print(
                    f"Running avg last 100 loss batch {sum(last_losses) / len(last_losses)}"
                )

        model.eval()
        eval_losses = []
        progress_bar_val = tqdm(
            val_dataloader, disable=not accelerator.is_local_main_process
        )
        print(f"Validation epoch: {epoch + 1} / {epochs}")
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                model_input, _ = batch
                flattened_patches = model_input.pop("flattened_patches")
                attention_mask = model_input.pop("attention_mask")
                labels = model_input.pop("labels")

                outputs = model(
                    flattened_patches=flattened_patches,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                loss_value = loss.detach().float()

                gathered_losses = accelerator.gather(loss_value)

                for val in gathered_losses:
                    eval_losses.append(float(val.item()))

                progress_bar_val.update(1)

        epoch_avg_train_loss = sum(training_losses) / len(training_losses)
        epoch_avg_val_loss = sum(eval_losses) / len(eval_losses)

        evaluation_metrics["avg_train_losses"].append(
            {"epoch": epoch, "loss": epoch_avg_train_loss}
        )
        evaluation_metrics["avg_val_losses"].append(
            {"epoch": epoch, "loss": epoch_avg_val_loss}
        )

        evaluation_metrics["all_val_losses"].append(
            {"epoch": epoch, "losses": eval_losses}
        )
        evaluation_metrics["all_train_losses"].append(
            {"epoch": epoch, "losses": training_losses}
        )

        # Save checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(
                accelerator,
                experiment_name,
                model,
                optimizer,
                epoch,
                len(train_dataloader),
                checkpoint_path,
                lora_on,
            )
            print(f"Training loss averge per epoch: {epoch_avg_train_loss}")
            print(f"Validation loss averge per epoch: {epoch_avg_val_loss}")
            if early_stopping.early_stop_check(epoch_avg_val_loss):
                print("Early stopping invoked.")
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        progress_bar.close()
        with open(
            os.path.join(experiment_path, "all_metrics.json"), "w"
        ) as output_file:
            json.dump(evaluation_metrics, output_file, indent=4)

        accelerator.end_training()
        print("Training completed successfully!")


if __name__ == "__main__":
    utils.set_all_seeds(0)

    parser = argparse.ArgumentParser(
        description="This is a script to finetune deplot on axis data with a parallel accelerate setup"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="deplot_finetuning_testing_local",
        help="Name of the experiment to be conducted.",
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        default="/src/model_tuning/03_deplot_finetune/output/",
        help="Path at which the evaluation data and the model weigths will be saved to.",
    )
    parser.add_argument(
        "--datasetpath",
        type=str,
        help="Path at which the Misviz Synthetic dataset is located at.",
    )
    parser.add_argument(
        "--axis_data_path",
        type=str,
        help="Path at which the additional axis metadata is located at. Root folder of the axis_variation dataset which can also be downloaded separately.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs for training the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training the model.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the model weights to be loaded",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="If set, the script will run in test mode with a very small train and val set and only 2 epochs.",
    )
    parser.add_argument(
        "--lora_on",
        action="store_true",
        help="If set, the script will use LoRA for model finetuning.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for the model output.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["fp8", "fp16", "no"],
        help="Mixed precision training.",
    )

    args = parser.parse_args()
    main(
        args.experiment_name,
        args.outputpath,
        args.datasetpath,
        args.axis_data_path,
        args.epochs,
        args.batch_size,
        args.checkpoint_path,
        False,
        args.lora_on,
        args.seq_length,
        args.mixed_precision,
    )
