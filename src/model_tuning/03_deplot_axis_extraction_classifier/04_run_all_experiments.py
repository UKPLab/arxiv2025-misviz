import argparse
import os
import sys
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.optim import AdamW
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

import commons

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
import utils


def collate_fn_misviz_synth(batch):
    embeddings = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    return embeddings, labels, metadata


def run_eval_for_misviz_synth_dataset(
    model,
    val_loader,
    criterion,
    class_weight_dict,
    label_to_idx,
    device,
):
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_actuals = []

    with torch.no_grad():
        validation_loss_epoch = []
        for batch in val_loader:
            embeddings, labels, _ = batch
            outputs = model(embeddings.to(device))
            loss = criterion(outputs.to(device), labels.to(device))
            validation_loss_epoch.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == actual.cpu()).sum().item()

            all_predicted.extend(predicted.cpu().tolist())
            all_actuals.extend(actual.cpu().tolist())
        avg_validation_loss_epoch = np.array(validation_loss_epoch).mean()

    evaluation_metrics_for_epoch = classification_report(
        all_actuals,
        all_predicted,
        labels=list(label_to_idx.values()),
        target_names=list(label_to_idx.keys()),
        output_dict=True,
        zero_division=0,
    )
    print(
        "Misviz Synth avg F1 score: ",
        evaluation_metrics_for_epoch["macro avg"]["f1-score"],
    )
    return {
        "avg_val_loss": avg_validation_loss_epoch,
        "imb_report": evaluation_metrics_for_epoch,
        "all_losses": validation_loss_epoch,
    }


def run_eval_for_misviz_dataset(
    model,
    val_dataset,
    criterion,
    class_weight_dict,
    device,
):
    model.eval()
    included_label_values_misviz = list(commons.label_mapping.values())
    all_predicted = []
    all_actuals = []
    ood_predictions = []
    ood_actuals = []
    num_ood_predictions = 0

    with torch.no_grad():
        validation_loss_epoch = []
        for embedding, true_labels, metadata in tqdm(val_dataset):
            outputs = model(embedding.to(device))
            _, predicted = torch.max(outputs.data, 0)
            if predicted.cpu().item() not in included_label_values_misviz:
                num_ood_predictions += 1
                ood_predictions.append(predicted.cpu().item())
                true_label_indices_mask = torch.zeros_like(outputs.data).to(device)
                for true_label in true_labels:
                    true_label_indices_mask = torch.maximum(
                        true_label_indices_mask, true_label.to(device)
                    )

                masked_values = outputs.data.clone()
                masked_values[true_label_indices_mask != 1] = float("-inf")

                max_index = torch.argmax(masked_values).item()
                label_for_loss_compute = torch.zeros_like(outputs.data)
                label_for_loss_compute[max_index] = 1

                _, actual = torch.max(label_for_loss_compute, 0)
                ood_actuals.append(actual.cpu().item())
                continue

            predicted_label_in_true_label_index = -1
            for idx, true_label in enumerate(true_labels):
                _, actual = torch.max(true_label, 0)
                if predicted.cpu() == actual.cpu():
                    predicted_label_in_true_label_index = idx

            if predicted_label_in_true_label_index != -1:
                _, actual = torch.max(
                    true_labels[predicted_label_in_true_label_index], 0
                )
                loss = criterion(
                    outputs.to(device),
                    true_labels[predicted_label_in_true_label_index].to(device),
                )
            else:
                true_label_indices_mask = torch.zeros_like(outputs.data).to(device)
                for true_label in true_labels:
                    true_label_indices_mask = torch.maximum(
                        true_label_indices_mask, true_label.to(device)
                    )

                masked_values = outputs.data.clone()
                masked_values[true_label_indices_mask != 1] = float("-inf")

                max_index = torch.argmax(masked_values).item()
                label_for_loss_compute = torch.zeros_like(outputs.data)
                label_for_loss_compute[max_index] = 1

                loss = criterion(
                    outputs.to(device),
                    label_for_loss_compute.to(device),
                )
                _, actual = torch.max(label_for_loss_compute, 0)

            # Append code for inverted axis if inverted x axis or y axis is predicted
            all_predicted.append(predicted.cpu().item())
            all_actuals.append(actual.cpu().item())

            validation_loss_epoch.append(loss.item())
        avg_validation_loss_epoch = np.array(validation_loss_epoch).mean()

    sample_weights = np.array([class_weight_dict[int(label)] for label in all_actuals])
    evaluation_metrics_for_epoch = classification_report(
        all_actuals,
        all_predicted,
        labels=list(commons.consolidated_labels_misviz.values()),
        target_names=list(commons.consolidated_labels_misviz.keys()),
        output_dict=True,
        zero_division=0,
    )
    print(
        "Misviz avg F1 score: ", evaluation_metrics_for_epoch["macro avg"]["f1-score"]
    )
    print("OOD rate misviz: ", (num_ood_predictions / len(val_dataset)) * 100)
    return {
        "avg_val_loss": avg_validation_loss_epoch,
        "imb_report": evaluation_metrics_for_epoch,
        "all_losses": validation_loss_epoch,
        "all_predicted": all_predicted,
        "all_actuals": all_actuals,
        "ood_predictions": ood_predictions,
        "ood_actuals": ood_actuals,
        "ood_rate": num_ood_predictions / len(val_dataset),
    }


def run_training_for_model(
    experiment_name,
    train_dataset,
    val_dataset_misviz_synth,
    val_dataset_misviz,
    label_to_idx,
    output_path,
    lr,
    epochs,
    batch_size,
    patience,
    min_delta,
    hidden_dim,
):
    # Load model onto specified device
    device = utils.get_available_device()
    output_path = os.path.join(output_path, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    config_params = {
        "experiment_name": experiment_name,
        "output_path": output_path,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "min_delta": min_delta,
        "hidden_dim": hidden_dim,
    }

    config_path = os.path.join(output_path, "config_params.json")
    with open(config_path, "w") as config_file:
        json.dump(config_params, config_file, indent=4)

    input_dim = train_dataset.input_length()
    output_dim = len(label_to_idx)

    model = utils.ClassifierHead(input_dim, hidden_dim, output_dim)

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "weights"), exist_ok=True)

    # Create class weights for f1 evaluation
    misviz_labels = []
    for metadata_entry in val_dataset_misviz.metadata:
        if metadata_entry["misleader"]:
            for misleader in metadata_entry["misleader"]:
                misviz_labels.append(commons.label_mapping[misleader])
        else:
            misviz_labels = misviz_labels + ["no misleader"]
    class_weights_misviz = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(commons.consolidated_labels_misviz.keys())),
        y=misviz_labels,
    )
    class_weight_dict_misviz = dict(
        zip(list(commons.consolidated_labels_misviz.values()), class_weights_misviz)
    )

    # Create class weights for misviz_synth
    labels = [metadata_entry["label"] for metadata_entry in train_dataset.metadata]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(label_to_idx.keys())),
        y=labels,
    )

    class_weight_dict = dict(zip(list(label_to_idx.values()), class_weights))

    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_misviz_synth,
        shuffle=True,
    )

    val_loader_misviz_synth = torch.utils.data.DataLoader(
        val_dataset_misviz_synth,
        batch_size=batch_size,
        collate_fn=collate_fn_misviz_synth,
    )

    # Start of training
    model.to(device)

    optimizer = AdamW(model.classifier.parameters(), lr=lr)
    all_metrics = []
    best_avg_f1_misviz_synth = 0
    best_avg_f1_misviz = 0

    early_stopping = utils.ValidationLossEarlyStopping(
        patience=patience, min_delta=min_delta
    )

    print(f"Training of model {experiment_name} is starting.")

    model.train()
    for epoch in range(0, epochs):
        training_loss_epoch = []
        print(f"Training epoch {epoch}/{epochs}")
        for idx, batch in enumerate(tqdm(train_loader)):
            embeddings, labels, _ = batch

            optimizer.zero_grad()

            outputs = model(embeddings.to(device))

            loss = criterion(outputs.to(device), labels.to(device))
            training_loss_epoch.append(loss.item())
            loss.backward()

            optimizer.step()

            if idx != 0 and idx % 100 == 0:
                print(
                    f"Epoch: [{epoch}/{epochs}], Batch: {idx}/{len(train_loader)}\t Loss {loss:.4f}\t"
                )
        print(f"Training finished for Epoch {epoch}/{epochs}")
        avg_training_loss_epoch = np.array(training_loss_epoch).mean()

        if epoch % 5 == 0 or epoch == epochs - 1:
            eval_metrics_misviz_synth = run_eval_for_misviz_synth_dataset(
                model,
                val_loader_misviz_synth,
                criterion,
                class_weight_dict,
                label_to_idx,
                device,
            )
            eval_metrics_misviz = run_eval_for_misviz_dataset(
                model,
                val_dataset_misviz,
                criterion,
                class_weight_dict_misviz,
                device,
            )
            metrics_for_epoch = {
                "epoch": epoch,
                "misviz_synth_avg_training_loss": avg_training_loss_epoch,
                "misviz_synth_val_metrics": eval_metrics_misviz_synth,
                "misviz_val_metrics": eval_metrics_misviz,
            }

            all_metrics.append(metrics_for_epoch)
            print(f"Eval finished for {epoch}/{epochs}")
            avg_val_loss = eval_metrics_misviz_synth["avg_val_loss"]
            early_stopping_activated = early_stopping.early_stop_check(avg_val_loss)
            avg_f1_score_misviz = eval_metrics_misviz["imb_report"]["macro avg"][
                "f1-score"
            ]
            avg_f1_score_misviz_synth = eval_metrics_misviz_synth["imb_report"][
                "macro avg"
            ]["f1-score"]

            if (
                epoch > 30
                or early_stopping_activated
                or epoch == epochs - 1
                or (
                    avg_f1_score_misviz > best_avg_f1_misviz
                    or avg_f1_score_misviz_synth > best_avg_f1_misviz_synth
                )
            ):
                if avg_f1_score_misviz > best_avg_f1_misviz:
                    best_avg_f1_misviz = avg_f1_score_misviz
                if avg_f1_score_misviz_synth > best_avg_f1_misviz_synth:
                    best_avg_f1_misviz_synth = avg_f1_score_misviz_synth

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                    },
                    os.path.join(
                        output_path,
                        "weights",
                        f"finetuned_epoch_{epoch}.pth",
                    ),
                )

        if early_stopping_activated:
            print("Early stopping invoked.")
            break

    with open(os.path.join(output_path, "all_metrics.json"), "w") as output_file:
        json.dump(all_metrics, output_file, indent=4)

    print(f"Finished training of model {experiment_name}. Script exiting.")


def run_all_train_experiments(
    misviz_synth_path,
    misviz_path,
    precomp_path,
    output_prev_steps_path,
    output_path,
    experiment_types,
    base_models,
    lr,
    epochs,
    batch_size,
    patience,
    min_delta,
    hidden_dim,
):
    device = utils.get_available_device()
    # Load the dataset

    with open(
        os.path.join(misviz_synth_path, "label_mapping.json"), "r"
    ) as label_mapping_file:
        label_to_idx = json.load(label_mapping_file)

    seeds = [123, 456, 789]
    for experiment in experiment_types:
        print(f"Running experiment for type {experiment}")
        for model in base_models:
            print(f"Running experiment {experiment} for model {model}")
            output_path_model = os.path.join(output_path, f"{model}_{experiment}")
            os.makedirs(output_path_model, exist_ok=True)
            for seed in seeds:
                print(f"Running experiment for seed {seed}")
                utils.set_all_seeds(seed)

                train_dataset_misviz_synth, val_dataset_misviz_synth, _ = (
                    commons.prepare_datasets_misviz_synth(
                        model,
                        precomp_path,
                        misviz_synth_path,
                        output_prev_steps_path,
                        experiment,
                        label_to_idx,
                        device,
                    )
                )

                val_dataset_misviz, _ = commons.prepare_datasets_misviz(
                    model,
                    precomp_path,
                    output_prev_steps_path,
                    misviz_path,
                    experiment,
                    label_to_idx,
                    device,
                )

                run_training_for_model(
                    f"{model}_{experiment}_{seed}",
                    train_dataset_misviz_synth,
                    val_dataset_misviz_synth,
                    val_dataset_misviz,
                    label_to_idx,
                    output_path_model,
                    lr,
                    epochs,
                    batch_size,
                    patience,
                    min_delta,
                    hidden_dim,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This is a script to train all model on the precomputed misviz_synth dataset."
    )
    parser.add_argument(
        "--outputpath",
        type=str,
        default= "src/model_tuning/03_deplot_axis_extraction_classifier/output/classifer_training/",
        help="Path at which the evaluation data and the model weigths will be saved to for every model configuration and seed.",
    )
    parser.add_argument(
        "--datasetpath_misviz_synth",
        type=str,
        default= "data/misviz_synth/",
        help="Path at which the synthetic misleader dataset is located at.",
    )
    parser.add_argument(
        "--datasetpath_misviz",
        type=str,
        default="data/misviz/",
        help="Path at which the misleader dataset is located at.",
    )
    parser.add_argument(
        "--precomp_path",
        type=str,
        default="data/precomp/",
        help="Path at which the precomputed misleader dataset is located at.",
    )
    parser.add_argument(
        "--output_prev_steps_path",
        type=str,
        default="src/model_tuning/03_deplot_axis_extraction_classifier/output/",
        help="Path at which the output of the previous two steps is located at. Simply the output root path of the previous two steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for training the model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs for training the model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training the model.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Patience iterations for early stopping of model training",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.004,
        help="Minimum delta per iteration of validation loss to be considered a big step and thus resetting the patience counter.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Size of the hidden layer in the model.",
    )
    parser.add_argument(
        "--base_models",
        type=str,
        nargs="+",
        default=[
            "tinychart",
        ],
        help="List of models for which vision encoders the experiments should be run.",
    )
    parser.add_argument(
        "--experiment_types",
        type=str,
        nargs="+",
        default=[
            "with_axis",
            "encoder_only",
        ],
        help="List of model configurations the experiments should be run for.",
    )

    args = parser.parse_args()

    run_all_train_experiments(
        args.datasetpath_misviz_synth,
        args.datasetpath_misviz,
        args.precomp_path,
        args.output_prev_steps_path,
        args.outputpath,
        args.experiment_types,
        args.base_models,
        args.lr,
        args.epochs,
        args.batch_size,
        args.patience,
        args.min_delta,
        args.hidden_dim,
    )
