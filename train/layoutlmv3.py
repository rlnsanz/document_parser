import torch
import torch.nn as nn
from torch.utils import data as torchdata
import numpy as np

from transformers import AutoProcessor, LayoutLMv3ForTokenClassification  # type: ignore
from datasets import load_dataset
from datasets.features import ClassLabel
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
import evaluate

import flor

# Device configuration
device = torch.device(
    flor.arg("device", "cuda" if torch.cuda.is_available() else "cpu")
)

# Hyper-parameters
num_epochs = flor.arg("epochs", default=5)
batch_size = flor.arg("batch_size", 4)
learning_rate = flor.arg("lr", 1e-5)

# Data loader
dataset = load_dataset("nielsr/funsd-layoutlmv3")
print(str(dataset))


features = dataset["train"].features  # type: ignore
print("Features: ", features)
column_names = dataset["train"].column_names  # type: ignore
print("Column Names:", column_names)
image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "bboxes"
label_column_name = "ner_tags"


# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    # No need to convert the labels since they are already ints.
    id2label = {k: v for k, v in enumerate(label_list)}
    label2id = {v: k for k, v in enumerate(label_list)}
else:
    label_list = get_label_list(dataset["train"][label_column_name])  # type: ignore
    id2label = {k: v for k, v in enumerate(label_list)}
    label2id = {v: k for k, v in enumerate(label_list)}
num_labels = len(label_list)

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base", id2label=id2label, label2id=label2id
)
assert isinstance(model, LayoutLMv3ForTokenClassification)
model = model.to(device)  # type: ignore

print(model)


def prepare_examples(examples):
    images = examples[image_column_name]
    words = examples[text_column_name]
    boxes = examples[boxes_column_name]
    word_labels = examples[label_column_name]

    encoding = processor(
        images,
        words,
        boxes=boxes,
        word_labels=word_labels,
        truncation=True,
        padding="max_length",
    )

    return encoding


features = Features(
    {
        "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
        "input_ids": Sequence(feature=Value(dtype="int64")),
        "attention_mask": Sequence(Value(dtype="int64")),
        "bbox": Array2D(dtype="int64", shape=(512, 4)),
        "labels": Sequence(feature=Value(dtype="int64")),
    }  # type: ignore
)

train_dataset = dataset["train"].map(  # type: ignore
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)
train_dataset.set_format("torch")

eval_dataset = dataset["test"].map(  # type: ignore
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)
eval_dataset.set_format("torch")


train_loader = torchdata.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=torchdata.default_collate)  # type: ignore
test_loader = torchdata.DataLoader(
    dataset=eval_dataset,  # type: ignore
    batch_size=batch_size,
    shuffle=False,
    collate_fn=torchdata.default_collate,
)


# Loss and optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # type: ignore

metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    assert results is not None
    return {
        "precision": flor.log("precision", results["overall_precision"]),
        "recall": flor.log("recall", results["overall_recall"]),
        "f1": flor.log("f1", results["overall_f1"]),
        "accuracy": flor.log("accuracy", results["overall_accuracy"]),
    }


# Train the model
total_step = len(train_loader)
num_steps = 1000
with flor.checkpointing(model=model, optimizer=optimizer):
    for epoch in flor.loop("epoch", range(num_epochs)):
        model.train()
        for i, batch in flor.loop("step", enumerate(train_loader)):
            # Move tensors to the configured device
            for k in batch:
                batch[k] = batch[k].to(device)

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i, total_step, flor.log("loss", loss.item())
                )
            )

        # Validate the model
        print("Model VALIDATE")
        model.eval()
        with torch.no_grad():
            preds = []
            labels = []
            # Valudate on 15% subsample of training data
            for i, batch in enumerate(train_loader):
                if i >= num_steps:
                    break
                for k in batch:
                    batch[k] = batch[k].to(device)

                # Forward pass
                outputs = model(**batch)
                preds.append(outputs.logits.cpu())
                labels.append(batch["labels"].cpu())

            # compute metrics
            p = np.concatenate(preds)
            l = np.concatenate(labels)
            result = compute_metrics((p, l))


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
print("Model TEST")
model.eval()
with torch.no_grad():
    preds = []
    labels = []
    for i, batch in enumerate(test_loader):
        for k in batch:
            batch[k] = batch[k].to(device)

        # Forward pass
        outputs = model(**batch)
        preds.append(outputs.logits.cpu())
        labels.append(batch["labels"].cpu())

        # compute metrics
    p = np.concatenate(preds)
    l = np.concatenate(labels)
    result = compute_metrics((p, l))

    print(result)
