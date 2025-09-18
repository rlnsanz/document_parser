import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List, Tuple, Dict
import torch
from PIL import Image
import pytesseract
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
import json

import flordb as flor

import torch.nn.functional as F
import matplotlib.pyplot as plt

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config as config

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_model(
    checkpoint: str,
    device: torch.device,
    base_model: str = "microsoft/layoutlmv3-base",
    labels: str = None,
):
    def _load_label_map(labels, num_labels_fallback: int = 2):
        if not labels:
            id2label = {i: f"LABEL_{i}" for i in range(num_labels_fallback)}
            label2id = {v: k for k, v in id2label.items()}
            return id2label, label2id
        else:
            data = labels
        # Support a list ["O","B-FOO",...] or dict {"O":0,...} or {"0":"O",...}
        if isinstance(data, list):
            id2label = {i: lbl for i, lbl in enumerate(data)}
            label2id = {lbl: i for i, lbl in enumerate(data)}
        elif all(isinstance(k, str) and k.isdigit() for k in data.keys()):
            id2label = {int(k): v for k, v in data.items()}
            label2id = {v: k for k, v in id2label.items()}
        else:
            label2id = {k: int(v) for k, v in data.items()}
            id2label = {v: k for k, v in label2id.items()}
        return id2label, label2id

    def _infer_num_labels(state_dict: Dict[str, torch.Tensor], default: int = 2) -> int:
        for key in (
            "classifier.weight",
            "model.classifier.weight",
            "layoutlmv3.classifier.weight",
        ):
            if key in state_dict:
                return int(state_dict[key].shape[0])
        for key in (
            "classifier.bias",
            "model.classifier.bias",
            "layoutlmv3.classifier.bias",
        ):
            if key in state_dict:
                return int(state_dict[key].shape[0])
        return default

    def _strip_prefix(
        d: Dict[str, torch.Tensor], prefix: str
    ) -> Dict[str, torch.Tensor]:
        return {
            (k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in d.items()
        }

    if os.path.isdir(checkpoint):
        # HF directory: load normally
        processor = AutoProcessor.from_pretrained(base_model, apply_ocr=False)
        model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint)
        model.to(device).eval()
        id2label = model.config.id2label
        label2id = model.config.label2id
        return processor, model, id2label, label2id

    # Raw PyTorch checkpoint file
    ckpt = torch.load(os.path.expanduser(checkpoint), map_location="cpu")
    # Unwrap common containers
    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(
            "Unsupported checkpoint format; expected a state_dict or dict with 'state_dict'."
        )

    # Clean common prefixes
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")

    num_labels = _infer_num_labels(state_dict, default=2)
    id2label, label2id = _load_label_map(labels, num_labels)

    # Processor from base model/repo
    processor = AutoProcessor.from_pretrained(base_model, apply_ocr=False)

    # Instantiate base model with correct label space, then load weights
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # helpful if head dims differ
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"Warning: {len(missing)} missing keys when loading checkpoint (e.g., {missing[:3]})"
        )
    if unexpected:
        print(
            f"Warning: {len(unexpected)} unexpected keys in checkpoint (e.g., {unexpected[:3]})"
        )

    model.to(device).eval()
    return processor, model, id2label, label2id


def _normalize_box(
    box: Tuple[int, int, int, int], width: int, height: int
) -> List[int]:
    # Convert pixel bbox to 0-1000 LayoutLM coordinates
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, width))
    x1 = max(0, min(x1, width))
    y0 = max(0, min(y0, height))
    y1 = max(0, min(y1, height))
    return [
        int(1000 * x0 / width) if width > 0 else 0,
        int(1000 * y0 / height) if height > 0 else 0,
        int(1000 * x1 / width) if width > 0 else 0,
        int(1000 * y1 / height) if height > 0 else 0,
    ]


def ocr_words_and_boxes(image: Image.Image) -> Tuple[List[str], List[List[int]]]:
    if pytesseract is None:
        raise RuntimeError(
            "pytesseract is not installed. Please install it to run OCR."
        )
    w, h = image.size
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words: List[str] = []
    boxes: List[List[int]] = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i]
        conf = data.get("conf", ["-1"] * n)[i]
        try:
            conf_val = float(conf)
        except Exception:
            conf_val = -1.0
        if text is None or text.strip() == "" or conf_val < 0:
            continue
        x, y, bw, bh = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        x0, y0, x1, y1 = x, y, x + bw, y + bh
        words.append(text.strip())
        boxes.append(_normalize_box((x0, y0, x1, y1), w, h))
    return words, boxes


def to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def predict_word_labels(
    processor,
    model,
    image: Image.Image,
    words: List[str],
    boxes: List[List[int]],
    device: torch.device,
):
    encoding = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
        max_length=512,
    )
    # forward
    with torch.no_grad():
        outputs = model(**to_device(encoding, device))
        logits = outputs.logits[0]  # (seq_len, num_labels)
    # Map sub-token predictions back to word-level
    word_ids = encoding.word_ids(batch_index=0)
    # Aggregate logits per word_id
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in sums:
            sums[wid] = logits[idx].detach().cpu()
            counts[wid] = 1
        else:
            sums[wid] += logits[idx].detach().cpu()
            counts[wid] += 1
    word_label_ids: List[int] = []
    word_confidences: List[float] = []
    num_labels = logits.shape[-1]
    for wid in range(len(words)):
        if wid in sums:
            avg_logits = sums[wid] / max(1, counts[wid])
        else:
            avg_logits = torch.zeros(num_labels)
        probs = F.softmax(avg_logits, dim=-1)
        label_id = int(torch.argmax(probs).item())
        conf = float(torch.max(probs).item())
        word_label_ids.append(label_id)
        word_confidences.append(conf)
    return word_label_ids, word_confidences


def label_colors(id2label: Dict[int, str]):
    labels = [id2label[i] for i in sorted(id2label)]
    cmap = plt.get_cmap("tab20")
    color_map: Dict[str, Tuple[float, float, float, float]] = {}
    j = 0
    for lbl in labels:
        if lbl.upper() == "O":
            color_map[lbl] = (0.2, 0.2, 0.2, 0.8)
        else:
            color_map[lbl] = cmap(j % 20)
            j += 1
    return color_map


def draw_boxes(
    image: Image.Image,
    words: List[str],
    boxes: List[List[int]],
    label_ids: List[int],
    confidences: List[float],
    id2label: Dict[int, str],
    output_path: str,
    conf_threshold: float = 0.0,
):
    w, h = image.size
    colors = label_colors(id2label)
    fig, ax = plt.subplots(1, 1, figsize=(min(16, w / 60 + 2), min(16, h / 60 + 2)))
    ax.imshow(image)
    ax.axis("off")
    for word, box, lid, conf in zip(words, boxes, label_ids, confidences):
        label = id2label.get(int(lid), str(lid))
        if conf < conf_threshold:
            continue
        x0 = int(box[0] * w / 1000.0)
        y0 = int(box[1] * h / 1000.0)
        x1 = int(box[2] * w / 1000.0)
        y1 = int(box[3] * h / 1000.0)
        rect_w = max(1, x1 - x0)
        rect_h = max(1, y1 - y0)
        color = colors.get(label, (1.0, 0.0, 0.0, 0.9))
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), rect_w, rect_h, fill=False, edgecolor=color, linewidth=1.5
            )
        )
        text = f"{label} ({conf:.2f})"
        ax.text(
            x0,
            max(0, y0 - 2),
            text,
            fontsize=8,
            color="white",
            bbox=dict(facecolor=color, alpha=0.6, pad=1, edgecolor="none"),
        )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main():
    df = flor.dataframe("accuracy")
    if df.empty:
        raise RuntimeError("No accuracy data found. Please run training first.")

    df = flor.utils.latest(df[df["filename"] == "layoutlmv3.py"])
    if df.empty:
        raise RuntimeError(
            "No accuracy data found for layoutlmv3.py. Please run training first."
        )

    max_row = df.loc[df["accuracy"].idxmax()]
    max_tstamp = str(max_row["tstamp"]).replace(" ", "T")
    check_point = os.path.join(
        os.path.expanduser("~/.flor"),
        "obj_store",
        str(max_row["projid"]),
        max_tstamp,
        f"model_epoch_{int(max_row['epoch_value'])}.pth",
    )
    print("Using checkpoint:", check_point)

    base_model = flor.arg("base_model", "microsoft/layoutlmv3-base")

    labels = flor.utils.latest(flor.dataframe("labels_layoutlmv3"))
    assert not labels.empty, "No labels found. Please run training first."
    labels = eval(labels.iloc[0]["labels_layoutlmv3"])
    print("Using labels:", labels)

    device = torch.device(flor.arg("device", config.device))
    processor, model, id2label, _ = load_model(
        check_point, device, base_model=base_model, labels=labels
    )

    documents = [
        each
        for each in os.listdir("private")
        if os.path.isdir(os.path.join("private", each))
    ]

    for doc in documents:
        pred_dir = os.path.join("private", doc, "layoutlmv3")
        os.makedirs(pred_dir, exist_ok=True)
        pages = sorted(
            [
                f
                for f in os.listdir(os.path.join("private", doc, "images"))
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp"))
            ],
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        for page in pages:
            image_path = os.path.join("private", doc, "images", page)
            output_path = os.path.join("private", doc, "layoutlmv3", page)
            print(f"Processing {image_path}")
            image = Image.open(image_path).convert("RGB")
            words, boxes = ocr_words_and_boxes(image)
            label_ids, confidences = predict_word_labels(
                processor, model, image, words, boxes, device
            )
            draw_boxes(
                image,
                words,
                boxes,
                label_ids,
                confidences,
                id2label,
                output_path,
            )


if __name__ == "__main__":
    main()
