# 📦 Visual Wake Words (VWW) Dataset Generator

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset: COCO 2014](https://img.shields.io/badge/Dataset-COCO%202014-orange)](https://cocodataset.org/)
[![Task: Binary Classification](https://img.shields.io/badge/Task-Binary%20Classification-green)]()

A clean, minimal script to generate the **Visual Wake Words (VWW)** dataset from [COCO 2014](https://cocodataset.org/), following the methodology described in the original [VWW paper](https://arxiv.org/abs/1906.05721). Designed for **TinyML**, **NanoNAS**, and **edge AI** experiments.

---

## 🧠 What is the VWW Dataset?

Visual Wake Words is a binary image classification benchmark derived from COCO. Each image is labelled:

| Label | Meaning |
|---|---|
| `person` | At least one person is present and large enough in the image |
| `not_person` | No person present (or persons are too small) |

The classification threshold is area-based: a person annotation is only counted if its bounding box covers more than **0.5%** of the total image area (configurable).

---

## ✨ Features

- ✅ Generates the full VWW dataset from COCO 2014 train/val splits
- ✅ Configurable area threshold for person detection
- ✅ Preserves original image resolution — no forced resizing
- ✅ Organized output in a `train/val` × `person/not_person` folder structure
- ✅ Progress bars via `tqdm` for real-time monitoring
- ✅ Minimal dependencies — easy to set up anywhere

---

## 📂 Repository Structure

```
vww-dataset-generator/
├── generate_vww_dataset.py   # Main script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 📁 Output Structure

After running the script, the following folder structure is created:

```
vww_dataset/
├── train/
│   ├── person/           # Train images containing a person
│   └── not_person/       # Train images without a person
└── val/
    ├── person/           # Val images containing a person
    └── not_person/       # Val images without a person
```

---

## ⚙️ Requirements

**Python 3.7+** is required.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
pycocotools
Pillow
tqdm
```

Or install manually:

```bash
pip install pycocotools pillow tqdm
```

> **Windows users:** `pycocotools` may require Visual C++ Build Tools. Consider using [pycocotools-windows](https://pypi.org/project/pycocotools-windows/) as a drop-in replacement.

---

## 🚀 Getting Started

Follow the steps below to generate the VWW dataset from scratch.

---

### 📥 Step 1 — Download the COCO 2014 Dataset

```bash
# Create the coco directory
mkdir -p coco && cd coco

# Download images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

---

### 📦 Step 2 — Extract the Files

```bash
unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip
cd ..
```

After extraction, your `coco/` directory should look like this:

```
coco/
├── train2014/                        # ~82,783 images
├── val2014/                          # ~40,504 images
└── annotations/
    ├── instances_train2014.json
    └── instances_val2014.json
```

> 💡 The total download size is approximately **~19 GB**. Make sure you have sufficient disk space.

---

### 🛠️ Step 3 — Configure the Script (Optional)

Open `generate_vww_dataset.py` and update the paths if your COCO data lives elsewhere:

```python
# ===== CONFIGURATION =====
COCO_ROOT   = os.path.join(os.getcwd(), "coco")        # Path to your COCO folder
OUTPUT_ROOT = os.path.join(os.getcwd(), "vww_dataset") # Where to save the VWW dataset

AREA_THRESHOLD = 0.005  # Minimum bbox/image area ratio to count as 'person' (0.5%)
# =========================
```

Alternatively, use absolute paths:

```python
COCO_ROOT   = os.path.expanduser("~/datasets/coco")
OUTPUT_ROOT = os.path.expanduser("~/datasets/vww_dataset")
```

---

### ▶️ Step 4 — Run the Script

```bash
python generate_vww_dataset.py
```

You should see output like:

```
Processing train set...
loading annotations into memory...
100%|████████████████████████████████| 82783/82783 [07:34<00:00, 182.1it/s]

Processing val set...
loading annotations into memory...
100%|████████████████████████████████| 40504/40504 [03:21<00:00, 200.8it/s]

Visual Wake Words dataset generation complete.
```

---

## 🔬 How It Works

For every image in the COCO dataset, the script:

1. Loads all annotations for that image
2. Filters annotations belonging to the `person` category
3. Computes the **area ratio** for each person bounding box:

$$\text{ratio} = \frac{bbox\_width \times bbox\_height}{image\_width \times image\_height}$$

4. Labels the image as **`person`** if any bounding box exceeds the area threshold, otherwise **`not_person`**
5. Copies the image (at original resolution) to the corresponding output folder

---

## ⚙️ Key Configuration Parameter

```python
AREA_THRESHOLD = 0.005
```

| Value | Effect |
|---|---|
| `0.005` | Standard VWW setting — persons must occupy ≥ 0.5% of the image (recommended) |
| `0.01` | Slightly stricter — filters out more distant/small persons |
| `0.05` | Relaxed — only large, prominent persons are counted |

Lowering the threshold increases the number of `person`-labelled images; raising it makes the dataset harder and more representative of challenging real-world conditions.

---

## ⚠️ No Resizing — By Design

This script intentionally **does not resize images**. The resize step is commented out:

```python
# image = image.resize(IMAGE_SIZE)
```

This is intentional because:

- It allows flexible input resolutions during training
- Resizing is best handled by your training pipeline (e.g., `tf.image.resize`, `torchvision.transforms`)
- It avoids baking in assumptions about model input size

To enable resizing, uncomment and set `IMAGE_SIZE` in the configuration block:

```python
IMAGE_SIZE = (96, 96)   # or (224, 224), etc.
# then in process_split():
image = image.resize(IMAGE_SIZE)
```

---

## ⏱️ Expected Runtime

| Split | Images | Estimated Time |
|---|---|---|
| `train` | ~82,783 | 5–10 min |
| `val` | ~40,504 | 3–5 min |

> Runtime depends on CPU speed, disk I/O, and whether your storage is SSD or HDD.

---

## 🔥 Use Cases

- 📱 **TinyML benchmarking** — standard benchmark for microcontroller-scale models
- 🔍 **Neural Architecture Search** — used in NanoNAS and similar automated design pipelines
- 🧪 **Edge AI experiments** — human presence detection for IoT and embedded systems
- 🎓 **Research reproducibility** — follow the original VWW paper's exact data pipeline

---

## 🐛 Troubleshooting

**`FileNotFoundError` when loading annotations:**
Verify that `coco/annotations/instances_train2014.json` exists and the `COCO_ROOT` path is correct.

**`pycocotools` install fails on Windows:**
Try `pip install pycocotools-windows` instead.

**Images are skipped silently:**
Corrupt or missing image files are caught and skipped via a `try/except` block. This is expected for a small number of files in COCO.

**Out of disk space:**
The generated `vww_dataset/` folder is roughly the same size as the source COCO images (~19 GB), since images are copied at original resolution.

---

## 📚 References

- **Visual Wake Words Dataset Paper:**
  Chowdhery et al., *"Visual Wake Words Dataset"*, 2019. [[arXiv:1906.05721]](https://arxiv.org/abs/1906.05721)

- **COCO Dataset:**
  Lin et al., *"Microsoft COCO: Common Objects in Context"*, ECCV 2014. [[cocodataset.org]](https://cocodataset.org/)

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

Note: The COCO dataset itself is subject to its own [terms of use](https://cocodataset.org/#termsofuse).

---

## 🙌 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
