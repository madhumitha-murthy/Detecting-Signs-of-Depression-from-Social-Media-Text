# 🧠 Depression Detection from Social Media Text

### Published Research · TechWhiz@LT-EDI · DepSign-LT-EDI@RANLP 2023

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Conference](https://img.shields.io/badge/RANLP-2023-blueviolet?style=flat-square)
![Task](https://img.shields.io/badge/Task-Multi--class%20Classification-orange?style=flat-square)

---

## 📄 Paper

> **TechWhiz@LT-EDI: Transformer Models to Detect Levels of Depression from Social Media Text**
> Madhumitha M, C. Jerin Mahibha
> Meenakshi Sundararajan Engineering College, Chennai
> Durairaj Thenmozhi — Sri Sivasubramaniya Nadar College of Engineering, Chennai
> *Proceedings of the Third Workshop on Language Technology for Equality, Diversity and Inclusion (LT-EDI), RANLP 2023*

---

## 📌 Overview

Depression is a serious mental health disorder that profoundly impacts emotions, thoughts, and daily behaviour. Social media platforms have become a significant contributor — social comparison, cyberbullying, and unrealistic standards all exacerbate depressive symptoms.

This project presents the system submitted by team **TechWhiz** to the **DepSign-LT-EDI@RANLP 2023** shared task on **Detecting Signs of Depression from Social Media Text**. The system fine-tunes two transformer models — **ALBERT** and **RoBERTa** — to classify English social media posts into three depression severity levels.

> Detecting levels of depression is a particularly challenging task because mental states shift over time and are highly dependent on individual context, sarcasm, and implicit language.

---

## 🎯 Task

Multi-class classification of social media text into:

| Label | Description |
|---|---|
| `not depression` | No signs of depression |
| `moderate` | Moderate depressive indicators |
| `severe` | Severe depressive symptoms |

**Example:**
> *"I didn't deserve all of this: I have been suffering from depression for 5 years…"* → **Severe**
> *"I constantly feel like anyone I talk to is just trying to get me to shut up."* → **Moderate**

---

## 📊 Results

| Model | Macro F1 | Accuracy |
|---|---|---|
| **ALBERT** | **0.258** | **0.421** |
| RoBERTa | 0.143 | 0.263 |

> ALBERT outperformed RoBERTa and was selected as the final submission. The system achieved **Rank 29** on the shared task leaderboard.

---

## 📂 Dataset

The dataset was provided by the shared task organizers (Sampath et al., 2023) and is **not included in this repository due to its size and licensing**.

| Category | Training | Evaluation |
|---|---|---|
| Moderate | 3,700 | 2,169 |
| Not depression | 2,755 | 848 |
| Severe | 768 | 316 |
| **Total** | **7,201** | **3,333** |
| Test | — | 499 |

The dataset is **heavily imbalanced** — the severe class has far fewer instances than moderate, which directly impacted model performance and led to increased false positives and negatives.

To obtain the dataset, refer to the [LT-EDI@RANLP 2023 shared task page](https://sites.google.com/view/lt-edi-2023/).

---

## 🧠 Models

### ALBERT (albert-base-v2)
A Lite BERT — introduced parameter-sharing and factorization techniques achieving up to 89% parameter reduction compared to BERT, while retaining expressive linguistic understanding. Architecture: 12 transformer layers, 768 hidden units, 12 attention heads.

### RoBERTa (roberta-base)
A Robustly Optimized BERT approach — pre-trained on large English corpora with larger mini-batches and longer training. Architecture: 12 layers, 768 hidden units, 12 attention heads, 125M parameters. Case-sensitive.

Both models were trained for **1 epoch** with the AdamW optimizer and cross-entropy loss.

---

## 🏗️ System Architecture

```
Raw Social Media Text
        │
        ▼
  Preprocessing
  (remove digits, symbols, URLs, tags, whitespace)
        │
        ▼
  Label Encoding
  (moderate / not depression / severe → integer)
        │
        ▼
  Tokenization
  (model-specific tokenizer, max_length=256)
        │
    ┌───┴───┐
    ▼       ▼
 ALBERT  RoBERTa
    │       │
    └───┬───┘
        ▼
  Evaluation on Dev Set
  (macro F1 + accuracy)
        │
        ▼
  ALBERT selected → Test Predictions
```

---

## 🏗️ Project Structure

```
depression-detection/
│
├── notebooks/
│   └── depression_detection.ipynb   # Full training and evaluation notebook
│
├── src/
│   ├── preprocess.py                # Text cleaning and label encoding
│   ├── train.py                     # Fine-tuning ALBERT / RoBERTa
│   ├── evaluate.py                  # Metrics, classification report, plots
│   └── predict.py                   # Inference on test set
│
├── data/                            # Dataset (not included — see above)
├── models/                          # Saved checkpoints (git-ignored)
├── outputs/                         # Predictions (git-ignored)
├── results/                         # Classification reports and plots
│
├── paper/
│   └── Paper.pdf                    # Published paper
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/madhumitha-murthy/Depression-Detection-Social-Media
cd Depression-Detection-Social-Media
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

**Preprocess the data**

```bash
python src/preprocess.py
```

**Train the model**

```bash
python src/train.py --model albert    # ALBERT (best performance)
python src/train.py --model roberta   # RoBERTa
```

**Evaluate on the development set**

```bash
python src/evaluate.py --model albert
```

**Run inference on the test set**

```bash
python src/predict.py --model albert
```

---

## 🔬 Key Findings

- ALBERT significantly outperformed RoBERTa on this task (F1: 0.258 vs 0.143), demonstrating better generalisation on imbalanced, short social media text.
- The **severe** class had the highest precision and recall due to more explicit depression markers ("kill myself", "don't want to be alive").
- The **not depression** class was most frequently misclassified — fewer training examples and subtler language made it harder to distinguish from moderate cases.
- **Sarcasm** proved a major challenge: texts like *"I'm fine"* used ironically were systematically misclassified.
- Data augmentation is identified as the most impactful next step to address class imbalance.

**Example misclassifications:**

| Text | Predicted | Actual |
|---|---|---|
| "I don't want to kill myself, I just don't want to be alive anymore" | Moderate | Severe |
| "But here I am, 24 years old man and doing exactly that" | Severe | Moderate |
| "I'm trapped inside. Does anyone else get that feeling?" | Severe | Moderate |

---

## ⚠️ Limitations

- Severe class imbalance (3,700 moderate vs 768 severe) biases predictions toward the majority class
- Models struggle with sarcasm and implicit language — common in real social media posts
- Single-epoch training due to resource constraints; more epochs may improve performance
- English-only dataset — no multilingual or code-mixed support
- Textual analysis only — no multimodal signals (images, engagement patterns)

---

## 🚀 Future Work

- [ ] Data augmentation (back-translation, synonym replacement) to address class imbalance
- [ ] Sarcasm-aware pre-processing and feature engineering
- [ ] Hybrid model combining transformer + BiLSTM for richer contextual understanding
- [ ] Experiment with mental-health-specific pre-trained models (MentalBERT, MentalRoBERTa)
- [ ] Extend to multilingual depression detection

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@inproceedings{madhumitha2023techwhiz,
  title     = {TechWhiz@LT-EDI: Transformer Models to Detect Levels of
               Depression from Social Media Text},
  author    = {Madhumitha M and C. Jerin Mahibha and Durairaj Thenmozhi},
  booktitle = {Proceedings of the Third Workshop on Language Technology for
               Equality, Diversity and Inclusion (LT-EDI), RANLP 2023},
  year      = {2023}
}
```

---

## 👩‍💻 Authors

**Madhumitha M** · **C. Jerin Mahibha**
Meenakshi Sundararajan Engineering College, Chennai

**Durairaj Thenmozhi**
Sri Sivasubramaniya Nadar College of Engineering, Chennai

---

> ⚠️ **Note:** This project involves detection of depression and mental health signals from text. The models are research prototypes and are **not intended for clinical use or diagnosis**. If you or someone you know is struggling, please reach out to a qualified mental health professional.

---

⭐ If this project helped your research, consider giving it a star!
