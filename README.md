# African Language Data Cleaning Pipeline

This repository contains a robust, checkpoint-enabled data cleaning pipeline designed to filter and improve the quality of a large-scale African language dataset. The pipeline is implemented in Python and utilizes the Hugging Face ecosystem for data loading, model inference, and checkpoint management.

## 1. System Requirements and Setup

The pipeline is resource-intensive, requiring a machine with sufficient RAM and a GPU (NVIDIA recommended) for efficient model inference.

### 1.1. Prerequisites

Ensure the following are installed on your server:
*   **Python 3.8+**
*   **Git**
*   **NVIDIA Drivers and CUDA Toolkit** (for GPU acceleration)

### 1.2. Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd african_data_cleaning
    ```

2.  **Install Dependencies**
    The required libraries, including `torch`, `transformers`, `datasets`, and `sentence-transformers`, are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## 2. Pipeline Configuration

The core configuration is managed within the `pipeline.py` script. The following parameters are pre-set based on your request:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | `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` | Your Hugging Face Write Token for checkpointing and pushing results. |
| `SOURCE_REPO` | `amanuelbyte/finetranslations-sentence-level` | The source dataset repository on Hugging Face. |
| `TARGET_REPO` | `amanuelbyte/finetranslations-sentence-level-cleaned` | The target repository where cleaned data and checkpoints are saved. |
| `BATCH_SIZE` | `25000` | Number of rows processed before a checkpoint is saved. |
| `TARGET_GOAL` | `5,000,000` | The target number of clean pairs to collect for each language configuration. |

## 3. Filtering Stages and Thresholds

The pipeline applies a rigorous, multi-stage filtering process to ensure high data quality.

### 3.1. Rule-Based Filtering and Deduplication

| Rule | Threshold/Logic |
| :--- | :--- |
| **Cleaning** | Removes HTML tags and collapses multiple whitespaces. |
| **Deduplication** | Uses a global hash set (`hashes.pkl`) to prevent duplicate pairs across all languages. |
| **Min Length** | Source and Target sentences must have at least 3 words. |
| **Max Length** | Source and Target sentences must have at most 200 words. |
| **Max Length Ratio** | The ratio of sentence lengths (words) must be less than 2.0. |

### 3.2. Model-Based Filtering

| Stage | Model | Logic | Threshold |
| :--- | :--- | :--- | :--- |
| **Language Identification (LID)** | `UBC-NLP/afrolid_1.5` | Verifies that the target sentence matches the expected ISO language code. | N/A (Match required) |
| **Semantic Filtering** | `sentence-transformers/LaBSE` | Cosine similarity between source and target sentence embeddings. | Score > 0.6 |
| **Quality Estimation (QE)** | `masakhane/africomet-qe-stl` | Estimates the quality of the translation pair. | Score > 0.55 |

## 4. Execution and Resumability

### 4.1. Running the Pipeline

To start or resume the pipeline, execute the main script:
```bash
python pipeline.py
```

### 4.2. Checkpointing

The script automatically manages checkpoints, ensuring that the process can be safely interrupted and resumed.

*   **State Tracking**: Progress is saved in `state.json`, tracking the `processed_idx` and `collected_count` for each language.
*   **Deduplication Tracking**: Seen hashes are saved in `hashes.pkl`.
*   **Storage**: Both files are uploaded to the root of your target Hugging Face repository (`amanuelbyte/finetranslations-sentence-level-cleaned`) after every batch. Upon restart, the script automatically downloads the latest checkpoint files to resume.

## 5. Languages Processed

The pipeline iterates through the following 10 language configurations:

*   `afr_Latn` (Afrikaans)
*   `amh_Ethi` (Amharic)
*   `arz_Arab` (Egyptian Arabic)
*   `hau_Latn` (Hausa)
*   `lin_Latn` (Lingala)
*   `som_Latn` (Somali)
*   `swh_Latn` (Swahili)
*   `wol_Latn` (Wolof)
*   `yor_Latn` (Yoruba)
*   `zul_Latn` (Zulu)
