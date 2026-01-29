import os
import json
import pickle
import torch
import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, hf_hub_download
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from comet import download_model, load_from_checkpoint
import re

# Configuration
HF_TOKEN = "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"
SOURCE_REPO = "amanuelbyte/finetranslations-sentence-level"
TARGET_REPO = "amanuelbyte/finetranslations-sentence-level-cleaned"
LANG_CONFIGS = [
    "afr_Latn", "amh_Ethi", "arz_Arab", "hau_Latn", "lin_Latn",
    "som_Latn", "swh_Latn", "wol_Latn", "yor_Latn", "zul_Latn"
]
BATCH_SIZE = 25000
TARGET_GOAL = 5000000
SIMILARITY_THRESHOLD = 0.6
QE_THRESHOLD = 0.55
MIN_LEN = 3
MAX_LEN = 200
MAX_RATIO = 2.0

# Initialize Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# LID Model
lid_pipeline = pipeline("text-classification", model="UBC-NLP/afrolid_1.5", device=0 if device == "cuda" else -1)

# Semantic Similarity Model
labse_model = SentenceTransformer('sentence-transformers/LaBSE').to(device)

# QE Model (Using unbabel-comet)
print("Loading QE model...")
qe_model_path = download_model("masakhane/africomet-qe-stl")
qe_model = load_from_checkpoint(qe_model_path)
if device == "cuda":
    qe_model = qe_model.to(device)

api = HfApi(token=HF_TOKEN)

def clean_text(text):
    if not text: return ""
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
    return text

def get_checkpoint():
    try:
        state_path = hf_hub_download(repo_id=TARGET_REPO, filename="state.json", repo_type="dataset", token=HF_TOKEN)
        with open(state_path, 'r') as f:
            state = json.load(f)
    except Exception as e:
        print(f"No state.json found, starting fresh. Error: {e}")
        state = {lang: {"processed_idx": 0, "collected_count": 0} for lang in LANG_CONFIGS}
    
    try:
        hashes_path = hf_hub_download(repo_id=TARGET_REPO, filename="hashes.pkl", repo_type="dataset", token=HF_TOKEN)
        with open(hashes_path, 'rb') as f:
            seen_hashes = pickle.load(f)
    except Exception as e:
        print(f"No hashes.pkl found, starting fresh. Error: {e}")
        seen_hashes = set()
        
    return state, seen_hashes

def save_checkpoint(state, seen_hashes):
    with open("state.json", "w") as f:
        json.dump(state, f)
    with open("hashes.pkl", "wb") as f:
        pickle.dump(seen_hashes, f)
    
    try:
        api.upload_file(path_or_fileobj="state.json", path_in_repo="state.json", repo_id=TARGET_REPO, repo_type="dataset")
        api.upload_file(path_or_fileobj="hashes.pkl", path_in_repo="hashes.pkl", repo_id=TARGET_REPO, repo_type="dataset")
    except Exception as e:
        print(f"Failed to upload checkpoint: {e}")

def process_language(lang_config, state, seen_hashes):
    print(f"Processing {lang_config}...")
    iso_code = lang_config.split('_')[0]
    
    try:
        dataset = load_dataset(SOURCE_REPO, lang_config, split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset for {lang_config}: {e}")
        return

    current_state = state.get(lang_config, {"processed_idx": 0, "collected_count": 0})
    if current_state["collected_count"] >= TARGET_GOAL:
        print(f"Goal already reached for {lang_config}")
        return

    dataset_iter = iter(dataset)
    # Skip already processed
    for _ in range(current_state["processed_idx"]):
        next(dataset_iter, None)

    batch_data = []
    for item in dataset_iter:
        batch_data.append(item)
        current_state["processed_idx"] += 1
        
        if len(batch_data) >= BATCH_SIZE:
            cleaned_batch = filter_batch(batch_data, iso_code, seen_hashes)
            if cleaned_batch:
                new_data = Dataset.from_list(cleaned_batch)
                new_data.push_to_hub(TARGET_REPO, config_name=lang_config, split="train", token=HF_TOKEN, append=True)
                current_state["collected_count"] += len(cleaned_batch)
            
            state[lang_config] = current_state
            save_checkpoint(state, seen_hashes)
            print(f"Progress for {lang_config}: {current_state['collected_count']}/{TARGET_GOAL}")
            
            if current_state["collected_count"] >= TARGET_GOAL:
                break
            batch_data = []
    
    # Process remaining
    if batch_data and current_state["collected_count"] < TARGET_GOAL:
        cleaned_batch = filter_batch(batch_data, iso_code, seen_hashes)
        if cleaned_batch:
            new_data = Dataset.from_list(cleaned_batch)
            new_data.push_to_hub(TARGET_REPO, config_name=lang_config, split="train", token=HF_TOKEN, append=True)
            current_state["collected_count"] += len(cleaned_batch)
        state[lang_config] = current_state
        save_checkpoint(state, seen_hashes)

def filter_batch(batch, expected_iso, seen_hashes):
    filtered = []
    
    src_texts = [clean_text(item['sentence_eng']) for item in batch]
    tgt_texts = [clean_text(item['sentence_tgt']) for item in batch]
    
    # 1. Rule-Based & Deduplication
    valid_indices = []
    for i, (s, t) in enumerate(zip(src_texts, tgt_texts)):
        if not s or not t: continue
        
        s_words, t_words = s.split(), t.split()
        if not (MIN_LEN <= len(s_words) <= MAX_LEN and MIN_LEN <= len(t_words) <= MAX_LEN):
            continue
        
        ratio = max(len(s_words), len(t_words)) / max(1, min(len(s_words), len(t_words)))
        if ratio > MAX_RATIO:
            continue
            
        pair_hash = hash(f"{s}|||{t}")
        if pair_hash in seen_hashes:
            continue
        
        seen_hashes.add(pair_hash)
        valid_indices.append(i)

    if not valid_indices: return []

    # 2. LID Filtering
    final_indices = []
    subset_tgt = [tgt_texts[i] for i in valid_indices]
    lid_results = lid_pipeline(subset_tgt, truncation=True)
    
    for idx, res in enumerate(lid_results):
        if res['label'] == expected_iso:
            final_indices.append(valid_indices[idx])

    if not final_indices: return []

    # 3. Semantic Similarity (LaBSE)
    subset_src = [src_texts[i] for i in final_indices]
    subset_tgt = [tgt_texts[i] for i in final_indices]
    
    with torch.no_grad():
        src_emb = labse_model.encode(subset_src, convert_to_tensor=True)
        tgt_emb = labse_model.encode(subset_tgt, convert_to_tensor=True)
        cos_sim = util.cos_sim(src_emb, tgt_emb).diagonal()
    
    # 4. QE Filtering (AfriCOMET)
    qe_data = [{"src": s, "mt": t} for s, t in zip(subset_src, subset_tgt)]
    # AfriCOMET-QE-STL returns a prediction object with 'scores'
    qe_outputs = qe_model.predict(qe_data, batch_size=32, gpus=1 if device == "cuda" else 0)
    qe_scores = qe_outputs.scores
    
    for i in range(len(final_indices)):
        if cos_sim[i] > SIMILARITY_THRESHOLD and qe_scores[i] > QE_THRESHOLD:
            filtered.append({
                "sentence_eng": subset_src[i],
                "sentence_tgt": subset_tgt[i],
                "score_sim": float(cos_sim[i]),
                "score_qe": float(qe_scores[i])
            })
            
    return filtered

if __name__ == "__main__":
    state, seen_hashes = get_checkpoint()
    for lang in LANG_CONFIGS:
        process_language(lang, state, seen_hashes)
