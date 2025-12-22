# ============================================================
#  Enhanced Dataset Generation Script for Diff A1 Assignment
#  FAST MODE VERSION + TOPIC-DRIVEN + STRONG UNIQUENESS (Option D)
# ============================================================

import os
import re
import random
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# ===========================
# 1. FULL REPRODUCIBILITY
# ===========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===========================
# 2. CONFIGURATION - OPTIMIZED
# ===========================
LLM_MODEL_NAME = "ytu-ce-cosmos/Turkish-Gemma-9b-T1"
EMBEDDING_MODEL_NAME = "ytu-ce-cosmos/turkish-e5-large"
NUM_QUESTIONS_PER_SET = 1  # keep small for testing; set to 50 for full run
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./assignment_data"
CHECKPOINT_INTERVAL = 5

RAW_DATA_FILE = f"{OUTPUT_DIR}/raw_qa_data.json"
EMBEDDING_DATA_FILE = f"{OUTPUT_DIR}/concatenated_embeddings.npz"
CHECKPOINT_TRAIN_FILE = f"{OUTPUT_DIR}/generation_checkpoint_train.json"
CHECKPOINT_TEST_FILE = f"{OUTPUT_DIR}/generation_checkpoint_test.json"

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    try:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
        print(f"Initial GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f}GB")
    except Exception:
        pass

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================
# TOPIC POOL & PARTITION (Option D)
# ===========================
# A reasonably large list of CS / Software Engineering topics in Turkish.
TOPIC_POOL = [
    "Veri Yapıları", "Algoritma Analizi", "Dinamik Programlama", "Graf Teorisi",
    "Ağ Protokolleri", "İşletim Sistemleri", "Bellek Yönetimi", "Çoklu İş Parçacığı",
    "Veritabanı Tasarımı", "SQL Optimizasyonu", "Dağıtık Sistemler",
    "Makine Öğrenmesi", "Derin Öğrenme", "Doğal Dil İşleme", "Bilgisayar Görüşü",
    "Yazılım Mimarisi", "Mikroservisler", "API Tasarımı", "Güvenlik ve Kriptografi",
    "Yazılım Testi", "Performans Ölçümü", "Derleyici Tasarımı",
    "Sistem Programlama", "Bilgi Kuramı", "Yığın & Kuyruk Yapıları",
    "Ağaçlar & İkili Arama", "Hash Tablolar", "Sıkıştırma Algoritmaları",
    "Şebeke Güvenliği", "Yapay Zeka Etiği", "Kubernetes ve Container",
    "Sürüm Kontrol Sistemleri", "Concurrent Data Structures", "Önbellekleme Stratejileri"
]
# Ensure deterministic topic sampling
random.seed(SEED)

# Safety: if topic pool is too small, we allow reuse but try to maximize uniqueness
min_required_topics = NUM_QUESTIONS_PER_SET * 2
if len(TOPIC_POOL) < min_required_topics:
    # expand by duplicating with index suffix (deterministic)
    extended = []
    idx = 0
    while len(extended) < min_required_topics:
        extended.append(f"{TOPIC_POOL[idx % len(TOPIC_POOL)]} (var{idx//len(TOPIC_POOL)})")
        idx += 1
    TOPIC_POOL = extended

# Partition topics deterministically
shuffled_topics = TOPIC_POOL[:]  # copy
random.shuffle(shuffled_topics)
topics_train = shuffled_topics[:NUM_QUESTIONS_PER_SET]
topics_test = shuffled_topics[NUM_QUESTIONS_PER_SET:NUM_QUESTIONS_PER_SET*2]

# ===========================
# 3. OPTIMIZED MODEL LOADING
# ===========================
print("\nLoading Gemma 9B (4-bit quantized)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

gemma_tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_NAME,
    padding_side="left"
)

gemma_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

if gemma_tokenizer.pad_token is None:
    gemma_tokenizer.pad_token = gemma_tokenizer.eos_token

print("Loading E5-Large Embedding Model...")
e5_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)


# ==========================================================
# 4. CLEANING FUNCTION FOR LLM OUTPUT
# ==========================================================
def clean_llm_output(text):
    """Remove model boilerplate and unwanted tags."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(assistant|user|system)\s*[:\n]", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==========================================================
# 5. OPTIMIZED TEXT GENERATION HELPER
# ==========================================================
def generate_text(prompt, max_new_tokens=256):
    """Optimized generation function with memory management"""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    inputs = gemma_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = gemma_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=gemma_tokenizer.pad_token_id,
            eos_token_id=gemma_tokenizer.eos_token_id,
            use_cache=True
        )

    # Decode generated tokens (exclude prompt tokens)
    generated_ids = outputs[:, input_ids.shape[-1]:]
    text = gemma_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return clean_llm_output(text)


# ==========================================================
# 6. EMBEDDING HELPERS
# ==========================================================
def get_question_embeddings(q_list):
    return e5_model.encode(
        [f"query: {q}" for q in q_list],
        convert_to_numpy=True,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True
    )


def get_answer_embeddings(a_list):
    return e5_model.encode(
        [f"passage: {a}" for a in a_list],
        convert_to_numpy=True,
        batch_size=8,
        show_progress_bar=True,
        normalize_embeddings=True
    )


# ==========================================================
# 7. NORMALIZATION AND GLOBAL TRACKING
# ==========================================================
def normalize_question(q):
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)
    return q


global_seen_questions = set()


# ==========================================================
# 8. OPTIMIZED QA GENERATION WITH TOPICS + DUP REPAIR
# ==========================================================
def generate_qa_set(set_type, num_questions, topics_for_set):
    """
    topics_for_set: list of topics (length == num_questions). Each generated question is guided by the topic.
    """
    print(f"\n--- Generating {set_type} set ({num_questions} questions) ---")
    if DEVICE == "cuda":
        try:
            print(f"GPU Memory before generation: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
        except Exception:
            pass

    checkpoint_file = CHECKPOINT_TRAIN_FILE if set_type == "train" else CHECKPOINT_TEST_FILE

    data = []
    question_bank = []
    start_idx = 0

    # Checkpoint loading (compatible)
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from: {checkpoint_file}")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)

        if checkpoint_data and isinstance(checkpoint_data[0], dict):
            if "status" in checkpoint_data[0] and checkpoint_data[0]["status"] == "question_only":
                question_bank = [d["question"] for d in checkpoint_data]
                data = []
            else:
                data = checkpoint_data
                question_bank = list({d["question"] for d in data})
        else:
            question_bank = []
            data = []

        start_idx = len(question_bank)
        global_seen_questions.update([normalize_question(q) for q in question_bank])
        print(f"  Resuming from question {start_idx + 1}/{num_questions}")

    # --- Question Generation (topic-guided; FAST) ---
    for i in tqdm(range(start_idx, num_questions), desc=f"Generating {set_type} questions"):
        attempts = 0
        topic = topics_for_set[i] if i < len(topics_for_set) else random.choice(topics_for_set)
        while attempts < 6:
            prompt = (
                f"<start_of_turn>user\n"
                f"Konu: {topic}\n"
                f"Bana, bilgisayar mühendisliği / bilgisayar bilimi alanında, daha önce sormadığın, "
                f"{i + 1}. yeni ve teknik bir soru sor (konu: {topic}). Cevap verme, sadece soruyu yaz.\n"
                f"<end_of_turn>\n<start_of_turn>assistant\n"
            )
            try:
                q = generate_text(prompt, max_new_tokens=64)  # FAST-ish generation for question
                q_clean = normalize_question(q)

                # Enforce: non-empty, length > 8 chars, not duplicate
                if q_clean and q_clean not in global_seen_questions and len(q_clean) > 8:
                    global_seen_questions.add(q_clean)
                    question_bank.append(q.strip())
                    break
            except Exception as e:
                print(f"  Warning: Generation attempt {attempts + 1} failed: {str(e)}")
                if DEVICE == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            attempts += 1

        if attempts >= 6:
            # As a last resort generate a simple template question (deterministic)
            fallback_q = f"Soru: {topic} hakkında kısa teknik bir soru?"
            fallback_q_clean = normalize_question(fallback_q)
            if fallback_q_clean not in global_seen_questions:
                global_seen_questions.add(fallback_q_clean)
                question_bank.append(fallback_q)

        # Periodic cache clear
        if DEVICE == "cuda" and (i + 1) % 2 == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Save question-only checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            temp_checkpoint = [{"question": q, "status": "question_only"} for q in question_bank]
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(temp_checkpoint, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(question_bank)} unique questions (topic-guided).")

    # --- Answer Generation ---
    if DEVICE == "cuda":
        try:
            print(f"GPU Memory before answers: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
        except Exception:
            pass

    for idx, q in enumerate(tqdm(question_bank, desc=f"Generating {set_type} answers")):
        # If checkpoint had some answers already, skip those
        if idx < len(data) // 2:
            continue

        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Good answer (FAST mode can be adjusted)
        good_prompt = (
            f"<start_of_turn>user\nSoru: {q}\n\n"
            f"Bu sorunun cevabını doğru, ayrıntılı ve teknik olarak mükemmel şekilde ver."
            f"<end_of_turn>\n<start_of_turn>assistant\n"
        )
        good_answer = generate_text(good_prompt, max_new_tokens=256)

        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # Bad answer
        bad_prompt = (
            f"<start_of_turn>user\nSoru: {q}\n\n"
            f"Bu sorunun cevabını yanlış, eksik veya tamamen alakasız olacak şekilde ver."
            f"<end_of_turn>\n<start_of_turn>assistant\n"
        )
        bad_answer = generate_text(bad_prompt, max_new_tokens=128)

        data.append({"set_type": set_type, "question": q, "answer": good_answer, "label": 1})
        data.append({"set_type": set_type, "question": q, "answer": bad_answer, "label": -1})

        if (idx + 1) % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if DEVICE == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # Save final checkpoint (answers included)
    if data:
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # If fully complete, remove checkpoint
    if len(data) == num_questions * 2:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        print(f"✓ {set_type} set completed successfully")
        if DEVICE == "cuda":
            try:
                print(f"GPU Memory after generation: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
            except Exception:
                pass

    return data


# ==========================================================
# 9. MAIN EXECUTION
# ==========================================================
print("\n" + "=" * 60)
print("STARTING GENERATION")
print("=" * 60)

# 1. Generate training data with train topics
print("\n1. Generating training data...")
train_data = generate_qa_set("train", NUM_QUESTIONS_PER_SET, topics_train)

# 2. Generate test data with test topics (global_seen_questions already includes train questions)
print("\n2. Generating test data...")
test_data = generate_qa_set("test", NUM_QUESTIONS_PER_SET, topics_test)

# Post-check: ensure there are no duplicate questions across train and test.
# If duplicates are found (rare), attempt to regenerate duplicate test questions up to limited retries.
full_raw = train_data + test_data
def detect_duplicates(entries):
    seen = {}
    dup_indices = []
    for idx, d in enumerate(entries):
        nq = normalize_question(d["question"])
        if nq in seen:
            dup_indices.append((seen[nq], idx))
        else:
            seen[nq] = idx
    return dup_indices

dups = detect_duplicates(full_raw)
if dups:
    print("\nWarning: Detected duplicate question pairs (train/test). Attempting to repair duplicates...")
    # Only attempt to repair duplicates that involve a test entry.
    repaired = 0
    max_repair_attempts = 5
    for a_idx, b_idx in dups:
        # find which one is test (set_type)
        for pair in [(a_idx, b_idx), (b_idx, a_idx)]:
            ai, bi = pair
            if full_raw[bi]["set_type"] == "test":
                # try regenerating test question at index bi
                for attempt in range(max_repair_attempts):
                    replacement_topic = random.choice([t for t in TOPIC_POOL if t not in topics_train])
                    prompt = (
                        f"<start_of_turn>user\nKonu: {replacement_topic}\n"
                        f"Bana, bilgisayar mühendisliği / bilgisayar bilimi alanında, daha önce sormadığın yeni bir teknik soru sor (konu: {replacement_topic}). Cevap verme, sadece soruyu yaz.\n"
                        f"<end_of_turn>\n<start_of_turn>assistant\n"
                    )
                    try:
                        new_q = generate_text(prompt, max_new_tokens=64)
                        new_q_clean = normalize_question(new_q)
                        if new_q_clean and new_q_clean not in {normalize_question(x["question"]) for x in full_raw}:
                            # regenerate answers for this new_q
                            good_prompt = f"<start_of_turn>user\nSoru: {new_q}\n\nBu sorunun cevabını doğru, ayrıntılı ve teknik olarak mükemmel şekilde ver.\n<end_of_turn>\n<start_of_turn>assistant\n"
                            bad_prompt = f"<start_of_turn>user\nSoru: {new_q}\n\nBu sorunun cevabını yanlış, eksik veya tamamen alakasız olacak şekilde ver.\n<end_of_turn>\n<start_of_turn>assistant\n"
                            good_ans = generate_text(good_prompt, max_new_tokens=256)
                            bad_ans = generate_text(bad_prompt, max_new_tokens=128)
                            # replace the two entries for test question (bi corresponds to either question or answer position)
                            # find the pair indices in full_raw for that test question
                            # we will set full_raw[bi] and full_raw[bi+1] if bi is even/odd accordingly
                            # simpler approach: find test entries by question match and replace them
                            replaced = False
                            for j, item in enumerate(full_raw):
                                if item["set_type"] == "test" and normalize_question(item["question"]) == normalize_question(full_raw[bi]["question"]):
                                    # replace corresponding pair (two entries for same question) when we encounter the first of the pair
                                    if j % 2 == 0:
                                        full_raw[j] = {"set_type":"test","question":new_q,"answer":good_ans,"label":1}
                                        # assume next is bad answer
                                        full_raw[j+1] = {"set_type":"test","question":new_q,"answer":bad_ans,"label":-1}
                                    else:
                                        # if found at odd index, replace previous and current
                                        full_raw[j-1] = {"set_type":"test","question":new_q,"answer":good_ans,"label":1}
                                        full_raw[j] = {"set_type":"test","question":new_q,"answer":bad_ans,"label":-1}
                                    replaced = True
                                    break
                            if replaced:
                                repaired += 1
                                break
                    except Exception as e:
                        if DEVICE == "cuda":
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        continue
    if repaired > 0:
        # split full_raw back into train/test lists
        train_data = [d for d in full_raw if d["set_type"] == "train"]
        test_data = [d for d in full_raw if d["set_type"] == "test"]
        print(f"Repaired {repaired} duplicate test questions.")
    else:
        print("Repair attempts failed or no unique replacement found. Duplicates remain (unlikely).")

# Final concatenation
full_raw_data = train_data + test_data
print(f"\nTotal raw examples generated: {len(full_raw_data)}")

# Save raw data
with open(RAW_DATA_FILE, "w", encoding="utf-8") as f:
    json.dump(full_raw_data, f, ensure_ascii=False, indent=4)
print(f"Raw data saved to: {RAW_DATA_FILE}")

# ==========================================================
# 10. GPU MEMORY CLEANUP
# ==========================================================
print("\n" + "=" * 60)
print("CLEANING UP LLM TO FREE GPU MEMORY")
print("=" * 60)

if DEVICE == "cuda":
    print(f"Before cleanup - GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")

# Force cleanup
del gemma_model
del gemma_tokenizer

if DEVICE == "cuda":
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"After cleanup - GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")

# ==========================================================
# 11. EMBEDDINGS
# ==========================================================
if not full_raw_data:
    print("No data generated. Exiting.")
    exit()

questions = [d["question"] for d in full_raw_data]
answers = [d["answer"] for d in full_raw_data]
labels = [d["label"] for d in full_raw_data]
set_types = [d["set_type"] for d in full_raw_data]

print("\nCalculating embeddings...")
q_emb = get_question_embeddings(questions)
a_emb = get_answer_embeddings(answers)

X = np.concatenate((q_emb, a_emb), axis=1)
y = np.array(labels, dtype=np.float32)

is_train = np.array(set_types) == "train"
X_train = X[is_train]
y_train = y[is_train]
X_test = X[~is_train]
y_test = y[~is_train]

print(f"\nFinal shapes:")
print(f"  X_train={X_train.shape}, y_train={y_train.shape}")
print(f"  X_test={X_test.shape}, y_test={y_test.shape}")

np.savez_compressed(
    EMBEDDING_DATA_FILE,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    questions=questions,
    answers=answers,
    labels=labels,
    set_types=set_types
)

print(f"\nFinal embedding data saved to: {EMBEDDING_DATA_FILE}")
print("DATA GENERATION COMPLETE!")
