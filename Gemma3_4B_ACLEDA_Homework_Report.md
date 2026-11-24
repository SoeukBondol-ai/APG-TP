# Homework Report — Fine-tuning Gemma 3 (4B) with LoRA / PEFT

**Student:** Soeuk Bondol
**Date:** 24 November 2026
**Model:** `unsloth/gemma-3-4b-it` (Gemma 3, 4B parameters, instruction-tuned, 4-bit)
**Notebook:** `Gemma3_4B_ACLEDA_Finetune.ipynb` (44 cells)
**New dataset:** `dataasilida_cleaned.csv` → repaired as `dataasilida_fixed.csv`
**Old dataset (baseline):** `data_kh.csv` (original notebook)

## 1. Objective 1 — Course summary

### 1.1 The fine-tuning workflow

```
Base model (Gemma 3 4B-it)
  → Dataset preparation (question/answer pairs → chat text)
  → LoRA adapters added (base weights frozen)
  → Supervised fine-tuning (SFT): predict the answer tokens
  → Evaluation: training loss + validation loss, before/after answers
  → Save LoRA adapter → push to Hugging Face
```

Fine-tuning does **not** teach the model a language or how to chat — Gemma 3 already does that. It teaches a model that already chats to answer **one kind of question in one style with one set of facts**: here, ACLEDA Bank questions in Khmer, in ACLEDA's wording.

### 1.2 Gemma 3

Google's open 4B-parameter instruction-tuned model (`-it`). It is multimodal (text + images); this homework uses **text only** (`finetune_vision_layers=False`). It expects the chat format:

```
<bos><start_of_turn>user
QUESTION<end_of_turn>
<start_of_turn>model
ANSWER<end_of_turn>
```

That is why the dataset must be rendered in exactly this format — the model only learns what it sees in training.

### 1.3 LoRA / PEFT

Full fine-tuning would update all ~4,000,000,000 weights — too much memory for a Colab T4. PEFT (Parameter-Efficient Fine-Tuning) freezes the base model and trains only LoRA adapters: for a weight matrix `W`, it learns a low-rank update `W + B·A`, where `A` and `B` are small matrices. Rank `r` is the size of that bottleneck.

| Run | LoRA rank `r` | Trainable parameters |
|---|---|---|
| Old run | 8 | **14,901,248 (0.37% of the model)** — printed by the trainer |
| New run | 16 | roughly double that, still **under 1%** |

Bigger `r` = more capacity, but more overfitting risk on a small dataset. Smaller `r` = safer, but it may underfit long answers.

### 1.4 Dataset preparation

1. **Load** the CSV (`input` = question, `output` = answer).
2. **Repair it.** The supplied file is broken in two ways: its header is the single quoted field `"input,output"` (not two columns), and 7 rows are a single field holding `question ,answer` (6 rows) or `question \n answer` (1 row). All 7 were repaired, 0 rows dropped; 2 exact duplicate pairs removed.
3. **Split by question, not by row** (90/10). 16 questions have several *different* answers (e.g. the generic loan-size question, whose answer depends on the loan type). Splitting by row would put the same question in train and validation and make validation loss look better than it really is.
4. **Format** each pair with the Gemma-3 chat template (see §1.2).
5. **Check length:** the longest formatted example is ~734 tokens — nothing is cut off at `max_seq_length = 2048`.

Real example from the new dataset:

- **Q:** តើប័ណ្ណឥណពន្ធចេស៊ីប៊ី អេស៊ីលីដា ធម្មតា ជាអ្វី
- **A:** ប័ណ្ណឥណពន្ធចេស៊ីប៊ី អេស៊ីលីដា ធម្មតា គឺជាប័ណ្ណ ឬកាតអេឡិចត្រូនិក បោះផ្សាយដោយ ធនាគារ អេស៊ីលីដា ភីអិលស៊ី …

## 2. Objective 2 — Notebook configuration

Every hyperparameter lives in one `CONFIG` dict at the top of the notebook. Change it there, re-run from §2.

| Hyperparameter | New run | Old run | What it does | Too high | Too low |
|---|---|---|---|---|---|
| `learning_rate` | 2e-4 | 2e-4 | Step size of each weight update | Loss jumps / diverges | Learns too slowly, underfits |
| `num_train_epochs` | 3 | — (60 steps only) | Full passes over the train set | Memorises train set → overfitting | Underfits, answers stay generic |
| `max_steps` | -1 (= use epochs) | 60 | Hard cap on steps | — | Stops before learning |
| `per_device_train_batch_size` | 2 | 2 | Rows per GPU step | Out of memory on a T4 | Noisy updates |
| `gradient_accumulation_steps` | 4 | 4 | Steps averaged per update; effective batch = **8** | Slower per update | Noisy, unstable loss |
| `lora_r` / `lora_alpha` | 16 / 16 | 8 / 8 | LoRA bottleneck / scaling | Overfits 519 rows | Cannot learn long answers |
| `weight_decay` | 0.01 | 0.01 | L2 penalty on weights | Underfits | Overfits |
| `warmup_steps` | 5 | 5 | Steps with LR rising from 0 | — | Early loss spike |
| `lr_scheduler_type` | linear | linear | LR decays linearly to 0 | — | — |
| `max_seq_length` | 2048 | 2048 | Texts longer than this are truncated (passed to `SFTConfig` as `max_length`) | Wastes memory | Long answers cut off |
| `eval_steps` | 20 | — (no eval) | Validation loss every 20 steps | Slower training | You miss the overfitting point |
| `optim` | adamw_8bit | adamw_8bit | 8-bit AdamW, saves memory | — | — |

**What changed vs the old run, and why:** epochs 3 instead of a 60-step cap (the old run saw only 16.4% of its data); LoRA `r` 8 → 16 (long Khmer answers need a little more capacity); a validation split and `eval_steps = 20` added (the old run had none). Everything else stayed the same on purpose, so the comparison in §4 measures the **dataset**, not a different setup.

**Note on the trainer API (fixed 24 Nov 2026):** the first version of this notebook used the March-2025 pattern (`SFTTrainer(..., tokenizer=..., dataset_text_field=..., args=TrainingArguments(...))`). That pattern raises `TypeError` on a fresh install — checked against the actual TRL 1.15.0 wheel, `SFTTrainer` has no `tokenizer` argument; the tokenizer is passed as `processing_class=`, and all data options live in `SFTConfig` (`max_length`, `dataset_text_field`, `packing`, `dataset_num_proc`). The install cell now only runs `pip install unsloth` (a separate `pip install trl` would fetch 1.18.2, above Unsloth's ≤1.15.0 cap).

## 3. Objective 3 — Loss analysis

### 3.1 How to read the two curves

- **Both fall together** → the model is learning *and* generalising. Good.
- **Train falls, validation flattens or rises** → **overfitting**: the model memorises the train rows instead of learning the facts. The best checkpoint is the one at the lowest validation loss (the notebook saves it automatically). Fix: fewer epochs, lower LoRA `r`, higher weight decay, more data.
- **Both stay high / flat** → **underfitting**: the model has not learned the task. Fix: more epochs, higher learning rate, higher LoRA `r`.
- **Both fall, but validation stays far above training** → some gap is normal (validation questions are unseen), but a large, growing gap is the early sign of overfitting.
- **Loss jumps up suddenly** → learning rate too high.

### 3.2 Worked analysis: the old run's real training loss (60 steps)

The old run logged all 60 training losses (figure: `old_run_loss_curve.png`). Verified numbers:

| Measure | Value |
|---|---|
| First loss (step 1) | **5.3791** |
| Last loss (step 60) | **0.5751** |
| Lowest loss | **0.449 at step 58** |
| Drop | 5.3791 → 0.5751 = **−4.80 (−89.3%)** |
| Mean, steps 1–10 | 4.593 |
| Mean, steps 51–60 | **0.508** |
| First step below 1.0 | **step 23** (0.9541) |
| Data actually seen | 60 steps × batch 8 = 480 of 2,930 rows = **16.4%** |

Reading: the model learned fast — loss fell below 1.0 by step 23 and flattened around 0.5 for the last 10 steps, which is the shape of a task being learned, not of a diverging run. The small upticks at the end (0.449 → 0.575) are normal batch noise at this scale, not a signal on their own. **But the honest limit is bigger than any number here: there was no validation loss at all** (`eval_dataset = None`). A falling training loss alone cannot tell overfitting from learning — the model could simply be memorising the 480 examples it saw. That is exactly the weakness the new run fixes.

### 3.3 What to expect from the new run — and what to write down after running it

- Training starts high (the base model has never seen this Q&A style) and should fall below ~1 within the first epoch.
- Validation loss is logged at steps 20, 40, 60, … (about 8–9 points over 174 steps) and plotted in the notebook as `loss_curve.png`.
- **Fill in after your run:**

| Measure | Your value |
|---|---|
| First train loss → last train loss | ____ → ____ |
| First val loss → best val loss → last val loss | ____ → ____ → ____ |
| Best val loss at step | ____ |
| Final gap (val − train) | ____ |
| Reading (learning well / overfitting / underfitting) + evidence | ____ |

## 4. Objective 4 — Practical experiment and comparison

### 4.1 Method

Same 3 held-out validation questions, asked twice with greedy decoding (`do_sample=False`, so the same question always gives the same answer): once to the **base** model before training (§3 of the notebook), once to the **fine-tuned** model after training (§6), next to the **gold** answer from the dataset. Two extra unseen questions (new wordings, not in the dataset) test generalisation. The notebook does this automatically; the comparison table below is the part you fill in from its output.

### 4.2 Comparison: old dataset vs new dataset

| | Old run (robot commands) | New run (ACLEDA Q&A) |
|---|---|---|
| Dataset | `data_kh.csv` | `dataasilida_cleaned.csv` → `dataasilida_fixed.csv` |
| Task | Khmer command → JSON action, e.g. `{'action':'move','params':{'distance':10,...}}` | Khmer question → Khmer answer about ACLEDA products |
| Rows | 2,930 | **519** (521 repaired − 2 exact duplicates) |
| Train / val | No split; `eval_dataset = None` | **470 train / 49 val**, grouped by question |
| Output length | Short JSON (tens of chars) | Median **231 chars**, p90 684, max 2,826 |
| LoRA rank | 8 (14.9M params, 0.37%) | 16 (under 1%) |
| Steps | 60 (16.4% of data seen) | **174** (3 full epochs) |
| Training loss | **5.3791 → 0.5751** | ____ → ____ (fill in) |
| Validation loss | **None logged** | ____ → ____ (fill in) |
| Overfitting? | **Could not tell** | Your reading from §3.3 |
| Training time / peak memory | ~8 min, 6.31 GB (Tesla T4) | ____ min, ____ GB (fill in) |
| Inference test | Only after training | Before **and** after, same 3 questions |

### 4.3 What the comparison shows (before the new run)

- The old run proves the pipeline works: a 4B model can learn a narrow output format in 60 steps (loss −89%). But it proves nothing about generalisation — no validation split, and only 16% of the data was seen.
- The new task is harder in the ways that matter: 5.6× less data (519 vs 2,930 rows), answers ~10× longer and in free Khmer instead of fixed JSON, and facts that must be right (a wrong fee or limit is worse than a wrong JSON key). That is why the new run uses 3 full epochs, a higher LoRA rank, and — most importantly — a validation loss: on this dataset, the validation curve is the result, and the training curve is only the check that learning happened.
- **Expected outcome:** the tuned model should answer the 3 test questions in ACLEDA's wording with the right facts, where the base model answers generically or in the wrong style. On the 2 unseen questions it will be weaker — that is normal for 519 pairs and should be reported, not hidden.

### 4.4 Status — what is done and what still needs your run

**Done:** notebook (44 cells, current TRL/SFTConfig API), repaired dataset (519 rows), notes, this report, and the old run's loss analysis above with verified numbers and figure.

**Not done — needs a GPU, which this machine does not have (no `nvidia-smi`, no `torch` here):** the new training run itself. Run the notebook on Colab (T4 GPU, ~10–20 min for training), then complete §3.3 and §4.2 above from its output:

1. Copy the first/last train loss and val loss from the §5 plot, and save `loss_curve.png`.
2. Copy the §6 BASE vs TUNED vs GOLD answers for the 3 test questions.
3. Push the LoRA adapter to Hugging Face (§7: accept the Gemma licence, `notebook_login()`, set `push_to_hub = True`) and paste the repo link here: ____

## 5. Conclusion

Fine-tuning did not teach Gemma 3 Khmer or how to chat — it taught a model that already chats to answer ACLEDA questions in ACLEDA's wording. The old run showed the method works on an easy, structured task; the new run tests it on a harder, real task with far less data and much longer answers. The single most important difference between the two runs is not the dataset or the hyperparameters — it is that the new run has a validation loss, so its result can actually be judged.

## References
- model : https://huggingface.co/mondolreal/gemma-3-4b-it-aclede-qa-lora
- Unsloth, Gemma 3 blog and notebooks: https://unsloth.ai/blog/gemma3 ; https://docs.unsloth.ai/get-started/unsloth-notebooks
- Hugging Face TRL, SFT Trainer docs: https://huggingface.co/docs/trl/sft_trainer
- Gemma 3 4B-IT model page (licence acceptance required): https://huggingface.co/google/gemma-3-4b-it
- Unsloth models: https://huggingface.co/unsloth
