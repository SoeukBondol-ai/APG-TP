# APG-TP
This Project really focus on the TP with bunch of file resource , assignment and others 

## LLM Performance Evaluation — Course Summary

Advanced Programming for Data Science | Institute of Technology of Cambodia

Evaluation metrics compare model predictions with trusted references or ground-truth labels. Choose the metric according to the task.

| Group | Metric | What it measures | Main application | Better score |
|---|---|---|---|---|
| 1 | Exact Match (EM) | Exact answer equality | Short-answer QA | Higher |
| 3 | Precision, Recall, F1 | Correctness and coverage of positive predictions | Classification | Higher |
| 4 | BLEU | Consecutive n-gram precision | Machine translation | Higher |
| 5 | ROUGE | Reference text coverage and overlap | Summarization | Higher |
| 6 | METEOR | Word, stem, synonym and order matching | Translation and text generation | Higher |
| 7 | BERTScore | Contextual token similarity | Paraphrasing and open-ended answers | Higher |
| 8 | WER | Word-level edit errors | Speech recognition | Lower |
| 9 | CER | Character-level edit errors | OCR and spelling correction | Lower |

### 1. Exact Match (EM) — Group 1

- Formula: $EM = \frac{\text{number of exact matches}}{N}$.
- Each answer scores 1 if it matches the reference and 0 otherwise. Normalized EM may ignore case, punctuation, articles and extra whitespace.
- Example: four exact matches out of five answers give EM = 80%.
- Limitation: correct answers with different wording can receive zero.

### 2. Precision, Recall and F1-score — Group 3

- Precision: $P = \frac{TP}{TP+FP}$ — how many predicted positives are correct?
- Recall: $R = \frac{TP}{TP+FN}$ — how many actual positives are found?
- F1-score: $F_1 = \frac{2PR}{P+R}$ — harmonic mean of precision and recall.
- TP = true positive, FP = false positive, FN = false negative, TN = true negative.
- Example: TP = 8, FP = 2, FN = 12 gives P = 80%, R = 40%, F1 = 53.3%.
- Limitation: F1 does not directly include TN, and these metrics do not evaluate language quality.

### 3. BLEU (Bilingual Evaluation Understudy) — Group 4

- Formula: $BLEU = BP \times \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$.
- Uses clipped n-gram precision ($p_n$) and a brevity penalty (BP) for short outputs; standard BLEU commonly uses 1–4 grams.
- Example: the presentation's simplified BLEU-2 example scores 0.709 (70.9/100).
- Limitation: penalizes valid paraphrases and does not directly measure meaning or factual correctness.

### 4. ROUGE (Recall-Oriented Understudy for Gisting Evaluation) — Group 5

- Recall = matched units / reference units; precision = matched units / candidate units. F1 balances both.
- ROUGE-N: consecutive n-grams (ROUGE-1 = words, ROUGE-2 = word pairs).
- ROUGE-L: longest common subsequence; ROUGE-Lsum: summary-level union LCS.
- ROUGE-W: weighted LCS; ROUGE-S: skip-bigrams; ROUGE-SU: skip-bigrams plus unigrams.
- Example: reference "the weather is cold outside", candidate "the weather is cold": ROUGE-1 recall = 4/5 = 0.8.
- Limitation: word overlap cannot guarantee meaning, factuality or readability.

### 5. METEOR — Group 6

- Matches exact words, stems and synonyms, then penalizes fragmented word order.
- Let $m$ be matched words and $c$ be matching chunks. The presentation uses $F = \frac{10PR}{R+9P}$, penalty $= 0.5(c/m)^3$, and $METEOR = F(1-\text{penalty})$.
- Example: "the cat sleeps" compared with itself scores 0.9815; reversing its words gives 0.5000 in the demonstration.
- Limitation: depends on language resources and references; cannot verify facts.

### 6. BERTScore — Group 7

- Uses pretrained contextual embeddings (e.g. BERT or RoBERTa) and cosine similarity to find each token's best match.
- Precision averages best candidate-to-reference similarities; recall averages best reference-to-candidate similarities; F1 combines them.
- Example: "The boy is happy" and "The kid feels joyful" can score highly despite different words.
- Limitation: higher computation cost, dependence on the encoder, and possible failure to detect negation or factual errors.

### 7. Word Error Rate (WER) — Group 8

- Formula: $WER = \frac{S+D+I}{N}$, where S = substitutions, D = deletions, I = insertions, N = reference word count.
- Uses word-level Levenshtein distance. Zero is perfect; insertion errors can push WER above 100%.
- Example: one incorrect word in five reference words gives WER = 20%.
- Reported small class experiment: Whisper-base 9.4% vs. Whisper-tiny 18.8% (32 reference words).
- Limitation: counts word errors, not meaning.

### 8. Character Error Rate (CER) — Group 9

- Formula: $CER = \frac{S+D+I}{N}$, using character edits and reference character count instead of words.
- Example: "cat" vs. "cut" requires one substitution, giving CER = 33.3%.
- Useful for OCR, speech transcription and spelling correction; does not require word splitting.
- Limitation: sensitive to character units, spaces and punctuation; does not evaluate meaning. CER can exceed 100%.

### Key takeaway

Use EM for exact answers, precision/recall/F1 for classification, BLEU for translation, ROUGE for summaries, METEOR for flexible word matching, BERTScore for semantic similarity, WER for transcripts and CER for character-level accuracy. No single metric proves factual correctness or overall quality; combine appropriate metrics with human evaluation.

*Source: supplied class presentations from Groups 1 and 3–9. Group 2 was not included in the provided materials.*
