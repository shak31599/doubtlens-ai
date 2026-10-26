# Model card

Baseline: scikit-learn word n-gram (1,2) TF-IDF followed by class-weighted Logistic Regression. Challenger: character `char_wb` n-grams (3,5) TF-IDF followed by class-weighted Logistic Regression. Train only on training split; validation determines the selected model. Do not reselect on held-out test. Probabilities are softmax outputs of a linear classifier, not externally calibrated. Uncertainty flag is a heuristic (<0.55 top score), not an independently validated reject option.

**Measured synthetic held-out scores:** baseline accuracy 0.9917 / macro F1 0.9917; character accuracy 1.0000 / macro F1 1.0000. Full metrics, latency and confusion counts appear in `artifacts/metrics.json`. The character model was selected from validation. Shared synthetic templates make reported metrics unrepresentative of performance on organically written questions.

**Optional experiment:** `src/doubtlens/embedding.py` uses `sentence-transformers/all-MiniLM-L6-v2` and logistic regression. The pretrained model's separate license/terms apply. Optional experiment not executed in the delivered bundle, and the default app does not claim embeddings are active.
