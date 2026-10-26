# Methodology

1. Run deterministic local synthetic generator; preserve original categorical terms and programming tokens.
2. Validate schema, trim repeated whitespace, remove exact case-insensitive duplicates and invalid labels.
3. Stratify 60% train, 20% validation, 20% untouched test. Fixed random seed 42. No exact text duplicates across splits; near-duplicate **templates remain** a known leakage/generalization concern.
4. Train word and character TF-IDF vectorizers on train only. Fit class-weighted multinomial classifiers on training labels.
5. Select model by validation macro F1. Evaluate each trained candidate exactly once on held-out test.
6. Build retrieval index on train-only questions; cosine retrieval scores are similarity, not confidence or semantic entailment.
7. Export actual classification metrics, per-class scores, confusion matrices, error list and comparison figure.

The default deployment intentionally favors compact zero-download offline dependencies. It is not a transformer-backed semantic search deployment. See separate opt-in embedding experiment.
