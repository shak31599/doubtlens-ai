# Error analysis

Misclassifications come directly from held-out synthetic predictions in `reports/misclassifications.csv`. The baseline misclassified one of 120 synthetic test questions; the character model misclassified none in this one split. Lack of observed test errors does **not** prove absence of failure modes. Synthetic template overlap, class-specific keywords, short inputs, ambiguous cross-domain questions, noisy real-world labels and unfamiliar acronyms are known risks that need prospective evaluation on externally sourced, human-labeled examples.

Test errors were not used to adjust the packaged model. Future work: create a genuinely independent manually labeled benchmark, group near-duplicates before splitting, add calibration testing, and measure retrieval top-k relevance with human judgments.
