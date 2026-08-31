# Training / Research Artifacts

Kept for reproducibility, separate from the deployed inference application.

- `notebooks/preprocessing.ipynb` — dataset preprocessing (im2latex-100k formulas +
  images → normalized/tokenized training data, vocab construction).
- `notebooks/training.ipynb` — model training loop for the encoder/seq_model/decoder
  architecture defined in `app/deep_learning/model.py`.
- `notebooks/training_all_metrics.ipynb` — extended training run with additional
  metric tracking.
- `notebooks/testing.ipynb` — evaluation of a trained checkpoint.
- `artifacts/` — logs produced during dataset preparation (missing/empty image
  lists, token/encoding error summaries) — useful context if you retrain or debug
  dataset issues, not required to run the deployed app.

None of this is required to run the Flask application in `app/` — it only needs
`models/full_checkpoint.pt` and `models/vocab.json`. The raw dataset these notebooks
consume (`dataset/`, `sample/`) is not committed to this repository; see the root
`README.md` for where to obtain im2latex-100k if you want to retrain.
