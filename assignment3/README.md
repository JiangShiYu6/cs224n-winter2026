# Assignment 3: Self-Attention and Transformers

The [written solutions](assignment3.pdf) cover attention, positional embeddings, and the training experiments. The LaTeX source is in `latex/`. Run `xelatex main.tex` twice from that directory to rebuild the PDF.

The PyTorch implementation is in `model_solution.py`. From this directory, install with `python -m pip install -r requirements.txt`, then run `python -m pytest` from `tests/`. All six course tests passed.

Run `python train.py` for the 100-step baseline. The first run uses the GPT-2 tokenizer and the first 1% of the TinyStories training split. The script writes `metrics.csv` and `losses_and_grad_norms.png` in the working directory. Save these outputs before starting another experiment.

| Experiment | Command | Training loss at step 100 |
| --- | --- | ---: |
| Baseline | `python train.py` | 10.7784 |
| Increase learning rate | `python train.py --experiment higher_lr` | 4.9214 |
| Also lower AdamW beta2 | `python train.py --experiment optimizer_betas` | 4.9105 |
| Also increase model width | `python train.py --experiment wider_model` | 4.8148 |

All runs used seed 42, identical batches, and 100 optimizer updates. The CSV files are in `results/`. These are training losses from one seed.

Starter materials: [Stanford CS224N Winter 2026](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1264/).
