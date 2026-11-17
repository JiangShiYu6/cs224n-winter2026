# CS224N Winter 2026

Personal self-study work for Stanford CS224N: Natural Language Processing with Deep Learning.

## Assignment 1: Exploring Word Vectors

The completed notebook is in [assignment1/exploring_word_vectors.ipynb](assignment1/exploring_word_vectors.ipynb). It covers co-occurrence matrices, truncated SVD, GloVe embeddings, cosine similarity, analogies, and bias analysis.

Create an environment with Python 3.12 and install the tested dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r assignment1/requirements.txt
.\.venv\Scripts\python.exe -m notebook assignment1/exploring_word_vectors.ipynb
```

In VS Code, select the repository's `.venv` Python interpreter as the notebook kernel. The official `env.yml` is included as an alternative Conda specification. `requirements-lock.txt` records the full tested Windows environment.

IMDb and GloVe are downloaded separately; datasets, model caches, and virtual environments are excluded from this repository. A compatible GloVe cache at `.cache/gensim/glove-wiki-gigaword-200/` is reused when available.

## Course source and submission

Original assignment materials are provided by the [Stanford CS224N Winter 2026 course](https://web.stanford.edu/class/cs224n/). This private repository is for personal study and version control. Official enrolled-student submission is through Gradescope; Assignment 1 requests a PDF exported from the completed notebook.
