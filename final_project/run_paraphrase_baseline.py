"""Run a reproducible GPT-2 Small QQP baseline using local course data."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "starter"
CHECKPOINT = ROOT / "checkpoints" / "gpt2"
sys.path.insert(0, str(STARTER))

import evaluation
import datasets
import paraphrase_detection as task
from transformers import GPT2TokenizerFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=1)
    opts = parser.parse_args()
    os.chdir(STARTER)
    task.seed_everything(11711)
    original_model = task.GPT2Model.from_pretrained
    task.GPT2Model.from_pretrained = lambda model, **kwargs: original_model(str(CHECKPOINT), **kwargs)

    class LocalTokenizer:
        @staticmethod
        def from_pretrained(_):
            return GPT2TokenizerFast.from_pretrained(str(CHECKPOINT))

    datasets.GPT2Tokenizer = LocalTokenizer
    load = task.load_paraphrase_data

    def load_data(path, split="train"):
        data = load(path, split=split)
        if path == "data/quora-train.csv" and opts.train_size:
            data = random.Random(11711).sample(data, opts.train_size)
            print(f"Training subset: {len(data)} pairs, seed 11711", flush=True)
        return data

    task.load_paraphrase_data = load_data
    task.TQDM_DISABLE = True
    evaluation.TQDM_DISABLE = True
    out = ROOT / "results" / "paraphrase" / f"{opts.train_size or 'full'}-train-{opts.epochs}epoch"
    out.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        para_train="data/quora-train.csv", para_dev="data/quora-dev.csv",
        para_test="data/quora-test-student.csv", para_dev_out=str(out / "dev.csv"),
        para_test_out=str(out / "test.csv"), filepath=str(out / "classifier.pt"),
        epochs=opts.epochs, batch_size=16, lr=1e-5, use_gpu=True, model_size="gpt2",
    )
    (out / "settings.json").write_text(json.dumps(dict(vars(args), train_size=opts.train_size, seed=11711),
                                                    indent=2), encoding="utf-8")
    task.train(args)
    task.test(args)


if __name__ == "__main__":
    main()
