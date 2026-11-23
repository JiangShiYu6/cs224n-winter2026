"""Run the course's frozen-encoder sentiment baseline with a local checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "starter"
CHECKPOINT = ROOT / "checkpoints" / "gpt2"
sys.path.insert(0, str(STARTER))

import torch
from transformers import GPT2TokenizerFast
import classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA.")
    os.chdir(STARTER)
    classifier.TQDM_DISABLE = True
    classifier.seed_everything(11711)

    # Route the starter's GPT-2 loader and tokenizer to the verified local files.
    original_load = classifier.GPT2Model.from_pretrained
    classifier.GPT2Model.from_pretrained = lambda: original_load(str(CHECKPOINT))
    classifier.GPT2Tokenizer = GPT2TokenizerFast
    for dataset_type in (classifier.SentimentDataset, classifier.SentimentTestDataset):
        def dataset_init(self, dataset, args):
            self.dataset = dataset
            self.p = args
            self.tokenizer = GPT2TokenizerFast.from_pretrained(str(CHECKPOINT))
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dataset_type.__init__ = dataset_init

    # The course evaluation functions only need inference.
    classifier.model_eval = torch.no_grad()(classifier.model_eval)
    classifier.model_test_eval = torch.no_grad()(classifier.model_test_eval)
    for dataset, batch_size in (("sst", 16), ("cfimdb", 8)):
        out = ROOT / "results" / "sentiment" / (dataset + f"-{args.epochs}epoch")
        out.mkdir(parents=True, exist_ok=True)
        config = SimpleNamespace(
            filepath=str(out / "classifier.pt"), lr=1e-3, use_gpu=True,
            epochs=args.epochs, batch_size=batch_size, hidden_dropout_prob=0.3,
            train=f"data/ids-{dataset}-train.csv", dev=f"data/ids-{dataset}-dev.csv",
            test=f"data/ids-{dataset}-test-student.csv", fine_tune_mode="last-linear-layer",
            dev_out=str(out / "dev.csv"), test_out=str(out / "test.csv"),
        )
        (out / "settings.json").write_text(json.dumps(vars(config), indent=2), encoding="utf-8")
        print(f"Training {dataset} on {torch.cuda.get_device_name()}", flush=True)
        classifier.train(config)
        classifier.test(config)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
