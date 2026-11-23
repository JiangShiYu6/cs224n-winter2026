# Final Project — Completed (baselines)

These runs use GPT-2 Small for the Stanford CS224N default project.

| Task | Baseline | Validation result |
| --- | --- | --- |
| SST sentiment | Frozen GPT-2, 10 epochs | 46.3% accuracy |
| CFIMDB sentiment | Frozen GPT-2, 10 epochs | 85.7% accuracy |
| Quora paraphrase detection | 8,192 training pairs, 1 epoch | 75.1% accuracy on 40,429 pairs |
| Shakespeare sonnet generation | Full fine-tuning, 10 epochs | 42.02 chrF on 12 poems |

The Quora run produced 80,858 test predictions. The sonnet run continued each three-line prompt, with 12 poems per split. The chrF score compares full poems, including the supplied opening lines. The [validation poems](results/sonnet/gpt2-10epochs/dev.txt), [test poems](results/sonnet/gpt2-10epochs/test.txt), and [metrics](results/sonnet/gpt2-10epochs/metrics.json) are included.

See the [starter README](starter/README.md) for setup and task entry points. The sentiment and Quora baseline runners use a local GPT-2 checkpoint under `checkpoints/gpt2` and course data under `starter/data`. Model weights, datasets, and the full classification prediction CSVs are stored locally.
