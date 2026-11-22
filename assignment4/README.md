# Assignment 4: LLM Evaluation

The [write-up](assignment4.pdf) includes the evaluation methods, figures, and human review. This self-study version uses GPT-5.4 in place of Gemini.

| Exercise | Result |
| --- | --- |
| Q2: GSM8K | A standard: 75/100; B standard: 48/100; A with a worked-solution prompt: 98/100. |
| Q3: LLM as judge | Z preferred E in 24/30 pairs and F in 6/30. Human and judge choices agreed on 1 of 5 reviewed pairs. |
| Q4: Defensive substitute | A local output filter blocked a synthetic marker without recording it in the audit log. All six checks passed. This replaces the course password extraction challenge; G, H, and I were not evaluated. |

Install with `python -m pip install -r requirements.txt`. For Q2 and Q3, copy `.env.template` to `.env` and set your own `OPENAI_API_KEY` locally.

```sh
python gsm8k.py both
python llm_judge.py --workers 4
python check_output_guard.py
```

Q2 makes 300 API requests for a full run; use `--limit 1` for a smoke test. Q3 reuses saved judgments by default; pass `--output-dir` with a new directory for a fresh run. Q4 runs offline. The [LaTeX source](latex/) can be compiled with XeLaTeX.
