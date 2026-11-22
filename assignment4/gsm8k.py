import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query
from client.query import query_model

INVALID_ANS = "[invalid]"
DATA_PATH = Path(__file__).resolve().parent / "data" / "gsm8k_first_100.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "gsm8k"
if os.getenv("A4_PROVIDER", "").strip().lower() == "openai":
    from client.providers import OpenAIProvider

    configured_model = os.getenv("A4_OPENAI_MODEL") or OpenAIProvider.DEFAULT_MODEL
    if configured_model != "gpt-5.4-nano":
        RESULTS_DIR = RESULTS_DIR / re.sub(r"[^a-zA-Z0-9_-]", "_", configured_model)

def standard_prompt_template(question: str) -> str:
    """
    Converts a gsm8k question into a standard model input

    Args:
        question: gsm8k question.
    Returns:
        prompt for a model to answer input question.
    """

    prompt = f"""Output a numerical answer to the following problem with two or fewer steps of reasoning. Output your numerical
answer as the only line of your output in the format "#### <numerical_answer>."

Problem: {question}
""".strip()

    return prompt

def standard_output_extractor(model_generation: str) -> str:
    """
    Extracts the string answer from a model generation, assuming it was prompted 
    using a prompt from `standard_prompt_template`.

    Args:
        model_generation: the string generation from the model
    Returns:
        String representing the numerical output of the model for the question, or "[invalid]" if
            no output can be extracted.
    """

    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    match = ANS_RE.search(model_generation)

    if match:
        match_str: str = match.group(1).strip()
        match_str = match_str.replace(",", "").removesuffix(".")
        return match_str
    else:
        return INVALID_ANS



def evaluate(model_id, prompt_template, run_name, limit=None):
    with DATA_PATH.open(encoding="utf-8") as data_file:
        examples = [json.loads(line) for line in data_file if line.strip()]
    if limit is not None and not 1 <= limit <= len(examples):
        raise ValueError(f"limit must be between 1 and {len(examples)}")

    selected = examples[:limit]

    def score_example(example):
        prompt = prompt_template(example["question"])
        response = query_model(model_id=model_id, query=Query(turns=[{"user": prompt}]))
        prediction = standard_output_extractor(response.text)
        expected = str(example["numerical_answer"])
        return {
            "question": example["question"],
            "prediction": prediction,
            "expected": expected,
            "correct": prediction == expected,
            "response": response.text,
            "cost": response.cost,
        }

    workers = int(os.getenv("A4_GSM8K_WORKERS", "1"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(tqdm(pool.map(score_example, selected), total=len(selected), desc=run_name))

    correct = sum(row["correct"] for row in rows)
    invalid = sum(row["prediction"] == INVALID_ANS for row in rows)
    result = {
        "model_id": model_id,
        "configured_model": os.getenv("A4_OPENAI_MODEL"),
        "estimated_cost": round(sum(row["cost"] for row in rows), 8),
        "total": len(rows),
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / len(rows),
        "examples": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{run_name}_{len(rows)}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{run_name}: {correct}/{len(rows)} ({result['accuracy']:.1%}); {invalid} invalid")
    return result


def eval_model_on_gsm8k(limit=None):
    """Compare A and B using the supplied standard prompt."""
    for model_id in ("A", "B"):
        evaluate(model_id, standard_prompt_template, f"{model_id}_standard", limit)


def superior_prompt_template(question: str) -> str:
    return f"""Solve the word problem. Write a short worked solution with one equation
per line. Track each quantity, convert units as needed, and check that the final
number answers the question. Put only the final number after the four hashes.
If the answer is a whole number, write it without a decimal point.

Example problem: A shop had 80 notebooks. It sold 12 in the morning and twice
as many in the afternoon. How many notebooks are left?
Afternoon sales: 2 × 12 = 24.
Remaining stock: 80 - 12 - 24 = 44.
#### 44

Problem: {question}"""


def eval_model_on_gsm8k_with_improved_prompt(limit=None):
    """Evaluate the candidate prompt and plot it against A's standard score."""
    improved = evaluate("A", superior_prompt_template, "A_worked_solution", limit)
    total = improved["total"]
    baseline_path = RESULTS_DIR / f"A_standard_{total}.json"
    if not baseline_path.exists():
        print("Run the standard evaluation with the same sample count to create the comparison chart.")
        return

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5, 4))
    scores = [100 * baseline["accuracy"], 100 * improved["accuracy"]]
    bars = ax.bar(["Standard", "Improved"], scores, color=["#35618a", "#4f8971"])
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 1, f"{score:.1f}%", ha="center")
    ax.set(ylim=(0, 108), ylabel="Accuracy (%)", title=f"Model A on GSM8K ({total} questions)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"A_worked_solution_comparison_{total}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["standard", "improved", "both"])
    parser.add_argument("--limit", type=int, help="Only evaluate the first N questions.")
    args = parser.parse_args()
    if args.mode in ("standard", "both"):
        eval_model_on_gsm8k(args.limit)
    if args.mode in ("improved", "both"):
        eval_model_on_gsm8k_with_improved_prompt(args.limit)
