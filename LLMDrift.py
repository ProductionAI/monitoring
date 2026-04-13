"""
LLM Drift Detection Demo
========================
Demonstrates four practical drift detection techniques:

  1. Canary Prompt Monitoring   – semantic similarity of outputs over time
  2. Output Distribution Drift  – statistical tests on token/length distributions
  3. Embedding Space Drift      – MMD (Maximum Mean Discrepancy) on query embeddings
  4. LLM-as-Judge Scoring       – a separate model scores output quality over time

Each section is self-contained and runnable independently.

Requirements:
    pip install anthropic numpy scipy scikit-learn python-dotenv
"""

import os
import json
import time
import hashlib
import statistics
from datetime import datetime
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel
from dotenv import load_dotenv
import anthropic

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("CLAUDKEY") or os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key)
MODEL = "claude-haiku-4-5-20251001"   # fast & cheap for demo


def call_model(prompt: str, system: str = "") -> str:
    """Single completion call; returns text content."""
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Utility: simple embedding via hash projection (no extra libraries)
# ---------------------------------------------------------------------------
def pseudo_embed(text: str, dim: int = 64) -> np.ndarray:
    """
    Deterministic pseudo-embedding using character n-gram hashing.
    Good enough to illustrate MMD; swap for a real embedding model in production.
    """
    vec = np.zeros(dim)
    for n in range(2, 5):
        for i in range(len(text) - n + 1):
            ngram = text[i : i + n]
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ============================================================================
# 1. CANARY PROMPT MONITORING
#    Run fixed probe prompts repeatedly; alert when semantic similarity drops.
# ============================================================================
CANARY_PROMPTS = [
    {
        "id": "factual_capital",
        "prompt": "What is the capital of France? Answer in one word.",
        "expected_keywords": ["paris"],
    },
    {
        "id": "tone_formal",
        "prompt": "Write one formal sentence introducing a software product.",
        "expected_keywords": ["pleased", "introduce", "present", "offer", "proud", "software", "product", "solution"],
    },
    {
        "id": "code_python",
        "prompt": "Write a Python one-liner that reverses a string called `s`.",
        "expected_keywords": ["[::-1]", "reversed", "s["],
    },
]


def run_canary_check(n_runs: int = 3) -> dict:
    """
    Runs each canary prompt n_runs times and measures:
      - keyword hit rate  (proportion of runs containing expected keywords)
      - semantic stability (cosine distance between run embeddings)
    """
    print("\n" + "=" * 60)
    print("1. CANARY PROMPT MONITORING")
    print("=" * 60)

    results = {}
    for canary in CANARY_PROMPTS:
        print(f"\n  Probe: {canary['id']}")
        outputs = []
        for i in range(n_runs):
            text = call_model(canary["prompt"])
            outputs.append(text)
            print(f"    Run {i+1}: {text[:80]}{'…' if len(text) > 80 else ''}")

        # Keyword hit rate
        hit_rate = sum(
            any(kw.lower() in out.lower() for kw in canary["expected_keywords"])
            for out in outputs
        ) / n_runs

        # Pairwise cosine distances between embeddings
        embeddings = [pseudo_embed(o) for o in outputs]
        distances = [
            cosine(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        avg_distance = statistics.mean(distances) if distances else 0.0

        results[canary["id"]] = {
            "keyword_hit_rate": round(hit_rate, 2),
            "avg_semantic_distance": round(avg_distance, 4),
            "drift_alert": hit_rate < 0.6 or avg_distance > 0.3,
        }
        status = "⚠️  DRIFT ALERT" if results[canary["id"]]["drift_alert"] else "✅ Stable"
        print(f"    Keyword hit rate: {hit_rate:.0%} | Avg distance: {avg_distance:.4f} | {status}")

    return results


# ============================================================================
# 2. OUTPUT DISTRIBUTION DRIFT
#    Compare response-length and vocabulary distributions across two time windows
#    using the Kolmogorov–Smirnov test.
# ============================================================================
DISTRIBUTION_PROMPT = (
    "In 2-3 sentences, explain why regular model evaluation matters in production ML systems."
)


def collect_distribution_sample(n: int = 8) -> list[str]:
    outputs = []
    for _ in range(n):
        outputs.append(call_model(DISTRIBUTION_PROMPT))
    return outputs


def ks_drift_test(baseline: list[str], current: list[str]) -> dict:
    """KS test on response length and unique-word-count distributions."""
    def features(samples):
        lengths = [len(s.split()) for s in samples]
        vocab   = [len(set(s.lower().split())) for s in samples]
        return np.array(lengths), np.array(vocab)

    bl_len, bl_voc = features(baseline)
    cu_len, cu_voc = features(current)

    ks_len = ks_2samp(bl_len, cu_len)
    ks_voc = ks_2samp(bl_voc, cu_voc)

    return {
        "length_ks_stat":  round(ks_len.statistic, 4),
        "length_p_value":  round(ks_len.pvalue, 4),
        "vocab_ks_stat":   round(ks_voc.statistic, 4),
        "vocab_p_value":   round(ks_voc.pvalue, 4),
        "drift_detected":  ks_len.pvalue < 0.05 or ks_voc.pvalue < 0.05,
    }


def run_distribution_drift(n: int = 6) -> dict:
    print("\n" + "=" * 60)
    print("2. OUTPUT DISTRIBUTION DRIFT  (KS Test)")
    print("=" * 60)

    print(f"  Collecting baseline ({n} samples)…")
    baseline = collect_distribution_sample(n)

    # Simulate a "later window" — in production these would come from different dates
    print(f"  Collecting current window ({n} samples)…")
    current = collect_distribution_sample(n)

    result = ks_drift_test(baseline, current)

    bl_lens = [len(s.split()) for s in baseline]
    cu_lens = [len(s.split()) for s in current]
    print(f"  Baseline avg length : {statistics.mean(bl_lens):.1f} words")
    print(f"  Current  avg length : {statistics.mean(cu_lens):.1f} words")
    print(f"  Length   KS stat={result['length_ks_stat']}  p={result['length_p_value']}")
    print(f"  Vocab    KS stat={result['vocab_ks_stat']}   p={result['vocab_p_value']}")
    status = "⚠️  DRIFT DETECTED" if result["drift_detected"] else "✅ No significant drift"
    print(f"  {status}")
    return result


# ============================================================================
# 3. EMBEDDING SPACE DRIFT  (Maximum Mean Discrepancy)
#    MMD measures whether two sets of embeddings come from the same distribution.
#    A high MMD score suggests the query/response space has shifted.
# ============================================================================
BASELINE_QUERIES = [
    "How do transformers handle long sequences?",
    "What is attention in deep learning?",
    "Explain positional encoding briefly.",
    "Why do LLMs hallucinate?",
    "What is RLHF?",
]

SHIFTED_QUERIES = [
    "How do I center a div in CSS?",
    "Write a SQL query for monthly revenue.",
    "What is a p-value in statistics?",
    "Explain gradient descent to a five-year-old.",
    "How does TCP/IP work?",
]


def mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Unbiased MMD² estimate using RBF kernel."""
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    n, m = len(X), len(Y)
    return (XX.sum() - np.trace(XX)) / (n * (n - 1)) \
         + (YY.sum() - np.trace(YY)) / (m * (m - 1)) \
         - 2 * XY.mean()


def run_embedding_drift() -> dict:
    print("\n" + "=" * 60)
    print("3. EMBEDDING SPACE DRIFT  (MMD)")
    print("=" * 60)

    print("  Embedding baseline queries (LLM-domain)…")
    X = np.array([pseudo_embed(q) for q in BASELINE_QUERIES])

    print("  Embedding shifted queries (mixed-domain)…")
    Y = np.array([pseudo_embed(q) for q in SHIFTED_QUERIES])

    print("  Embedding same-distribution control…")
    Z = np.array([pseudo_embed(q + " detail") for q in BASELINE_QUERIES])

    mmd_shifted  = mmd(X, Y)
    mmd_control  = mmd(X, Z)
    threshold    = 0.02  # tunable; set empirically on your data

    result = {
        "mmd_baseline_vs_shifted": round(mmd_shifted, 6),
        "mmd_baseline_vs_control": round(mmd_control, 6),
        "threshold": threshold,
        "drift_detected": mmd_shifted > threshold,
    }

    print(f"  MMD (baseline vs shifted) : {mmd_shifted:.6f}")
    print(f"  MMD (baseline vs control) : {mmd_control:.6f}  ← expect near 0")
    status = "⚠️  DISTRIBUTION SHIFT" if result["drift_detected"] else "✅ Similar distribution"
    print(f"  {status}")
    return result


# ============================================================================
# 4. LLM-AS-JUDGE SCORING
#    A judge model scores responses on three axes over time.
#    A drop in average score signals quality drift.
# ============================================================================
SCORED_PROMPTS = [
    "Explain overfitting in machine learning in two sentences.",
    "What's the difference between precision and recall?",
    "Briefly describe what a REST API is.",
]

JUDGE_SYSTEM = """You are an objective evaluator. Score the provided AI response on three criteria,
each from 1 (poor) to 5 (excellent):
  - accuracy:   factual correctness
  - clarity:    ease of understanding
  - conciseness: appropriate brevity without losing meaning

Respond ONLY with valid JSON, no markdown, exactly like:
{"accuracy": <int>, "clarity": <int>, "conciseness": <int>}"""


def judge_response(response_text: str) -> dict:
    judge_prompt = f"Rate this AI response:\n\n{response_text}"
    raw = call_model(judge_prompt, system=JUDGE_SYSTEM)
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"accuracy": 0, "clarity": 0, "conciseness": 0, "parse_error": raw}
    return scores


def run_llm_judge_scoring() -> dict:
    print("\n" + "=" * 60)
    print("4. LLM-AS-JUDGE SCORING")
    print("=" * 60)

    all_scores = []
    for prompt in SCORED_PROMPTS:
        response = call_model(prompt)
        scores   = judge_response(response)
        all_scores.append(scores)
        avg = statistics.mean(
            v for k, v in scores.items()
            if k in ("accuracy", "clarity", "conciseness") and isinstance(v, (int, float))
        )
        print(f"\n  Prompt : {prompt[:60]}…")
        print(f"  Response: {response[:80]}{'…' if len(response) > 80 else ''}")
        print(f"  Scores : {scores}  →  avg={avg:.2f}/5")

    valid = [
        statistics.mean(v for k, v in s.items() if k in ("accuracy","clarity","conciseness") and isinstance(v,(int,float)))
        for s in all_scores
        if "parse_error" not in s
    ]
    overall_avg = statistics.mean(valid) if valid else 0.0
    drift_alert = overall_avg < 3.5

    result = {
        "overall_avg_score": round(overall_avg, 2),
        "drift_alert": drift_alert,
        "per_prompt_scores": all_scores,
    }
    status = "⚠️  QUALITY DEGRADATION" if drift_alert else "✅ Quality stable"
    print(f"\n  Overall avg score: {overall_avg:.2f}/5  |  {status}")
    return result


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "█" * 60)
    print("  LLM DRIFT DETECTION DEMO")
    print(f"  Model: {MODEL}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("█" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "sections": {},
    }

    report["sections"]["canary_prompts"]       = run_canary_check(n_runs=3)
    report["sections"]["distribution_drift"]   = run_distribution_drift(n=5)
    report["sections"]["embedding_mmd"]        = run_embedding_drift()
    report["sections"]["llm_judge_scores"]     = run_llm_judge_scoring()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    alerts = []
    for section, data in report["sections"].items():
        if isinstance(data, dict):
            if data.get("drift_alert") or data.get("drift_detected"):
                alerts.append(section)

    if alerts:
        print(f"  ⚠️  Drift signals detected in: {', '.join(alerts)}")
    else:
        print("  ✅  No drift signals detected across all checks.")

    out_path = "drift_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved → {out_path}")


if __name__ == "__main__":
    main()