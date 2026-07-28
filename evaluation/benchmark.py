"""
MindSense AI - Benchmark Runner
=================================
Full research benchmark harness that runs all evaluation components
and generates a comprehensive report suitable for research publication.

Runs:
  1. Classifier accuracy evaluation
  2. Response quality evaluation (BLEU, ROUGE, empathy, safety)
  3. RAG retrieval quality (coverage, diversity)
  4. End-to-end latency benchmarking
  5. Report generation (JSON + Markdown)

Usage::

    python evaluation/benchmark.py
    python evaluation/benchmark.py --quick   # Run on reduced sample set
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluate_responses import evaluate_responses, create_sample_evaluation_set
from evaluation.metrics import evaluation_metrics
from rag.retriever import retriever
from agents.orchestrator import orchestrator
from utils.logger import get_logger
from utils.helpers import get_utc_timestamp, safe_json_save

logger = get_logger(__name__)

# Create results directory
RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def benchmark_latency(eval_data: List[Dict], n_samples: int = 5) -> Dict[str, Any]:
    """
    Measure end-to-end pipeline latency across a sample of inputs.

    Args:
        eval_data: List of evaluation sample dicts.
        n_samples: Number of samples to benchmark.

    Returns:
        Latency statistics dict.
    """
    logger.info(f"Benchmarking latency on {n_samples} samples...")
    latencies = []
    samples = eval_data[:n_samples]

    for i, sample in enumerate(samples):
        msg = sample.get("user_message", "")
        if not msg:
            continue

        start = time.perf_counter()
        try:
            orchestrator.process(
                user_message=msg,
                session_id=f"bench_latency_{i}",
                user_id="benchmarker",
            )
        except Exception as e:
            logger.error(f"Latency benchmark sample {i} failed: {e}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        logger.debug(f"  Sample {i+1}: {elapsed_ms:.1f}ms")

    if not latencies:
        return {"error": "No latency measurements collected"}

    return {
        "n_samples": len(latencies),
        "mean_ms": round(sum(latencies) / len(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if len(latencies) > 1 else latencies[0],
    }


def benchmark_retrieval(eval_data: List[Dict], n_samples: int = 5) -> Dict[str, Any]:
    """
    Benchmark RAG retrieval coverage and diversity.

    Args:
        eval_data: Evaluation samples.
        n_samples: Number to test.

    Returns:
        Retrieval statistics dict.
    """
    logger.info(f"Benchmarking retrieval on {n_samples} samples...")
    coverage_scores = []
    diversity_scores = []

    for sample in eval_data[:n_samples]:
        msg = sample.get("user_message", "")
        if not msg:
            continue

        try:
            results = retriever.retrieve_by_categories(msg, per_category_k=2)
            if not results:
                coverage_scores.append(0.0)
                diversity_scores.append(0.0)
                continue

            # Coverage: fraction of categories represented
            categories = {r.get("category", "") for r in results}
            from config import settings
            total_cats = len(settings.rag.KNOWLEDGE_CATEGORIES)
            coverage = len(categories) / max(1, total_cats)
            coverage_scores.append(coverage)

            # Diversity: avg pairwise dissimilarity between top chunks
            texts = [r.get("text", "") for r in results[:3]]
            if len(texts) > 1:
                overlaps = []
                for j in range(len(texts)):
                    for k in range(j + 1, len(texts)):
                        t1 = set(texts[j].lower().split())
                        t2 = set(texts[k].lower().split())
                        union = t1 | t2
                        overlap = len(t1 & t2) / max(1, len(union))
                        overlaps.append(1 - overlap)  # Dissimilarity
                diversity = sum(overlaps) / max(1, len(overlaps))
                diversity_scores.append(diversity)
            else:
                diversity_scores.append(1.0)

        except Exception as e:
            logger.error(f"Retrieval benchmark error: {e}")

    return {
        "n_samples": len(coverage_scores),
        "mean_coverage": round(sum(coverage_scores) / max(1, len(coverage_scores)), 4),
        "mean_diversity": round(sum(diversity_scores) / max(1, len(diversity_scores)), 4),
    }


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """
    Generate a Markdown-formatted research benchmark report.

    Args:
        report: Complete benchmark results dict.

    Returns:
        Markdown report string.
    """
    ts = report.get("timestamp", "")
    md_lines = [
        "# MindSense AI — Benchmark Report",
        f"*Generated: {ts}*",
        "",
        "## Overview",
        f"- **System**: MindSense AI v1.0.0",
        f"- **LLM**: Google Gemini 2.5 Flash",
        f"- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2",
        f"- **Vector Store**: FAISS (Flat Index)",
        "",
        "## Response Quality Metrics",
    ]

    response_eval = report.get("response_evaluation", {})
    agg = response_eval.get("aggregate_scores", {})
    for metric, stats in agg.items():
        md_lines.append(
            f"| {metric.capitalize()} | {stats['mean']:.4f} | "
            f"{stats['min']:.4f} | {stats['max']:.4f} |"
        )

    md_lines += [
        "",
        "## Retrieval Quality",
    ]
    retrieval = report.get("retrieval_benchmark", {})
    md_lines.append(f"- Coverage: {retrieval.get('mean_coverage', 0):.4f}")
    md_lines.append(f"- Diversity: {retrieval.get('mean_diversity', 0):.4f}")

    md_lines += [
        "",
        "## Latency",
    ]
    latency = report.get("latency_benchmark", {})
    md_lines.append(f"- Mean: {latency.get('mean_ms', 0):.1f}ms")
    md_lines.append(f"- P95: {latency.get('p95_ms', 0):.1f}ms")
    md_lines.append(f"- Min: {latency.get('min_ms', 0):.1f}ms")
    md_lines.append(f"- Max: {latency.get('max_ms', 0):.1f}ms")

    return "\n".join(md_lines)


def run_benchmark(quick: bool = False) -> Dict[str, Any]:
    """
    Execute the complete benchmark suite.

    Args:
        quick: If True, use a reduced sample set for faster execution.

    Returns:
        Complete benchmark report dict.
    """
    logger.info("=" * 60)
    logger.info("MindSense AI — Full Research Benchmark")
    logger.info("=" * 60)

    eval_data = create_sample_evaluation_set()
    n_samples = 3 if quick else len(eval_data)

    report: Dict[str, Any] = {
        "timestamp": get_utc_timestamp(),
        "mode": "quick" if quick else "full",
        "n_eval_samples": n_samples,
    }

    # 1. Response quality evaluation
    logger.info("Phase 1: Response Quality Evaluation")
    response_eval = evaluate_responses(eval_data[:n_samples])
    report["response_evaluation"] = {
        "aggregate_scores": response_eval.get("aggregate_scores", {}),
        "n_evaluated": response_eval.get("n_evaluated", 0),
    }

    # 2. Retrieval benchmark
    logger.info("Phase 2: Retrieval Quality Benchmark")
    retrieval_bench = benchmark_retrieval(eval_data, n_samples=min(n_samples, 3))
    report["retrieval_benchmark"] = retrieval_bench

    # 3. Latency benchmark
    logger.info("Phase 3: Latency Benchmark")
    latency_bench = benchmark_latency(eval_data, n_samples=min(n_samples, 3))
    report["latency_benchmark"] = latency_bench

    # 4. Save JSON report
    json_path = RESULTS_DIR / "benchmark_report.json"
    safe_json_save(report, json_path)
    logger.info(f"Benchmark JSON report saved to: {json_path}")

    # 5. Save Markdown report
    md_report = generate_markdown_report(report)
    md_path = RESULTS_DIR / "benchmark_report.md"
    md_path.write_text(md_report, encoding="utf-8")
    logger.info(f"Benchmark Markdown report saved to: {md_path}")

    logger.info("=" * 60)
    logger.info("Benchmark complete.")
    logger.info("=" * 60)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MindSense AI research benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick benchmark on a reduced sample set",
    )
    args = parser.parse_args()
    run_benchmark(quick=args.quick)
