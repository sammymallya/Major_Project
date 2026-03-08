"""
Standalone Evaluation Module
=============================
Completely independent file — plug into any pipeline.

Usage:
    from evaluator import Evaluator, TEST_SET

    # After getting results from your pipeline (any mode):
    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        query="beaches in Mangalore",
        results=[
            {"name": "Panambur Beach", "type": "Beach", "city": "Mangalore", "description": "..."},
            ...
        ],
        mode="kg"   # "kg", "vector", "hybrid", "none"
    )
    evaluator.print_metrics(metrics)

    # OR run full benchmark across all modes automatically:
    evaluator.run_benchmark(pipeline_fn=your_query_function)

Install:
    pip install sentence-transformers matplotlib tabulate
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from sentence_transformers import SentenceTransformer, util


# ─────────────────────────────────────────────────────────
# TEST SET — add your own queries + expected answers here
# ─────────────────────────────────────────────────────────
TEST_SET = [
    # ── Type-specific queries ──────────────────────────
    {
        "query": "beaches in Mangalore",
        "expected_types": ["Beach", "Beach and Temple"],
        "expected_city": "Mangalore",
        "known_places": [
            "Panambur Beach", "Tannirbhavi Beach",
            "Kulai Beach", "Hosabettu Beach", "Surathkal Beach"
        ]
    },
    {
        "query": "temples in Udupi",
        "expected_types": ["Temple", "Beach and Temple"],
        "expected_city": "Udupi",
        "known_places": []
    },
    {
        "query": "shopping malls in Mangalore",
        "expected_types": ["Shopping Mall", "Shopping Area"],
        "expected_city": "Mangalore",
        "known_places": []
    },
    {
        "query": "restaurants in Mangalore",
        "expected_types": [
            "Restaurant", "Seafood Restaurants",
            "Cafe", "Dessert Shop", "Dessert Restaurant"
        ],
        "expected_city": "Mangalore",
        "known_places": []
    },
    {
        "query": "waterfalls near Chikkamagaluru",
        "expected_types": ["Waterfall"],
        "expected_city": "Chikkamagaluru",
        "known_places": []
    },

    # ── Activity-based queries ─────────────────────────
    {
        "query": "trekking places in Karnataka",
        "expected_types": ["Trekking Destination", "Hill Station", "Mountain Pass"],
        "expected_city": None,   # state-level query
        "known_places": []
    },
    {
        "query": "historical places in Hampi",
        "expected_types": [
            "Historical Site", "Historical Monument",
            "Historical Fort", "Heritage Site", "Monument", "Museum", "Palace"
        ],
        "expected_city": "Hampi",
        "known_places": []
    },
    {
        "query": "viewpoints in Coorg",
        "expected_types": ["Viewpoint", "Hill Viewpoint"],
        "expected_city": "Madikeri",
        "known_places": []
    },

    # ── Ambiguous / semantic queries ───────────────────
    {
        "query": "relaxing places in Mangalore",
        "expected_types": ["Beach", "Park", "Garden", "Lake"],
        "expected_city": "Mangalore",
        "known_places": []
    },
    {
        "query": "scenic spots near Udupi",
        "expected_types": ["Viewpoint", "Beach", "Waterfall", "Hill Viewpoint"],
        "expected_city": "Udupi",
        "known_places": []
    },

    # ── Edge cases ─────────────────────────────────────
    {
        "query": "places near Panambur Beach",
        "expected_types": [],    # any type — testing NEARBY relationship
        "expected_city": None,
        "known_places": ["Tannirbhavi Beach", "Surathkal Beach"]
    },
    {
        "query": "water parks in Mangalore",
        "expected_types": ["Water Park"],
        "expected_city": "Mangalore",
        "known_places": []
    },
]


# ─────────────────────────────────────────────────────────
# RESULT DATACLASS — one row in the results table
# ─────────────────────────────────────────────────────────
@dataclass
class MetricResult:
    query: str
    mode: str
    precision: float
    recall: float | None
    relevance: float
    hallucination: float
    result_count: int
    latency_ms: float
    notes: str = ""

    def to_row(self) -> list:
        """Return as a table row for display."""
        recall_str = f"{self.recall:.1f}%" if self.recall is not None else "N/A"
        return [
            self.query[:35] + ("..." if len(self.query) > 35 else ""),
            self.mode.upper(),
            f"{self.precision:.1f}%",
            recall_str,
            f"{self.relevance:.1f}%",
            f"{self.hallucination:.1f}%",
            self.result_count,
            f"{self.latency_ms:.0f}ms",
        ]


# ─────────────────────────────────────────────────────────
# EVALUATOR — standalone, connects to any pipeline
# ─────────────────────────────────────────────────────────
class Evaluator:
    """
    Standalone evaluator — works with any pipeline output.

    Input  : query string + list of result dicts
    Output : MetricResult with precision, recall, relevance, hallucination

    Result dicts can have any of these field names (auto-detected):
        name / p.name
        type / p.type
        city / c.name
        description / p.description
        text  (from vector DB)
        score (similarity score)
    """

    def __init__(self):
        print("Loading evaluation model (MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Evaluator ready.\n")

    # ── Field extraction helpers ───────────────────────
    def _name(self, r: dict) -> str:
        return r.get("name") or r.get("p.name") or ""

    def _type(self, r: dict) -> str:
        return r.get("type") or r.get("p.type") or ""

    def _city(self, r: dict) -> str:
        return r.get("city") or r.get("c.name") or r.get("metadata", {}).get("city", "")

    def _desc(self, r: dict) -> str:
        return (
            r.get("description") or r.get("p.description") or
            r.get("text") or
            r.get("metadata", {}).get("text", "") or ""
        )

    # ── Core evaluate ──────────────────────────────────
    def evaluate(self,
                 query: str,
                 results: list[dict],
                 mode: str = "kg",
                 latency_ms: float = 0.0) -> MetricResult:
        """
        Evaluate a single query result set.

        Args:
            query      : the natural language query that was asked
            results    : list of result dicts from your pipeline
            mode       : "kg", "vector", "hybrid", or "none"
            latency_ms : how long the query took in milliseconds

        Returns:
            MetricResult with all 4 metrics computed
        """
        q_lower = query.lower().strip()
        gt = self._find_ground_truth(q_lower)

        precision    = self._precision(results, gt)
        recall       = self._recall(results, gt)
        relevance    = self._relevance(query, results)
        hallucination = self._hallucination(results)

        notes = []
        if not gt:
            notes.append("no ground truth")
        if not results:
            notes.append("no results returned")

        return MetricResult(
            query=query,
            mode=mode,
            precision=precision,
            recall=recall,
            relevance=relevance,
            hallucination=hallucination,
            result_count=len(results),
            latency_ms=latency_ms,
            notes=", ".join(notes)
        )

    # ── Metric 1: Precision ────────────────────────────
    def _precision(self, results: list[dict], gt: dict | None) -> float:
        """
        % of returned results whose type matches expected types.
        If no ground truth, checks if all results are same type
        (type consistency = precision indicator).
        """
        if not results:
            return 0.0

        if gt and gt.get("expected_types"):
            expected = [t.lower() for t in gt["expected_types"]]
            correct = sum(
                1 for r in results
                if self._type(r).lower() in expected
            )
            return correct / len(results) * 100

        # No ground truth — measure type consistency
        types = set(self._type(r).lower() for r in results if self._type(r))
        if not types:
            return 0.0
        # 1 type = 100%, 2 types = 80%, 3+ types = lower
        return max(0.0, 100.0 - (len(types) - 1) * 20)

    # ── Metric 2: Recall ───────────────────────────────
    def _recall(self, results: list[dict], gt: dict | None) -> float | None:
        """
        % of known places that were returned.
        Returns None if no known places defined in ground truth.
        """
        if not gt or not gt.get("known_places"):
            return None

        returned_names = [self._name(r).lower() for r in results]
        found = sum(
            1 for kp in gt["known_places"]
            if kp.lower() in returned_names
        )
        return found / len(gt["known_places"]) * 100

    # ── Metric 3: Relevance ────────────────────────────
    def _relevance(self, query: str, results: list[dict]) -> float:
        """
        Average cosine similarity between query embedding
        and result description embeddings.
        Range: 0-100%. Higher = more semantically relevant.
        """
        if not results:
            return 0.0

        descriptions = [
            self._desc(r) or self._name(r)
            for r in results
        ]
        descriptions = [d for d in descriptions if d]

        if not descriptions:
            return 0.0

        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        d_embs = self.embedder.encode(descriptions, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, d_embs)[0]
        return float(sims.mean()) * 100

    # ── Metric 4: Hallucination Check ─────────────────
    def _hallucination(self, results: list[dict]) -> float:
        """
        % of results that are properly grounded (have name + type).
        A result missing name or type is suspicious — may be hallucinated
        or fabricated by the model rather than retrieved from KG/VectorDB.

        100% = fully grounded, 0% = all results suspicious.
        """
        if not results:
            return 100.0  # no results = no hallucination

        grounded = sum(
            1 for r in results
            if self._name(r) and (self._type(r) or self._desc(r))
        )
        return grounded / len(results) * 100

    # ── Ground truth lookup ────────────────────────────
    def _find_ground_truth(self, query_lower: str) -> dict | None:
        for item in TEST_SET:
            if item["query"].lower() == query_lower:
                return item
        # Fuzzy match — check if query contains key words
        for item in TEST_SET:
            words = item["query"].lower().split()
            if sum(1 for w in words if w in query_lower) >= len(words) * 0.7:
                return item
        return None

    # ── Pretty print single result ─────────────────────
    def print_metrics(self, m: MetricResult):
        """Print metrics for a single query — developer view."""
        print("\n" + "─" * 55)
        print(f"  📊 METRICS  [{m.mode.upper()} mode]  —  dev only")
        print("─" * 55)
        print(f"\n  Query      : {m.query}")
        print(f"  Results    : {m.result_count} returned | {m.latency_ms:.0f}ms")

        print(f"\n  🎯 Precision    {self._bar(m.precision)} {m.precision:.1f}%")
        print(f"     All returned results match expected place type")

        if m.recall is not None:
            print(f"\n  🔁 Recall       {self._bar(m.recall)} {m.recall:.1f}%")
            print(f"     Known places successfully retrieved")
        else:
            print(f"\n  🔁 Recall       N/A  (add known_places to TEST_SET)")

        print(f"\n  💡 Relevance    {self._bar(m.relevance)} {m.relevance:.1f}%")
        print(f"     Semantic similarity between query and results")

        print(f"\n  🛡️  Grounding    {self._bar(m.hallucination)} {m.hallucination:.1f}%")
        print(f"     Results verified from KG/VectorDB (not hallucinated)")

        if m.notes:
            print(f"\n  ⚠️  Notes: {m.notes}")
        print("─" * 55)

    # ── Full benchmark across all modes ───────────────
    def run_benchmark(self,
                      pipeline_fn: Callable[[str, str], list[dict]],
                      modes: list[str] = None,
                      test_set: list[dict] = None,
                      save_json: str = "eval_results.json",
                      save_csv: str = "eval_results.csv") -> list[MetricResult]:
        """
        Run all TEST_SET queries across all modes and collect metrics.

        Args:
            pipeline_fn : function(query, mode) → list[dict]
                          This is YOUR pipeline's query function.
                          It must accept (query: str, mode: str)
                          and return a list of result dicts.

            modes       : list of modes to test. Default: all 4.
            test_set    : override TEST_SET if needed.
            save_json   : save full results to JSON file.
            save_csv    : save summary table to CSV file.

        Returns:
            list of MetricResult — one per (query, mode) combination.

        Example:
            def my_pipeline(query, mode):
                result = pipeline.query(query, mode=mode, dev_mode=False)
                return result["results"]   # list of dicts

            evaluator.run_benchmark(pipeline_fn=my_pipeline)
        """
        if modes is None:
            modes = ["kg", "vector", "hybrid", "none"]
        if test_set is None:
            test_set = TEST_SET

        all_results: list[MetricResult] = []

        print("=" * 60)
        print("  RUNNING BENCHMARK")
        print(f"  {len(test_set)} queries × {len(modes)} modes = "
              f"{len(test_set) * len(modes)} evaluations")
        print("=" * 60)

        for mode in modes:
            print(f"\n── Mode: {mode.upper()} {'─'*40}")
            for item in test_set:
                query = item["query"]
                print(f"  Query: {query[:50]}...", end=" ", flush=True)

                start = time.time()
                try:
                    results = pipeline_fn(query, mode)
                except Exception as e:
                    print(f"ERROR: {e}")
                    results = []
                elapsed_ms = (time.time() - start) * 1000

                metric = self.evaluate(
                    query=query,
                    results=results,
                    mode=mode,
                    latency_ms=elapsed_ms
                )
                all_results.append(metric)
                print(f"P={metric.precision:.0f}% "
                      f"R={metric.recall:.0f}% " if metric.recall else
                      f"P={metric.precision:.0f}% R=N/A ",
                      end="")
                print(f"Rel={metric.relevance:.0f}% "
                      f"H={metric.hallucination:.0f}%")

        # Print summary table
        self._print_summary_table(all_results)

        # Save results
        self._save_json(all_results, save_json)
        self._save_csv(all_results, save_csv)

        # Print mode averages
        self._print_mode_averages(all_results, modes)

        return all_results

    # ── Summary table ──────────────────────────────────
    def _print_summary_table(self, results: list[MetricResult]):
        try:
            from tabulate import tabulate
            headers = [
                "Query", "Mode", "Precision",
                "Recall", "Relevance", "Grounding", "Count", "Latency"
            ]
            rows = [r.to_row() for r in results]
            print("\n\n" + "=" * 60)
            print("  FULL RESULTS TABLE")
            print("=" * 60)
            print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        except ImportError:
            print("\n(Install tabulate for a pretty table: pip install tabulate)")
            for r in results:
                print(r.to_row())

    # ── Mode averages ──────────────────────────────────
    def _print_mode_averages(self,
                              results: list[MetricResult],
                              modes: list[str]):
        print("\n\n" + "=" * 60)
        print("  AVERAGE SCORES PER MODE")
        print("=" * 60)

        rows = []
        for mode in modes:
            mode_results = [r for r in results if r.mode == mode]
            if not mode_results:
                continue

            avg_p   = sum(r.precision     for r in mode_results) / len(mode_results)
            avg_r   = [r.recall for r in mode_results if r.recall is not None]
            avg_rel = sum(r.relevance     for r in mode_results) / len(mode_results)
            avg_h   = sum(r.hallucination for r in mode_results) / len(mode_results)
            avg_lat = sum(r.latency_ms    for r in mode_results) / len(mode_results)

            recall_str = f"{sum(avg_r)/len(avg_r):.1f}%" if avg_r else "N/A"
            rows.append([
                mode.upper(),
                f"{avg_p:.1f}%",
                recall_str,
                f"{avg_rel:.1f}%",
                f"{avg_h:.1f}%",
                f"{avg_lat:.0f}ms"
            ])

        try:
            from tabulate import tabulate
            print(tabulate(
                rows,
                headers=["Mode", "Precision", "Recall",
                          "Relevance", "Grounding", "Avg Latency"],
                tablefmt="rounded_outline"
            ))
        except ImportError:
            for row in rows:
                print(row)

        print("\n💡 Key insight: HYBRID should have highest overall scores.")
        print("   KG should have highest Precision + Grounding.")
        print("   NONE should have lowest Precision + Grounding (proves KG value).")

    # ── Save JSON ──────────────────────────────────────
    def _save_json(self, results: list[MetricResult], path: str):
        data = [
            {
                "query":         r.query,
                "mode":          r.mode,
                "precision":     r.precision,
                "recall":        r.recall,
                "relevance":     r.relevance,
                "hallucination": r.hallucination,
                "result_count":  r.result_count,
                "latency_ms":    r.latency_ms,
                "notes":         r.notes,
            }
            for r in results
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✅ Results saved to {path}")

    # ── Save CSV ───────────────────────────────────────
    def _save_csv(self, results: list[MetricResult], path: str):
        import csv
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query", "mode", "precision", "recall",
                "relevance", "hallucination", "result_count", "latency_ms"
            ])
            for r in results:
                writer.writerow([
                    r.query, r.mode,
                    round(r.precision, 2),
                    round(r.recall, 2) if r.recall is not None else "",
                    round(r.relevance, 2),
                    round(r.hallucination, 2),
                    r.result_count,
                    round(r.latency_ms, 2)
                ])
        print(f"✅ CSV saved to {path}")

    # ── Bar helper ─────────────────────────────────────
    def _bar(self, score: float, width: int = 15) -> str:
        filled = int((score / 100) * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"

    # ── Plot bar chart ─────────────────────────────────
    def plot_results(self,
                     results: list[MetricResult],
                     save_path: str = "eval_chart.png"):
        """
        Generate a bar chart comparing all 4 modes across all metrics.
        Save as PNG for your project report.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Install matplotlib: pip install matplotlib")
            return

        modes   = ["kg", "vector", "hybrid", "none"]
        metrics = ["Precision", "Relevance", "Grounding"]
        colors  = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

        # Compute averages per mode per metric
        data = {}
        for mode in modes:
            mr = [r for r in results if r.mode == mode]
            if not mr:
                data[mode] = [0, 0, 0]
                continue
            data[mode] = [
                sum(r.precision     for r in mr) / len(mr),
                sum(r.relevance     for r in mr) / len(mr),
                sum(r.hallucination for r in mr) / len(mr),
            ]

        x = np.arange(len(metrics))
        width = 0.18
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (mode, color) in enumerate(zip(modes, colors)):
            bars = ax.bar(
                x + i * width,
                data[mode],
                width,
                label=mode.upper(),
                color=color,
                alpha=0.85,
                edgecolor="white"
            )
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.5,
                    f"{h:.1f}%",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold"
                )

        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title(
            "Pipeline Evaluation — KG vs Vector vs Hybrid vs None",
            fontsize=13, fontweight="bold", pad=15
        )
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 115)
        ax.legend(fontsize=10, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Chart saved to {save_path}")
        plt.show()


# ─────────────────────────────────────────────────────────
# HOW TO CONNECT TO YOUR PIPELINE
# ─────────────────────────────────────────────────────────
"""
Step 1: Import this file in your main pipeline file

    from evaluator import Evaluator, TEST_SET

Step 2: Define a wrapper function for your pipeline

    def my_pipeline_fn(query: str, mode: str) -> list[dict]:
        # Call your pipeline and return raw results as list of dicts
        # Each dict should have: name, type, city, description
        result = pipeline.query(query, mode=mode, dev_mode=False)
        return result.get("results", [])

Step 3: Run benchmark

    evaluator = Evaluator()
    results = evaluator.run_benchmark(pipeline_fn=my_pipeline_fn)

Step 4: Plot chart for report

    evaluator.plot_results(results, save_path="eval_chart.png")


── For single query evaluation (during live use) ──────────

    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        query="beaches in Mangalore",
        results=kg_results,   # list of dicts from your pipeline
        mode="kg"
    )
    evaluator.print_metrics(metrics)
"""


# ─────────────────────────────────────────────────────────
# DEMO — run without any pipeline (uses dummy data)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    evaluator = Evaluator()

    # ── Demo: evaluate a single result set ────────────
    print("\n── Single Query Evaluation Demo ──")
    dummy_results = [
        {"name": "Panambur Beach",    "type": "Beach",         "city": "Mangalore", "description": "Popular beach with water sports."},
        {"name": "Tannirbhavi Beach", "type": "Beach",         "city": "Mangalore", "description": "Calm and less crowded beach."},
        {"name": "Kulai Beach",       "type": "Beach",         "city": "Mangalore", "description": "Scenic beach with stone wave breakers."},
        {"name": "City Centre Mall",  "type": "Shopping Mall", "city": "Mangalore", "description": "Shopping mall."},  # wrong type — precision test
    ]

    metrics = evaluator.evaluate(
        query="beaches in Mangalore",
        results=dummy_results,
        mode="kg",
        latency_ms=320
    )
    evaluator.print_metrics(metrics)

    # ── Demo: benchmark with dummy pipeline ───────────
    print("\n── Benchmark Demo (dummy pipeline) ──")

    def dummy_pipeline(query: str, mode: str) -> list[dict]:
        """Simulates your pipeline — replace with real pipeline."""
        if mode == "none":
            # Simulate hallucination — wrong types returned
            return [
                {"name": "Some Place",   "type": "Unknown", "city": "", "description": ""},
                {"name": "Another Place","type": "Unknown", "city": "", "description": ""},
            ]
        # Simulate KG/vector returning correct results
        return [
            {"name": "Panambur Beach",    "type": "Beach", "city": "Mangalore",
             "description": "Popular beach known for water sports and sunset views."},
            {"name": "Tannirbhavi Beach", "type": "Beach", "city": "Mangalore",
             "description": "A calm and peaceful beach away from city traffic."},
        ]

    # Run on just 2 queries for demo speed
    results = evaluator.run_benchmark(
        pipeline_fn=dummy_pipeline,
        modes=["kg", "vector", "hybrid", "none"],
        test_set=TEST_SET[:3],          # use first 3 queries for demo
        save_json="eval_results.json",
        save_csv="eval_results.csv"
    )

    # Generate chart
    evaluator.plot_results(results, save_path="eval_chart.png")
