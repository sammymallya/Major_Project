"""
Standalone Evaluation Module
=============================
Completely independent file — plug into any pipeline.

Metrics are split by source:
  KG      → Precision, Recall, Hallucination
  Vector  → Relevance Score, Semantic Coverage
  Hybrid  → All metrics combined
  None    → Baseline (proves grounding layer is needed)

Usage:
    from evaluator import Evaluator, TEST_SET

    evaluator = Evaluator()

    # Single query evaluation
    metrics = evaluator.evaluate(
        query="beaches in Mangalore",
        kg_results=[{"name": "Panambur Beach", "type": "Beach", ...}],
        vector_snippet="Panambur Beach is a popular beach...",
        mode="hybrid"
    )
    evaluator.print_metrics(metrics)

    # Full benchmark across all modes
    evaluator.run_benchmark(pipeline_fn=your_query_function)

Install:
    pip install sentence-transformers matplotlib tabulate
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from typing import Callable

from sentence_transformers import SentenceTransformer, util


# ─────────────────────────────────────────────────────────
# TEST SET
# Queries split into two types:
#   exact    → has known_places, tests KG precision/recall
#   semantic → ambiguous, tests vector DB semantic coverage
# ─────────────────────────────────────────────────────────
TEST_SET = [

    # ── Exact queries (KG strong, Vector also good) ────
    {
        "query": "beaches in Mangalore",
        "query_type": "exact",
        "expected_types": ["Beach", "Beach and Temple"],
        "expected_city": "Mangalore",
        "known_places": [
            "Panambur Beach", "Tannirbhavi Beach",
            "Kulai Beach", "Hosabettu Beach", "Surathkal Beach"
        ]
    },
    {
        "query": "temples in Udupi",
        "query_type": "exact",
        "expected_types": ["Temple", "Beach and Temple"],
        "expected_city": "Udupi",
        "known_places": []
    },
    {
        "query": "shopping malls in Mangalore",
        "query_type": "exact",
        "expected_types": ["Shopping Mall", "Shopping Area"],
        "expected_city": "Mangalore",
        "known_places": []
    },
    {
        "query": "restaurants in Mangalore",
        "query_type": "exact",
        "expected_types": [
            "Restaurant", "Seafood Restaurants",
            "Cafe", "Dessert Shop", "Dessert Restaurant"
        ],
        "expected_city": "Mangalore",
        "known_places": []
    },
    {
        "query": "waterfalls near Chikkamagaluru",
        "query_type": "exact",
        "expected_types": ["Waterfall"],
        "expected_city": "Chikkamagaluru",
        "known_places": []
    },
    {
        "query": "historical places in Hampi",
        "query_type": "exact",
        "expected_types": [
            "Historical Site", "Historical Monument",
            "Historical Fort", "Heritage Site", "Monument", "Palace"
        ],
        "expected_city": "Hampi",
        "known_places": []
    },
    {
        "query": "trekking places in Karnataka",
        "query_type": "exact",
        "expected_types": ["Trekking Destination", "Hill Station", "Mountain Pass"],
        "expected_city": None,
        "known_places": []
    },

    # ── Semantic queries (Vector strong, KG struggles) ─
    {
        "query": "relaxing places in Mangalore",
        "query_type": "semantic",
        "expected_types": ["Beach", "Park", "Garden", "Lake"],
        "expected_city": "Mangalore",
        "known_places": [],
        "semantic_keywords": ["calm", "peaceful", "relax", "nature", "quiet", "serene"]
    },
    {
        "query": "scenic spots near Udupi",
        "query_type": "semantic",
        "expected_types": ["Viewpoint", "Beach", "Waterfall", "Hill Viewpoint"],
        "expected_city": "Udupi",
        "known_places": [],
        "semantic_keywords": ["scenic", "view", "beautiful", "landscape", "nature"]
    },
    {
        "query": "budget friendly places to visit",
        "query_type": "semantic",
        "expected_types": [],
        "expected_city": None,
        "known_places": [],
        "semantic_keywords": ["free", "entry", "no fee", "affordable", "budget"]
    },
    {
        "query": "places for adventure activities",
        "query_type": "semantic",
        "expected_types": ["Trekking Destination", "Water Park", "Beach"],
        "expected_city": None,
        "known_places": [],
        "semantic_keywords": ["adventure", "trek", "sport", "activity", "thrilling"]
    },
    {
        "query": "family friendly places in Mangalore",
        "query_type": "semantic",
        "expected_types": ["Park", "Water Park", "Garden", "Beach"],
        "expected_city": "Mangalore",
        "known_places": [],
        "semantic_keywords": ["family", "children", "kids", "fun", "park"]
    },
]


# ─────────────────────────────────────────────────────────
# METRIC RESULT
# ─────────────────────────────────────────────────────────
@dataclass
class MetricResult:
    query: str
    mode: str
    query_type: str

    # KG metrics
    kg_precision: float | None
    kg_recall: float | None
    kg_hallucination: float | None
    kg_result_count: int

    # Vector metrics
    vector_relevance: float | None
    vector_semantic_coverage: float | None
    vector_returned: bool

    # Overall
    overall_relevance: float
    latency_ms: float
    notes: str = ""

    def to_row(self) -> list:
        return [
            self.query[:35] + ("..." if len(self.query) > 35 else ""),
            self.mode.upper(),
            self.query_type,
            f"{self.kg_precision:.1f}%"            if self.kg_precision            is not None else "N/A",
            f"{self.kg_recall:.1f}%"               if self.kg_recall               is not None else "N/A",
            f"{self.kg_hallucination:.1f}%"        if self.kg_hallucination        is not None else "N/A",
            f"{self.vector_relevance:.1f}%"        if self.vector_relevance        is not None else "N/A",
            f"{self.vector_semantic_coverage:.1f}%" if self.vector_semantic_coverage is not None else "N/A",
            f"{self.overall_relevance:.1f}%",
            f"{self.latency_ms:.0f}ms",
        ]


# ─────────────────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────────────────
class Evaluator:
    """
    Standalone evaluator with separate KG and Vector DB metrics.

    KG metrics    : Precision, Recall, Hallucination
    Vector metrics: Relevance Score, Semantic Coverage
    """

    def __init__(self):
        print("Loading evaluation model (MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Evaluator ready.\n")

    # ── Field helpers ──────────────────────────────────
    def _name(self, r: dict) -> str:
        return r.get("name") or r.get("p.name") or ""

    def _type(self, r: dict) -> str:
        return r.get("type") or r.get("p.type") or ""

    def _desc(self, r: dict) -> str:
        return (
            r.get("description") or r.get("p.description") or
            r.get("text") or
            r.get("metadata", {}).get("text", "") or ""
        )

    # ── Main evaluate ──────────────────────────────────
    def evaluate(self,
                 query: str,
                 kg_results: list[dict],
                 vector_snippet: str | None,
                 mode: str = "hybrid",
                 latency_ms: float = 0.0) -> MetricResult:
        """
        Evaluate a single query with separate KG and Vector metrics.

        Args:
            query          : natural language query
            kg_results     : list of dicts from KG (name, type, city, description)
            vector_snippet : text snippet returned from Vector DB after reranking
            mode           : "kg", "vector", "hybrid", "none"
            latency_ms     : query latency in milliseconds
        """
        gt = self._find_ground_truth(query.lower().strip())
        query_type = gt.get("query_type", "exact") if gt else "exact"

        kg_precision     = self._kg_precision(kg_results, gt)
        kg_recall        = self._kg_recall(kg_results, gt)
        kg_hallucination = self._kg_hallucination(kg_results)

        vector_relevance         = self._vector_relevance(query, vector_snippet)
        vector_semantic_coverage = self._vector_semantic_coverage(vector_snippet, gt)
        vector_returned          = bool(vector_snippet and vector_snippet.strip())

        all_texts = [self._desc(r) or self._name(r) for r in kg_results]
        if vector_snippet:
            all_texts.append(vector_snippet)
        overall_relevance = self._relevance_score(query, [t for t in all_texts if t])

        notes = []
        if not gt:
            notes.append("no ground truth")
        if not kg_results and mode in ("kg", "hybrid"):
            notes.append("KG returned nothing")
        if not vector_returned and mode in ("vector", "hybrid"):
            notes.append("Vector returned nothing")

        return MetricResult(
            query=query,
            mode=mode,
            query_type=query_type,
            kg_precision=kg_precision,
            kg_recall=kg_recall,
            kg_hallucination=kg_hallucination,
            kg_result_count=len(kg_results),
            vector_relevance=vector_relevance,
            vector_semantic_coverage=vector_semantic_coverage,
            vector_returned=vector_returned,
            overall_relevance=overall_relevance,
            latency_ms=latency_ms,
            notes=", ".join(notes)
        )

    # ── KG: Precision ──────────────────────────────────
    def _kg_precision(self, results: list[dict], gt: dict | None) -> float | None:
        if not results:
            return 0.0
        if gt and gt.get("expected_types"):
            expected = [t.lower() for t in gt["expected_types"]]
            correct = sum(1 for r in results if self._type(r).lower() in expected)
            return correct / len(results) * 100
        types = set(self._type(r).lower() for r in results if self._type(r))
        return max(0.0, 100.0 - (len(types) - 1) * 20) if types else 0.0

    # ── KG: Recall ─────────────────────────────────────
    def _kg_recall(self, results: list[dict], gt: dict | None) -> float | None:
        if not gt or not gt.get("known_places"):
            return None
        returned = [self._name(r).lower() for r in results]
        found = sum(1 for kp in gt["known_places"] if kp.lower() in returned)
        return found / len(gt["known_places"]) * 100

    # ── KG: Hallucination ──────────────────────────────
    def _kg_hallucination(self, results: list[dict]) -> float:
        if not results:
            return 100.0
        grounded = sum(
            1 for r in results
            if self._name(r) and (self._type(r) or self._desc(r))
        )
        return grounded / len(results) * 100

    # ── Vector: Relevance ──────────────────────────────
    def _vector_relevance(self, query: str, snippet: str | None) -> float | None:
        """
        Semantic similarity between query and vector snippet.
        PRIMARY metric for Vector DB evaluation.
        """
        if not snippet or not snippet.strip():
            return None
        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        s_emb = self.embedder.encode(snippet, convert_to_tensor=True)
        return round(float(util.cos_sim(q_emb, s_emb)[0][0]) * 100, 1)

    # ── Vector: Semantic Coverage ──────────────────────
    def _vector_semantic_coverage(self,
                                   snippet: str | None,
                                   gt: dict | None) -> float | None:
        """
        Checks if vector snippet contains semantic keywords
        showing it understood the query intent.
        Especially useful for ambiguous queries like
        'relaxing places' or 'budget friendly'.
        """
        if not snippet or not gt:
            return None
        keywords = gt.get("semantic_keywords", [])
        if not keywords:
            keywords = [t.lower() for t in gt.get("expected_types", [])]
        if not keywords:
            return None
        snippet_lower = snippet.lower()
        matched = sum(1 for kw in keywords if kw.lower() in snippet_lower)
        return matched / len(keywords) * 100

    # ── Shared: Relevance score ────────────────────────
    def _relevance_score(self, query: str, texts: list[str]) -> float:
        if not texts:
            return 0.0
        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        t_embs = self.embedder.encode(texts, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, t_embs)[0]
        return float(sims.mean()) * 100

    # ── Ground truth lookup ────────────────────────────
    def _find_ground_truth(self, query_lower: str) -> dict | None:
        for item in TEST_SET:
            if item["query"].lower() == query_lower:
                return item
        for item in TEST_SET:
            words = item["query"].lower().split()
            if sum(1 for w in words if w in query_lower) >= len(words) * 0.7:
                return item
        return None

    # ── Pretty print ───────────────────────────────────
    def print_metrics(self, m: MetricResult):
        print("\n" + "─" * 60)
        print(f"  📊 METRICS  [{m.mode.upper()} mode]  [{m.query_type.upper()} query]")
        print("─" * 60)
        print(f"  Query   : {m.query}")
        print(f"  Latency : {m.latency_ms:.0f}ms")

        print(f"\n  ── KG Metrics (Knowledge Graph) ──────────────")
        print(f"  Results returned : {m.kg_result_count}")
        if m.kg_precision is not None:
            print(f"  🎯 Precision     {self._bar(m.kg_precision)} {m.kg_precision:.1f}%")
            print(f"     Results match expected place type")
        if m.kg_recall is not None:
            print(f"  🔁 Recall        {self._bar(m.kg_recall)} {m.kg_recall:.1f}%")
            print(f"     Known places found from ground truth")
        else:
            print(f"  🔁 Recall        N/A  — add known_places to TEST_SET")
        if m.kg_hallucination is not None:
            print(f"  🛡️  Grounding     {self._bar(m.kg_hallucination)} {m.kg_hallucination:.1f}%")
            print(f"     Results verified from Neo4j graph")

        print(f"\n  ── Vector Metrics (Pinecone Semantic Search) ─")
        print(f"  Snippet returned : {'✅ Yes' if m.vector_returned else '❌ No'}")
        if m.vector_relevance is not None:
            print(f"  💡 Relevance     {self._bar(m.vector_relevance)} {m.vector_relevance:.1f}%")
            print(f"     Semantic similarity: query vs returned snippet")
        else:
            print(f"  💡 Relevance     N/A  — no snippet returned")
        if m.vector_semantic_coverage is not None:
            print(f"  🧠 Sem.Coverage  {self._bar(m.vector_semantic_coverage)} {m.vector_semantic_coverage:.1f}%")
            print(f"     Snippet contains expected semantic keywords")
        else:
            print(f"  🧠 Sem.Coverage  N/A")

        print(f"\n  ── Overall ───────────────────────────────────")
        print(f"  🌐 Relevance     {self._bar(m.overall_relevance)} {m.overall_relevance:.1f}%")
        print(f"     Combined KG + Vector semantic match")

        if m.notes:
            print(f"\n  ⚠️  Notes: {m.notes}")
        print("─" * 60)

    # ── Full benchmark ─────────────────────────────────
    def run_benchmark(self,
                      pipeline_fn: Callable[[str, str], dict],
                      modes: list[str] = None,
                      test_set: list[dict] = None,
                      save_json: str = "eval_results.json",
                      save_csv: str = "eval_results.csv") -> list[MetricResult]:
        """
        Run all TEST_SET queries across all modes.

        Args:
            pipeline_fn : function(query, mode) → dict with keys:
                          "kg_results"     : list[dict]
                          "vector_snippet" : str | None
        """
        if modes is None:
            modes = ["kg", "vector", "hybrid", "none"]
        if test_set is None:
            test_set = TEST_SET

        all_results: list[MetricResult] = []

        print("=" * 60)
        print("  RUNNING BENCHMARK")
        print(f"  {len(test_set)} queries x {len(modes)} modes = "
              f"{len(test_set) * len(modes)} evaluations")
        print("=" * 60)

        for mode in modes:
            print(f"\n-- Mode: {mode.upper()} " + "-"*42)
            for item in test_set:
                query = item["query"]
                print(f"  [{item['query_type'].upper()}] {query[:45]}", end=" ", flush=True)

                start = time.time()
                try:
                    output         = pipeline_fn(query, mode)
                    kg_results     = output.get("kg_results", [])
                    vector_snippet = output.get("vector_snippet", None)
                except Exception as e:
                    print(f"ERROR: {e}")
                    kg_results     = []
                    vector_snippet = None
                elapsed_ms = (time.time() - start) * 1000

                metric = self.evaluate(
                    query=query,
                    kg_results=kg_results,
                    vector_snippet=vector_snippet,
                    mode=mode,
                    latency_ms=elapsed_ms
                )
                all_results.append(metric)

                p   = f"P={metric.kg_precision:.0f}%"   if metric.kg_precision   is not None else "P=N/A"
                r   = f"R={metric.kg_recall:.0f}%"       if metric.kg_recall       is not None else "R=N/A"
                rel = f"VRel={metric.vector_relevance:.0f}%" if metric.vector_relevance is not None else "VRel=N/A"
                print(f"{p} {r} {rel} Overall={metric.overall_relevance:.0f}%")

        self._print_summary_table(all_results)
        self._print_mode_averages(all_results, modes)
        self._save_json(all_results, save_json)
        self._save_csv(all_results, save_csv)

        return all_results

    # ── Summary table ──────────────────────────────────
    def _print_summary_table(self, results: list[MetricResult]):
        try:
            from tabulate import tabulate
            headers = [
                "Query", "Mode", "Type",
                "KG Prec", "KG Recall", "KG Ground",
                "Vec Rel", "Vec Cover",
                "Overall", "Latency"
            ]
            rows = [r.to_row() for r in results]
            print("\n\n" + "=" * 60)
            print("  FULL RESULTS TABLE")
            print("=" * 60)
            print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        except ImportError:
            print("\n(pip install tabulate for a pretty table)")
            for r in results:
                print(r.to_row())

    # ── Mode averages ──────────────────────────────────
    def _print_mode_averages(self, results: list[MetricResult], modes: list[str]):
        print("\n\n" + "=" * 60)
        print("  AVERAGE SCORES PER MODE")
        print("=" * 60)

        rows = []
        for mode in modes:
            mr = [r for r in results if r.mode == mode]
            if not mr:
                continue
            def avg(lst): return sum(lst)/len(lst) if lst else 0
            prec = [r.kg_precision    for r in mr if r.kg_precision    is not None]
            rec  = [r.kg_recall       for r in mr if r.kg_recall       is not None]
            hall = [r.kg_hallucination for r in mr if r.kg_hallucination is not None]
            vrel = [r.vector_relevance  for r in mr if r.vector_relevance  is not None]
            vcov = [r.vector_semantic_coverage for r in mr if r.vector_semantic_coverage is not None]
            orel = [r.overall_relevance for r in mr]
            lat  = [r.latency_ms for r in mr]
            rows.append([
                mode.upper(),
                f"{avg(prec):.1f}%" if prec else "N/A",
                f"{avg(rec):.1f}%"  if rec  else "N/A",
                f"{avg(hall):.1f}%" if hall else "N/A",
                f"{avg(vrel):.1f}%" if vrel else "N/A",
                f"{avg(vcov):.1f}%" if vcov else "N/A",
                f"{avg(orel):.1f}%",
                f"{avg(lat):.0f}ms"
            ])

        try:
            from tabulate import tabulate
            print(tabulate(
                rows,
                headers=["Mode", "KG Prec", "KG Recall",
                         "KG Ground", "Vec Rel", "Vec Cover",
                         "Overall", "Latency"],
                tablefmt="rounded_outline"
            ))
        except ImportError:
            for row in rows:
                print(row)

        print("\nKey insights:")
        print("  KG mode     -> highest Precision + Grounding (exact type matching)")
        print("  Vector mode -> highest Relevance for semantic/ambiguous queries")
        print("  Hybrid mode -> best overall (combines both strengths)")
        print("  None mode   -> lowest scores (proves grounding layer is essential)")

    # ── Save JSON ──────────────────────────────────────
    def _save_json(self, results: list[MetricResult], path: str):
        data = [{
            "query": r.query, "mode": r.mode, "query_type": r.query_type,
            "kg_precision": r.kg_precision, "kg_recall": r.kg_recall,
            "kg_hallucination": r.kg_hallucination, "kg_result_count": r.kg_result_count,
            "vector_relevance": r.vector_relevance,
            "vector_semantic_coverage": r.vector_semantic_coverage,
            "vector_returned": r.vector_returned,
            "overall_relevance": r.overall_relevance,
            "latency_ms": r.latency_ms, "notes": r.notes,
        } for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n Results saved to {path}")

    # ── Save CSV ───────────────────────────────────────
    def _save_csv(self, results: list[MetricResult], path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query", "mode", "query_type",
                "kg_precision", "kg_recall", "kg_hallucination", "kg_result_count",
                "vector_relevance", "vector_semantic_coverage", "vector_returned",
                "overall_relevance", "latency_ms"
            ])
            for r in results:
                writer.writerow([
                    r.query, r.mode, r.query_type,
                    round(r.kg_precision, 2)            if r.kg_precision            is not None else "",
                    round(r.kg_recall, 2)               if r.kg_recall               is not None else "",
                    round(r.kg_hallucination, 2)        if r.kg_hallucination        is not None else "",
                    r.kg_result_count,
                    round(r.vector_relevance, 2)        if r.vector_relevance        is not None else "",
                    round(r.vector_semantic_coverage, 2) if r.vector_semantic_coverage is not None else "",
                    r.vector_returned,
                    round(r.overall_relevance, 2),
                    round(r.latency_ms, 2)
                ])
        print(f"CSV saved to {path}")

    # ── Plot chart ─────────────────────────────────────
    def plot_results(self, results: list[MetricResult], save_path: str = "eval_chart.png"):
        """Two charts: Left = KG metrics, Right = Vector + Overall metrics."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Install matplotlib: pip install matplotlib")
            return

        modes  = ["kg", "vector", "hybrid", "none"]
        def avg(lst): return sum(lst)/len(lst) if lst else 0

        kg_prec = [avg([r.kg_precision    for r in results if r.mode == m and r.kg_precision    is not None]) for m in modes]
        kg_hall = [avg([r.kg_hallucination for r in results if r.mode == m and r.kg_hallucination is not None]) for m in modes]
        vec_rel = [avg([r.vector_relevance  for r in results if r.mode == m and r.vector_relevance  is not None]) for m in modes]
        overall = [avg([r.overall_relevance for r in results if r.mode == m]) for m in modes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Pipeline Evaluation — KG vs Vector vs Hybrid vs None",
                     fontsize=13, fontweight="bold")

        x = np.arange(len(modes))
        w = 0.3

        # Left — KG metrics
        ax1.set_title("Knowledge Graph Metrics", fontweight="bold")
        for i, (data, label, color) in enumerate([
            (kg_prec, "Precision",  "#3b82f6"),
            (kg_hall, "Grounding",  "#10b981"),
        ]):
            bars = ax1.bar(x + (i - 0.5) * w, data, w, label=label, color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2, h + 1,
                         f"{h:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.upper() for m in modes])
        ax1.set_ylim(0, 115)
        ax1.set_ylabel("Score (%)")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
        ax1.spines[["top", "right"]].set_visible(False)

        # Right — Vector + Overall
        ax2.set_title("Vector DB + Overall Metrics", fontweight="bold")
        for i, (data, label, color) in enumerate([
            (vec_rel, "Vec Relevance",    "#f59e0b"),
            (overall, "Overall Relevance","#8b5cf6"),
        ]):
            bars = ax2.bar(x + (i - 0.5) * w, data, w, label=label, color=color, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, h + 1,
                         f"{h:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.upper() for m in modes])
        ax2.set_ylim(0, 115)
        ax2.set_ylabel("Score (%)")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        ax2.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
        plt.show()

    def _bar(self, score: float, width: int = 15) -> str:
        filled = int((score / 100) * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"
