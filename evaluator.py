"""
evaluator.py — Karnataka Tourism Pipeline Evaluator
=====================================================
Tests KG, Vector DB, and Hybrid performance separately.

Test data built from:
  - KG   : Neo4j export (neo4j_query_table_data_2026-3-16.csv)
  - Vector: vectordb_dataset.json (200 records)

Metrics:
  KG     → Precision, Recall, Grounding
  Vector → Relevance Score, Semantic Coverage
  Hybrid → All combined

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


# ═══════════════════════════════════════════════════════════
# KG TEST SET
# Built from your Neo4j database export
# known_places = exact names confirmed in Neo4j
# ═══════════════════════════════════════════════════════════
KG_TEST_SET = [
    {
        "query": "beaches in Mangalore",
        "expected_types": ["Beach", "Beach and Temple"],
        "expected_city": "Mangalore",
        "known_places": [
            "Thannirbavi Tree Park Beach",
            "Bengre Beach",
            "Kulai Beach",
            "Tannirbhavi Beach",
            "Shashitlu Beach",
            "Surathkal Beach"
        ]
    },
    {
        "query": "trekking places in Karnataka",
        "expected_types": ["Trekking Destination", "Hill Station"],
        "expected_city": None,
        "known_places": [
            "Kodikallu Gudda",
            "Skandagiri",
            "Mullayanagiri",
            "Kodachadri",
            "Kudremukh Peak",
            "Tadiandamol",
            "Kurunjimalai"
        ]
    },
    {
        "query": "waterfalls in Karnataka",
        "expected_types": ["Waterfall"],
        "expected_city": None,
        "known_places": [
            "Ermayi Falls",
            "Jomlu Theertha",
            "Iruppu Falls",
            "Didupe Falls",
            "Gokak Falls",
            "Kudlu Theertha Falls",
            "Hebbe Falls",
            "Abbey Falls"
        ]
    },
    {
        "query": "temples in Karnataka",
        "expected_types": ["Temple"],
        "expected_city": None,
        "known_places": [
            "Chennakesava Temple",
            "Dharmasthala",
            "Hoysaleswara Temple",
            "Hattiyangadi Siddhivinayaka Temple",
            "Chaturmukha Basadi",
            "Anegudde Vinayaka Temple"
        ]
    },
    {
        "query": "historical places in Karnataka",
        "expected_types": [
            "Historical Site", "Historical Fort",
            "Historical Monument", "Monument"
        ],
        "expected_city": None,
        "known_places": [
            "Badami Cave Temples",
            "Barkur Fort",
            "Bekal Fort",
            "Bidar Fort",
            "Hampi",
            "Daria Bahadurgad Fort",
            "Karkala Gomateshwara Statue"
        ]
    },
    {
        "query": "national parks in Karnataka",
        "expected_types": ["National Park"],
        "expected_city": None,
        "known_places": [
            "Bandipur National Park",
            "Bannerghatta National Park",
            "Kudremukh National Park"
        ]
    },
    {
        "query": "trekking in Chikkamagaluru",
        "expected_types": ["Trekking Destination", "Hill Station"],
        "expected_city": "Chikkamagaluru",
        "known_places": [
            "Mullayanagiri",
            "Baba Budangiri"
        ]
    },
    {
        "query": "waterfalls near Madikeri",
        "expected_types": ["Waterfall"],
        "expected_city": "Madikeri",
        "known_places": ["Abbey Falls"]
    },
    {
        "query": "historical places in Hampi",
        "expected_types": [
            "Historical Site", "Historical Fort",
            "Historical Monument", "Heritage Site", "Monument", "Palace"
        ],
        "expected_city": "Hampi",
        "known_places": ["Hampi"]
    },
    {
        "query": "churches in Karnataka",
        "expected_types": ["Church"],
        "expected_city": None,
        "known_places": ["St. Lawrence Basilica"]
    },
]


# ═══════════════════════════════════════════════════════════
# VECTOR DB TEST SET
# Built from vectordb_dataset.json (200 records)
# Tests semantic retrieval — exact + ambiguous queries
# known_places = exact names confirmed in Vector DB
# semantic_keywords = words that should appear in returned text
# ═══════════════════════════════════════════════════════════
VECTOR_TEST_SET = [

    # ── Exact queries — Vector DB should return correct places ──
    {
        "query": "beaches in Mangalore",
        "query_type": "exact",
        "known_places": [
            "Panambur Beach",
            "Tannirbhavi Beach",
            "Someshwara Beach",
            "Ullal Beach",
            "Sasihithlu Beach"
        ],
        "semantic_keywords": ["beach", "sea", "shore", "coastal", "sand", "water"]
    },
    {
        "query": "beaches in Udupi",
        "query_type": "exact",
        "known_places": [
            "Malpe Beach",
            "Kapu Beach",
            "Padubidri Blue Flag Beach"
        ],
        "semantic_keywords": ["beach", "sea", "shore", "udupi", "coastal"]
    },
    {
        "query": "temples in Udupi",
        "query_type": "exact",
        "known_places": [
            "Udupi Sri Krishna Temple",
            "Anantheshwara Temple",
            "Bada Yermal Temple",
            "Chandramouleshwara Temple"
        ],
        "semantic_keywords": ["temple", "worship", "deity", "religious", "krishna"]
    },
    {
        "query": "restaurants in Mangalore",
        "query_type": "exact",
        "known_places": [
            "Machali Restaurant",
            "Gajalee Seafood Restaurant",
            "Pallkhi Restaurant",
            "New Taj Mahal Cafe",
            "Shetty Lunch Home"
        ],
        "semantic_keywords": ["restaurant", "food", "eat", "seafood", "cuisine", "cafe"]
    },
    {
        "query": "heritage places in Mangalore",
        "query_type": "exact",
        "known_places": [
            "Sultan Battery",
            "St Aloysius Chapel",
            "Pilikula Heritage Village"
        ],
        "semantic_keywords": ["heritage", "history", "historic", "fort", "chapel", "colonial"]
    },
    {
        "query": "viewpoints near Mangalore",
        "query_type": "exact",
        "known_places": [
            "Bajpe Airport Viewpoint",
            "Netravati River Bridge",
            "Mangalore Lighthouse Hill",
            "Mangalore Harbor Sunset"
        ],
        "semantic_keywords": ["view", "viewpoint", "scenic", "panoramic", "sunset", "hill"]
    },
    {
        "query": "water activities in Mangalore",
        "query_type": "exact",
        "known_places": [
            "Panambur Water Sports",
            "Sultan Battery Ferry",
            "Mangalore Harbor Cruise",
            "Netravati River Boating",
            "Pilikula Lake Kayaking"
        ],
        "semantic_keywords": ["water", "sport", "boat", "ferry", "kayak", "river", "cruise"]
    },
    {
        "query": "markets in Udupi",
        "query_type": "exact",
        "known_places": [
            "Udupi Anantheshwara Temple Street Market",
            "Udupi Temple Street",
            "Car Street Udupi",
            "Udupi Night Market"
        ],
        "semantic_keywords": ["market", "shop", "street", "bazaar", "local"]
    },

    # ── Semantic queries — Vector DB understanding tested ──
    {
        "query": "peaceful places to relax in Mangalore",
        "query_type": "semantic",
        "known_places": [
            "Tannirbhavi Beach",
            "Pilikula Lake",
            "Mangalore Marina Walk"
        ],
        "semantic_keywords": ["peaceful", "calm", "relax", "quiet", "serene", "nature"]
    },
    {
        "query": "adventure activities near Udupi",
        "query_type": "semantic",
        "known_places": [
            "Malpe Beach",
            "Kapu Beach"
        ],
        "semantic_keywords": ["adventure", "sport", "activity", "trek", "water sports", "exciting"]
    },
    {
        "query": "nature and wildlife in Karnataka",
        "query_type": "semantic",
        "known_places": [
            "Kudremukh National Park",
            "Sammilan Shetty Butterfly Park",
            "Western Ghats Bird Watching",
            "Agumbe Rainforest Research Station"
        ],
        "semantic_keywords": ["nature", "wildlife", "forest", "bird", "butterfly", "rainforest"]
    },
    {
        "query": "sunset spots near Udupi and Mangalore",
        "query_type": "semantic",
        "known_places": [
            "Malpe Beach Sunset Point",
            "Agumbe Sunset Viewpoint",
            "Kodi Backwater Sunset",
            "Mulki River Sunset",
            "Mangalore Harbor Sunset"
        ],
        "semantic_keywords": ["sunset", "view", "evening", "golden", "horizon", "scenic"]
    },
    {
        "query": "local food experience in Mangalore",
        "query_type": "semantic",
        "known_places": [
            "Mangalore Fish Market",
            "Shetty Lunch Home",
            "New Taj Mahal Cafe",
            "Ideal Ice Cream Parlour"
        ],
        "semantic_keywords": ["food", "local", "cuisine", "eat", "taste", "traditional", "fish"]
    },
    {
        "query": "religious and spiritual places in Udupi",
        "query_type": "semantic",
        "known_places": [
            "Udupi Sri Krishna Temple",
            "Anantheshwara Temple",
            "Udupi Temple Chariot Storage"
        ],
        "semantic_keywords": ["temple", "religious", "spiritual", "worship", "devotion", "pilgrimage"]
    },
]


# ═══════════════════════════════════════════════════════════
# METRIC RESULTS
# ═══════════════════════════════════════════════════════════
@dataclass
class KGMetricResult:
    query: str
    known_total: int
    returned_count: int
    precision: float
    recall: float
    grounding: float
    relevance: float
    latency_ms: float
    found_places: list
    missed_places: list

    def to_row(self):
        return [
            self.query[:38] + ("..." if len(self.query) > 38 else ""),
            f"{self.precision:.1f}%",
            f"{self.recall:.1f}% ({len(self.found_places)}/{self.known_total})",
            f"{self.grounding:.1f}%",
            f"{self.relevance:.1f}%",
            self.returned_count,
            f"{self.latency_ms:.0f}ms"
        ]


@dataclass
class VectorMetricResult:
    query: str
    query_type: str
    known_total: int
    vector_relevance: float | None
    semantic_coverage: float | None
    snippet_returned: bool
    found_in_snippet: list
    latency_ms: float

    def to_row(self):
        return [
            self.query[:38] + ("..." if len(self.query) > 38 else ""),
            self.query_type,
            f"{self.vector_relevance:.1f}%" if self.vector_relevance is not None else "N/A",
            f"{self.semantic_coverage:.1f}%" if self.semantic_coverage is not None else "N/A",
            "✅" if self.snippet_returned else "❌",
            f"{len(self.found_in_snippet)}/{self.known_total}",
            f"{self.latency_ms:.0f}ms"
        ]


# ═══════════════════════════════════════════════════════════
# EVALUATOR
# ═══════════════════════════════════════════════════════════
class Evaluator:
    """
    Evaluates KG and Vector DB separately.

    KG   pipeline_fn : function(query) → list[dict]
                       Each dict: {name, type, city, description}

    Vector pipeline_fn: function(query) → str | None
                        Returns the text snippet from Pinecone after reranking
    """

    def __init__(self):
        print("Loading evaluation model (MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Evaluator ready.\n")

    # ── Helpers ───────────────────────────────────────
    def _name(self, r): return (r.get("name") or r.get("p.name") or "").strip()
    def _type(self, r): return (r.get("type") or r.get("p.type") or "").strip()
    def _desc(self, r): return (r.get("description") or r.get("p.description") or r.get("text") or "").strip()

    def _normalize(self, text: str) -> str:
        """
        Normalize text for fuzzy matching.
        Handles plural/singular, case, extra spaces.
        e.g. "Beaches" → "beach", "Historical Sites" → "historical site"
        """
        import re
        t = text.lower().strip()
        t = re.sub(r"\s+", " ", t)
        # Remove common suffixes for plural/singular matching
        suffixes = ["ies", "es", "s"]
        for suf in suffixes:
            if t.endswith(suf) and len(t) > len(suf) + 2:
                t = t[:-len(suf)]
                break
        return t

    def _types_match(self, result_type: str, expected_types: list[str]) -> bool:
        """
        Check if result type matches any expected type using fuzzy matching.
        Handles: case, plural/singular, partial match.
        e.g. "Beach" matches ["Beach", "Beaches", "beach and temple"]
        """
        if not result_type or not expected_types:
            return False
        rt_norm = self._normalize(result_type)
        for et in expected_types:
            et_norm = self._normalize(et)
            # Exact normalized match
            if rt_norm == et_norm:
                return True
            # Partial match — one contains the other
            if rt_norm in et_norm or et_norm in rt_norm:
                return True
        return False

    def _names_match(self, result_name: str, known_name: str) -> bool:
        """
        Check if result name matches known place name using fuzzy matching.
        Handles: case, extra spaces, minor spelling differences.
        e.g. "Tannirbhavi Beach" matches "Tannirbhavi beach"
        """
        import re
        def clean(s):
            return re.sub(r"\s+", " ", s.lower().strip())
        rn = clean(result_name)
        kn = clean(known_name)
        if rn == kn:
            return True
        # One contains the other (handles prefix/suffix differences)
        if rn in kn or kn in rn:
            return True
        # Remove common words and compare
        stopwords = ["the", "a", "an", "and", "or", "of", "in", "at", "near"]
        rn_words = [w for w in rn.split() if w not in stopwords]
        kn_words = [w for w in kn.split() if w not in stopwords]
        # If 70%+ words match
        if rn_words and kn_words:
            common = sum(1 for w in rn_words if w in kn_words)
            if common / max(len(rn_words), len(kn_words)) >= 0.7:
                return True
        return False

    # ── KG: evaluate single query ──────────────────────
    def evaluate_kg(self,
                    query: str,
                    kg_results: list[dict],
                    latency_ms: float = 0.0) -> KGMetricResult:

        tc = self._find_kg_test_case(query)
        expected_types = tc["expected_types"] if tc else []
        known_places   = tc["known_places"]   if tc else []

        # Precision — uses fuzzy type matching
        # handles "beaches" vs "Beach", "historical" vs "Historical Site" etc.
        if kg_results and expected_types:
            correct   = [r for r in kg_results
                         if self._types_match(self._type(r), expected_types)]
            precision = len(correct) / len(kg_results) * 100
        else:
            precision = 100.0 if kg_results else 0.0

        # Recall — uses fuzzy name matching
        # handles "Tannirbhavi Beach" vs "Tannirbhavi beach", minor spelling diffs
        found_places  = [kp for kp in known_places
                         if any(self._names_match(self._name(r), kp) for r in kg_results)]
        missed_places = [kp for kp in known_places
                         if not any(self._names_match(self._name(r), kp) for r in kg_results)]
        recall = len(found_places) / len(known_places) * 100 if known_places else 0.0

        # Grounding
        if kg_results:
            grounded  = sum(1 for r in kg_results if self._name(r) and self._type(r))
            grounding = grounded / len(kg_results) * 100
        else:
            grounding = 100.0

        # Relevance
        texts = [self._desc(r) or self._name(r) for r in kg_results if (self._desc(r) or self._name(r))]
        if texts:
            q_emb  = self.embedder.encode(query, convert_to_tensor=True)
            t_embs = self.embedder.encode(texts, convert_to_tensor=True)
            relevance = float(util.cos_sim(q_emb, t_embs)[0].mean()) * 100
        else:
            relevance = 0.0

        return KGMetricResult(
            query=query,
            known_total=len(known_places),
            returned_count=len(kg_results),
            precision=round(precision, 1),
            recall=round(recall, 1),
            grounding=round(grounding, 1),
            relevance=round(relevance, 1),
            latency_ms=round(latency_ms, 1),
            found_places=found_places,
            missed_places=missed_places
        )

    # ── Vector: evaluate single query ─────────────────
    def evaluate_vector(self,
                        query: str,
                        vector_snippet: str | None,
                        latency_ms: float = 0.0) -> VectorMetricResult:

        tc = self._find_vector_test_case(query)
        known_places     = tc["known_places"]      if tc else []
        semantic_keywords = tc["semantic_keywords"] if tc else []
        query_type       = tc["query_type"]         if tc else "exact"

        snippet_returned = bool(vector_snippet and vector_snippet.strip())

        # Relevance — semantic similarity query vs snippet
        if snippet_returned:
            q_emb = self.embedder.encode(query,          convert_to_tensor=True)
            s_emb = self.embedder.encode(vector_snippet, convert_to_tensor=True)
            vector_relevance = round(float(util.cos_sim(q_emb, s_emb)[0][0]) * 100, 1)
        else:
            vector_relevance = None

        # Semantic coverage — keywords in snippet
        if snippet_returned and semantic_keywords:
            sl = vector_snippet.lower()
            matched = sum(1 for kw in semantic_keywords if kw.lower() in sl)
            semantic_coverage = round(matched / len(semantic_keywords) * 100, 1)
        else:
            semantic_coverage = None

        # Found known places in snippet — fuzzy match
        # handles "Tannirbhavi Beach" vs "tannirbhavi beach" in snippet text
        found_in_snippet = []
        if snippet_returned and known_places:
            import re
            sl = re.sub(r"\s+", " ", vector_snippet.lower())
            for kp in known_places:
                kp_norm = re.sub(r"\s+", " ", kp.lower())
                # Direct match or key words match
                words = [w for w in kp_norm.split() if len(w) > 3]
                if kp_norm in sl or (words and sum(1 for w in words if w in sl) >= len(words) * 0.7):
                    found_in_snippet.append(kp)

        return VectorMetricResult(
            query=query,
            query_type=query_type,
            known_total=len(known_places),
            vector_relevance=vector_relevance,
            semantic_coverage=semantic_coverage,
            snippet_returned=snippet_returned,
            found_in_snippet=found_in_snippet,
            latency_ms=round(latency_ms, 1)
        )

    # ── KG Benchmark ──────────────────────────────────
    def run_kg_benchmark(self,
                         kg_pipeline_fn: Callable[[str], list[dict]],
                         save_json: str = "kg_eval_results.json",
                         save_csv:  str = "kg_eval_results.csv") -> list[KGMetricResult]:
        """
        Run all KG_TEST_SET queries.

        Args:
            kg_pipeline_fn: function(query) → list[dict]
                            Returns KG results with name, type, city, description.

        Example:
            def my_kg_fn(query):
                # your TYPE_MAP + Cypher logic here
                return run_query(cypher, params)

            evaluator.run_kg_benchmark(kg_pipeline_fn=my_kg_fn)
        """
        results = []

        print("=" * 62)
        print("  KG BENCHMARK")
        print(f"  {len(KG_TEST_SET)} queries | Ground truth from Neo4j export")
        print("=" * 62)

        for tc in KG_TEST_SET:
            query = tc["query"]
            known = len(tc["known_places"])
            print(f"\n  Query : {query}")
            print(f"  Known : {known} places")

            start = time.time()
            try:
                kg_results = kg_pipeline_fn(query)
            except Exception as e:
                print(f"  ERROR: {e}")
                kg_results = []
            elapsed = (time.time() - start) * 1000

            m = self.evaluate_kg(query, kg_results, elapsed)
            results.append(m)
            print(f"  → Returned={m.returned_count} P={m.precision:.0f}% "
                  f"R={m.recall:.0f}% ({len(m.found_places)}/{known}) "
                  f"Ground={m.grounding:.0f}%")

        self._print_kg_table(results)
        self._print_kg_averages(results)
        self._save_json([self._kg_to_dict(r) for r in results], save_json)
        self._save_kg_csv(results, save_csv)
        return results

    # ── Vector Benchmark ──────────────────────────────
    def run_vector_benchmark(self,
                             vector_pipeline_fn: Callable[[str], str | None],
                             save_json: str = "vector_eval_results.json",
                             save_csv:  str = "vector_eval_results.csv") -> list[VectorMetricResult]:
        """
        Run all VECTOR_TEST_SET queries.

        Args:
            vector_pipeline_fn: function(query) → str | None
                                 Returns text snippet from Pinecone after reranking.

        Example:
            def my_vector_fn(query):
                candidates, _ = fetch_top_vectordb(n=10, query=query)
                if candidates:
                    reranked, _ = rerank_top_cross_encoder(query=query, candidates=candidates, top_n=1)
                    return reranked[0].text if reranked else None
                return None

            evaluator.run_vector_benchmark(vector_pipeline_fn=my_vector_fn)
        """
        results = []

        print("=" * 62)
        print("  VECTOR DB BENCHMARK")
        print(f"  {len(VECTOR_TEST_SET)} queries | Ground truth from vectordb_dataset.json")
        print("=" * 62)

        for tc in VECTOR_TEST_SET:
            query = tc["query"]
            known = len(tc["known_places"])
            print(f"\n  [{tc['query_type'].upper()}] {query}")
            print(f"  Known: {known} places")

            start = time.time()
            try:
                snippet = vector_pipeline_fn(query)
            except Exception as e:
                print(f"  ERROR: {e}")
                snippet = None
            elapsed = (time.time() - start) * 1000

            m = self.evaluate_vector(query, snippet, elapsed)
            results.append(m)

            rel = f"Rel={m.vector_relevance:.0f}%" if m.vector_relevance else "Rel=N/A"
            cov = f"Cov={m.semantic_coverage:.0f}%" if m.semantic_coverage else "Cov=N/A"
            print(f"  → Snippet={'YES' if m.snippet_returned else 'NO'} "
                  f"{rel} {cov} Found={len(m.found_in_snippet)}/{known}")

        self._print_vector_table(results)
        self._print_vector_averages(results)
        self._save_json([self._vec_to_dict(r) for r in results], save_json)
        self._save_vector_csv(results, save_csv)
        return results

    # ── Print single KG result ─────────────────────────
    def print_kg_metrics(self, m: KGMetricResult):
        print("\n" + "─" * 60)
        print(f"  📊 KG METRICS")
        print("─" * 60)
        print(f"  Query      : {m.query}")
        print(f"  Known      : {m.known_total} | Returned: {m.returned_count} | {m.latency_ms:.0f}ms")
        print(f"\n  🎯 Precision  {self._bar(m.precision)} {m.precision:.1f}%")
        print(f"  🔁 Recall     {self._bar(m.recall)} {m.recall:.1f}% ({len(m.found_places)}/{m.known_total})")
        if m.found_places:
            print(f"     ✅ Found : {', '.join(m.found_places)}")
        if m.missed_places:
            print(f"     ❌ Missed: {', '.join(m.missed_places)}")
        print(f"  🛡️  Grounding  {self._bar(m.grounding)} {m.grounding:.1f}%")
        print(f"  💡 Relevance  {self._bar(m.relevance)} {m.relevance:.1f}%")
        print("─" * 60)

    # ── Print single Vector result ─────────────────────
    def print_vector_metrics(self, m: VectorMetricResult):
        print("\n" + "─" * 60)
        print(f"  📊 VECTOR METRICS [{m.query_type.upper()}]")
        print("─" * 60)
        print(f"  Query    : {m.query}")
        print(f"  Snippet  : {'✅ Returned' if m.snippet_returned else '❌ Not returned'}")
        if m.vector_relevance is not None:
            print(f"\n  💡 Relevance      {self._bar(m.vector_relevance)} {m.vector_relevance:.1f}%")
        if m.semantic_coverage is not None:
            print(f"  🧠 Sem. Coverage  {self._bar(m.semantic_coverage)} {m.semantic_coverage:.1f}%")
        if m.found_in_snippet:
            print(f"  ✅ Places in snippet: {', '.join(m.found_in_snippet)}")
        print("─" * 60)

    # ── KG table ───────────────────────────────────────
    def _print_kg_table(self, results: list[KGMetricResult]):
        try:
            from tabulate import tabulate
            headers = ["Query", "Precision", "Recall", "Grounding", "Relevance", "Returned", "Latency"]
            print("\n\n" + "=" * 62)
            print("  KG FULL RESULTS TABLE")
            print("=" * 62)
            print(tabulate([r.to_row() for r in results], headers=headers, tablefmt="rounded_outline"))
        except ImportError:
            for r in results:
                print(r.to_row())

    # ── KG averages ────────────────────────────────────
    def _print_kg_averages(self, results: list[KGMetricResult]):
        if not results: return
        avg_p   = sum(r.precision  for r in results) / len(results)
        avg_r   = sum(r.recall     for r in results) / len(results)
        avg_g   = sum(r.grounding  for r in results) / len(results)
        avg_rel = sum(r.relevance  for r in results) / len(results)
        avg_lat = sum(r.latency_ms for r in results) / len(results)
        total_known  = sum(r.known_total        for r in results)
        total_found  = sum(len(r.found_places)  for r in results)
        total_missed = sum(len(r.missed_places) for r in results)

        print("\n\n" + "=" * 62)
        print("  KG OVERALL SUMMARY")
        print("=" * 62)
        print(f"  Test queries    : {len(results)}")
        print(f"  Total known     : {total_known}")
        print(f"  Total found     : {total_found}")
        print(f"  Total missed    : {total_missed}")
        print()
        print(f"  🎯 Avg Precision  {self._bar(avg_p)} {avg_p:.1f}%")
        print(f"  🔁 Avg Recall     {self._bar(avg_r)} {avg_r:.1f}%")
        print(f"  🛡️  Avg Grounding  {self._bar(avg_g)} {avg_g:.1f}%")
        print(f"  💡 Avg Relevance  {self._bar(avg_rel)} {avg_rel:.1f}%")
        print(f"  ⚡ Avg Latency    {avg_lat:.0f}ms")
        print("=" * 62)

    # ── Vector table ───────────────────────────────────
    def _print_vector_table(self, results: list[VectorMetricResult]):
        try:
            from tabulate import tabulate
            headers = ["Query", "Type", "Vec Relevance", "Sem Coverage", "Snippet", "Found", "Latency"]
            print("\n\n" + "=" * 62)
            print("  VECTOR DB FULL RESULTS TABLE")
            print("=" * 62)
            print(tabulate([r.to_row() for r in results], headers=headers, tablefmt="rounded_outline"))
        except ImportError:
            for r in results:
                print(r.to_row())

    # ── Vector averages ────────────────────────────────
    def _print_vector_averages(self, results: list[VectorMetricResult]):
        if not results: return
        rel = [r.vector_relevance  for r in results if r.vector_relevance  is not None]
        cov = [r.semantic_coverage for r in results if r.semantic_coverage is not None]
        lat = [r.latency_ms        for r in results]
        returned = sum(1 for r in results if r.snippet_returned)

        avg_rel = sum(rel)/len(rel) if rel else 0
        avg_cov = sum(cov)/len(cov) if cov else 0
        avg_lat = sum(lat)/len(lat) if lat else 0

        print("\n\n" + "=" * 62)
        print("  VECTOR DB OVERALL SUMMARY")
        print("=" * 62)
        print(f"  Test queries     : {len(results)}")
        print(f"  Snippets returned: {returned}/{len(results)}")
        print()
        print(f"  💡 Avg Relevance      {self._bar(avg_rel)} {avg_rel:.1f}%")
        print(f"  🧠 Avg Sem. Coverage  {self._bar(avg_cov)} {avg_cov:.1f}%")
        print(f"  ⚡ Avg Latency        {avg_lat:.0f}ms")
        print("=" * 62)

    # ── Plot KG chart ──────────────────────────────────
    def plot_kg_results(self, results: list[KGMetricResult], save_path: str = "kg_eval_chart.png"):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("pip install matplotlib")
            return

        queries   = [r.query[:20] + "..." if len(r.query) > 20 else r.query for r in results]
        precision = [r.precision  for r in results]
        recall    = [r.recall     for r in results]
        grounding = [r.grounding  for r in results]
        relevance = [r.relevance  for r in results]

        x = np.arange(len(queries))
        w = 0.2

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.bar(x - 1.5*w, precision, w, label="Precision",  color="#3b82f6", alpha=0.85)
        ax.bar(x - 0.5*w, recall,    w, label="Recall",     color="#10b981", alpha=0.85)
        ax.bar(x + 0.5*w, grounding, w, label="Grounding",  color="#f59e0b", alpha=0.85)
        ax.bar(x + 1.5*w, relevance, w, label="Relevance",  color="#8b5cf6", alpha=0.85)

        ax.set_title("KG Pipeline Performance per Query", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(queries, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Score (%)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ KG chart saved to {save_path}")
        plt.show()

    # ── Plot Vector chart ──────────────────────────────
    def plot_vector_results(self, results: list[VectorMetricResult], save_path: str = "vector_eval_chart.png"):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("pip install matplotlib")
            return

        queries  = [r.query[:20] + "..." if len(r.query) > 20 else r.query for r in results]
        rel      = [r.vector_relevance  if r.vector_relevance  is not None else 0 for r in results]
        cov      = [r.semantic_coverage if r.semantic_coverage is not None else 0 for r in results]
        colors_r = ["#f59e0b" if r.query_type == "exact" else "#ef4444" for r in results]

        x = np.arange(len(queries))
        w = 0.35

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.bar(x - w/2, rel, w, label="Relevance",        color="#f59e0b", alpha=0.85)
        ax.bar(x + w/2, cov, w, label="Semantic Coverage", color="#10b981", alpha=0.85)

        ax.set_title("Vector DB Performance per Query\n(Orange=Exact, Red=Semantic)",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(queries, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Score (%)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ Vector chart saved to {save_path}")
        plt.show()

    # ── Ground truth lookups ───────────────────────────
    def _find_kg_test_case(self, query: str) -> dict | None:
        q = query.lower().strip()
        for tc in KG_TEST_SET:
            if tc["query"].lower() == q:
                return tc
        for tc in KG_TEST_SET:
            words = tc["query"].lower().split()
            if sum(1 for w in words if w in q) >= len(words) * 0.7:
                return tc
        return None

    def _find_vector_test_case(self, query: str) -> dict | None:
        q = query.lower().strip()
        for tc in VECTOR_TEST_SET:
            if tc["query"].lower() == q:
                return tc
        for tc in VECTOR_TEST_SET:
            words = tc["query"].lower().split()
            if sum(1 for w in words if w in q) >= len(words) * 0.7:
                return tc
        return None

    # ── Save helpers ───────────────────────────────────
    def _kg_to_dict(self, r: KGMetricResult) -> dict:
        return {
            "query": r.query, "known_total": r.known_total,
            "returned_count": r.returned_count,
            "precision": r.precision, "recall": r.recall,
            "grounding": r.grounding, "relevance": r.relevance,
            "latency_ms": r.latency_ms,
            "found_places": r.found_places, "missed_places": r.missed_places
        }

    def _vec_to_dict(self, r: VectorMetricResult) -> dict:
        return {
            "query": r.query, "query_type": r.query_type,
            "known_total": r.known_total,
            "vector_relevance": r.vector_relevance,
            "semantic_coverage": r.semantic_coverage,
            "snippet_returned": r.snippet_returned,
            "found_in_snippet": r.found_in_snippet,
            "latency_ms": r.latency_ms
        }

    def _save_json(self, data: list, path: str):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✅ Saved to {path}")

    def _save_kg_csv(self, results: list[KGMetricResult], path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "known_total", "returned_count",
                             "precision", "recall", "grounding", "relevance",
                             "latency_ms", "found_places", "missed_places"])
            for r in results:
                writer.writerow([r.query, r.known_total, r.returned_count,
                                 r.precision, r.recall, r.grounding, r.relevance,
                                 r.latency_ms,
                                 " | ".join(r.found_places),
                                 " | ".join(r.missed_places)])
        print(f"  ✅ CSV saved to {path}")

    def _save_vector_csv(self, results: list[VectorMetricResult], path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query", "query_type", "known_total",
                             "vector_relevance", "semantic_coverage",
                             "snippet_returned", "found_in_snippet", "latency_ms"])
            for r in results:
                writer.writerow([r.query, r.query_type, r.known_total,
                                 r.vector_relevance, r.semantic_coverage,
                                 r.snippet_returned,
                                 " | ".join(r.found_in_snippet),
                                 r.latency_ms])
        print(f"  ✅ CSV saved to {path}")

    def _bar(self, score: float, width: int = 15) -> str:
        filled = int((score / 100) * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"
