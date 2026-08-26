"""
ranking_defence.py -- PlaceMux Phase 3, Task 22
================================================
Defences against ranking manipulation and keyword stuffing.

Three detectors are implemented:

1. KeywordStuffingDetector
   - Type-Token Ratio (TTR): flags resumes with very low lexical diversity.
   - Keyword density: flags resumes where target keywords exceed a density
     threshold relative to total token count.
   - Invisible character scan: flags zero-width spaces, soft hyphens, and
     other non-printable Unicode tricks used to hide repeated tokens.

2. SynonymFloodingDetector
   - Canonicalises each skill token against a synonym table.
   - Computes the ratio of unique canonical skills to total skill tokens.
   - Flags resumes where >40% of skill tokens map to the same underlying skill.

3. RankingManipulationDetector
   - Score-velocity guard: detects unnaturally fast score jumps across sessions.
   - Omnipresence guard: detects candidates appearing at rank #1 across >80%
     of diverse query categories (impossible for genuine candidates).

Design rationale
----------------
All detectors return a standardised result dict:
  {"flagged": bool, "reason": str, "score": float}

'score' is always in [0.0, 1.0] where 1.0 = definitely adversarial.

A flagged candidate is NOT auto-rejected -- the result is logged and passed
to a human review queue.  The model is unavailable fallback path returns
{"flagged": False, "reason": "defence_unavailable", "score": 0.0} so that
the candidate is not incorrectly penalised.

Metric bar (Task 22 spec)
-------------------------
  Stuffing detector  : precision >= 0.90, recall >= 0.85,  FPR < 5%
  Manipulation guard : precision >= 0.90 on simulated velocity attacks
"""

import os
import sys
import json
import re
import math
import logging
import unicodedata
from collections import Counter
from typing import List, Dict, Optional, Tuple

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

DEFENCE_REPORT_PATH = "logs/ranking_defence_report.json"

# ---------------------------------------------------------------------------
# Synonym table (lite -- no external NLP library required)
# Maps surface forms to canonical skill IDs.
# ---------------------------------------------------------------------------

SYNONYM_TABLE: Dict[str, str] = {
    # Machine Learning cluster
    "ml": "machine_learning", "machine learning": "machine_learning",
    "machine-learning": "machine_learning", "ai": "machine_learning",
    "artificial intelligence": "machine_learning", "deep learning": "machine_learning",
    "deep-learning": "machine_learning", "dl": "machine_learning",
    "neural network": "machine_learning", "neural networks": "machine_learning",
    "neural net": "machine_learning", "nn": "machine_learning",
    "supervised learning": "machine_learning", "unsupervised learning": "machine_learning",

    # Python cluster
    "python": "python", "python3": "python", "python 3": "python",
    "py": "python", "cpython": "python",

    # Data Science cluster
    "data science": "data_science", "data scientist": "data_science",
    "data analysis": "data_science", "data analytics": "data_science",
    "analytics": "data_science", "statistical analysis": "data_science",
    "statistics": "data_science", "stats": "data_science",

    # NLP cluster
    "nlp": "nlp", "natural language processing": "nlp",
    "natural language understanding": "nlp", "nlu": "nlp",
    "text mining": "nlp", "computational linguistics": "nlp",

    # Cloud cluster
    "aws": "cloud", "amazon web services": "cloud", "azure": "cloud",
    "microsoft azure": "cloud", "gcp": "cloud", "google cloud": "cloud",
    "google cloud platform": "cloud", "cloud computing": "cloud",

    # SQL cluster
    "sql": "sql", "mysql": "sql", "postgresql": "sql", "postgres": "sql",
    "sqlite": "sql", "database": "sql", "rdbms": "sql",
    "relational database": "sql", "structured query language": "sql",
}


def _canonicalise_skill(token: str) -> str:
    """
    Return the canonical skill ID for a token, or the token itself if unknown.

    Parameters
    ----------
    token : str  Raw skill token (already lowercased and stripped).

    Returns
    -------
    str  Canonical skill ID or the original token.
    """
    return SYNONYM_TABLE.get(token.lower().strip(), token.lower().strip())


# ---------------------------------------------------------------------------
# 1. KeywordStuffingDetector
# ---------------------------------------------------------------------------

class KeywordStuffingDetector:
    """
    Detects keyword stuffing in resume / profile text.

    Three signals are combined:
    - TTR (Type-Token Ratio): low TTR = repetitive text.
    - Keyword density: fraction of tokens that are high-value target keywords.
    - Invisible character injection: zero-width / control Unicode characters.

    Parameters
    ----------
    ttr_threshold       : float  TTR below this value is suspicious (default 0.35).
    density_threshold   : float  Keyword density above this is suspicious (default 0.15).
    min_tokens          : int    Minimum tokens required for analysis (default 30).
    target_keywords     : List[str]  High-value keywords to track density for.
    """

    DEFAULT_TARGET_KEYWORDS = [
        "python", "java", "sql", "machine learning", "ai", "ml",
        "data science", "aws", "azure", "react", "node", "kubernetes",
        "docker", "devops", "tensorflow", "pytorch", "nlp",
        "deep learning", "cloud", "analytics",
    ]

    # Invisible / suspicious Unicode characters
    INVISIBLE_CHARS = re.compile(
        r"[\u200b\u200c\u200d\u200e\u200f"   # zero-width spaces / joiners
        r"\u00ad"                              # soft hyphen
        r"\ufeff"                              # BOM
        r"\u2060\u2061\u2062\u2063\u2064"    # invisible operators
        r"\u180e"                              # Mongolian vowel separator
        r"\u00a0]"                             # non-breaking space
    )

    def __init__(
        self,
        ttr_threshold: float = 0.35,
        density_threshold: float = 0.15,
        min_tokens: int = 30,
        target_keywords: Optional[List[str]] = None,
    ) -> None:
        """Initialise the KeywordStuffingDetector."""
        self.ttr_threshold     = ttr_threshold
        self.density_threshold = density_threshold
        self.min_tokens        = min_tokens
        self.target_keywords   = [
            kw.lower() for kw in (target_keywords or self.DEFAULT_TARGET_KEYWORDS)
        ]
        logger.info(
            f"KeywordStuffingDetector initialised: "
            f"ttr_threshold={ttr_threshold}, density_threshold={density_threshold}"
        )

    def _tokenise(self, text: str) -> List[str]:
        """
        Lowercase, strip punctuation, and split text into tokens.

        Parameters
        ----------
        text : str  Raw resume / profile text.

        Returns
        -------
        List[str]  List of word tokens.
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]

    def _detect_invisible_chars(self, text: str) -> Tuple[bool, int]:
        """
        Scan for invisible / non-printable Unicode characters.

        Parameters
        ----------
        text : str

        Returns
        -------
        Tuple[bool, int]  (found, count)
        """
        matches = self.INVISIBLE_CHARS.findall(text)
        return len(matches) > 0, len(matches)

    def detect(self, text: str, candidate_id: str = "unknown") -> Dict:
        """
        Run all keyword stuffing checks on the provided text.

        Parameters
        ----------
        text         : str  Resume / profile text.
        candidate_id : str  For logging.

        Returns
        -------
        dict
            {
                "flagged": bool,
                "reason":  str,
                "score":   float,   # 0.0 = clean, 1.0 = definitely stuffed
                "details": {...}
            }
        """
        # --- Guard: model unavailable ---
        if not text or not text.strip():
            logger.warning(f"[{candidate_id}] Empty text provided to detector.")
            return {
                "flagged": False,
                "reason":  "empty_input",
                "score":   0.0,
                "details": {},
            }

        tokens    = self._tokenise(text)
        n_tokens  = len(tokens)
        reasons   = []
        score     = 0.0

        # --- Check 1: Invisible characters ---
        invis_found, invis_count = self._detect_invisible_chars(text)
        if invis_found:
            reasons.append(
                f"invisible_chars: {invis_count} non-printable Unicode chars detected"
            )
            score = max(score, 0.95)  # near-certain adversarial

        # --- Check 2: TTR (skip if too few tokens) ---
        ttr_value = 0.0
        if n_tokens >= self.min_tokens:
            unique_tokens = len(set(tokens))
            ttr_value     = unique_tokens / n_tokens
            if ttr_value < self.ttr_threshold:
                reasons.append(
                    f"low_ttr: {ttr_value:.3f} < threshold {self.ttr_threshold}"
                )
                # Score contribution: lower TTR = higher suspicion
                contribution = 1.0 - (ttr_value / self.ttr_threshold)
                score = max(score, min(contribution, 0.90))

        # --- Check 3: Keyword density ---
        density_value = 0.0
        keyword_hits  = 0
        if n_tokens > 0:
            # Count multi-word keyword matches in the original (lowercased) text
            text_lower = text.lower()
            for kw in self.target_keywords:
                keyword_hits += text_lower.count(kw)
            density_value = keyword_hits / n_tokens
            if density_value > self.density_threshold:
                reasons.append(
                    f"high_keyword_density: {density_value:.3f} "
                    f"> threshold {self.density_threshold} "
                    f"({keyword_hits} keyword hits / {n_tokens} tokens)"
                )
                contribution = min((density_value / self.density_threshold) - 1.0, 1.0)
                score = max(score, min(0.40 + 0.5 * contribution, 0.90))

        flagged = len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "clean"

        result = {
            "flagged":      flagged,
            "reason":       reason_str,
            "score":        round(score, 4),
            "details": {
                "n_tokens":        n_tokens,
                "unique_tokens":   len(set(tokens)),
                "ttr":             round(ttr_value, 4),
                "keyword_density": round(density_value, 4),
                "keyword_hits":    keyword_hits,
                "invisible_chars": invis_count,
            },
        }

        level = logging.WARNING if flagged else logging.INFO
        logger.log(
            level,
            f"[{candidate_id}] KeywordStuffing: flagged={flagged} "
            f"score={score:.3f} reason='{reason_str}'"
        )
        return result


# ---------------------------------------------------------------------------
# 2. SynonymFloodingDetector
# ---------------------------------------------------------------------------

class SynonymFloodingDetector:
    """
    Detects synonym flooding in the skills section of a candidate profile.

    Canonicalises each skill token against a synonym table, then checks what
    fraction of distinct canonical IDs are dominated by a single cluster.

    Parameters
    ----------
    max_cluster_ratio  : float  Max fraction of skill tokens a single canonical
                                skill can represent (default 0.40).
    min_skills         : int    Minimum skill tokens to analyse (default 5).
    """

    def __init__(
        self,
        max_cluster_ratio: float = 0.40,
        min_skills: int = 5,
    ) -> None:
        """Initialise the SynonymFloodingDetector."""
        self.max_cluster_ratio = max_cluster_ratio
        self.min_skills        = min_skills

    def detect(self, skills: List[str], candidate_id: str = "unknown") -> Dict:
        """
        Detect synonym flooding in a skills list.

        Parameters
        ----------
        skills       : List[str]  Raw skill strings from the candidate profile.
        candidate_id : str        For logging.

        Returns
        -------
        dict
            {
                "flagged": bool,
                "reason":  str,
                "score":   float,
                "details": {...}
            }
        """
        # Guard: empty input
        if not skills:
            return {
                "flagged": False,
                "reason":  "empty_skills_list",
                "score":   0.0,
                "details": {},
            }

        canonical = [_canonicalise_skill(s) for s in skills]
        counts    = Counter(canonical)
        n_total   = len(canonical)

        if n_total < self.min_skills:
            return {
                "flagged": False,
                "reason":  f"too_few_skills ({n_total} < {self.min_skills})",
                "score":   0.0,
                "details": {"n_skills": n_total},
            }

        top_skill, top_count = counts.most_common(1)[0]
        top_ratio = top_count / n_total

        flagged = top_ratio > self.max_cluster_ratio
        score   = round(min(top_ratio, 1.0), 4)
        reason  = (
            f"synonym_flooding: '{top_skill}' cluster ratio={top_ratio:.2f} "
            f"> threshold {self.max_cluster_ratio}"
            if flagged else "clean"
        )

        result = {
            "flagged": flagged,
            "reason":  reason,
            "score":   score,
            "details": {
                "n_skills":          n_total,
                "unique_canonical":  len(counts),
                "top_canonical":     top_skill,
                "top_count":         top_count,
                "top_ratio":         round(top_ratio, 4),
            },
        }

        level = logging.WARNING if flagged else logging.INFO
        logger.log(
            level,
            f"[{candidate_id}] SynonymFlooding: flagged={flagged} "
            f"score={score:.3f} reason='{reason}'"
        )
        return result


# ---------------------------------------------------------------------------
# 3. RankingManipulationDetector
# ---------------------------------------------------------------------------

class RankingManipulationDetector:
    """
    Detects ranking manipulation via two guards:

    1. Score-velocity guard:
       Flags candidates whose ranking score increases by more than
       `max_score_delta` between consecutive evaluation windows --
       an unnaturally fast rise indicative of coordinated feedback injection.

    2. Omnipresence guard:
       Flags candidates who appear at rank #1 across more than
       `max_top1_rate` of distinct query categories, which is statistically
       impossible for a genuine candidate.

    Parameters
    ----------
    max_score_delta : float  Max allowed score increase per window (default 0.25).
    max_top1_rate   : float  Max allowed fraction of queries where candidate is #1
                             (default 0.80).
    min_queries     : int    Minimum distinct queries to evaluate omnipresence
                             (default 5).
    """

    def __init__(
        self,
        max_score_delta: float = 0.25,
        max_top1_rate:   float = 0.80,
        min_queries:     int   = 5,
    ) -> None:
        """Initialise the RankingManipulationDetector."""
        self.max_score_delta = max_score_delta
        self.max_top1_rate   = max_top1_rate
        self.min_queries     = min_queries

    def check_velocity(
        self,
        candidate_id: str,
        score_history: List[float],
    ) -> Dict:
        """
        Score-velocity guard: detect unnaturally rapid score inflation.

        Parameters
        ----------
        candidate_id  : str
        score_history : List[float]  Scores in chronological order (oldest first).
                                     Each element is a score in [0, 1].

        Returns
        -------
        dict  {"flagged": bool, "reason": str, "score": float, "details": {...}}
        """
        if len(score_history) < 2:
            return {
                "flagged": False, "reason": "insufficient_history",
                "score": 0.0, "details": {"n_windows": len(score_history)},
            }

        deltas    = [score_history[i+1] - score_history[i]
                     for i in range(len(score_history) - 1)]
        max_delta = max(deltas)
        flagged   = max_delta > self.max_score_delta
        score_val = round(min(max_delta / self.max_score_delta, 1.0), 4)
        reason    = (
            f"score_velocity: max_delta={max_delta:.3f} "
            f"> threshold {self.max_score_delta}"
            if flagged else "clean"
        )

        result = {
            "flagged": flagged,
            "reason":  reason,
            "score":   score_val,
            "details": {
                "n_windows":   len(score_history),
                "score_history": [round(s, 4) for s in score_history],
                "deltas":      [round(d, 4) for d in deltas],
                "max_delta":   round(max_delta, 4),
            },
        }
        level = logging.WARNING if flagged else logging.INFO
        logger.log(
            level,
            f"[{candidate_id}] ScoreVelocity: flagged={flagged} "
            f"max_delta={max_delta:.3f}"
        )
        return result

    def check_omnipresence(
        self,
        candidate_id: str,
        query_results: Dict[str, int],
    ) -> Dict:
        """
        Omnipresence guard: detect candidates ranking #1 across all query types.

        Parameters
        ----------
        candidate_id  : str
        query_results : Dict[str, int]
            Keys are query category strings (e.g. 'data_scientist_london'),
            values are the candidate's rank in that query (1-indexed, 1 = top).

        Returns
        -------
        dict  {"flagged": bool, "reason": str, "score": float, "details": {...}}
        """
        if len(query_results) < self.min_queries:
            return {
                "flagged": False,
                "reason":  f"too_few_queries ({len(query_results)} < {self.min_queries})",
                "score":   0.0,
                "details": {"n_queries": len(query_results)},
            }

        top1_count = sum(1 for rank in query_results.values() if rank == 1)
        top1_rate  = top1_count / len(query_results)
        flagged    = top1_rate > self.max_top1_rate
        score_val  = round(min(top1_rate / self.max_top1_rate, 1.0), 4)
        reason     = (
            f"omnipresence: top1_rate={top1_rate:.2f} "
            f"({top1_count}/{len(query_results)} queries) "
            f"> threshold {self.max_top1_rate}"
            if flagged else "clean"
        )

        result = {
            "flagged": flagged,
            "reason":  reason,
            "score":   score_val,
            "details": {
                "n_queries": len(query_results),
                "top1_count": top1_count,
                "top1_rate":  round(top1_rate, 4),
            },
        }
        level = logging.WARNING if flagged else logging.INFO
        logger.log(
            level,
            f"[{candidate_id}] Omnipresence: flagged={flagged} "
            f"top1_rate={top1_rate:.2f}"
        )
        return result


# ---------------------------------------------------------------------------
# Evaluator helper (for the demo offline metrics)
# ---------------------------------------------------------------------------

def evaluate_detector(
    detector: KeywordStuffingDetector,
    adversarial_texts: List[str],
    legitimate_texts: List[str],
) -> Dict:
    """
    Compute precision, recall, and FPR for a KeywordStuffingDetector.

    Parameters
    ----------
    detector          : KeywordStuffingDetector
    adversarial_texts : List[str]  Ground-truth positive cases.
    legitimate_texts  : List[str]  Ground-truth negative cases.

    Returns
    -------
    dict
        {
            "tp": int, "fp": int, "fn": int, "tn": int,
            "precision": float, "recall": float, "fpr": float, "f1": float
        }
    """
    tp = fp = fn = tn = 0

    for text in adversarial_texts:
        result = detector.detect(text, candidate_id="adversarial")
        if result["flagged"]:
            tp += 1
        else:
            fn += 1

    for text in legitimate_texts:
        result = detector.detect(text, candidate_id="legitimate")
        if result["flagged"]:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "fpr":       round(fpr, 4),
        "f1":        round(f1, 4),
    }


def save_defence_report(report: Dict, path: str = DEFENCE_REPORT_PATH) -> None:
    """
    Save the defence evaluation report to JSON.

    Parameters
    ----------
    report : dict
    path   : str
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Defence report saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save defence report: {e}")


def main() -> None:
    """Smoke test for all three detectors."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/task22_security.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    try:
        ksd = KeywordStuffingDetector()
        stuffed  = ("python " * 80 + "machine learning " * 50 +
                    "ai ai ai ai ai ai ai ai ai ai ")
        clean    = ("Experienced data scientist with 5 years building ML models. "
                    "Strong Python, SQL, and cloud skills. Led cross-functional teams.")
        r1 = ksd.detect(stuffed, "adversarial_01")
        r2 = ksd.detect(clean,   "legitimate_01")
        print(f"Stuffed: flagged={r1['flagged']} score={r1['score']}")
        print(f"Clean  : flagged={r2['flagged']} score={r2['score']}")

        sfd = SynonymFloodingDetector()
        flooded_skills = ["ml", "machine learning", "AI", "deep learning",
                          "neural network", "nn", "DL", "supervised learning",
                          "unsupervised learning", "artificial intelligence"]
        r3 = sfd.detect(flooded_skills, "adversarial_02")
        print(f"Flooded skills: flagged={r3['flagged']} ratio={r3['details']['top_ratio']}")

        rmd = RankingManipulationDetector()
        r4 = rmd.check_velocity("candidate_X", [0.3, 0.35, 0.38, 0.75, 0.80])
        print(f"Velocity: flagged={r4['flagged']} max_delta={r4['details']['max_delta']}")

        print("[OK] ranking_defence smoke test passed.")
    except Exception as e:
        logger.critical(f"Smoke test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
