"""Match GoEmotions labels to the closest ANEW word using spaCy en_core_web_lg word vectors
(cosine similarity), then output normalised valence/arousal scores.

ANEW scale: 1-9  →  normalised to [-1, 1] via  (x - 5) / 4

Output: emotion_va_scores.csv
  emotion       – GoEmotions label (28 labels)
  anew_word     – most similar ANEW entry by cosine similarity
  similarity    – cosine similarity (0–1)
  valence_raw   – ANEW valence (1-9)
  arousal_raw   – ANEW arousal (1-9)
  valence_norm  – (valence_raw - 5) / 4  →  [-1, 1]
  arousal_norm  – (arousal_raw - 5) / 4  →  [-1, 1]
"""

import csv
from pathlib import Path

import numpy as np
import spacy

# ── GoEmotions labels ──────────────────────────────────────────────────────────
GO_EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

HERE = Path(__file__).parent
ANEW_CSV   = HERE / "anew_scores.csv"
OUTPUT_CSV = HERE / "emotion_va_scores.csv"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize(val: float) -> float:
    """Map ANEW 1-9 scale to [-1, 1]."""
    return (val - 5.0) / 4.0


def main() -> None:
    print("Loading spaCy en_core_web_lg …")
    nlp = spacy.load("en_core_web_lg")

    # ── Load ANEW ──────────────────────────────────────────────────────────────
    anew_words: list[str] = []
    anew_valence: list[float] = []
    anew_arousal: list[float] = []
    with open(ANEW_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["Word"].strip().lower()
            try:
                v = float(row["Valence"])
                a = float(row["Arousal"])
            except ValueError:
                continue
            tok = nlp(word)[0]
            if not tok.has_vector:
                continue
            anew_words.append(word)
            anew_valence.append(v)
            anew_arousal.append(a)

    print(f"Loaded {len(anew_words)} ANEW words with vectors.")

    # Pre-stack ANEW vectors for batch cosine similarity (n_anew × 300)
    anew_vecs = np.stack([nlp(w)[0].vector for w in anew_words])
    anew_norms = np.linalg.norm(anew_vecs, axis=1, keepdims=True)
    anew_vecs_normed = anew_vecs / np.maximum(anew_norms, 1e-9)

    # ── Match each GoEmotions label ────────────────────────────────────────────
    results: list[dict] = []
    for emotion in GO_EMOTIONS:
        # Use spaCy to get a vector for the emotion word (may be multi-token for future-proofing)
        doc = nlp(emotion)
        # Average over tokens that have vectors
        valid_vecs = [t.vector for t in doc if t.has_vector]
        if not valid_vecs:
            print(f"  WARNING: no vector for '{emotion}', skipping.")
            continue
        emot_vec = np.mean(valid_vecs, axis=0)
        emot_norm = np.linalg.norm(emot_vec)
        if emot_norm < 1e-9:
            print(f"  WARNING: zero vector for '{emotion}', skipping.")
            continue
        emot_vec_n = emot_vec / emot_norm

        sims = anew_vecs_normed @ emot_vec_n          # (n_anew,)
        best_idx = int(np.argmax(sims))
        best_word = anew_words[best_idx]
        best_sim  = float(sims[best_idx])
        v_raw = anew_valence[best_idx]
        a_raw = anew_arousal[best_idx]

        results.append({
            "emotion":      emotion,
            "anew_word":    best_word,
            "similarity":   round(best_sim, 4),
            "valence_raw":  v_raw,
            "arousal_raw":  a_raw,
            "valence_norm": round(normalize(v_raw), 4),
            "arousal_norm": round(normalize(a_raw), 4),
        })
        print(f"  {emotion:18s}  →  {best_word:18s}  sim={best_sim:.3f}  "
              f"V={v_raw:.2f}({normalize(v_raw):+.2f})  A={a_raw:.2f}({normalize(a_raw):+.2f})")

    # ── Manual overrides for known-bad cosine matches ────────────────────────
    # "disappointment": matched to "anger" (arousal far too high); ANEW has
    #   "disappoint" directly — V=2.39, A=4.92 (low-arousal negative, correct).
    # "relief": matched to "pain" (valence flipped); ANEW "relaxed" is the
    #   canonical low-arousal positive anchor — V=7.00, A=2.39.
    overrides: dict[str, tuple[str, float, float]] = {
        "disappointment": ("disappoint", 2.39, 4.92),
        "relief":         ("relaxed",    7.00, 2.39),
    }
    for row in results:
        emotion = row["emotion"]
        if emotion in overrides:
            anew_word, v_raw, a_raw = overrides[emotion]
            print(f"  [OVERRIDE] {emotion:18s}  →  {anew_word:18s}  "
                  f"V={v_raw:.2f}({normalize(v_raw):+.2f})  A={a_raw:.2f}({normalize(a_raw):+.2f})")
            row["anew_word"]    = anew_word
            row["similarity"]   = None   # not a cosine match
            row["valence_raw"]  = v_raw
            row["arousal_raw"]  = a_raw
            row["valence_norm"] = round(normalize(v_raw), 4)
            row["arousal_norm"] = round(normalize(a_raw), 4)

    # ── Write output ───────────────────────────────────────────────────────────
    fieldnames = ["emotion", "anew_word", "similarity", "valence_raw", "arousal_raw",
                  "valence_norm", "arousal_norm"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
