"""Flesch Reading Ease / Flesch-Kincaid grade for the report prose.

Skips code blocks, tables, headings and image lines — those are not prose and
would distort the score. Target for this project: Reading Ease >= 65.

  Reading Ease = 206.835 - 1.015*(words/sentence) - 84.6*(syllables/word)

Usage: python -m analysis.readability report/report.md [--verbose]
"""
from __future__ import annotations

import argparse
import re
import sys

VOWELS = "aeiouy"


def syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    count, prev_vowel = 0, False
    for ch in w:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e") and count > 1 and not w.endswith(("le", "ee")):
        count -= 1
    return max(count, 1)


def prose_lines(text: str) -> list[str]:
    out, in_code = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not s:
            continue
        if s.startswith(("#", "|", ">", "![", "---", "***")):
            continue
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)          # images
        s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)      # links -> text
        s = re.sub(r"`[^`]*`", "code", s)                   # inline code -> 1 word
        s = re.sub(r"[*_]{1,3}", "", s)                     # emphasis
        s = re.sub(r"^[-*+]\s+", "", s)                     # bullets
        s = re.sub(r"^\d+\.\s+", "", s)                     # numbered
        if s:
            out.append(s)
    return out


def score(text: str):
    lines = prose_lines(text)
    blob = " ".join(lines)
    # Sentence split; treat a bullet line without terminal punctuation as one sentence.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", blob) if re.search(r"[A-Za-z]", s)]
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", blob)
    syl = sum(syllables(w) for w in words)
    n_s, n_w = max(len(sentences), 1), max(len(words), 1)
    asl, asw = n_w / n_s, syl / n_w
    ease = 206.835 - 1.015 * asl - 84.6 * asw
    grade = 0.39 * asl + 11.8 * asw - 15.59
    return {"reading_ease": ease, "grade": grade, "words": n_w, "sentences": n_s,
            "words_per_sentence": asl, "syllables_per_word": asw}, sentences, words


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--verbose", action="store_true",
                    help="list the longest sentences and the heaviest words")
    a = ap.parse_args()
    text = open(a.path).read()
    st, sentences, words = score(text)
    print(f"Flesch Reading Ease : {st['reading_ease']:.1f}   (target >= 65)")
    print(f"Flesch-Kincaid grade: {st['grade']:.1f}")
    print(f"{st['words']} words, {st['sentences']} sentences, "
          f"{st['words_per_sentence']:.1f} words/sentence, "
          f"{st['syllables_per_word']:.2f} syllables/word")
    if a.verbose:
        print("\nLongest sentences:")
        for s in sorted(sentences, key=lambda x: -len(x.split()))[:8]:
            print(f"  [{len(s.split()):>3} words] {s[:110]}")
        heavy = {}
        for w in words:
            if syllables(w) >= 4:
                heavy[w.lower()] = heavy.get(w.lower(), 0) + 1
        print("\nHeaviest words (4+ syllables, by count):")
        for w, c in sorted(heavy.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {c:>3}x {w}")
    sys.exit(0 if st["reading_ease"] >= 65 else 1)


if __name__ == "__main__":
    main()
