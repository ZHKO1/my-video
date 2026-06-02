#!/usr/bin/env python3
"""Compare text similarity across professional libraries vs difflib."""

import difflib

from rapidfuzz import fuzz, distance
import jellyfish
import textdistance


def levenshtein_similarity(a: str, b: str) -> float:
    """Standard Levenshtein similarity: 1 - distance/max_len."""
    dist = distance.Levenshtein.distance(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - dist / max_len if max_len else 1.0


def main():
    original = (
        "Now if you're interested about like little hidden things like that in games, "
        "there's a really cool website called the cutting room floor or tcrf.net for short, "
        "and it's basically like a wikipedia where you can go look up any game you pretty much want, "
        "and then you'll be able to find what is hidden in there, "
        "and or if there are like beta stuff or unused stuff, very cool to see."
    )

    optimized = (
        "Now if you're interested about little hidden things like that in games, "
        "there's a really cool website called the cutting room floor or tcrf.net for short, "
        "and it's basically like a wikipedia where you can go look up any game you pretty much want, "
        "and then you'll be able to find what is hidden in there, "
        "and or if there are beta stuff or unused stuff, very cool to see."
    )

    # Additional test cases
    cases = [
        ("删除两个 like", original, optimized),
        ("改一个词", "hello world", "hallo world"),
        ("删一个词", "hello beautiful world", "hello world"),
        ("改标点", "hello, world!", "hello world"),
        ("大段改写", original, "This is a completely different sentence about nothing related."),
        ("完全相同", original, original),
    ]

    print("=" * 100)
    print(f"{'Case':<20} {'difflib':>10} {'Levenshtein':>12} {'fuzz.ratio':>12} {'fuzz.WRatio':>12} {'jaro_winkl':>12} {'hamming':>10}")
    print("=" * 100)

    for name, a, b in cases:
        # difflib (current)
        difflib_ratio = difflib.SequenceMatcher(None, a, b).ratio()

        # rapidfuzz
        lev_sim = levenshtein_similarity(a, b)
        fuzz_ratio = fuzz.ratio(a, b) / 100.0
        fuzz_wratio = fuzz.WRatio(a, b) / 100.0

        # jellyfish
        jaro_wink = jellyfish.jaro_winkler_similarity(a, b)
        # hamming only works on same-length strings
        try:
            hamming = jellyfish.hamming_distance(a, b)
            hamming_sim = 1.0 - hamming / len(a) if len(a) else 1.0
        except ValueError:
            hamming_sim = 0.0

        print(
            f"{name:<20} {difflib_ratio:>10.2%} {lev_sim:>12.2%} {fuzz_ratio:>12.2%} "
            f"{fuzz_wratio:>12.2%} {jaro_wink:>12.2%} {hamming_sim:>10.2%}"
        )

    print("=" * 100)
    print("\nNotes:")
    print("- difflib:    Ratcliff/Obershelp, prone to misalignment on long repeated text")
    print("- Levenshtein: Edit distance / max_len. Intuitive for character-level changes.")
    print("- fuzz.ratio:  Optimal alignment Levenshtein (rapidfuzz). Handles substrings well.")
    print("- fuzz.WRatio: Same but preprocesses (strips, lowercases).")
    print("- jaro_winkler: Good for short strings, names. Not ideal for long sentences.")
    print("- hamming:     Only for same-length strings.")

    print("\n" + "=" * 100)
    print("Recommendation:")
    print("=" * 100)

    # Show what happens with the "delete two likes" case
    print("\nFor the 'delete two likes' case:")
    print(f"  difflib ratio:    {difflib.SequenceMatcher(None, original, optimized).ratio():.2%}  ← WRONG")
    print(f"  Levenshtein sim:  {levenshtein_similarity(original, optimized):.2%}  ← correct")
    print(f"  fuzz.ratio:       {fuzz.ratio(original, optimized)/100:.2%}  ← correct")

    print("\nBest option for optimize.py:")
    print("  rapidfuzz.distance.Levenshtein.similarity()  or  fuzz.ratio()")
    print("  Both give intuitive results and are very fast (C extension).")


if __name__ == "__main__":
    main()
