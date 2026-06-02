#!/usr/bin/env python3
"""Verify the similarity calculation for the user's specific case."""

import difflib
import re


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

    # Show raw lengths
    print(f"Original length:  {len(original)} chars")
    print(f"Optimized length: {len(optimized)} chars")
    print()

    # 1. Raw ratio (as used in optimize.py)
    matcher_raw = difflib.SequenceMatcher(None, original, optimized)
    ratio_raw = matcher_raw.ratio()
    print(f"1. Raw SequenceMatcher.ratio(): {ratio_raw:.4f} = {ratio_raw:.1%}")

    # 2. With whitespace normalization (as used in optimize.py)
    original_cleaned = re.sub(r"\s+", " ", original).strip()
    optimized_cleaned = re.sub(r"\s+", " ", optimized).strip()
    matcher_clean = difflib.SequenceMatcher(None, original_cleaned, optimized_cleaned)
    ratio_clean = matcher_clean.ratio()
    print(f"2. After whitespace normalization: {ratio_clean:.4f} = {ratio_clean:.1%}")

    # 3. Show the diff in detail
    print("\n" + "=" * 60)
    print("Detailed diff (character-level opcodes):")
    print("=" * 60)

    total_chars = len(original_cleaned)
    equal_chars = 0
    replace_chars = 0
    delete_chars = 0
    insert_chars = 0

    for tag, i1, i2, j1, j2 in matcher_clean.get_opcodes():
        if tag == "equal":
            segment = original_cleaned[i1:i2]
            # Truncate very long equal segments
            display = segment if len(segment) <= 60 else segment[:30] + "..." + segment[-30:]
            print(f"\n[EQUAL] {len(segment)} chars: {display!r}")
            equal_chars += (i2 - i1)
        elif tag == "replace":
            old_seg = original_cleaned[i1:i2]
            new_seg = optimized_cleaned[j1:j2]
            print(f"\n[REPLACE] original[{i1}:{i2}] -> optimized[{j1}:{j2}]")
            print(f"  old: {old_seg!r}")
            print(f"  new: {new_seg!r}")
            replace_chars += max(len(old_seg), len(new_seg))
        elif tag == "delete":
            seg = original_cleaned[i1:i2]
            print(f"\n[DELETE] original[{i1}:{i2}]: {seg!r}")
            delete_chars += (i2 - i1)
        elif tag == "insert":
            seg = optimized_cleaned[j1:j2]
            print(f"\n[INSERT] optimized[{j1}:{j2}]: {seg!r}")
            insert_chars += (j2 - j1)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total chars (original cleaned): {total_chars}")
    print(f"  Equal chars:  {equal_chars}")
    print(f"  Replace chars (max): {replace_chars}")
    print(f"  Delete chars: {delete_chars}")
    print(f"  Insert chars: {insert_chars}")

    # difflib ratio formula: 2*M / (len(a) + len(b))
    a, b = original_cleaned, optimized_cleaned
    M = sum((i2 - i1) for tag, i1, i2, j1, j2 in matcher_clean.get_opcodes() if tag == "equal")
    T = len(a) + len(b)
    computed_ratio = 2.0 * M / T
    print(f"\n  Formula check: 2*{M} / ({len(a)} + {len(b)}) = {2*M}/{T} = {computed_ratio:.4f}")

    # 4. Word-level comparison for perspective
    print("\n" + "=" * 60)
    print("Word-level comparison:")
    print("=" * 60)
    words_orig = original_cleaned.split()
    words_opt = optimized_cleaned.split()
    matcher_words = difflib.SequenceMatcher(None, words_orig, words_opt)
    ratio_words = matcher_words.ratio()
    print(f"Word-level ratio: {ratio_words:.4f} = {ratio_words:.1%}")

    for tag, i1, i2, j1, j2 in matcher_words.get_opcodes():
        if tag == "equal":
            continue
        print(f"\n[{tag.upper()}]")
        if tag in ("replace", "delete"):
            print(f"  original words[{i1}:{i2}]: {words_orig[i1:i2]}")
        if tag in ("replace", "insert"):
            print(f"  optimized words[{j1}:{j2}]: {words_opt[j1:j2]}")

    print("\n" + "=" * 60)
    print("Conclusion:")
    print(f"  Character-level similarity: {ratio_clean:.1%}")
    print(f"  Word-level similarity:      {ratio_words:.1%}")
    print()
    print("  The two texts differ by only 2 words ('like' removed twice).")
    print("  A similarity of 13.1% is IMPOSSIBLE with standard difflib.")
    print("  The actual similarity should be around 96-97% character-level")
    print("  and even higher word-level.")
    print("=" * 60)


if __name__ == "__main__":
    main()
