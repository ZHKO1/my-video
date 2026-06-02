#!/usr/bin/env python3
"""Debug _is_valid_punctuation_rewrite for the given texts."""

ALLOWED_PUNCTUATION = ",.?"


def _is_valid_punctuation_rewrite_debug(original_text: str, optimized_text: str):
    """带详细日志的调试版本。"""
    i = 0
    j = 0
    pending_sentence_start = False
    seen_nonspace = False
    step = 0

    print(f"original length: {len(original_text)}, optimized length: {len(optimized_text)}")
    print("-" * 80)

    while i < len(original_text) and j < len(optimized_text):
        orig_char = original_text[i]
        opt_char = optimized_text[j]
        step += 1

        if orig_char == opt_char:
            if not opt_char.isspace():
                seen_nonspace = True
                pending_sentence_start = False
            i += 1
            j += 1
            continue

        if opt_char in ALLOWED_PUNCTUATION and (orig_char != opt_char):
            if not seen_nonspace:
                print(f"\nStep {step}: i={i}, j={j}")
                print(f"  orig_char={orig_char!r}")
                print(f"  opt_char ={opt_char!r}")
                print(f"  -> FAIL: seen_nonspace is False, cannot add punctuation before any non-space char")
                return False
            pending_sentence_start = opt_char in ".?"
            j += 1
            continue

        if (
            pending_sentence_start
            and orig_char.isalpha()
            and opt_char.isalpha()
            and orig_char.lower() == opt_char.lower()
            and opt_char == orig_char.upper()
        ):
            seen_nonspace = True
            pending_sentence_start = False
            i += 1
            j += 1
            continue

        if pending_sentence_start and orig_char.isspace() and opt_char.isspace():
            i += 1
            j += 1
            continue

        # 这是失败点
        print(f"\nStep {step}: i={i}, j={j}")
        print(f"  orig_char={orig_char!r} (context: ...{original_text[max(0,i-30):i+31]!r}...)")
        print(f"  opt_char ={opt_char!r} (context: ...{optimized_text[max(0,j-30):j+31]!r}...)")
        print(f"  seen_nonspace={seen_nonspace}, pending_sentence_start={pending_sentence_start}")
        print(f"  -> FAIL: No rule matched this pair")
        print(f"     Conditions check:")
        print(f"       - opt_char in ALLOWED_PUNCTUATION: {opt_char in ALLOWED_PUNCTUATION}")
        print(f"       - pending_sentence_start: {pending_sentence_start}")
        if pending_sentence_start:
            print(f"       - orig_char.isalpha(): {orig_char.isalpha()}")
            print(f"       - opt_char.isalpha(): {opt_char.isalpha()}")
            print(f"       - orig_char.lower() == opt_char.lower(): {orig_char.lower() == opt_char.lower()}")
            print(f"       - opt_char == orig_char.upper(): {opt_char == orig_char.upper()}")
            print(f"       - orig_char.isspace() and opt_char.isspace(): {orig_char.isspace() and opt_char.isspace()}")
        return False

    print(f"\nLoop ended: i={i} (len_orig={len(original_text)}), j={j} (len_opt={len(optimized_text)})")

    if i != len(original_text):
        print(f"  -> FAIL: i != len(original_text), remaining original: {original_text[i:]!r}")
        return False

    while j < len(optimized_text):
        opt_char = optimized_text[j]
        if opt_char in ALLOWED_PUNCTUATION and seen_nonspace:
            pending_sentence_start = opt_char in ".?"
            j += 1
            continue
        print(f"  -> FAIL: Trailing char {opt_char!r} not allowed or seen_nonspace=False")
        return False

    print("  -> SUCCESS")
    return True


def main():
    text1 = """and you sadly you don't really find that information anywhere like if you go on on youtube for example you look on you type in the making of you most likely find videos about making off of buildings or bridges or maybe about um how roller coasters were built for example in the esterling but not really about how a game is made you don't find that information anywhere it's always company secrets now sometimes information like that does get leaked and it's usually done via leaks outside of the company and then you can get an insight in how games were made for example last year there were lots of internal documents being leaked and that gave us an insight into how older games were made and more specifically what didn't make it into the final game so let's say for example this image you see here this is a sprite table for one of the for super mario worlds um and it's showing here for example a sprite of uh like a grandpa yoshi that never ended up being in a final game but at some point nintendo was like planning to have it in the game and we would never have known about it if it wasn't leaked and at some functions like information we never get to see now that's via leaks we get information like that but what if you still want to know more about the game and there aren't any leaks about it what do you do then Well, that's where reverse engineering comes in. """

    text2 = """and you, sadly, you don't really find that information anywhere. Like if you go on on youtube, for example, you look on, you type in the making of, you most likely find videos about making off of buildings or bridges, or maybe about um how roller coasters were built, for example, in the esterling, but not really about how a game is made. You don't find that information anywhere. It's always company secrets. Now sometimes information like that does get leaked, and it's usually done via leaks outside of the company, and then you can get an insight in how games were made. For example, last year there were lots of internal documents being leaked, and that gave us an insight into how older games were made, and more specifically what didn't make it into the final game. So let's say, for example, this image you see here, this is a sprite table for one of the for super mario worlds um, and it's showing here, for example, a sprite of uh like a grandpa yoshi that never ended up being in a final game, but at some point nintendo was like planning to have it in the game, and we would never have known about it if it wasn't leaked. And at some functions, like information we never get to see. Now that's via leaks we get information like that, but what if you still want to know more about the game and there aren't any leaks about it? What do you do then? Well, that's where reverse engineering comes in."""

    result = _is_valid_punctuation_rewrite_debug(text1, text2)
    print("\n" + "=" * 80)
    print(f"Final result: {result}")


if __name__ == "__main__":
    main()
