#!/usr/bin/env python3
"""Compare two text segments with detailed change statistics."""

import difflib
import re
import string


def main():
    text1 = """and you sadly you don't really find that information anywhere like if you go on on youtube for example you look on you type in the making of you most likely find videos about making off of buildings or bridges or maybe about um how roller coasters were built for example in the esterling but not really about how a game is made you don't find that information anywhere it's always company secrets now sometimes information like that does get leaked and it's usually done via leaks outside of the company and then you can get an insight in how games were made for example last year there were lots of internal documents being leaked and that gave us an insight into how older games were made and more specifically what didn't make it into the final game so let's say for example this image you see here this is a sprite table for one of the for super mario worlds um and it's showing here for example a sprite of uh like a grandpa yoshi that never ended up being in a final game but at some point nintendo was like planning to have it in the game and we would never have known about it if it wasn't leaked and at some functions like information we never get to see now that's via leaks we get information like that but what if you still want to know more about the game and there aren't any leaks about it what do you do then Well, that's where reverse engineering comes in."""

    text2 = """and you, sadly, you don't really find that information anywhere. Like if you go on on youtube, for example, you look on, you type in the making of, you most likely find videos about making off of buildings or bridges, or maybe about um how roller coasters were built, for example, in the esterling, but not really about how a game is made. You don't find that information anywhere. It's always company secrets. Now sometimes information like that does get leaked, and it's usually done via leaks outside of the company, and then you can get an insight in how games were made. For example, last year there were lots of internal documents being leaked, and that gave us an insight into how older games were made, and more specifically what didn't make it into the final game. So let's say, for example, this image you see here, this is a sprite table for one of the for super mario worlds um, and it's showing here, for example, a sprite of uh like a grandpa yoshi that never ended up being in a final game, but at some point nintendo was like planning to have it in the game, and we would never have known about it if it wasn't leaked. And at some functions, like information we never get to see. Now that's via leaks we get information like that, but what if you still want to know more about the game and there aren't any leaks about it? What do you do then? Well, that's where reverse engineering comes in."""

    # 1. 原文字符数
    original_chars = len(text1)

    # 2. 新增的符号数量（text2 中有而 text1 中没有的标点）
    # 这里通过 diff 统计插入操作中的标点符号
    matcher = difflib.SequenceMatcher(None, text1, text2)
    added_punctuation = 0
    changed_letters = 0
    replaced_chars = 0
    deleted_chars = 0
    inserted_chars = 0

    context_size = 20

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'replace':
            # 替换区域：text1[i1:i2] 被 text2[j1:j2] 替换
            old_part = text1[i1:i2]
            new_part = text2[j1:j2]
            replaced_chars += max(len(old_part), len(new_part))
            # 统计字母变化（大小写变化也算）
            # 用更细粒度的比较
            min_len = min(len(old_part), len(new_part))
            for k in range(min_len):
                if old_part[k] != new_part[k]:
                    if old_part[k].isalpha() or new_part[k].isalpha():
                        changed_letters += 1
            # 如果长度不同，多出来的部分也算变化
            if len(old_part) != len(new_part):
                extra = abs(len(old_part) - len(new_part))
                changed_letters += extra

            # ---- 详细输出 replace ----
            before = text1[max(0, i1 - context_size) : i1]
            after = text1[i2 : i2 + context_size]
            print(f"\n[replace] text1[{i1}:{i2}] → text2[{j1}:{j2}]")
            print(f"  before: ...{before!r}")
            print(f"  old:    {old_part!r}")
            print(f"  new:    {new_part!r}")
            print(f"  after:  {after!r}...")

        elif tag == 'delete':
            deleted_chars += (i2 - i1)
        elif tag == 'insert':
            inserted_chars += (j2 - j1)
            inserted_text = text2[j1:j2]
            for ch in inserted_text:
                if ch in string.punctuation:
                    added_punctuation += 1

            # ---- 详细输出 insert ----
            before = text2[max(0, j1 - context_size) : j1]
            after = text2[j2 : j2 + context_size]
            print(f"\n[insert] text2[{j1}:{j2}]")
            print(f"  before: ...{before!r}")
            print(f"  inserted: {inserted_text!r}")
            print(f"  after:  {after!r}...")

    # 总变化字符数（replace 取 max，避免重复计算）
    total_changed_chars = replaced_chars + deleted_chars + inserted_chars

    # 变化百分比（基于原文长度）
    change_percentage = (total_changed_chars / original_chars) * 100 if original_chars else 0

    print("\n" + "=" * 50)
    print("文本对比详细统计")
    print("=" * 50)
    print(f"原文（第一段）总字符数: {original_chars}")
    print(f"修改后（第二段）总字符数: {len(text2)}")
    print("-" * 50)
    print(f"新增符号数: {added_punctuation}")
    print(f"  - 新增标点字符统计")
    print("-" * 50)
    print(f"替换区域字符数: {replaced_chars}")
    print(f"  - 其中字母变化数: {changed_letters}")
    print(f"删除字符数: {deleted_chars}")
    print(f"插入字符数: {inserted_chars}")
    print(f"总变化字符数: {total_changed_chars}")
    print("-" * 50)
    print(f"原文变化占比: {change_percentage:.2f}%")
    print("=" * 50)

    # 更直观的：统计 text2 相比 text1 多出的标点
    all_punct = set(string.punctuation)
    text1_punct_count = sum(1 for ch in text1 if ch in all_punct)
    text2_punct_count = sum(1 for ch in text2 if ch in all_punct)
    print(f"\n补充统计:")
    print(f"text1 标点总数: {text1_punct_count}")
    print(f"text2 标点总数: {text2_punct_count}")
    print(f"标点净增加: {text2_punct_count - text1_punct_count}")

    # 相似率
    similarity = matcher.ratio() * 100
    print(f"\n相似率: {similarity:.2f}%")


if __name__ == "__main__":
    main()
