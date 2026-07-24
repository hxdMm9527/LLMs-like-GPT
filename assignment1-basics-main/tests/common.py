from __future__ import annotations

import pathlib
from functools import lru_cache

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    # 返回每个可能的字节（0 到 255 的整数）到可打印 unicode 字符串字符表示的映射。
    # 此函数取自 GPT-2 代码。
    # 例如，`chr(0)` 是 `\x00`，这是一个不可打印的字符：
    #     >>> chr(0)
    #     '\x00'
    #     >>> print(chr(0))
    # 因此，此函数返回一个字典 `d`，其中 `d[0]` 返回 `Ā`。
    # 视觉上可打印的字节保留其原始字符串表示 [1]。
    # 例如，`chr(33)` 返回 `!`，因此 `d[33]` 返回 `!`。
    # 特别注意，空格字符 `chr(32)` 变为 `d[32]`，返回 'Ġ'。
    # 对于不可打印的字符，该函数取该字符的 Unicode 码点（由 Python `ord` 返回），
    # 并将其偏移 256。例如，`ord(" ")` 返回 `32`，因此空格字符 ' ' 被偏移为 `256 + 32`。
    # 由于 `chr(256 + 32)` 返回 `Ġ`，我们将其用空格的字符串表示。
    # 此函数可以简化 BPE 实现，并使将生成的 merges 序列化到文件后更容易手动检查。
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # 这 188 个整数可以直接使用，因为它们不是空白字符或控制字符。
    # 参见 https://www.ssec.wisc.edu/~tomw/java/unicode.html。
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # 现在获取其他 68 个需要偏移的整数的表示
    # 每个将被映射为 chr(256 + n)，其中 n 在循环中从 0...67 递增
    # 获取剩余 68 个整数的可打印表示。
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # 如果此整数不在我们的视觉可表示字符列表中，
            # 则将其映射到下一个合适的字符（偏移 256）
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
