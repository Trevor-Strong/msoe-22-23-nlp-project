
from typing import TypeVar, Sequence, Iterable, Optional


_T = TypeVar('_T')
_U = TypeVar('_U')


def unzip_unique(seq: Sequence[tuple[_T, _U]]) -> tuple[set[_T], set[_U]]:
    a, b = set(), set()
    for x, y in seq:
        a.add(x)
        b.add(y)
    return a, b


def three_windowed(source: Iterable[_T], /, default=None) -> Iterable[tuple[Optional[_T], _T, Optional[_T]]]:
    it = iter(source)
    prev: Optional[_T] = default
    try:
        curr: Optional[_T] = next(it)
    except StopIteration:
        return  # no elements in source

    for next_item in it:
        yield prev, curr, next_item
        prev = curr
        curr = next_item

    yield prev, curr, default

