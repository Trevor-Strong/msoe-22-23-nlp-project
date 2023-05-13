
from typing import TypeVar, Sequence

_T = TypeVar('_T')
_U = TypeVar('_U')

def unzip_unique(seq: Sequence[tuple[_T, _U]]) -> tuple[set[_T], set[_U]]:
    a, b = set(), set()
    for x, y in seq:
        a.add(x)
        b.add(y)
    return a, b
    
