
from typing import Sequence, TypeVar, Literal, Any, Iterator, overload, Self

_T = TypeVar('_T')
_LiteralBool = Literal[True] | Literal[False]


class _AlwaysEqual:

    __slots__ = ()

    _instance: Self | None = None

    @classmethod
    def instance(cls):
        instance = cls._instance
        if instance is None:
            instance = cls()
            cls._instance = instance
        return instance

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


@overload
def _make_pair_seq_item(value: _T, /, *, is_first: Literal[True]) -> tuple[_T, _AlwaysEqual]: ...
@overload
def _make_pair_seq_item(value: _T, /, *, is_first: Literal[False]) -> tuple[_AlwaysEqual, _T]: ...


def _make_pair_seq_item(value: _T, /, *, is_first: _LiteralBool) -> tuple:
    return (value, _AlwaysEqual.instance()) if is_first else (_AlwaysEqual.instance(), value)


class PairSeq(Sequence[_T]):

    @overload
    def __init__(self, base: Sequence[tuple[_T, Any]], first: Literal[True]): ...
    @overload
    def __init__(self, base: Sequence[tuple[Any, _T]], first: Literal[False]): ...

    def __init__(self, base: Sequence[tuple[_T, Any]] | Sequence[tuple[Any, _T]], first: _LiteralBool):
        self.base = base
        self.is_first = first

    @overload
    def __getitem__(self, index: int, /) -> _T: ...
    @overload
    def __getitem__(self, index: slice, /) -> Sequence[_T]: ...

    def __getitem__(self, index: int | slice, /) -> _T | Sequence[_T]:
        if isinstance(index, slice):
            start = slice.start
            stop = slice.stop
            step = slice.step
            the_slice = PairSeq(self.base[start:stop:step], self.is_first)
            return the_slice
        else:
            return self.base[index][int(not self.is_first)]

    def __iter__(self) -> Iterator[_T]:
        idx = int(not self.is_first)
        seq = self.base
        return (item[idx] for item in seq)

    def __reversed__(self) -> Iterator[_T]:
        idx = int(not self.is_first)
        seq = self.base
        return (item[idx] for item in reversed(seq))

    @overload
    def index(self, value: Any, /, start: int = ..., stop: int = ...) -> int: ...

    def index(self, value: Any, /, *args, **kwargs) -> int:
        return self.base.index(_make_pair_seq_item(value, is_first=self.is_first), *args, **kwargs)

    def count(self, value: Any) -> int:
        return self.base.count(_make_pair_seq_item(value, is_first=self.is_first))

    def __contains__(self, item: Any, /) -> bool:
        return _make_pair_seq_item(item, is_first=self.is_first) in self.base

    def __len__(self) -> int:
        return len(self.base)


