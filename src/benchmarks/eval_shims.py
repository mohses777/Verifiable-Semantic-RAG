"""Run-time stand-ins for third-party pieces of the official evaluator that do not work in this environment.

``datasets.Dataset`` is only used by the evaluator as ``Dataset.from_dict(columns)``, column access by name and
``select(indices)``. ``datasets==3.3.2`` pins an old ``dill`` whose pickler fails on Python 3.14
(``Pickler._batch_setitems() takes 2 positional arguments but 3 were given``), so a minimal in-memory class with
the same three behaviours is substituted. No metric logic is touched. Documented in
docs/benchmark-investigation.md.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class InMemoryDataset:
    def __init__(self, columns: Mapping[str, Sequence[Any]]) -> None:
        lengths = {len(values) for values in columns.values()}
        if len(lengths) > 1:
            raise ValueError(f"Columns have different lengths: { {k: len(v) for k, v in columns.items()} }")
        self._columns = {key: list(values) for key, values in columns.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence[Any]]) -> "InMemoryDataset":
        return cls(data)

    def __getitem__(self, key: str) -> list[Any]:
        if not isinstance(key, str):
            raise TypeError("InMemoryDataset supports column access by name only")
        return self._columns[key]

    def __len__(self) -> int:
        return len(next(iter(self._columns.values()), []))

    def select(self, indices: Sequence[int]) -> "InMemoryDataset":
        return InMemoryDataset({key: [values[i] for i in indices] for key, values in self._columns.items()})
