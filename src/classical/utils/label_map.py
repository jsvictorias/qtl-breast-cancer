from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class LabelMap:
    """Mapeamento único para problemas binários, ternários ou multiclasse."""

    class_to_index: Mapping[str, int]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, int]) -> LabelMap:
        normalized = {str(name): int(index) for name, index in mapping.items()}
        expected_indices = list(range(len(normalized)))
        actual_indices = sorted(normalized.values())

        if actual_indices != expected_indices:
            raise ValueError(
                "class_to_index precisa usar índices contíguos começando em 0. "
                f"Recebido: {normalized}"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("Existem nomes de classe duplicados.")

        return cls(class_to_index=MappingProxyType(normalized))

    @property
    def index_to_class(self) -> dict[int, str]:
        return {index: name for name, index in self.class_to_index.items()}

    @property
    def classes(self) -> list[str]:
        return [self.index_to_class[index] for index in range(self.num_classes)]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_index)

    def encode(self, label: str) -> int:
        try:
            return int(self.class_to_index[label])
        except KeyError as error:
            raise KeyError(f"Classe desconhecida: {label!r}") from error

    def decode(self, index: int) -> str:
        try:
            return self.index_to_class[int(index)]
        except KeyError as error:
            raise KeyError(f"Índice de classe desconhecido: {index}") from error

    def to_dict(self) -> dict[str, int]:
        return dict(self.class_to_index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)
