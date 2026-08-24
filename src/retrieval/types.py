"""Shared retrieval types and backend-independent metadata filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping


Metadata = dict[str, Any]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Canonical result returned by dense, sparse, and hybrid retrieval."""

    score: float
    chunk_id: str
    text: str

    page_title: str
    section_heading: str | None
    source_url: str

    metadata: Metadata


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Exact-match constraints applied before ranking and reranking.

    Values inside one field use OR semantics. Different fields use AND
    semantics. For example, two ``page_ids`` and one ``unit_type`` mean::

        (page_id == first OR page_id == second) AND unit_type == value

    A field left empty adds no constraint.
    """

    chunk_ids: frozenset[str] = field(default_factory=frozenset)
    parent_ids: frozenset[str] = field(default_factory=frozenset)
    semantic_unit_ids: frozenset[str] = field(default_factory=frozenset)

    page_ids: frozenset[str] = field(default_factory=frozenset)
    page_titles: frozenset[str] = field(default_factory=frozenset)
    page_urls: frozenset[str] = field(default_factory=frozenset)

    unit_types: frozenset[str] = field(default_factory=frozenset)
    section_indexes: frozenset[int] = field(default_factory=frozenset)
    section_headings: frozenset[str | None] = field(default_factory=frozenset)
    section_levels: frozenset[int] = field(default_factory=frozenset)

    parent_indexes: frozenset[int] = field(default_factory=frozenset)
    child_indexes: frozenset[int] = field(default_factory=frozenset)

    _FIELD_MAP: ClassVar[dict[str, str]] = {
        "chunk_ids": "chunk_id",
        "parent_ids": "parent_id",
        "semantic_unit_ids": "semantic_unit_id",
        "page_ids": "page_id",
        "page_titles": "page_title",
        "page_urls": "page_url",
        "unit_types": "unit_type",
        "section_indexes": "section_index",
        "section_headings": "section_heading",
        "section_levels": "section_level",
        "parent_indexes": "parent_index",
        "child_indexes": "child_index",
    }

    _STRING_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "chunk_ids",
            "parent_ids",
            "semantic_unit_ids",
            "page_ids",
            "page_titles",
            "page_urls",
            "unit_types",
        }
    )

    _INTEGER_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "section_indexes",
            "section_levels",
            "parent_indexes",
            "child_indexes",
        }
    )

    def __post_init__(self) -> None:
        """Normalize single values and iterables to immutable sets."""
        for field_name in self._FIELD_MAP:
            raw_value = getattr(self, field_name)

            if field_name in self._STRING_FIELDS:
                normalized = self._normalize_values(
                    raw_value,
                    expected_type=str,
                    field_name=field_name,
                )
            elif field_name in self._INTEGER_FIELDS:
                normalized = self._normalize_values(
                    raw_value,
                    expected_type=int,
                    field_name=field_name,
                )
            else:
                normalized = self._normalize_values(
                    raw_value,
                    expected_type=(str, type(None)),
                    field_name=field_name,
                )

            object.__setattr__(
                self,
                field_name,
                normalized,
            )

    @staticmethod
    def _normalize_values(
        value: Any,
        expected_type: type | tuple[type, ...],
        field_name: str,
    ) -> frozenset[Any]:
        if value is None:
            return frozenset()

        if isinstance(value, expected_type):
            values = frozenset({value})
        else:
            try:
                values = frozenset(value)
            except TypeError as error:
                raise TypeError(
                    f"{field_name} must be a value or an iterable of values."
                ) from error

        for item in values:
            # bool is an int subclass, but it is never a valid index here.
            if isinstance(item, bool) or not isinstance(item, expected_type):
                raise TypeError(
                    f"Invalid value {item!r} in {field_name}."
                )

        return values

    @property
    def is_empty(self) -> bool:
        """Return True when the filter contains no constraints."""
        return all(
            not getattr(self, field_name)
            for field_name in self._FIELD_MAP
        )

    def active_constraints(self) -> dict[str, frozenset[Any]]:
        """Return active constraints keyed by stored metadata field name."""
        return {
            metadata_key: getattr(self, field_name)
            for field_name, metadata_key in self._FIELD_MAP.items()
            if getattr(self, field_name)
        }

    def matches(self, metadata: Mapping[str, Any]) -> bool:
        """Evaluate the filter locally, primarily for BM25 documents."""
        for metadata_key, allowed_values in self.active_constraints().items():
            if metadata_key not in metadata:
                return False

            if metadata[metadata_key] not in allowed_values:
                return False

        return True


__all__ = [
    "Metadata",
    "MetadataFilter",
    "RetrievalResult",
]