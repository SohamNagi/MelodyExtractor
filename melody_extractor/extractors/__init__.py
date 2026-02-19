"""Melody extraction algorithm registry."""

from melody_extractor.extractors.base import MelodyExtractor
from melody_extractor.extractors.pyin import PYINExtractor

_ALL_EXTRACTORS: list[type[MelodyExtractor]] = [PYINExtractor]

try:
    from melody_extractor.extractors.melodia import MelodiaExtractor

    _ALL_EXTRACTORS.append(MelodiaExtractor)
except ImportError:
    pass

try:
    from melody_extractor.extractors.crepe import CrepeExtractor

    _ALL_EXTRACTORS.append(CrepeExtractor)
except ImportError:
    pass

EXTRACTORS: dict[str, type[MelodyExtractor]] = {
    cls.name: cls for cls in _ALL_EXTRACTORS if cls.available
}


def get_available_extractors() -> dict[str, type[MelodyExtractor]]:
    """Return a copy of the registry mapping name → extractor class."""
    return dict(EXTRACTORS)


def get_extractor(name: str) -> MelodyExtractor:
    """Instantiate and return the extractor registered under *name*.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    return EXTRACTORS[name]()
