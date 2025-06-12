"""Public package API for :mod:`sign_indexing`.

Importing this module exposes the two main classes used by the library:

``IndexSC``
    Performs sign-concordance filtering of candidate vectors.
``IndexRR``
    Re-ranks those candidates using exact dot-product similarity.
"""

from .indexing import IndexSC, IndexRR

__all__ = ["IndexSC", "IndexRR"]
