"""Retrieval stage: dense (Qdrant) + BM25, fused in Python.

The two indexes stay where the index stage put them — dense vectors in one
Qdrant database, BM25 in ``artifacts/indexes/bm25/<strategy>/``. This package
only adds the layer the previous stage deliberately left out: one query in,
one ranked list out, with the chunk text attached so a hit can be cited.
"""
