import math

from ask_ashish.embeddings import HashEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    embedder = HashEmbedder()
    text = "ledgerline exactly-once payments"
    first, second = embedder.embed([text, text])
    assert first == second
    assert embedder.model_name == "hash-bow-256"
    norm = math.sqrt(sum(value * value for value in first))
    assert abs(norm - 1.0) < 1e-6


def test_related_text_is_closer_than_unrelated_text() -> None:
    embedder = HashEmbedder()
    topic, query, other = embedder.embed(
        [
            "ledgerline is an exactly-once payments backend with idempotency keys",
            "What is ledgerline?",
            "favorite color telephone initech",
        ]
    )
    assert _cosine(topic, query) > _cosine(topic, other)
