from ask_ashish.chunking import chunk_text


def test_short_paragraph_is_one_chunk() -> None:
    chunks = chunk_text("Hello ledgerline.", "notes.md", size=800, overlap=120, url="https://ex")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].source == "notes.md"
    assert chunks[0].url == "https://ex"
    assert chunks[0].text == "Hello ledgerline."


def test_overlap_keeps_a_boundary_word() -> None:
    words = ["alpha"] * 40 + ["BOUNDARY"] + ["omega"] * 40
    chunks = chunk_text(" ".join(words), "long.md", size=80, overlap=30)
    assert len(chunks) > 1
    assert chunks[0].index == 0
    assert chunks[1].index == 1
    host = next(chunk for chunk in chunks if "BOUNDARY" in chunk.text)
    if host.index + 1 < len(chunks):
        tail = host.text.split()[-1]
        assert tail in chunks[host.index + 1].text or "BOUNDARY" in chunks[host.index + 1].text


def test_overlap_must_be_smaller_than_size() -> None:
    try:
        chunk_text("hello", "a.md", size=20, overlap=20)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("expected ValueError")
