from pathlib import Path

from ask_ashish.ingest import discover, ingest_corpus, load_text


def _pdf_bytes() -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        None,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = b"BT /F1 18 Tf 72 720 Td (Ashish Kosana public resume note) Tj ET"
    objects[3] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_ingest_md_txt_and_pdf(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("Ledgerline retries.\n", encoding="utf-8")
    (tmp_path / "plain.txt").write_text("Tick leases jobs.\n", encoding="utf-8")
    (tmp_path / "resume.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "SOURCES.md").write_text("not indexed\n", encoding="utf-8")
    (tmp_path / "sources.json").write_text(
        '{"notes.md": "https://example.com/notes", "plain.txt": "https://example.com/plain"}\n',
        encoding="utf-8",
    )

    found = {path.name for path in discover(tmp_path)}
    assert found == {"notes.md", "plain.txt", "resume.pdf"}
    assert "Ashish Kosana public resume note" in load_text(tmp_path / "resume.pdf")

    chunks = ingest_corpus(tmp_path, size=800, overlap=120)
    by_source = {chunk.source: chunk for chunk in chunks}
    assert set(by_source) == {"notes.md", "plain.txt", "resume.pdf"}
    assert by_source["notes.md"].url == "https://example.com/notes"
    assert by_source["resume.pdf"].url is None
    assert "not indexed" not in " ".join(chunk.text for chunk in chunks)
