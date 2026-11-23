import io

import pytest
from PIL import Image, ImageDraw, ImageFont

from vidya.ingest import Page, chunk_pages, clean_text, extract_pages, split_sentences, IngestError


def test_clean_text_rejoins_hyphenation_and_lines():
    assert clean_text("photo-\nsynthesis is a pro-\ncess.\nIt makes food.") == "photosynthesis is a process. It makes food."


def test_split_sentences_handles_devanagari_danda():
    s = split_sentences("यह पहला वाक्य है। यह दूसरा वाक्य है। Third one.")
    assert len(s) == 3


def test_chunks_keep_page_numbers_and_overlap():
    text = " ".join(f"Sentence number {i} talks about plants and sunlight." for i in range(60))
    chunks = chunk_pages([Page(3, text), Page(4, "Page four has its own text about the water cycle in nature.")],
                         "book.pdf", "abc", size=300, overlap=80)
    assert all(c["page"] in (3, 4) for c in chunks)
    assert chunks[-1]["page"] == 4
    p3 = [c for c in chunks if c["page"] == 3]
    assert len(p3) > 3
    assert all(len(c["text"]) <= 300 for c in chunks)
    # overlap: end of chunk n shows up at the start of chunk n+1
    tail = p3[0]["text"].split(". ")[-1]
    assert tail in p3[1]["text"]
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_very_long_sentence_is_hard_split():
    chunks = chunk_pages([Page(1, "word " * 1000)], "x.txt", "id", size=400, overlap=50)
    assert len(chunks) >= 10 and all(len(c["text"]) <= 400 for c in chunks)


def test_unsupported_extension():
    with pytest.raises(IngestError):
        extract_pages("virus.exe", b"x")


def test_txt_and_md():
    pages, w = extract_pages("notes.txt", ("Cells are the basic unit of life. " * 3).encode())
    assert pages[0].text.startswith("Cells")


def _pdf_bytes(pages):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for t in pages:
        y = 800
        for line in t:
            c.drawString(60, y, line)
            y -= 18
        c.showPage()
    c.save()
    return buf.getvalue()


def test_pdf_text_extraction_with_pages():
    data = _pdf_bytes([["Photosynthesis converts light energy into chemical energy.", "It happens in chloroplasts."],
                       ["The water cycle includes evaporation, condensation and precipitation."]])
    pages, warns = extract_pages("bio.pdf", data, ocr=False)
    assert [p.number for p in pages] == [1, 2]
    assert "chloroplasts" in pages[0].text


def test_scanned_pdf_falls_back_to_ocr():
    img = Image.new("RGB", (1000, 400), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
    except OSError:
        pytest.skip("no TTF font available")
    d.text((40, 60), "Gravity pulls objects", font=font, fill="black")
    d.text((40, 160), "towards the Earth", font=font, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PDF")  # image-only PDF => no text layer
    pages, warns = extract_pages("scan.pdf", buf.getvalue(), ocr=True)
    assert pages, warns
    assert "gravity" in pages[0].text.lower()
    assert any("OCR" in w for w in warns)


def test_image_ocr():
    img = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(img).text((30, 60), "Newton first law", font=ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 56), fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    pages, _ = extract_pages("pic.png", buf.getvalue(), ocr=True)
    assert "newton" in pages[0].text.lower()


def test_docx_and_pptx():
    import docx
    from pptx import Presentation
    d = docx.Document()
    d.add_paragraph("Atoms are made of protons, neutrons and electrons.")
    b = io.BytesIO()
    d.save(b)
    pages, _ = extract_pages("a.docx", b.getvalue())
    assert "protons" in pages[0].text

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Light"
    s.placeholders[1].text = "Light travels in straight lines."
    b = io.BytesIO()
    prs.save(b)
    pages, _ = extract_pages("a.pptx", b.getvalue())
    assert pages[0].number == 1 and "straight lines" in pages[0].text
