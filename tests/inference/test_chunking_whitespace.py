"""Chunked and single-pass surfaces must follow one whitespace/offset policy.

Guards the long-text chunking contract: a span detected inside a chunk, once
remapped, must index the same document characters (and yield the same surface
text) as the equivalent single-pass offset — including at chunk boundaries with
leading/trailing whitespace.
"""

from __future__ import annotations

import pytest

from gliner2.inference.chunking import (
    iter_word_offsets,
    remap_result_spans,
    split_text_into_chunks,
)
from gliner2.processing.word_splitter import CharLevelSplitter, WhitespaceTokenSplitter


TEXTS = [
    "  hello   world  foo bar baz  ",
    "leading word then   irregular    spacing here",
    "\tTabbed and\nnewline separated words across lines\n",
    "single",
    "  padded single  ",
]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("chunk_size,overlap", [(2, 1), (3, 1), (4, 2)])
def test_chunk_remap_matches_single_pass_offsets(text, chunk_size, overlap):
    doc_words = list(iter_word_offsets(text))
    # Document-level (start, end) for each surface word — the single-pass truth.
    single_pass = {(w, s, e) for w, s, e in doc_words}

    chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=overlap)
    for chunk in chunks:
        for word, local_s, local_e in iter_word_offsets(chunk.text):
            span = {"text": chunk.text[local_s:local_e], "start": local_s, "end": local_e}
            remapped = remap_result_spans(dict(span), text, chunk)
            start, end = remapped["start"], remapped["end"]
            # Surface parity: remapped offsets index the same document characters.
            assert text[start:end] == word
            assert remapped["text"] == word
            # Offset parity: the remapped span is a real single-pass word span.
            assert (word, start, end) in single_pass


def test_chunk_text_starts_at_word_not_whitespace():
    text = "  alpha beta gamma delta  "
    chunks = split_text_into_chunks(text, chunk_size=2, chunk_overlap=1)
    for chunk in chunks:
        # Chunks begin exactly at a word boundary, never on stripped whitespace.
        assert chunk.text[:1].strip() != ""
        assert text[chunk.start_char] == chunk.text[0]


def test_default_offsets_match_whitespace_splitter():
    text = "visit https://example.com and foo@bar.com now"
    assert list(iter_word_offsets(text)) == list(
        WhitespaceTokenSplitter()(text, lower=False)
    )


def test_char_splitter_chunks_on_character_boundaries():
    text = "我爱北京Tiananmen"
    tokens = list(iter_word_offsets(text, word_splitter="char"))
    assert [tok for tok, _, _ in tokens] == ["我", "爱", "北", "京", "Tiananmen"]
    chunks = split_text_into_chunks(
        text, chunk_size=2, chunk_overlap=1, word_splitter=CharLevelSplitter()
    )
    assert chunks[0].text == "我爱"
    assert text[chunks[0].start_char:chunks[0].end_char] == "我爱"


def test_custom_callable_splitter_drives_chunk_windows():
    def every_char(text, lower=True):
        for index, char in enumerate(text):
            if not char.isspace():
                yield char, index, index + 1

    text = "ab cd"
    chunks = split_text_into_chunks(
        text, chunk_size=2, chunk_overlap=0, word_splitter=every_char
    )
    assert [chunk.text for chunk in chunks] == ["ab", "cd"]
