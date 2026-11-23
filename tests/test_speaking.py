"""Wadia speaks sentence by sentence while the model is still writing, so these
functions decide where to cut the streamed text and what not to say aloud."""
from brain import clean_for_speech, is_filler_question, sentences_from_stream


def stream(text, size=3):
    """Simulates the language model sending its answer a few characters at a time."""
    return (text[i : i + size] for i in range(0, len(text), size))


def test_splits_streamed_text_into_sentences():
    text = "It's sunny today. Highs of 24 degrees. Enjoy the beach!"
    assert list(sentences_from_stream(stream(text))) == [
        "It's sunny today.",
        "Highs of 24 degrees.",
        "Enjoy the beach!",
    ]


def test_first_piece_is_cut_early_at_a_comma_to_start_talking_sooner():
    text = "Well, the match ended two to one, with a late goal in the last minute. Great game."
    pieces = list(sentences_from_stream(stream(text)))
    assert pieces[0] == "Well, the match ended two to one,"
    assert pieces[1] == "with a late goal in the last minute."


def test_short_opening_is_not_cut_too_early():
    pieces = list(sentences_from_stream(stream("Sure, it's ten past three.")))
    assert pieces == ["Sure, it's ten past three."]


def test_drops_generic_follow_up_offers():
    text = "Paris is the capital of France. Anything else I can help you with?"
    assert list(sentences_from_stream(stream(text))) == ["Paris is the capital of France."]


def test_keeps_real_questions():
    assert not is_filler_question("Do you mean Paris in Texas or in France?")
    assert is_filler_question("Want to know more?")
    assert is_filler_question("¿Quieres saber más?")


def test_removes_markdown_and_emoji_before_speaking():
    assert clean_for_speech("**Great** news 🎉") == "Great news"
