from bsg.representations.extract import target_token_indices, unique_target_char_span


class SyntheticTokenizer:
    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):
        # Contextual, whitespace-preserving-ish offsets with a multi-token target span.
        offsets = [(0, 0)] if add_special_tokens else []
        offsets += [(0, 4), (5, 7), (8, 10), (11, 13), (13, 15), (15, 17), (17, 20), (20, 21)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


def test_target_span_supports_multi_token_overlap():
    text = "The day is Wednesday."
    assert unique_target_char_span(text, "Wednesday") == (11, 20)
    assert target_token_indices(text, "Wednesday", SyntheticTokenizer()) == [4, 5, 6, 7]


def test_target_span_repeated_target_fails_clearly():
    text = "Monday follows Monday."
    try:
        unique_target_char_span(text, "Monday")
    except ValueError as exc:
        assert "Ambiguous" in str(exc)
    else:
        raise AssertionError("expected ambiguous target failure")
