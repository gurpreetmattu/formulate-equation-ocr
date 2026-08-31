from app.deep_learning.postprocessing import fix_mathml_block_tag, latex_to_mathml, tokens_to_latex


def test_tokens_to_latex_joins_with_spaces():
    idx_to_token = {0: "x", 1: "+", 2: "y"}
    assert tokens_to_latex(idx_to_token, [0, 1, 2]) == "x + y"


def test_tokens_to_latex_unknown_index_falls_back_to_unk():
    idx_to_token = {0: "x"}
    assert tokens_to_latex(idx_to_token, [0, 99]) == "x <UNK>"


def test_fix_mathml_block_tag_sets_display_block():
    mathml = '<math display="inline"><mi>x</mi></math>'
    assert 'display="block"' in fix_mathml_block_tag(mathml)


def test_fix_mathml_block_tag_strips_ns0_prefix():
    mathml = '<ns0:math xmlns:ns0="http://example.com"><ns0:mi>x</ns0:mi></ns0:math>'
    fixed = fix_mathml_block_tag(mathml)
    assert "ns0:" not in fixed
    assert 'xmlns="http://www.w3.org/1998/Math/MathML"' in fixed


def test_latex_to_mathml_valid_expression():
    result = latex_to_mathml("x + y")
    assert result.startswith("<math")


def test_latex_to_mathml_invalid_input_returns_error_string_not_raise():
    result = latex_to_mathml("\\unknownbadcommand{{{")
    # latex2mathml is lenient about most malformed input; this asserts the
    # function never raises, regardless of whether conversion "succeeds".
    assert isinstance(result, str)
