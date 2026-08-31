"""Post-processing of decoded token sequences: token joining and MathML fixup."""
import re

from latex2mathml.converter import convert as latex2mathml_convert


def tokens_to_latex(idx_to_token: dict, token_ids: list[int]) -> str:
    return " ".join(idx_to_token.get(idx, "<UNK>") for idx in token_ids)


def fix_mathml_block_tag(mathml: str) -> str:
    """Make MathML display="block" and normalize the ns0 namespace prefix
    that latex2mathml sometimes emits."""
    mathml = re.sub(r'display="inline"', 'display="block"', mathml)
    mathml = re.sub(r'<(/?)ns0:', r'<\1', mathml)
    mathml = re.sub(r'xmlns:ns0="[^"]+"', 'xmlns="http://www.w3.org/1998/Math/MathML"', mathml)
    return mathml


def latex_to_mathml(latex: str) -> str:
    """Converts LaTeX to MathML, returning a bracketed error string on failure
    (matching the original app's error-surfacing behavior) rather than raising."""
    try:
        mathml = latex2mathml_convert(latex)
        return fix_mathml_block_tag(mathml)
    except Exception as exc:
        return f"[Conversion error: {exc}]"
