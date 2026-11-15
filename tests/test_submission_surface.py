"""Small adversarial manuscript fixtures; no scientific code or data is executed."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_submission_surface.py'
SPEC = importlib.util.spec_from_file_location('submission_surface', SCRIPT)
SURFACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SURFACE)


class SubmissionSurfaceTests(unittest.TestCase):
    def inspect(self, main, files=None):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'main.tex').write_text(main)
            for name, content in (files or {}).items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            return SURFACE.inspect_tex(root)

    def test_nested_caption_and_commented_false_references(self):
        result = self.inspect(r'''
% \input{missing} \ref{missing}
\begin{table}[t]
\caption{A {nested \textbf{caption}}, with 95\% coverage.}
\label{tab:a}
\begin{tabular}{l}A\\\end{tabular}
\end{table}
See \cref{tab:a}.
''')
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['tables'], 1)

    def test_caption_below_table_is_rejected_even_with_nested_braces(self):
        result = self.inspect(r'''
\begin{table}\begin{tabular}{l}A\end{tabular}
\caption{\textbf{Nested} caption}\end{table}
''')
        self.assertTrue(any('caption order' in e for e in result['errors']))

    def test_caption_above_input_body_is_accepted(self):
        result = self.inspect(r'\begin{table}\caption{A}\input{body}\end{table}',
                              {'body.tex': r'\begin{tabular}{l}1\end{tabular}'})
        self.assertEqual(result['errors'], [])
        self.assertEqual(len(result['active_tex_files']), 2)

    def test_caption_below_input_body_is_rejected(self):
        result = self.inspect(r'\begin{table}\input{body}\caption{A}\end{table}',
                              {'body.tex': r'\begin{tabular}{l}1\end{tabular}'})
        self.assertTrue(any('caption order' in e for e in result['errors']))

    def test_missing_reference_is_not_hidden_by_valid_neighbor(self):
        result = self.inspect(r'\label{good}See \cref{good,missing}.')
        self.assertIn('Missing reference label: missing', result['errors'])
        self.assertNotIn('Missing reference label: good', result['errors'])

    def test_duplicate_label_in_two_inputs(self):
        result = self.inspect(r'\input{a}\input{b}',
                              {'a.tex': r'\label{same}', 'b.tex': r'\label{same}'})
        self.assertIn('Duplicate label: same', result['errors'])

    def test_range_checks_both_endpoints(self):
        result = self.inspect(r'\label{first}\crefrange{first}{last}')
        self.assertIn('Missing reference label: last', result['errors'])

    def test_citation_optional_notes_and_missing_key(self):
        result = self.inspect(r'\citep[see][pp. {2--3}]{known,missing}\bibliography{refs}',
                              {'refs.bib': '@article{known,title={A title}}'})
        self.assertEqual(result['errors'], ['Missing citation key: missing'])

    def test_hyperlinks_use_explicit_anchor_not_label(self):
        result = self.inspect(r'\hypertarget{anchor}{}\hyperlink{anchor}{Here}\label{section}\hyperref[section]{There}')
        self.assertEqual(result['errors'], [])

    def test_input_cannot_escape_project(self):
        result = self.inspect(r'\input{../external}')
        self.assertTrue(any('Nonportable' in e for e in result['errors']))

    def test_figure_caption_must_follow_every_image(self):
        good = self.inspect(r'\begin{figure}\includegraphics{a.pdf}\includegraphics{b.pdf}\caption{Both}\end{figure}',
                            {'a.pdf': '', 'b.pdf': ''})
        bad = self.inspect(r'\begin{figure}\includegraphics{a.pdf}\caption{Both}\includegraphics{b.pdf}\end{figure}',
                           {'a.pdf': '', 'b.pdf': ''})
        self.assertEqual(good['errors'], [])
        self.assertTrue(any('caption order' in e for e in bad['errors']))

    def test_escaped_and_real_percent_are_distinguished(self):
        source = r'95\% retained % \ref{absent}' + '\n' + r'\label{actual}\ref{actual}'
        result = self.inspect(source)
        self.assertEqual(result['errors'], [])
        self.assertIn(r'95\%', SURFACE.strip_comments(source))


if __name__ == '__main__':
    unittest.main()
