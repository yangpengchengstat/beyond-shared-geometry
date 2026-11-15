"""Privacy and archive-boundary tests only; no scientific workers imported."""
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_submission import validate_members, write_zip


class SubmissionPackagingTests(unittest.TestCase):
    def test_relative_path_escape_is_rejected(self):
        for name in ('../secret', '/absolute/file', 'paper/../../file', '.git/config'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_members({name: b'x'})

    def test_personal_and_temporary_paths_are_rejected(self):
        for raw in (b'/' + b'Users/owner/private.txt', b'/' + b'private/tmp/run/record'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_members({'record.txt': raw})

    def test_attributed_public_source_is_not_anonymous_source(self):
        raw = b'Peng' + b'cheng Yang'
        validate_members({'LICENSE': raw})
        with self.assertRaises(ValueError):
            validate_members({'LICENSE': raw}, anonymous=True)

    def test_credential_like_string_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_members({'config.txt': b'hf_' + b'x' * 25})

    def test_zip_is_deterministic_and_never_overwrites(self):
        members = {'main.tex': b'unchanged', 'data/source.csv': b'x,y\n1,2\n'}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            a = write_zip(root/'a.zip', members)
            b = write_zip(root/'b.zip', dict(reversed(list(members.items()))))
            self.assertEqual(a['sha256'], b['sha256'])
            with zipfile.ZipFile(root/'a.zip') as archive:
                self.assertEqual(archive.read('data/source.csv'), members['data/source.csv'])
            with self.assertRaises(FileExistsError):
                write_zip(root/'a.zip', members)


if __name__ == '__main__':
    unittest.main()
