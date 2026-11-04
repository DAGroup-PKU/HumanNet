"""Check release scanning against accidental disclosure in documentation and code."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check_release.py'
spec = importlib.util.spec_from_file_location('release_check', SCRIPT)
release_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)


@pytest.mark.parametrize('prefix', ['pfs', 'mnt', 'home', 'Users', 'workspace'])
def test_internal_path_in_docstring_is_flagged(prefix):
    sample = '"""Notes\n' + '/' + prefix + '/example/report.md\n"""'
    assert release_check.privacy_findings(sample) == [(2, 'internal absolute path')]


def test_public_references_and_portable_paths_are_allowed():
    sample = '\n'.join([
        './data/video_cache', '../configs/robotwin.yaml',
        'https://github.com/DAGroup-PKU/ViGAR', '/proc/self/status',
        '/usr/share/fonts/truetype/noto', '${paths.resume}',
    ])
    assert not release_check.privacy_findings(sample)


def test_sensitive_matches_are_never_echoed():
    token = 'hf_' + 'A' * 32
    result = release_check.privacy_findings('# ' + token)
    assert result == [(1, 'credential token')]
    assert token not in repr(result)


def test_private_address_and_literal_secret_are_detected():
    address = '.'.join(['172', '20', '1', '2'])
    secret = 'password' + ' = ' + repr('example-secret-value')
    assert release_check.privacy_findings(address) == [(1, 'private IPv4 address')]
    assert release_check.privacy_findings(secret) == [(1, 'literal credential assignment')]


def test_curand_version_field_is_not_misidentified_as_an_address():
    version = '.'.join(['10', '3', '9', '90'])
    assert not release_check.privacy_findings('  "nvidia-curand-cu12": "' + version + '",')
    assert release_check.privacy_findings('"server": "' + version + '"') == [
        (1, 'private IPv4 address'),
    ]


def test_scan_includes_metadata_and_scripts_but_not_build_cache(tmp_path):
    (tmp_path / 'release').mkdir()
    (tmp_path / 'release' / 'metadata.json').write_text('{}')
    (tmp_path / 'tool.py').write_text('# public code')
    (tmp_path / 'build').mkdir()
    (tmp_path / 'build' / 'generated.py').write_text('# generated')
    files = {p.relative_to(tmp_path).as_posix()
             for p, _ in release_check.release_text_files(tmp_path)}
    assert files == {'release/metadata.json', 'tool.py'}
