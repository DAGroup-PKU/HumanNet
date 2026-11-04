#!/usr/bin/env python3
"""Check source imports and portable resource paths without importing ML libraries."""
import ast
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


PRIVACY_PATTERNS = {
    'internal absolute path': re.compile(
        r'(?<![\w./:])/(?:mnt|pfs|home|Users|root|workspace|scratch|nas|nfs|lustre)/'
        r'[^\s\"\'<>),;`]+|[A-Za-z]:\\Users\\[^\s\"\']+'
    ),
    'private IPv4 address': re.compile(
        r'(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|'
        r'172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\d.])'
    ),
    'credential token': re.compile(
        r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|'
        r'hf_[A-Za-z0-9]{20,}|olp_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b'
    ),
    'private key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'credential in URL': re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
    'literal credential assignment': re.compile(
        r'''(?i)\b(?:api[_-]?key|access[_-]?key(?:[_-]?id)?|secret[_-]?key|password|auth[_-]?token|access[_-]?token)\s*[:=]\s*["'][^"'\n]{8,}["']'''
    ),
}


def privacy_findings(text):
    """Report locations and categories, never the matched sensitive values."""
    findings = []
    for category, pattern in PRIVACY_PATTERNS.items():
        for match in pattern.finditer(text):
            if category == 'private IPv4 address':
                start = text.rfind('\n', 0, match.start()) + 1
                end = text.find('\n', match.end())
                line = text[start:end if end >= 0 else len(text)]
                # NVIDIA cuRAND uses four-component versions that resemble IPs.
                # Only exempt a complete package-version JSON field, not bare addresses.
                if re.fullmatch(r'\s*"nvidia-curand-cu\d+"\s*:\s*"\d+(?:\.\d+){3}"\s*,?\s*', line):
                    continue
            findings.append((text.count('\n', 0, match.start()) + 1, category))
    return findings


def release_text_files(repo_root):
    """Include tracked and untracked release files, excluding local build output."""
    if (repo_root / '.git').exists():
        output = subprocess.check_output(
            ['git', '-C', str(repo_root), 'ls-files', '-co', '--exclude-standard', '-z']
        )
        candidates = [repo_root / name for name in set(output.decode().split('\0')) if name]
    else:
        excluded = {'.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'dist'}
        candidates = [p for p in repo_root.rglob('*')
                      if not any(part in excluded or part.endswith('.egg-info')
                                 for part in p.relative_to(repo_root).parts)]
    for path in sorted(candidates):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeError:
            continue
        yield path, text


def check(root=ROOT):
    errors = []
    modules = {}
    namespaces = set()
    sources = sorted((root / 'src').rglob('*.py'))
    for path in sources:
        parts = list(path.relative_to(root / 'src').with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        module = '.'.join(parts)
        modules[module] = path
        namespaces.update('.'.join(parts[:i]) for i in range(1, len(parts)))
    for path in [*sources, *sorted((root / 'scripts').glob('*.py'))]:
        try:
            tree = ast.parse(path.read_bytes())
        except SyntaxError as exc:
            errors.append(f'{path.relative_to(root)}: {exc}')
            continue
        if path not in modules.values():
            continue
        module = next(k for k, v in modules.items() if v == path)
        package = module if path.name == '__init__.py' else module.rsplit('.', 1)[0]
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    base = importlib.util.resolve_name('.' * node.level + base, package)
                names = [base]
                # Resolve relative `from . import module` references as well.
                if node.module is None:
                    names += [base + '.' + a.name for a in node.names]
            for name in names:
                if name.split('.')[0] in ('humanwam', 'openhumanscale') and name not in modules and name not in namespaces:
                    errors.append(f'{path.relative_to(root)}: missing local import {name}')
    # Cover comments, docstrings, configs, scripts, and release metadata alike.
    repo_root = root.parent
    scanned_text_files = 0
    for path, text in release_text_files(repo_root):
        scanned_text_files += 1
        for line, category in privacy_findings(text):
            errors.append(f'{path.relative_to(repo_root)}:{line}: {category}')
    hf = json.loads((root / 'release/huggingface.json').read_text())
    if hf['namespace'] != 'DAGroup-PKU' or len(hf['collections']) != 1:
        errors.append('HF layout must use one collection under DAGroup-PKU')
    return dict(source_files=len(sources),
                scanned_text_files=scanned_text_files,
                reference_profiles=len(list((root / 'configs/reference').glob('*.yaml'))), errors=errors,
                static_checks_passed=not errors, runtime_verified=False)


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, indent=2))
    sys.exit(1 if result['errors'] else 0)
