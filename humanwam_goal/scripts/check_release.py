#!/usr/bin/env python3
"""Check local imports, release metadata and accidental disclosure, without importing ML libraries."""

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRIVACY_PATTERNS = {
    "internal absolute path": re.compile(
        r"(?<![\w./:])/(?:mnt|pfs|home|Users|root|workspace|scratch|nas|nfs|lustre|cpfs\d*)/"
        r"[^\s\"'<>),;`]+|[A-Za-z]:\\Users\\[^\s\"']+"
    ),
    "object storage URL": re.compile(r"\boss://[^\s\"'<>]+"),
    "private IPv4 address": re.compile(
        r"(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\d.])"
    ),
    "credential token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"hf_[A-Za-z0-9]{20,}|olp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|LTAI[A-Za-z0-9]{12,})\b"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "credential in URL": re.compile(r"https?://[^\s/:@]+:[^\s/@]+@"),
    "literal credential assignment": re.compile(
        r"""(?i)\b(?:api[_-]?key|access[_-]?key(?:[_-]?id)?|secret[_-]?key|password|auth[_-]?token|access[_-]?token)\s*[:=]\s*["'][^"'\n]{8,}["']"""
    ),
}


def privacy_findings(text):
    """Report locations and categories, never the matched sensitive values."""
    return [
        (text.count("\n", 0, match.start()) + 1, category)
        for category, pattern in PRIVACY_PATTERNS.items()
        for match in pattern.finditer(text)
    ]


def release_text_files(root):
    """Tracked and untracked release files under this directory, excluding ignored build output."""
    output = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z", "."]
    )
    for name in sorted(set(output.decode().split("\0")) - {""}):
        path = root / name
        if not path.is_file():
            continue
        try:
            yield path, path.read_text(encoding="utf-8")
        except UnicodeError:
            continue


def local_import_errors(root):
    errors, modules, namespaces = [], {}, set()
    sources = sorted((root / "src").rglob("*.py"))
    for path in sources:
        parts = list(path.relative_to(root / "src").with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
        namespaces.update(".".join(parts[:i]) for i in range(1, len(parts)))
    for module, path in modules.items():
        package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_bytes())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                names = [base] + (
                    [base + "." + a.name for a in node.names] if node.module is None else []
                )
            errors += [
                f"{path.relative_to(root)}: missing local import {name}"
                for name in names
                if name.split(".")[0] == "goalwam"
                and name not in modules
                and name not in namespaces
            ]
            errors += [
                f"{path.relative_to(root)}: imports the training framework {name}"
                for name in names
                if name.split(".")[0] in ("recipes", "veomni", "cosmos_framework")
            ]
    return errors, len(sources)


def check(root=ROOT):
    errors, sources = local_import_errors(root)
    scanned = 0
    for path, text in release_text_files(root):
        scanned += 1
        errors += [
            f"{path.relative_to(root)}:{line}: {category}"
            for line, category in privacy_findings(text)
        ]
    hf = json.loads((root / "release/huggingface.json").read_text())
    repos = [r["repo_id"] for r in hf["repositories"]]
    if (
        hf["namespace"] != "DAGroup-PKU"
        or len(hf["collections"]) != 1
        or hf["repository_count"] != len(repos)
        or repos != hf["collections"][0]["repo_ids"]
        or any(not r.startswith("DAGroup-PKU/") for r in repos)
    ):
        errors.append(
            "The Hugging Face layout must list its DAGroup-PKU repositories in one collection"
        )
    paper = json.loads((root / "release/paper_alignment.json").read_text())
    scores = paper["main_result"]["scores"]
    if any(round(s["successes"] / s["episodes"] * 100, 2) != s["sr"] for s in scores) or round(
        sum(s["sr"] for s in scores) / len(scores), 2
    ) != float(paper["main_result"]["mean"]):
        errors.append("paper_alignment.json success rates do not match their counts")
    return dict(
        source_files=sources,
        scanned_text_files=scanned,
        errors=errors,
        static_checks_passed=not errors,
    )


if __name__ == "__main__":
    result = check()
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["errors"] else 0)
