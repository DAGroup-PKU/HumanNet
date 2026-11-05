"""Release metadata and disclosure checks; no model, GPU or network."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_check", ROOT / "scripts/check_release.py")
release_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)


def test_release_tree_passes_its_static_checks():
    result = release_check.check()
    assert result["errors"] == [] and result["scanned_text_files"] > 20


def test_main_result_matches_the_benchmark_counts():
    paper = json.loads((ROOT / "release/paper_alignment.json").read_text())
    scores = {s["phase"]: s for s in paper["main_result"]["scores"]}
    assert (scores["clean"]["successes"], scores["random"]["successes"]) == (4635, 4230)
    assert paper["main_result"]["mean"] == "88.65" and paper["source_profile"] == "midtrain52k20k"


def test_evaluation_profile_matches_the_released_serving_protocol():
    profile = yaml.safe_load((ROOT / "configs/robotwin.yaml").read_text())
    assert profile == dict(
        schema="humanwam_goal.robotwin.v1",
        profile="midtrain52k20k",
        action_type="qpos",
        replan_steps=32,
        generation_seed=9000,
    )


@pytest.mark.parametrize("prefix", ["cpfs01", "home", "mnt", "workspace"])
def test_internal_paths_are_flagged(prefix):
    sample = "notes\n" + "/" + prefix + "/someone/checkpoints/run"
    assert release_check.privacy_findings(sample) == [(2, "internal absolute path")]


def test_storage_urls_and_tokens_are_flagged_without_echoing_them():
    token = "hf_" + "A" * 32
    url = "oss" + "://bucket/private/run"
    findings = release_check.privacy_findings(f"{token}\n{url}")
    assert sorted(findings) == [(1, "credential token"), (2, "object storage URL")]
    assert token not in repr(findings)


def test_public_references_are_allowed():
    sample = "\n".join(
        [
            "./checkpoints/humanwam-goal",
            "https://huggingface.co/collections/DAGroup-PKU/open-humanscale",
            "/proc/self/maps",
            "/usr/local/cuda",
            "${target}/runtime",
        ]
    )
    assert release_check.privacy_findings(sample) == []
