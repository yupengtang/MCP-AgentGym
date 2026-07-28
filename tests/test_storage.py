"""Artifact store, local and S3.

The S3 tests run against `moto`, which intercepts the AWS API inside botocore. That means
the boto3 client construction, key composition, pagination and error handling in
`S3Store` are all real code under test -- only the remote service is simulated. What these
tests do *not* cover is a live AWS account: IAM, bucket policy, and network behaviour are
unverified, and DESIGN.md says so rather than letting the green checkmark imply otherwise.
"""

from __future__ import annotations

import json

import pytest

from agentgym.storage import LocalStore, S3Store, open_store

moto = pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

BUCKET = "agentgym-test"


@pytest.fixture
def local(tmp_path):
    return LocalStore(tmp_path / "artifacts")


@pytest.fixture
def s3():
    with mock_aws():
        store = S3Store(bucket=BUCKET, prefix="runs/exp1", region="us-east-1")
        store.ensure_bucket()
        yield store


def _roundtrip(store):
    store.put_text("report.md", "# results\n")
    assert store.get_text("report.md") == "# results\n"
    assert store.exists("report.md")
    assert not store.exists("missing.md")


def test_local_roundtrip(local):
    _roundtrip(local)


def test_s3_roundtrip(s3):
    _roundtrip(s3)


def test_s3_uri_includes_prefix(s3):
    assert s3.uri("report.md") == f"s3://{BUCKET}/runs/exp1/report.md"


def test_listing_is_prefix_scoped_and_strips_the_store_prefix(s3):
    s3.put_text("trajectories/a.jsonl", "{}")
    s3.put_text("trajectories/b.jsonl", "{}")
    s3.put_text("adapters/config.json", "{}")

    assert s3.list("trajectories/") == ["trajectories/a.jsonl", "trajectories/b.jsonl"]
    assert "adapters/config.json" in s3.list()


def test_s3_listing_paginates_beyond_one_page(s3):
    """A run's trajectories routinely exceed S3's 1000-key page limit; a truncated
    listing would look exactly like missing data."""
    for i in range(1105):
        s3.put_text(f"many/{i:05d}.json", "{}")
    keys = s3.list("many/")
    assert len(keys) == 1105, f"pagination dropped keys: got {len(keys)}"


@pytest.mark.parametrize("kind", ["local", "s3"])
def test_tree_upload_and_download_preserves_structure(kind, tmp_path, s3, local):
    store = s3 if kind == "s3" else local
    src = tmp_path / "adapter"
    (src / "nested").mkdir(parents=True)
    (src / "adapter_config.json").write_text(json.dumps({"r": 32}))
    (src / "nested" / "weights.bin").write_bytes(b"\x00\x01\x02")

    store.put_tree("adapters/sft-v4", src)
    out = store.get_tree("adapters/sft-v4", tmp_path / "restored")

    assert json.loads((out / "adapter_config.json").read_text())["r"] == 32
    assert (out / "nested" / "weights.bin").read_bytes() == b"\x00\x01\x02"


def test_local_store_rejects_path_traversal(local):
    with pytest.raises(ValueError):
        local.put_text("../escaped.txt", "nope")


def test_open_store_dispatches_on_scheme(tmp_path):
    assert isinstance(open_store(f"file://{tmp_path}"), LocalStore)
    store = open_store("s3://my-bucket/some/prefix")
    assert isinstance(store, S3Store)
    assert store.bucket == "my-bucket" and store.prefix == "some/prefix"
    with pytest.raises(ValueError):
        open_store("gs://unsupported/bucket")


def test_missing_key_raises_rather_than_returning_empty(s3):
    """Reading an absent artifact must fail loudly. Returning b'' would let a downstream
    stage train on nothing and report success."""
    from botocore.exceptions import ClientError

    with pytest.raises(ClientError):
        s3.get_bytes("does/not/exist.json")
