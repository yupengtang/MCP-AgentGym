#!/usr/bin/env python3
"""Verify the S3 artifact store against a real AWS account.

The S3 backend is unit-tested against `moto`, which proves the boto3 call paths are
correct but says nothing about IAM policy, bucket policy, or region behaviour. This
script closes that gap by exercising the real service, and it is deliberately kept
runnable from wherever the credentials actually live -- it needs nothing from the
MCP-AgentGym cluster environment beyond `storage.py`.

    python scripts/verify_aws.py --bucket agentgym-artifacts-<suffix>
    python scripts/verify_aws.py --bucket <existing-bucket> --no-create

What it does, in order, reporting each step:
    1. sts:GetCallerIdentity                     (read-only, free)
    2. s3:ListBuckets                            (read-only, free)
    3. create the bucket if absent               (only with --create, the default)
    4. put / head / get / list(paginated) / tree round-trip
    5. delete every object it created, and the bucket if it created it

Cost: a handful of PUT/GET requests on a few KB. Well under one US cent, and step 5
removes everything unless --keep is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if REPO_SRC.exists():
    sys.path.insert(0, str(REPO_SRC))


def main() -> int:
    p = argparse.ArgumentParser(description="Verify S3 artifact storage against real AWS")
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="agentgym-verify")
    p.add_argument("--region", default=None, help="defaults to the profile's region")
    p.add_argument("--create", dest="create", action="store_true", default=True)
    p.add_argument("--no-create", dest="create", action="store_false",
                   help="Use an existing bucket; never call CreateBucket.")
    p.add_argument("--keep", action="store_true", help="Leave the test objects in place.")
    args = p.parse_args()

    import boto3
    from botocore.exceptions import ClientError

    from agentgym.storage import S3Store

    # Fail with an instruction rather than a stack trace: the overwhelmingly common
    # reason this script is run somewhere it cannot work is that the credentials live on
    # a different machine than the code.
    if boto3.session.Session().get_credentials() is None:
        print("No AWS credentials found in this environment.\n"
              "Run this on the machine where `aws configure` was done, or export\n"
              "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION there.\n"
              "Do not copy long-lived keys onto a shared HPC login node.",
              file=sys.stderr)
        return 2

    results: dict[str, object] = {}

    def step(name: str, fn):
        try:
            value = fn()
            print(f"  OK    {name}" + (f"  -> {value}" if value else ""))
            results[name] = value if value is not None else True
            return value
        except Exception as exc:  # noqa: BLE001 - report, do not abort the whole run
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            results[name] = f"FAILED: {type(exc).__name__}: {exc}"
            raise

    print("== identity ==")
    ident = step("sts:GetCallerIdentity",
                 lambda: boto3.client("sts").get_caller_identity()["Arn"])
    region = args.region or boto3.session.Session().region_name or "us-east-1"
    print(f"  region: {region}")

    print("== bucket ==")
    s3 = boto3.client("s3", region_name=region)
    step("s3:ListBuckets",
         lambda: len(s3.list_buckets().get("Buckets", [])))

    created_bucket = False
    try:
        s3.head_bucket(Bucket=args.bucket)
        print(f"  OK    bucket exists: {args.bucket}")
    except ClientError:
        if not args.create:
            print(f"  FAIL  bucket {args.bucket} not found and --no-create was given")
            return 1
        kwargs: dict = {"Bucket": args.bucket}
        if region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        step("s3:CreateBucket", lambda: s3.create_bucket(**kwargs) and args.bucket)
        created_bucket = True

    store = S3Store(bucket=args.bucket, prefix=args.prefix, region=region)

    print("== round-trip ==")
    step("put_text", lambda: store.put_text("hello.txt", "agentgym"))
    step("get_text", lambda: store.get_text("hello.txt"))
    assert store.get_text("hello.txt") == "agentgym", "content mismatch on round-trip"
    step("exists(present)", lambda: store.exists("hello.txt"))
    step("exists(absent)", lambda: not store.exists("nope.txt"))

    print("== tree round-trip ==")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "adapter"
        (src / "nested").mkdir(parents=True)
        (src / "adapter_config.json").write_text(json.dumps({"r": 32}))
        (src / "nested" / "weights.bin").write_bytes(b"\x00\x01\x02")
        step("put_tree", lambda: len(store.put_tree("adapter", src)))
        out = step("get_tree", lambda: store.get_tree("adapter", Path(tmp) / "back"))
        assert (out / "nested" / "weights.bin").read_bytes() == b"\x00\x01\x02"

    print("== pagination (>1000 keys) ==")
    # The one behaviour moto cannot fully vouch for: real ListObjectsV2 continuation.
    step("put 1050 keys", lambda: [store.put_text(f"page/{i:05d}", "x")
                                   for i in range(1050)] and 1050)
    listed = step("list(page/)", lambda: len(store.list("page/")))
    assert listed == 1050, f"pagination dropped keys: {listed}/1050"

    if not args.keep:
        print("== cleanup ==")
        keys = store.list("")
        step("delete objects", lambda: _delete_all(s3, args.bucket, store, keys))
        if created_bucket:
            step("s3:DeleteBucket", lambda: s3.delete_bucket(Bucket=args.bucket)
                 and args.bucket)

    print("\nall checks passed against real AWS")
    print(json.dumps({"identity": ident, "region": region,
                      "bucket": args.bucket, "created": created_bucket}, indent=2))
    return 0


def _delete_all(s3, bucket: str, store, keys: list[str]) -> int:
    full = [{"Key": store._key(k)} for k in keys]
    for i in range(0, len(full), 1000):          # DeleteObjects caps at 1000 per call
        s3.delete_objects(Bucket=bucket, Delete={"Objects": full[i:i + 1000]})
    return len(full)


if __name__ == "__main__":
    raise SystemExit(main())
