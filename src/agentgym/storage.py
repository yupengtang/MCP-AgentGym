"""Artifact storage: local filesystem or S3, behind one interface.

Trajectories, adapters, datasets and reports are produced on whatever node Slurm
allocated and consumed somewhere else -- by a training job, by an evaluation sweep, or by
a reader months later. On a single cluster a shared filesystem is enough. Anywhere else
(a second cluster, an EKS deployment, a colleague reproducing a result) it is not.

`open_store("file:///path")` and `open_store("s3://bucket/prefix")` return the same
interface, so the rest of the codebase never learns which one it is talking to.

The S3 backend is exercised in tests against `moto`, which intercepts the AWS API at the
botocore layer -- the boto3 call paths, pagination, and error handling here are all real
code under test. What has *not* been verified is behaviour against a live AWS account
(IAM, bucket policies, network); that is stated in DESIGN.md rather than implied away.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ArtifactStore(ABC):
    """Put, get, list. Deliberately small -- artifacts are written once and read whole."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> str: ...
    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...
    @abstractmethod
    def exists(self, key: str) -> bool: ...
    @abstractmethod
    def list(self, prefix: str = "") -> list[str]: ...
    @abstractmethod
    def uri(self, key: str) -> str: ...

    def put_file(self, key: str, path: Path) -> str:
        return self.put_bytes(key, Path(path).read_bytes())

    def get_file(self, key: str, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.get_bytes(key))
        return path

    def put_text(self, key: str, text: str) -> str:
        return self.put_bytes(key, text.encode())

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode()

    def put_tree(self, key_prefix: str, directory: Path) -> list[str]:
        """Upload a directory (an adapter, a run's trajectories) preserving structure."""
        directory = Path(directory)
        out = []
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            rel = path.relative_to(directory).as_posix()
            out.append(self.put_file(f"{key_prefix.rstrip('/')}/{rel}", path))
        return out

    def get_tree(self, key_prefix: str, directory: Path) -> Path:
        directory = Path(directory)
        prefix = key_prefix.rstrip("/") + "/"
        for key in self.list(prefix):
            self.get_file(key, directory / key[len(prefix):])
        return directory


@dataclass
class LocalStore(ArtifactStore):
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Reject traversal: a key is a relative location inside the store, never a way out.
        resolved = (self.root / key).resolve()
        if not str(resolved).startswith(str(self.root.resolve())):
            raise ValueError(f"key escapes the store root: {key!r}")
        return resolved

    def put_bytes(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.uri(key)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str = "") -> list[str]:
        base = self.root
        return sorted(
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file() and p.relative_to(base).as_posix().startswith(prefix)
        )

    def uri(self, key: str) -> str:
        return f"file://{self._path(key)}"

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)


@dataclass
class S3Store(ArtifactStore):
    """S3-backed artifacts. Works against AWS, or any S3-compatible endpoint."""

    bucket: str
    prefix: str = ""
    region: str | None = None
    endpoint_url: str | None = None
    _client: object = field(default=None, init=False, repr=False)

    @property
    def client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                region_name=self.region or os.environ.get("AWS_REGION", "us-east-1"),
                endpoint_url=self.endpoint_url or os.environ.get("AGENTGYM_S3_ENDPOINT"),
            )
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.prefix.strip('/')}/{key}".lstrip("/") if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> str:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)
        return self.uri(key)

    def get_bytes(self, key: str) -> bytes:
        obj = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str = "") -> list[str]:
        # Paginate: a run's trajectories routinely exceed the 1000-key page limit, and a
        # silently truncated listing would look like missing data.
        paginator = self.client.get_paginator("list_objects_v2")
        full = self._key(prefix)
        base = (self.prefix.strip("/") + "/") if self.prefix else ""
        out: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for item in page.get("Contents", []):
                out.append(item["Key"][len(base):] if base else item["Key"])
        return sorted(out)

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"

    def ensure_bucket(self) -> None:
        """Create the bucket if absent. Convenience for tests and first-time setup."""
        from botocore.exceptions import ClientError

        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError:
            region = self.region or os.environ.get("AWS_REGION", "us-east-1")
            kwargs = {"Bucket": self.bucket}
            if region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
            self.client.create_bucket(**kwargs)


def open_store(uri: str | None = None) -> ArtifactStore:
    """Build a store from a URI. Defaults to $AGENTGYM_ARTIFACT_URI, else local artifacts."""
    from agentgym import config

    uri = uri or os.environ.get("AGENTGYM_ARTIFACT_URI")
    if not uri:
        return LocalStore(config.ARTIFACTS_DIR)

    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return LocalStore(Path(parsed.path or uri))
    if parsed.scheme == "s3":
        return S3Store(bucket=parsed.netloc, prefix=parsed.path.strip("/"))
    raise ValueError(f"unsupported artifact URI scheme: {uri!r}")


__all__ = ["ArtifactStore", "LocalStore", "S3Store", "open_store"]
