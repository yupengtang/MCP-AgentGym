"""Minimal PKI for a throwaway control plane.

kube-apiserver insists on real TLS and a real client CA even for a single-user local
cluster, so every episode mints its own short-lived CA. Generation is ~200ms with
openssl, which is cheap relative to etcd start-up.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

_SAN_CONF = """\
[req]
distinguished_name=dn
[dn]
[ext]
subjectAltName=DNS:kubernetes,DNS:kubernetes.default,DNS:kubernetes.default.svc,DNS:kubernetes.default.svc.cluster.local,DNS:localhost,IP:127.0.0.1,IP:10.0.0.1
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
"""


def _run(*args: str) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"openssl failed: {' '.join(args)}\n{proc.stderr}")


@dataclass(frozen=True)
class PKI:
    dir: Path

    @property
    def ca_crt(self) -> Path: return self.dir / "ca.crt"
    @property
    def ca_key(self) -> Path: return self.dir / "ca.key"
    @property
    def apiserver_crt(self) -> Path: return self.dir / "apiserver.crt"
    @property
    def apiserver_key(self) -> Path: return self.dir / "apiserver.key"
    @property
    def sa_key(self) -> Path: return self.dir / "sa.key"
    @property
    def sa_pub(self) -> Path: return self.dir / "sa.pub"
    @property
    def front_proxy_ca_crt(self) -> Path: return self.dir / "front-proxy-ca.crt"
    @property
    def front_proxy_ca_key(self) -> Path: return self.dir / "front-proxy-ca.key"
    @property
    def front_proxy_client_crt(self) -> Path: return self.dir / "front-proxy-client.crt"
    @property
    def front_proxy_client_key(self) -> Path: return self.dir / "front-proxy-client.key"

    def client_crt(self, name: str) -> Path: return self.dir / f"{name}.crt"
    def client_key(self, name: str) -> Path: return self.dir / f"{name}.key"


# Clients the control plane needs, as (file stem, subject).
# O=system:masters grants cluster-admin; the component identities get their
# permissions from the default RBAC bootstrap policy.
_CLIENTS: tuple[tuple[str, str], ...] = (
    ("admin", "/CN=agentgym-admin/O=system:masters"),
    ("controller-manager", "/CN=system:kube-controller-manager"),
    ("scheduler", "/CN=system:kube-scheduler"),
    # The simulated kubelet acts as a node; Node authorization expects this shape.
    ("kubelet-sim", "/CN=system:node:kubelet-sim/O=system:nodes"),
    # Least-privilege identity the agent's MCP servers use (RBAC-bound per episode).
    ("agent", "/CN=agentgym-agent/O=agentgym"),
)


def generate(pki_dir: Path, *, rsa_bits: int = 2048) -> PKI:
    """Create a complete CA + leaf certs for one episode. Idempotent."""
    pki_dir.mkdir(parents=True, exist_ok=True)
    pki = PKI(pki_dir)
    if pki.ca_crt.exists() and pki.client_crt("agent").exists():
        return pki

    b = str(rsa_bits)
    _run("openssl", "genrsa", "-out", str(pki.ca_key), b)
    _run("openssl", "req", "-x509", "-new", "-nodes", "-key", str(pki.ca_key),
         "-subj", "/CN=agentgym-ca", "-days", "3650", "-out", str(pki.ca_crt))

    san = pki_dir / "san.cnf"
    san.write_text(_SAN_CONF)
    _run("openssl", "genrsa", "-out", str(pki.apiserver_key), b)
    _run("openssl", "req", "-new", "-key", str(pki.apiserver_key),
         "-subj", "/CN=kube-apiserver", "-out", str(pki_dir / "apiserver.csr"))
    _run("openssl", "x509", "-req", "-in", str(pki_dir / "apiserver.csr"),
         "-CA", str(pki.ca_crt), "-CAkey", str(pki.ca_key), "-CAcreateserial",
         "-out", str(pki.apiserver_crt), "-days", "3650",
         "-extfile", str(san), "-extensions", "ext")

    for stem, subject in _CLIENTS:
        key, crt = pki.client_key(stem), pki.client_crt(stem)
        csr = pki_dir / f"{stem}.csr"
        _run("openssl", "genrsa", "-out", str(key), b)
        _run("openssl", "req", "-new", "-key", str(key), "-subj", subject, "-out", str(csr))
        _run("openssl", "x509", "-req", "-in", str(csr), "-CA", str(pki.ca_crt),
             "-CAkey", str(pki.ca_key), "-CAcreateserial", "-out", str(crt), "-days", "3650")

    _run("openssl", "genrsa", "-out", str(pki.sa_key), b)
    _run("openssl", "rsa", "-in", str(pki.sa_key), "-pubout", "-out", str(pki.sa_pub))

    # Aggregation-layer CA. Nothing here uses an aggregated API server, but without it
    # the kube-system/extension-apiserver-authentication ConfigMap has no
    # requestheader CA bundle and the scheduler and controller-manager log a
    # "missing content for CA bundle" error several times a second.
    _run("openssl", "genrsa", "-out", str(pki.front_proxy_ca_key), b)
    _run("openssl", "req", "-x509", "-new", "-nodes", "-key", str(pki.front_proxy_ca_key),
         "-subj", "/CN=front-proxy-ca", "-days", "3650", "-out", str(pki.front_proxy_ca_crt))
    fp_csr = pki_dir / "front-proxy-client.csr"
    _run("openssl", "genrsa", "-out", str(pki.front_proxy_client_key), b)
    _run("openssl", "req", "-new", "-key", str(pki.front_proxy_client_key),
         "-subj", "/CN=front-proxy-client", "-out", str(fp_csr))
    _run("openssl", "x509", "-req", "-in", str(fp_csr),
         "-CA", str(pki.front_proxy_ca_crt), "-CAkey", str(pki.front_proxy_ca_key),
         "-CAcreateserial", "-out", str(pki.front_proxy_client_crt), "-days", "3650")
    return pki


def kubeconfig_yaml(pki: PKI, user: str, server: str) -> str:
    """A kubeconfig referencing cert files on disk (no embedding: faster, and the
    files outlive the episode only inside its own runtime dir)."""
    return f"""apiVersion: v1
kind: Config
clusters:
- name: agentgym
  cluster:
    server: {server}
    certificate-authority: {pki.ca_crt}
users:
- name: {user}
  user:
    client-certificate: {pki.client_crt(user)}
    client-key: {pki.client_key(user)}
contexts:
- name: agentgym
  context:
    cluster: agentgym
    user: {user}
current-context: agentgym
"""
