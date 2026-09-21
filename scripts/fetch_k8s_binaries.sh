#!/usr/bin/env bash
# Fetch the Kubernetes control-plane binaries AgentGym runs as user processes.
# Idempotent; safe to re-run.
set -euo pipefail

K8S_VERSION="${K8S_VERSION:-v1.31.4}"
ETCD_VERSION="${ETCD_VERSION:-v3.5.16}"
DEST="${AGENTGYM_BIN:-${AGENTGYM_HOME:-$HOME/.agentgym}/bin}"

mkdir -p "$DEST"
cd "$DEST"

for b in kube-apiserver kube-controller-manager kube-scheduler kubectl; do
  if [ -x "$b" ]; then echo "have $b"; continue; fi
  echo "fetching $b ($K8S_VERSION)"
  curl -fsSLO "https://dl.k8s.io/$K8S_VERSION/bin/linux/amd64/$b"
  chmod +x "$b"
done

if [ ! -x etcd ]; then
  echo "fetching etcd ($ETCD_VERSION)"
  tmp=$(mktemp -d)
  curl -fsSL "https://github.com/etcd-io/etcd/releases/download/$ETCD_VERSION/etcd-$ETCD_VERSION-linux-amd64.tar.gz" \
    | tar xz -C "$tmp" --strip-components=1
  mv "$tmp/etcd" "$tmp/etcdctl" .
  rm -rf "$tmp"
  chmod +x etcd etcdctl
fi

echo "--- installed in $DEST ---"
./kubectl version --client 2>/dev/null | head -2
./etcd --version | head -1
