"""Runs a real Kubernetes control plane as unprivileged user processes.

etcd + kube-apiserver + kube-controller-manager + kube-scheduler, no root, no
containers, no privileged ports. Each instance owns a disjoint port block and its own
etcd data directory, so many control planes run concurrently on one node -- that is what
makes parallel episode rollout possible.

Why this instead of kind/k3s: the HPC cluster has no Docker and no /etc/subuid range, so
rootless Podman cannot map multiple UIDs and neither kind nor k3s can start. Running the
components directly sidesteps the container runtime entirely while keeping the API
semantics that actually matter (real objects, CRDs, RBAC, admission, watch/reconcile).
"""

from __future__ import annotations

import atexit
import contextlib
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from agentgym import config
from agentgym.cluster import pki as pki_mod

_LIVE: list[ControlPlane] = []


def _free_port(preferred: int, *, attempts: int = 200) -> int:
    """Return `preferred` if bindable, else scan upward. Racy by nature, so callers
    should still tolerate a component failing to bind and retry."""
    for i in range(attempts):
        port = preferred + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port near {preferred}")


class ControlPlaneError(RuntimeError):
    pass


@dataclass
class _Proc:
    name: str
    popen: subprocess.Popen
    log: Path

    def alive(self) -> bool:
        return self.popen.poll() is None

    def tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return "(no log)"


@dataclass
class ControlPlane:
    """A single-tenant Kubernetes control plane rooted at `root`."""

    root: Path
    offset: int = 0
    bin_dir: Path = field(default_factory=lambda: config.BIN_DIR)
    keep_data: bool = False

    _procs: list[_Proc] = field(default_factory=list, init=False)
    _pki: pki_mod.PKI | None = field(default=None, init=False)
    _ports: dict[str, int] = field(default_factory=dict, init=False)
    _started: bool = field(default=False, init=False)

    # --- lifecycle ----------------------------------------------------------

    def __enter__(self) -> ControlPlane:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def kubeconfig(self) -> Path:
        return self.root / "admin.kubeconfig"

    @property
    def server_url(self) -> str:
        return f"https://127.0.0.1:{self._ports['apiserver']}"

    @property
    def pki(self) -> pki_mod.PKI:
        if self._pki is None:
            raise ControlPlaneError("control plane not started")
        return self._pki

    def _bin(self, name: str) -> str:
        p = self.bin_dir / name
        if not p.exists():
            raise ControlPlaneError(
                f"missing binary {p}. Run scripts/fetch_k8s_binaries.sh first."
            )
        return str(p)

    def _spawn(self, name: str, argv: list[str]) -> _Proc:
        log = self.root / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("wb")
        popen = subprocess.Popen(
            argv, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
        )
        proc = _Proc(name, popen, log)
        self._procs.append(proc)
        return proc

    def start(self, *, attempts: int = 4) -> ControlPlane:
        """Bring up the control plane, retrying past port races.

        `_free_port` can only prove a port was free a moment ago: between the probe and
        the component's own bind, another worker -- or an orphan from a previous run --
        can take it. With N parallel shards each starting a control plane, that race is
        routine rather than exotic, so a bind failure retries on a fresh port block
        instead of failing the shard.
        """
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                return self._start_once()
            except ControlPlaneError as exc:
                last = exc
                if "address already in use" not in str(exc) or attempt == attempts - 1:
                    raise
                self.stop()
                # Jump well clear of the contended block rather than probing adjacents.
                self.offset += 17
        raise ControlPlaneError(f"control plane could not bind after {attempts} attempts: {last}")

    def _start_once(self) -> ControlPlane:
        if self._started:
            return self
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)

        want = config.PortAllocation(self.offset)
        self._ports = {
            "etcd_client": _free_port(want.etcd_client),
            "etcd_peer": _free_port(want.etcd_peer),
            "apiserver": _free_port(want.apiserver),
            "controller_manager": _free_port(want.controller_manager),
            "scheduler": _free_port(want.scheduler),
        }
        self._pki = pki_mod.generate(self.root / "pki")
        self._write_kubeconfigs()

        _LIVE.append(self)
        try:
            self._start_etcd()
            self._start_apiserver()
            self._start_controller_manager()
            self._start_scheduler()
        except Exception:
            self.stop()
            raise
        self._started = True
        return self

    def stop(self) -> None:
        for proc in reversed(self._procs):
            if proc.alive():
                with contextlib.suppress(Exception):
                    proc.popen.send_signal(signal.SIGTERM)
        deadline = time.time() + 10
        for proc in reversed(self._procs):
            with contextlib.suppress(Exception):
                proc.popen.wait(timeout=max(0.1, deadline - time.time()))
        for proc in self._procs:
            if proc.alive():
                with contextlib.suppress(Exception):
                    proc.popen.kill()
        self._procs.clear()
        self._started = False
        with contextlib.suppress(ValueError):
            _LIVE.remove(self)
        if not self.keep_data:
            shutil.rmtree(self.root / "etcd", ignore_errors=True)

    # --- components ---------------------------------------------------------

    def _start_etcd(self) -> None:
        data = self.root / "etcd"
        client = f"http://127.0.0.1:{self._ports['etcd_client']}"
        peer = f"http://127.0.0.1:{self._ports['etcd_peer']}"
        self._spawn("etcd", [
            self._bin("etcd"),
            "--name", "agentgym", "--data-dir", str(data),
            "--listen-client-urls", client, "--advertise-client-urls", client,
            "--listen-peer-urls", peer, "--initial-advertise-peer-urls", peer,
            "--initial-cluster", f"agentgym={peer}",
            "--initial-cluster-state", "new",
            "--log-level", "error",
            # Episodes are disposable: skipping fsync is a large, safe speedup here.
            "--unsafe-no-fsync",
            "--quota-backend-bytes", str(2 * 1024**3),
            "--auto-compaction-retention", "1",
        ])
        self._await(lambda: self._etcd_healthy(client), "etcd", timeout=30)

    def _etcd_healthy(self, client_url: str) -> bool:
        proc = subprocess.run(
            [self._bin("etcdctl"), f"--endpoints={client_url}", "endpoint", "health"],
            capture_output=True, text=True, timeout=5,
        )
        return proc.returncode == 0

    def _start_apiserver(self) -> None:
        p = self.pki
        self._spawn("kube-apiserver", [
            self._bin("kube-apiserver"),
            "--etcd-servers", f"http://127.0.0.1:{self._ports['etcd_client']}",
            "--secure-port", str(self._ports["apiserver"]),
            "--bind-address", "127.0.0.1", "--advertise-address", "127.0.0.1",
            "--tls-cert-file", str(p.apiserver_crt),
            "--tls-private-key-file", str(p.apiserver_key),
            "--client-ca-file", str(p.ca_crt),
            "--service-account-key-file", str(p.sa_pub),
            "--service-account-signing-key-file", str(p.sa_key),
            "--service-account-issuer", "https://kubernetes.default.svc.cluster.local",
            "--authorization-mode", "Node,RBAC",
            "--enable-admission-plugins",
            "NamespaceLifecycle,ResourceQuota,LimitRanger,ServiceAccount,DefaultStorageClass,MutatingAdmissionWebhook,ValidatingAdmissionWebhook",
            "--service-cluster-ip-range", "10.0.0.0/24",
            "--allow-privileged=true",
            # Populates extension-apiserver-authentication so the scheduler and
            # controller-manager stop erroring about a missing requestheader CA bundle.
            "--requestheader-client-ca-file", str(p.front_proxy_ca_crt),
            "--requestheader-allowed-names", "front-proxy-client",
            "--requestheader-extra-headers-prefix", "X-Remote-Extra-",
            "--requestheader-group-headers", "X-Remote-Group",
            "--requestheader-username-headers", "X-Remote-User",
            "--proxy-client-cert-file", str(p.front_proxy_client_crt),
            "--proxy-client-key-file", str(p.front_proxy_client_key),
            "--profiling=false",
            "--v", "0",
        ])
        self._await(self._apiserver_healthy, "kube-apiserver",
                    timeout=config.CONTROL_PLANE_TIMEOUT_S)

    def _apiserver_healthy(self) -> bool:
        proc = subprocess.run(
            [self._bin("kubectl"), "--kubeconfig", str(self.kubeconfig),
             "get", "--raw", "/healthz"],
            capture_output=True, text=True, timeout=5,
        )
        return proc.returncode == 0 and proc.stdout.strip() == "ok"

    def _start_controller_manager(self) -> None:
        kc = str(self.root / "controller-manager.kubeconfig")
        # The default flexvolume dir is under /usr/libexec and unwritable for us; the
        # prober logs an error every start unless it is pointed somewhere we own.
        flex = self.root / "flexvolume"
        flex.mkdir(exist_ok=True)
        self._spawn("kube-controller-manager", [
            self._bin("kube-controller-manager"),
            "--kubeconfig", kc,
            "--authentication-kubeconfig", kc, "--authorization-kubeconfig", kc,
            "--leader-elect=false",
            "--bind-address", "127.0.0.1",
            "--secure-port", str(self._ports["controller_manager"]),
            "--service-account-private-key-file", str(self.pki.sa_key),
            "--root-ca-file", str(self.pki.ca_crt),
            "--use-service-account-credentials=true",
            "--flex-volume-plugin-dir", str(flex),
            # Node lifecycle would evict our simulated pods; kubelet_sim owns node health.
            "--controllers",
            "*,-node-lifecycle-controller,-node-ipam-controller,-cloud-node-lifecycle-controller,-taint-eviction-controller",
            "--profiling=false", "--v", "0",
        ])
        # A ready controller-manager is one that has created the default ServiceAccount.
        # That single object proves the process is authenticated and its caches are live,
        # which a bare /healthz does not.
        self._await(self._default_sa_exists, "kube-controller-manager", timeout=60)

    def _default_sa_exists(self) -> bool:
        proc = self.kubectl("get", "sa", "default", "-n", "default",
                            check=False, timeout=5)
        return proc.returncode == 0

    def _start_scheduler(self) -> None:
        kc = str(self.root / "scheduler.kubeconfig")
        # Without the auth kubeconfigs the scheduler cannot resolve the
        # extension-apiserver-authentication configmap and exits with
        # "finished without leader elect" a few seconds after start.
        self._spawn("kube-scheduler", [
            self._bin("kube-scheduler"),
            "--kubeconfig", kc,
            "--authentication-kubeconfig", kc, "--authorization-kubeconfig", kc,
            "--leader-elect=false",
            "--bind-address", "127.0.0.1",
            "--secure-port", str(self._ports["scheduler"]),
            "--profiling=false", "--v", "0",
        ])

    # --- helpers ------------------------------------------------------------

    def _write_kubeconfigs(self) -> None:
        server = f"https://127.0.0.1:{self._ports['apiserver']}"
        for user, filename in (
            ("admin", "admin.kubeconfig"),
            ("controller-manager", "controller-manager.kubeconfig"),
            ("scheduler", "scheduler.kubeconfig"),
            ("kubelet-sim", "kubelet-sim.kubeconfig"),
            ("agent", "agent.kubeconfig"),
        ):
            (self.root / filename).write_text(
                pki_mod.kubeconfig_yaml(self.pki, user, server)
            )

    def _await(self, probe, what: str, *, timeout: float) -> None:
        deadline = time.time() + timeout
        last: Exception | None = None
        while time.time() < deadline:
            for proc in self._procs:
                if not proc.alive():
                    raise ControlPlaneError(
                        f"{proc.name} exited (rc={proc.popen.returncode}) while waiting "
                        f"for {what}:\n{proc.tail()}"
                    )
            try:
                if probe():
                    return
            except Exception as exc:  # probe tools may transiently fail
                last = exc
            time.sleep(0.25)
        tail = self._procs[-1].tail() if self._procs else ""
        raise ControlPlaneError(f"{what} not ready in {timeout}s (last={last})\n{tail}")

    def kubectl(self, *args: str, check: bool = True, stdin: str | None = None,
                timeout: float = 60) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [self._bin("kubectl"), "--kubeconfig", str(self.kubeconfig), *args],
            capture_output=True, text=True, input=stdin, timeout=timeout,
        )
        if check and proc.returncode != 0:
            raise ControlPlaneError(
                f"kubectl {' '.join(args)} failed: {proc.stderr.strip()}"
            )
        return proc


@atexit.register
def _cleanup_live() -> None:
    for cp in list(_LIVE):
        with contextlib.suppress(Exception):
            cp.stop()
