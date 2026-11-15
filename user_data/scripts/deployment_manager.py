"""
Docker SDK wrapper for the deployment reconciler (Phase 1 of the
deployment-lifecycle work — see ``docs/deployment-lifecycle.md``).

Phase 1 is pure infrastructure: this module exists, has tests, and is
NOT called from anywhere except those tests. Phase 2 will introduce
``job_reconcile_deployments`` which uses it in observe-only mode.
Phase 3 will start acting for real.

Design constraints baked in here:

  * **Label-scoped operations.** Every container we create carries
    ``first_duck.role=deployed-strategy`` plus per-strategy labels.
    Every list/stop/remove method filters by that role label and will
    refuse to touch a container without it. This is the primary
    safety net against accidentally killing the orchestrator itself,
    ft-monitor, or the legacy ft-momentum / ft-sweep containers
    during migration.

  * **Dry-run mode.** Every action method accepts ``dry_run=True``.
    Under dry-run we log exactly what would happen but never call the
    docker daemon to start/stop/remove. The reconciler runs in
    dry-run mode through all of Phase 2 (observe-only).

  * **Internal Docker network only.** We attach deployed containers
    to the existing project network (``FT_DOCKER_NETWORK`` env, default
    ``first-duck-trade_default``). No host port mapping — every deployed
    strategy exposes its REST API on container port 8080 reachable only
    over the Docker network. The orchestrator talks to them via
    container name (``http://ft-deployed-<slug>:8080``). This sidesteps
    the host-port collision problem that would hit instantly if N
    containers tried to bind 8080 on the host.

  * **No registry mutations from here.** This module is the
    container-side abstraction only. The reconciler (Phase 2) decides
    when to call which method based on the registry and writes the
    state transitions back to the registry separately. Pure separation
    of concerns.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("deployment_manager")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLE_LABEL = "first_duck.role"
ROLE_VALUE = "deployed-strategy"
STRATEGY_ID_LABEL = "first_duck.strategy_id"
STRATEGY_NAME_LABEL = "first_duck.strategy_name"
DEPLOYMENT_GENERATION_LABEL = "first_duck.deployment_generation"

CONTAINER_NAME_PREFIX = "ft-deployed-"
FREQTRADE_IMAGE = "freqtradeorg/freqtrade:stable"
DEFAULT_NETWORK = "first-duck-trade_default"

# Names + paths INSIDE the container. The host paths come via the
# bind-mount of user_data — same as ft-momentum / ft-sweep.
TEMPLATE_PATH_IN_CONTAINER = "/freqtrade/user_data/configs/config-deployed.json.template"
RENDER_SCRIPT_IN_CONTAINER = "/freqtrade/user_data/scripts/render_config.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def strategy_slug(strategy_name: str) -> str:
    """Lowercase, hyphen-separated, safe for use in container names + file
    paths. ``FundingContrarianBreakout_v3`` → ``funding-contrarian-breakout-v3``.
    Idempotent.

    We split on camelCase boundaries BEFORE lowercasing — otherwise
    ``FundingContrarianBreakout_v3`` collapses to
    ``fundingcontrarianbreakout-v3`` (only the underscore inserts a
    hyphen), which is unreadable in container names + logs.
    """
    # Insert hyphen at camelCase boundaries (lower→Upper, Upper→UpperLower)
    spaced = _CAMEL_BOUNDARY_RE.sub("-", strategy_name)
    # Then lowercase + collapse any non-alphanumeric runs
    s = _NON_ALNUM_RE.sub("-", spaced.lower()).strip("-")
    if not s:
        raise ValueError(f"strategy_name {strategy_name!r} produces empty slug")
    return s


def container_name_for(strategy_name: str) -> str:
    """Canonical container name. Same input always → same name, so the
    reconciler can do exact-match lookups against ``docker ps``."""
    return f"{CONTAINER_NAME_PREFIX}{strategy_slug(strategy_name)}"


# ---------------------------------------------------------------------------
# Container spec
# ---------------------------------------------------------------------------

@dataclass
class DeployedContainerSpec:
    """Everything needed to start one deployed-strategy container.

    Held as data so tests can build specs deterministically and so the
    spec → docker.run() translation is the only piece that needs the
    real SDK. The spec itself is pure Python.
    """
    strategy_id: int
    strategy_name: str         # the Freqtrade class name
    deployment_generation: int  # bumps on every redeploy; helps spot stale state

    # Env propagated INTO the container. The render script substitutes
    # ${VAR} placeholders in the template, so these must include OKX
    # creds + Freqtrade REST API password + JWT secret + STRATEGY_NAME
    # + STRATEGY_SLUG. The reconciler builds this dict from its own env.
    env: dict[str, str] = field(default_factory=dict)

    # Bind mounts. By default we mirror ft-momentum's setup.
    volumes: dict[str, dict] = field(default_factory=dict)

    # Docker network the container joins so the orchestrator can reach
    # its REST API by container name.
    network: str = DEFAULT_NETWORK

    # Restart policy. unless-stopped = auto-restart on crash, but
    # respect explicit `docker stop` from the reconciler.
    restart_policy: dict = field(default_factory=lambda: {"Name": "unless-stopped"})

    image: str = FREQTRADE_IMAGE

    @property
    def container_name(self) -> str:
        return container_name_for(self.strategy_name)

    @property
    def slug(self) -> str:
        return strategy_slug(self.strategy_name)

    @property
    def labels(self) -> dict[str, str]:
        """Labels Docker sees on the container. The reconciler filters
        ``docker ps`` on the role label, so every container we manage
        MUST carry it — without it, the container is invisible to the
        reconciler (which means we'd silently lose it)."""
        return {
            ROLE_LABEL: ROLE_VALUE,
            STRATEGY_ID_LABEL: str(self.strategy_id),
            STRATEGY_NAME_LABEL: self.strategy_name,
            DEPLOYMENT_GENERATION_LABEL: str(self.deployment_generation),
        }

    # Entrypoint MUST be overridden — the freqtrade docker image's default
    # entrypoint is `freqtrade`, which would then interpret our shell
    # script as a freqtrade subcommand (and fail with "invalid choice").
    # Shipping entrypoint=/bin/sh -c lets us run the render step then
    # exec freqtrade with the right args.
    @property
    def container_entrypoint(self) -> list[str]:
        return ["/bin/sh", "-c"]

    def freqtrade_command(self) -> list[str]:
        """The shell script the container runs at startup. Renders the
        template to /tmp (NOT host-mounted — see PR #38) then execs
        freqtrade. Per-strategy db and log files DO go to bind-mounted
        user_data so they persist across container restarts and are
        inspectable from the host.

        Returns a single-element list — Docker SDK ``containers.run``
        with our ``container_entrypoint`` set to ``["/bin/sh", "-c"]``
        expects exactly one shell-string argument as the command.
        """
        slug = self.slug
        # The render script's placeholder for the strategy slug is
        # ${STRATEGY_SLUG}; the container env (see DeployedContainerSpec.env
        # built in orchestrator._build_deployed_env) supplies it.
        script = (
            "set -e\n"
            f"python {RENDER_SCRIPT_IN_CONTAINER} "
            f"{TEMPLATE_PATH_IN_CONTAINER} "
            f"/tmp/config-deployed-{slug}.json\n"
            "exec freqtrade trade "
            f"--logfile /freqtrade/user_data/logs/ft-deployed-{slug}.log "
            f"--db-url sqlite:////freqtrade/user_data/tradesv3-deployed-{slug}.sqlite "
            f"--config /tmp/config-deployed-{slug}.json "
            # Generated strategies live under candidates/, not the default
            # user_data/strategies/. Without this freqtrade can't import the
            # class and dies at startup with "This class does not exist".
            "--strategy-path /freqtrade/user_data/strategies/candidates "
            f"--strategy {self.strategy_name}"
        )
        return [script]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class DeploymentManager:
    """Thin wrapper over the Docker SDK with the safety constraints
    described in the module docstring. All actions go through this
    object so labels, dry-run, and refuse-unlabeled checks happen
    exactly once."""

    def __init__(self, docker_client=None):
        """``docker_client`` is injectable for tests. Real callers pass
        nothing and we lazy-import ``docker`` from the SDK package."""
        self._client_override = docker_client
        self._client_cache = None

    @property
    def _client(self):
        if self._client_override is not None:
            return self._client_override
        if self._client_cache is None:
            import docker  # noqa: lazy import — tests inject a mock instead
            self._client_cache = docker.from_env()
        return self._client_cache

    # -----------------------------------------------------------------
    # Read-only
    # -----------------------------------------------------------------

    def list_deployed(self) -> list[dict]:
        """Containers carrying our role label. The reconciler's view of
        "what's actually running". Includes stopped/exited containers
        so the reconciler can detect crashed strategies and clean them
        up; the caller filters on ``status`` if it only wants live."""
        containers = self._client.containers.list(
            all=True,
            filters={"label": f"{ROLE_LABEL}={ROLE_VALUE}"},
        )
        return [
            {
                "name": c.name,
                "id": c.id,
                "status": c.status,
                "labels": c.labels,
                "strategy_id": int(c.labels.get(STRATEGY_ID_LABEL, "-1")),
                "strategy_name": c.labels.get(STRATEGY_NAME_LABEL, ""),
                "deployment_generation": int(
                    c.labels.get(DEPLOYMENT_GENERATION_LABEL, "0")
                ),
            }
            for c in containers
        ]

    def inspect(self, strategy_name: str) -> Optional[dict]:
        """Find the container for one strategy (if any) by exact name
        match. Returns the same dict shape as ``list_deployed`` items,
        or None if not present."""
        target = container_name_for(strategy_name)
        for entry in self.list_deployed():
            if entry["name"] == target:
                return entry
        return None

    # -----------------------------------------------------------------
    # Mutations (all support dry_run)
    # -----------------------------------------------------------------

    def start(self, spec: DeployedContainerSpec, *, dry_run: bool = True) -> dict:
        """Spin up a new deployed-strategy container per the spec.

        Returns a description dict including the would-be container name
        and the dry-run flag. When dry_run=True (default) no real Docker
        call happens — we log the action and return.

        Safety: if a container with the same name already exists, we
        refuse to overwrite it. The reconciler must explicitly stop the
        existing container first. This prevents a silent generation
        bump from clobbering a still-running strategy.
        """
        existing = self.inspect(spec.strategy_name)
        if existing is not None:
            raise RuntimeError(
                f"refusing to start {spec.container_name}: a container with that "
                f"name already exists (status={existing['status']!r}). Stop it "
                f"first via stop_*() before starting a new one."
            )

        action = {
            "action": "start",
            "container_name": spec.container_name,
            "strategy_id": spec.strategy_id,
            "strategy_name": spec.strategy_name,
            "deployment_generation": spec.deployment_generation,
            "image": spec.image,
            "network": spec.network,
            "labels": spec.labels,
            "command": spec.freqtrade_command(),
            "env_var_names_only": sorted(spec.env.keys()),  # don't log values
            "volume_count": len(spec.volumes),
            "dry_run": dry_run,
        }

        if dry_run:
            log.info(f"[dry_run] would start: {action}")
            return action

        self._client.containers.run(
            spec.image,
            name=spec.container_name,
            detach=True,
            labels=spec.labels,
            environment=spec.env,
            volumes=spec.volumes,
            network=spec.network,
            restart_policy=spec.restart_policy,
            entrypoint=spec.container_entrypoint,
            command=spec.freqtrade_command(),
        )
        log.info(f"started container {spec.container_name} (gen={spec.deployment_generation})")
        return action

    def stop_graceful(
        self,
        strategy_name: str,
        *,
        timeout_seconds: int = 30,
        dry_run: bool = True,
    ) -> dict:
        """Stop the container with a SIGTERM + grace period. Falls back
        to SIGKILL after the timeout. Freqtrade traps SIGTERM and
        attempts to close open positions before exiting cleanly.

        Safety: refuses to touch any container that doesn't carry our
        role label, even if its NAME matches our convention. (E.g. a
        manual ``docker run`` someone did for debugging.)
        """
        entry = self._require_managed(strategy_name)
        action = {
            "action": "stop_graceful",
            "container_name": entry["name"],
            "strategy_id": entry["strategy_id"],
            "strategy_name": strategy_name,
            "timeout_seconds": timeout_seconds,
            "dry_run": dry_run,
        }
        if dry_run:
            log.info(f"[dry_run] would stop_graceful: {action}")
            return action

        c = self._client.containers.get(entry["name"])
        c.stop(timeout=timeout_seconds)
        log.info(f"stopped container {entry['name']} (graceful, timeout={timeout_seconds}s)")
        return action

    def stop_hard(self, strategy_name: str, *, dry_run: bool = True) -> dict:
        """Send SIGKILL immediately, no grace period. Use only when
        ``stop_graceful`` has already failed."""
        entry = self._require_managed(strategy_name)
        action = {
            "action": "stop_hard",
            "container_name": entry["name"],
            "strategy_id": entry["strategy_id"],
            "strategy_name": strategy_name,
            "dry_run": dry_run,
        }
        if dry_run:
            log.info(f"[dry_run] would stop_hard: {action}")
            return action

        c = self._client.containers.get(entry["name"])
        c.kill()
        log.info(f"killed container {entry['name']} (hard stop)")
        return action

    def remove(self, strategy_name: str, *, dry_run: bool = True) -> dict:
        """Delete the container record (after it's stopped). Same label
        safety as stop_*."""
        entry = self._require_managed(strategy_name)
        action = {
            "action": "remove",
            "container_name": entry["name"],
            "strategy_id": entry["strategy_id"],
            "strategy_name": strategy_name,
            "dry_run": dry_run,
        }
        if dry_run:
            log.info(f"[dry_run] would remove: {action}")
            return action

        c = self._client.containers.get(entry["name"])
        c.remove(force=False)
        log.info(f"removed container {entry['name']}")
        return action

    # -----------------------------------------------------------------
    # Internal safety
    # -----------------------------------------------------------------

    def _require_managed(self, strategy_name: str) -> dict:
        """Look up the container by canonical name AND confirm it
        carries our role label. Raises if either check fails. This is
        the single chokepoint that prevents the reconciler from acting
        on something it didn't create.
        """
        target = container_name_for(strategy_name)

        # First check label-scoped list — fast and safe.
        for entry in self.list_deployed():
            if entry["name"] == target:
                return entry

        # If the name exists at all but DOESN'T carry our label, that's
        # a different beast and we must refuse. Without this branch we'd
        # raise "not found" while a same-named unlabeled container sits
        # right there, which would be confusing.
        try:
            unlabeled = self._client.containers.get(target)
        except Exception:
            raise LookupError(
                f"no managed container named {target!r} (no deployed-strategy "
                f"label match)"
            )
        raise PermissionError(
            f"refusing to operate on container {target!r}: it exists but "
            f"does not carry {ROLE_LABEL}={ROLE_VALUE!r} "
            f"(labels={unlabeled.labels!r}). Only containers created by the "
            f"reconciler are touchable."
        )
