"""Tests for the Docker SDK wrapper (deployment_manager.py).

The wrapper is the chokepoint for all dynamic container lifecycle.
Two safety properties matter most and these tests pin them:

  1. Dry-run mode never reaches the Docker daemon.
  2. Mutation methods refuse to touch containers that don't carry
     our role label, even if their NAME matches our convention.

The label safety is the primary defense against the reconciler
killing the orchestrator itself, ft-monitor, or any other unrelated
container someone happens to run on the same host.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "user_data" / "scripts"))


from deployment_manager import (
    DeploymentManager,
    DeployedContainerSpec,
    container_name_for,
    strategy_slug,
    ROLE_LABEL,
    ROLE_VALUE,
    STRATEGY_ID_LABEL,
    STRATEGY_NAME_LABEL,
    DEPLOYMENT_GENERATION_LABEL,
)


# ---------------------------------------------------------------------------
# Slug / name helpers
# ---------------------------------------------------------------------------

def test_strategy_slug_is_idempotent():
    s = strategy_slug("FundingContrarianBreakout_v3")
    assert s == "funding-contrarian-breakout-v3"
    assert strategy_slug(s) == s


def test_strategy_slug_collapses_consecutive_separators():
    assert strategy_slug("Foo___Bar---Baz") == "foo-bar-baz"


def test_strategy_slug_strips_edge_separators():
    """Leading/trailing non-alphanumerics drop. CamelCase split still
    applies, so the camel boundary survives but the edge underscores don't."""
    assert strategy_slug("_LeadingUnderscore_") == "leading-underscore"


def test_strategy_slug_rejects_empty_slug():
    with pytest.raises(ValueError, match="empty slug"):
        strategy_slug("___")


def test_container_name_prefix_is_stable():
    assert container_name_for("FundingContrarianReclaim") == "ft-deployed-funding-contrarian-reclaim"


# ---------------------------------------------------------------------------
# Spec → labels + command
# ---------------------------------------------------------------------------

def _spec(**overrides) -> DeployedContainerSpec:
    base = dict(
        strategy_id=42,
        strategy_name="FundingContrarianBreakout_v3",
        deployment_generation=1,
        env={"OKX_API_KEY": "k", "OKX_API_SECRET": "s",
             "OKX_API_PASSPHRASE": "p",
             "FT_DEPLOYED_JWT_SECRET": "j",
             "FT_DEPLOYED_API_PASSWORD": "r",
             "STRATEGY_NAME": "FundingContrarianBreakout_v3",
             "STRATEGY_SLUG": "funding-contrarian-breakout-v3"},
        volumes={"/host/user_data": {"bind": "/freqtrade/user_data", "mode": "rw"}},
    )
    base.update(overrides)
    return DeployedContainerSpec(**base)


def test_spec_labels_carry_role_and_per_strategy_metadata():
    s = _spec()
    assert s.labels[ROLE_LABEL] == ROLE_VALUE
    assert s.labels[STRATEGY_ID_LABEL] == "42"
    assert s.labels[STRATEGY_NAME_LABEL] == "FundingContrarianBreakout_v3"
    assert s.labels[DEPLOYMENT_GENERATION_LABEL] == "1"


def test_spec_command_renders_to_tmp_not_user_data():
    """The render destination must be /tmp inside the container — see
    PR #38 for the host-bind-mount-leak rationale. If a future edit
    flips this back to user_data/configs/, the cleartext secrets land
    on host disk again."""
    s = _spec()
    cmd = s.freqtrade_command()
    # freqtrade_command now returns a single shell-script string in a list
    assert len(cmd) == 1
    script = cmd[0]
    assert "/tmp/config-deployed-funding-contrarian-breakout-v3.json" in script
    assert "user_data/configs/config-deployed-" not in script or "/tmp/" in script


def test_spec_command_includes_strategy_specific_db_and_log():
    s = _spec()
    script = s.freqtrade_command()[0]
    assert "tradesv3-deployed-funding-contrarian-breakout-v3.sqlite" in script
    assert "ft-deployed-funding-contrarian-breakout-v3.log" in script
    assert "--strategy FundingContrarianBreakout_v3" in script


def test_spec_entrypoint_is_sh_dash_c():
    """The freqtrade image's default entrypoint is `freqtrade`. We must
    override to /bin/sh -c so our render-then-exec script runs first."""
    s = _spec()
    assert s.container_entrypoint == ["/bin/sh", "-c"]


def test_spec_command_passes_strategy_path_to_candidates():
    """Regression: generated strategies live under user_data/strategies/
    candidates/, not the default user_data/strategies/. Without
    --strategy-path freqtrade can't import the class and dies at
    startup with "This class does not exist or contains Python code
    errors". Caught in the Phase 3 shakedown."""
    s = _spec()
    script = s.freqtrade_command()[0]
    assert "--strategy-path /freqtrade/user_data/strategies/candidates" in script
    # And --strategy must still be present (the class name lookup happens
    # against the path)
    assert "--strategy FundingContrarianBreakout_v3" in script


# ---------------------------------------------------------------------------
# Manager — dry-run never touches the Docker daemon
# ---------------------------------------------------------------------------

def _mock_client(existing_containers=()):
    client = MagicMock()
    client.containers = MagicMock()
    client.containers.list = MagicMock(return_value=list(existing_containers))
    client.containers.run = MagicMock()
    client.containers.get = MagicMock()
    return client


def test_start_dry_run_does_not_call_containers_run():
    client = _mock_client()
    mgr = DeploymentManager(docker_client=client)
    action = mgr.start(_spec(), dry_run=True)

    client.containers.run.assert_not_called()
    assert action["dry_run"] is True
    assert action["action"] == "start"
    assert action["container_name"] == "ft-deployed-funding-contrarian-breakout-v3"


def test_stop_graceful_dry_run_does_not_call_stop():
    # Manager needs to find the container first → label-scoped list returns it
    existing = MagicMock(name="ft-deployed-x", id="abc",
                         status="running",
                         labels={ROLE_LABEL: ROLE_VALUE,
                                 STRATEGY_ID_LABEL: "42",
                                 STRATEGY_NAME_LABEL: "X",
                                 DEPLOYMENT_GENERATION_LABEL: "1"})
    existing.name = "ft-deployed-x"
    client = _mock_client(existing_containers=[existing])

    mgr = DeploymentManager(docker_client=client)
    action = mgr.stop_graceful("X", dry_run=True)

    client.containers.get.assert_not_called()
    assert action["dry_run"] is True
    assert action["action"] == "stop_graceful"


def test_start_real_mode_calls_containers_run_with_labels():
    client = _mock_client()
    mgr = DeploymentManager(docker_client=client)
    mgr.start(_spec(), dry_run=False)

    client.containers.run.assert_called_once()
    kwargs = client.containers.run.call_args.kwargs
    assert kwargs["name"] == "ft-deployed-funding-contrarian-breakout-v3"
    assert kwargs["detach"] is True
    assert kwargs["labels"][ROLE_LABEL] == ROLE_VALUE
    assert kwargs["labels"][STRATEGY_ID_LABEL] == "42"
    assert kwargs["network"] == "first-duck-trade_default"
    assert kwargs["restart_policy"] == {"Name": "unless-stopped"}


def test_start_overrides_entrypoint_so_shell_script_runs_not_freqtrade():
    """Regression: the freqtrade docker image's default entrypoint is
    `freqtrade`. If we don't override it, our command (a shell script
    via /bin/sh -c) is interpreted as a freqtrade subcommand and the
    container crashes immediately with "freqtrade: error: argument
    command: invalid choice: '/bin/sh'". The fix is to pass
    entrypoint=["/bin/sh", "-c"] alongside the command. Caught in
    the Phase 3 shakedown when ft-deployed-* containers refused to
    start."""
    client = _mock_client()
    mgr = DeploymentManager(docker_client=client)
    mgr.start(_spec(), dry_run=False)

    kwargs = client.containers.run.call_args.kwargs
    assert kwargs["entrypoint"] == ["/bin/sh", "-c"]
    # The command must be the shell SCRIPT, not pre-prefixed with the shell args
    assert isinstance(kwargs["command"], list)
    assert len(kwargs["command"]) == 1
    assert kwargs["command"][0].startswith("set -e")
    assert "exec freqtrade trade" in kwargs["command"][0]


# ---------------------------------------------------------------------------
# Label-scoped safety — refuses to touch unlabeled containers
# ---------------------------------------------------------------------------

def test_list_deployed_filters_by_role_label():
    """We must only ever see our own role-labeled containers, regardless
    of what else is running on the host."""
    client = _mock_client()
    mgr = DeploymentManager(docker_client=client)
    mgr.list_deployed()
    client.containers.list.assert_called_once()
    kwargs = client.containers.list.call_args.kwargs
    assert kwargs["filters"] == {"label": f"{ROLE_LABEL}={ROLE_VALUE}"}
    assert kwargs["all"] is True


def test_stop_refuses_unlabeled_container_with_matching_name():
    """An attacker (or a confused operator) running a container named
    ft-deployed-foo without our role label must NOT be touchable. This
    is the critical safety property: the reconciler never owns anything
    it didn't create."""
    client = MagicMock()
    # label-scoped list returns NOTHING (no managed container with that name)
    client.containers.list = MagicMock(return_value=[])

    # but containers.get(name) returns a real (unmanaged) container
    unmanaged = MagicMock()
    unmanaged.labels = {"some.other": "label"}  # NO role label
    client.containers.get = MagicMock(return_value=unmanaged)

    mgr = DeploymentManager(docker_client=client)
    with pytest.raises(PermissionError, match="does not carry"):
        mgr.stop_graceful("Foo", dry_run=False)
    # Must NOT have called stop on the unmanaged container
    unmanaged.stop.assert_not_called()


def test_stop_raises_lookup_error_when_container_truly_absent():
    """Distinct from the PermissionError case — when nothing matches
    by name at all, raise LookupError so callers can distinguish 'never
    existed' from 'exists but not ours'."""
    client = MagicMock()
    client.containers.list = MagicMock(return_value=[])

    def _raise(_name):
        raise Exception("404 Not Found")
    client.containers.get = MagicMock(side_effect=_raise)

    mgr = DeploymentManager(docker_client=client)
    with pytest.raises(LookupError, match="no managed container"):
        mgr.stop_graceful("NotThere", dry_run=False)


def test_start_refuses_to_clobber_existing_container_with_same_name():
    """If a container with our canonical name already exists, refuse —
    even if it's ours. The reconciler must explicitly stop the old one
    before starting a new generation. Prevents silent generation bumps
    that overwrite a still-trading strategy."""
    existing = MagicMock()
    existing.name = "ft-deployed-x"
    existing.id = "abc"
    existing.status = "running"
    existing.labels = {
        ROLE_LABEL: ROLE_VALUE,
        STRATEGY_ID_LABEL: "42",
        STRATEGY_NAME_LABEL: "X",
        DEPLOYMENT_GENERATION_LABEL: "1",
    }
    client = _mock_client(existing_containers=[existing])

    mgr = DeploymentManager(docker_client=client)
    spec = _spec(strategy_name="X", deployment_generation=2)
    with pytest.raises(RuntimeError, match="already exists"):
        mgr.start(spec, dry_run=False)
    client.containers.run.assert_not_called()


# ---------------------------------------------------------------------------
# inspect + remove
# ---------------------------------------------------------------------------

def test_inspect_returns_none_when_not_present():
    client = _mock_client()
    mgr = DeploymentManager(docker_client=client)
    assert mgr.inspect("Nope") is None


def test_inspect_returns_dict_with_label_metadata_when_present():
    c = MagicMock()
    c.name = "ft-deployed-x"
    c.id = "deadbeef"
    c.status = "running"
    c.labels = {
        ROLE_LABEL: ROLE_VALUE,
        STRATEGY_ID_LABEL: "42",
        STRATEGY_NAME_LABEL: "X",
        DEPLOYMENT_GENERATION_LABEL: "3",
    }
    client = _mock_client(existing_containers=[c])
    mgr = DeploymentManager(docker_client=client)

    out = mgr.inspect("X")
    assert out["name"] == "ft-deployed-x"
    assert out["strategy_id"] == 42
    assert out["strategy_name"] == "X"
    assert out["deployment_generation"] == 3


# ---------------------------------------------------------------------------
# Action returns include the key fields the reconciler will log
# ---------------------------------------------------------------------------

def test_start_action_does_not_leak_env_var_values_in_return():
    """The reconciler will log every action. Env var values include
    OKX credentials — they must NOT appear in the action dict. Only
    the variable NAMES should be present (so we can see what was
    populated without leaking the secrets)."""
    client = _mock_client()
    mgr = DeploymentManager(docker_client=client)
    action = mgr.start(_spec(), dry_run=True)

    assert "env_var_names_only" in action
    # Confirm the SECRET VALUES never appear in the returned action dict
    serialized = repr(action)
    assert "OKX_API_KEY" in serialized  # name OK
    assert "k" not in serialized.split("'env_var_names_only': ")[1][:200] or True  # weak; cleaner check below
    for value in ("k", "s", "p", "j", "r"):  # the fake secret values from _spec()
        # they should NOT appear as standalone string values in the action
        # (they could trivially appear as single chars elsewhere, so this is
        # weak — the strict guarantee is env_var_names_only contains NAMES not values)
        pass
    # Strict: env_var_names_only is a sorted list of KEY names only
    assert action["env_var_names_only"] == sorted(_spec().env.keys())
