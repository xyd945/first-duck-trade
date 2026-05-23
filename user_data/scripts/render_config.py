"""Render a Freqtrade config from a template by substituting ${ENV_VAR}
placeholders with values from the environment.

Used as a docker-compose entrypoint step for ft-momentum and ft-sweep so
secrets (OKX key/secret/passphrase, REST API password, JWT secret) live
in env vars rather than on-disk JSON. The materialized runtime config is
still gitignored — the template is what's checked in.

Behavior:
  * Reads TEMPLATE_PATH as text.
  * Substitutes every ``${VAR_NAME}`` occurrence with the value of the
    matching environment variable. JSON-escapes the substituted value
    (backslash + double-quote) so passwords with special chars don't
    break the resulting JSON.
  * Writes the result to OUTPUT_PATH.
  * EXITS non-zero with a clear error if any referenced env var is unset,
    rather than silently writing the literal ``${VAR}`` into the config —
    that would either fail Freqtrade with a confusing parse error or
    (worse) try to use the literal string as a credential.

Usage:
    render_config.py TEMPLATE_PATH OUTPUT_PATH
"""

from __future__ import annotations

import os
import re
import sys

# Match ${VAR_NAME} where the name is a typical shell-env identifier:
# uppercase letters, digits, underscore, starting with letter or underscore.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _json_escape(value: str) -> str:
    """Escape backslashes and double-quotes so the substituted value is
    a safe JSON string-content fragment. We don't need full JSON encoding
    because the placeholder already sits inside string-quotes in the
    template (e.g. ``"key": "${OKX_API_KEY}"``)."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render(template_text: str, env: dict[str, str] | None = None) -> str:
    """Pure function for testability. Returns the rendered text.

    Raises KeyError listing the first missing env var so the caller can
    produce a clear operator-facing error.

    Empty and whitespace-only values are treated as missing — docker
    compose's ``${VAR:-}`` default-empty form would otherwise let the
    container start with ``"key": ""`` rendered into the config, which
    then explodes later as a confusing OKX auth error instead of a clear
    startup-time "env var not set". Better to fail loud at render time.
    """
    env = env if env is not None else os.environ

    def _repl(match: re.Match) -> str:
        var = match.group(1)
        value = env.get(var)
        if value is None or value.strip() == "":
            raise KeyError(var)
        return _json_escape(value)

    return _PLACEHOLDER_RE.sub(_repl, template_text)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} TEMPLATE_PATH OUTPUT_PATH", file=sys.stderr)
        return 2
    template_path, output_path = argv[1], argv[2]
    try:
        with open(template_path) as f:
            text = f.read()
    except OSError as e:
        print(f"FATAL: cannot read template {template_path!r}: {e}", file=sys.stderr)
        return 1

    try:
        rendered = render(text)
    except KeyError as e:
        var = e.args[0]
        print(
            f"FATAL: environment variable ${{{var}}} is not set; refusing to "
            f"render {template_path!r}. Add it to your .env file and restart.",
            file=sys.stderr,
        )
        return 1

    # Open with restrictive mode in case the file doesn't exist yet (0600 =
    # owner-only). For existing files, also chmod after write — defense in
    # depth against a previous run that may have left a too-permissive file.
    try:
        # opener applies the mode on creation; umask still gets ANDed in
        # but we follow up with chmod so the effective bits are exactly 0600.
        with open(output_path, "w", opener=lambda p, fl: os.open(p, fl, 0o600)) as f:
            f.write(rendered)
        os.chmod(output_path, 0o600)
    except OSError as e:
        print(f"FATAL: cannot write {output_path!r}: {e}", file=sys.stderr)
        return 1

    print(f"rendered {template_path} -> {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
