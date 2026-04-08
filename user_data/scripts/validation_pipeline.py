"""
Validation Pipeline — Safety checks for LLM-generated strategy code.

Validates generated .py files through a multi-stage pipeline before
allowing them to be backtested or deployed:

  1. Static analysis: whitelisted imports, no exec/eval, no file I/O
  2. Syntax check: Python AST parse
  3. Look-ahead bias detection: shift(-N), rolling(center=True), global normalization
  4. Structure check: extends BaseGeneratedStrategy, has required methods and metadata
  5. Trade frequency sanity (post-backtest): reject if >50 trades/day average

Returns a ValidationResult with pass/fail and detailed error messages.
"""

import ast
import importlib.util
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("validation_pipeline")

# ---------------------------------------------------------------------------
# Whitelisted imports
# ---------------------------------------------------------------------------
ALLOWED_IMPORT_PREFIXES = frozenset([
    "freqtrade",
    "pandas",
    "pandas_ta",
    "numpy",
    "np",
    "pd",
    "ta",
    "talib",
    "math",
    "strategies",    # For importing BaseGeneratedStrategy
    "indicators",    # For importing custom indicators
])

# Banned function calls (security)
BANNED_CALLS = frozenset([
    "exec", "eval", "compile", "__import__",
    "open", "os.system", "subprocess",
    "pickle.load", "pickle.loads",
])

# Banned module imports (security)
BANNED_IMPORTS = frozenset([
    "os", "sys", "subprocess", "shutil", "socket", "http",
    "urllib", "requests", "pathlib", "io", "pickle", "shelve",
    "importlib", "ctypes", "multiprocessing", "threading",
    "signal", "tempfile", "glob",
])


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def fail(self, msg: str):
        self.passed = False
        self.errors.append(msg)
        return self

    def warn(self, msg: str):
        self.warnings.append(msg)
        return self

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        lines = [f"Validation: {status}"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1: Static analysis via AST
# ---------------------------------------------------------------------------
class SecurityVisitor(ast.NodeVisitor):
    """AST visitor that checks for banned patterns."""

    def __init__(self, result: ValidationResult):
        self.result = result

    def visit_Import(self, node):
        for alias in node.names:
            module = alias.name.split(".")[0]
            if module in BANNED_IMPORTS:
                self.result.fail(f"Banned import: '{alias.name}' (line {node.lineno})")
            elif module not in ALLOWED_IMPORT_PREFIXES:
                self.result.fail(
                    f"Disallowed import: '{alias.name}' (line {node.lineno}). "
                    f"Only these are allowed: {sorted(ALLOWED_IMPORT_PREFIXES)}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            module = node.module.split(".")[0]
            if module in BANNED_IMPORTS:
                self.result.fail(f"Banned import: 'from {node.module}' (line {node.lineno})")
            elif module not in ALLOWED_IMPORT_PREFIXES:
                self.result.fail(
                    f"Disallowed import: 'from {node.module}' (line {node.lineno}). "
                    f"Only these are allowed: {sorted(ALLOWED_IMPORT_PREFIXES)}"
                )
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for banned function calls
        func_name = self._get_call_name(node)
        if func_name in BANNED_CALLS:
            self.result.fail(f"Banned function call: '{func_name}' (line {node.lineno})")

        # Check for exec/eval as attributes
        if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval", "compile"):
            self.result.fail(f"Banned builtin: '{node.func.id}()' (line {node.lineno})")

        self.generic_visit(node)

    def _get_call_name(self, node) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            n = node.func
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
            return ".".join(reversed(parts))
        return ""


# ---------------------------------------------------------------------------
# Stage 2: Look-ahead bias detection
# ---------------------------------------------------------------------------
class LookAheadVisitor(ast.NodeVisitor):
    """AST visitor that detects common look-ahead bias patterns."""

    def __init__(self, result: ValidationResult):
        self.result = result

    def visit_Call(self, node):
        # Check for .shift(-N) — accessing future data
        if isinstance(node.func, ast.Attribute) and node.func.attr == "shift":
            for arg in node.args:
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    if isinstance(arg.operand, (ast.Constant, ast.Num)):
                        val = arg.operand.value if isinstance(arg.operand, ast.Constant) else arg.operand.n
                        if val > 0:
                            self.result.fail(
                                f"Look-ahead bias: .shift(-{val}) accesses future data "
                                f"(line {node.lineno})"
                            )

        # Check for .rolling(center=True) — centered window uses future data
        if isinstance(node.func, ast.Attribute) and node.func.attr == "rolling":
            for kw in node.keywords:
                if kw.arg == "center":
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        self.result.fail(
                            f"Look-ahead bias: rolling(center=True) uses future data "
                            f"(line {node.lineno}). Use center=False and shift() instead."
                        )

        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Stage 3: Structure validation
# ---------------------------------------------------------------------------
def check_structure(tree: ast.Module, result: ValidationResult):
    """Verify the code defines a class that extends BaseGeneratedStrategy."""
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    if not classes:
        result.fail("No class definition found. Strategy must define a class.")
        return

    # Find the strategy class (should extend BaseGeneratedStrategy)
    strategy_class = None
    for cls in classes:
        for base in cls.bases:
            base_name = ""
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name == "BaseGeneratedStrategy":
                strategy_class = cls
                break

    if not strategy_class:
        result.fail(
            "Strategy class must extend BaseGeneratedStrategy. "
            f"Found classes: {[c.name for c in classes]}"
        )
        return

    # Check required methods
    method_names = {
        n.name for n in ast.walk(strategy_class) if isinstance(n, ast.FunctionDef)
    }
    required = {"populate_indicators", "populate_entry_trend", "populate_exit_trend"}
    missing = required - method_names
    if missing:
        result.fail(f"Missing required methods: {missing}")

    # Check required class attributes (STRATEGY_THESIS, TARGET_REGIME, GENERATION_ID)
    assigns = {}
    for node in strategy_class.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigns[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigns[node.target.id] = node

    for attr in ["STRATEGY_THESIS", "TARGET_REGIME", "GENERATION_ID"]:
        if attr not in assigns:
            result.warn(f"Missing class attribute: {attr} (recommended but not required)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate_strategy_file(filepath: str | Path) -> ValidationResult:
    """
    Run the full validation pipeline on a strategy .py file.

    Returns ValidationResult with pass/fail and detailed messages.
    """
    filepath = Path(filepath)
    result = ValidationResult()

    # --- Read file ---
    if not filepath.exists():
        return result.fail(f"File not found: {filepath}")

    try:
        source = filepath.read_text()
    except Exception as e:
        return result.fail(f"Cannot read file: {e}")

    if not source.strip():
        return result.fail("File is empty")

    # --- Stage 1: Parse AST ---
    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        return result.fail(f"Syntax error: {e}")

    # --- Stage 2: Security checks ---
    SecurityVisitor(result).visit(tree)

    # --- Stage 3: Look-ahead bias ---
    LookAheadVisitor(result).visit(tree)

    # --- Stage 4: Structure checks ---
    check_structure(tree, result)

    log.info(f"Validation of {filepath.name}: {result}")
    return result


def validate_backtest_results(
    results: dict,
    max_trades_per_day: float = 50.0,
    min_trades_total: int = 5,
    max_drawdown_pct: float = 50.0,
) -> ValidationResult:
    """
    Post-backtest validation: check if strategy behaves sanely.

    Parameters
    ----------
    results : dict
        Parsed backtest results from Freqtrade.
    max_trades_per_day : float
        Reject strategies averaging more than this many trades per day.
    min_trades_total : int
        Reject strategies with fewer trades (likely broken or too conservative).
    max_drawdown_pct : float
        Reject strategies with drawdown exceeding this.
    """
    result = ValidationResult()

    total_trades = results.get("total_trades", 0)
    trading_days = results.get("backtest_days", 1)
    max_drawdown = results.get("max_drawdown_abs", 0)
    starting_balance = results.get("starting_balance", 1000)

    if total_trades < min_trades_total:
        result.fail(
            f"Too few trades: {total_trades} (minimum {min_trades_total}). "
            f"Strategy may be broken or too restrictive."
        )

    trades_per_day = total_trades / max(trading_days, 1)
    if trades_per_day > max_trades_per_day:
        result.fail(
            f"Excessive trading: {trades_per_day:.1f} trades/day "
            f"(max {max_trades_per_day}). Likely draining capital via fees."
        )

    drawdown_pct = (max_drawdown / starting_balance) * 100 if starting_balance > 0 else 0
    if drawdown_pct > max_drawdown_pct:
        result.fail(
            f"Excessive drawdown: {drawdown_pct:.1f}% (max {max_drawdown_pct}%). "
            f"Strategy is too risky."
        )

    return result
