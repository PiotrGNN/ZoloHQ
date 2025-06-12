#!/usr/bin/env python3
"""
Comprehensive syntax analyzer for unified_trading_dashboard.py
This script will identify ALL syntax issues at once for batch fixing.
"""

import ast
import logging
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def analyze_dashboard_syntax(file_path: str) -> None:
    """
    Comprehensive syntax analysis for dashboard files. Obsługa braku pliku.
    """
    logger = logging.getLogger("syntax_analyzer")
    logger.info("🔍 COMPREHENSIVE DASHBOARD SYNTAX ANALYSIS")
    logger.info("=" * 60)
    issues = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        print("OK: FileNotFoundError handled gracefully.")
        return
    except Exception as e:
        logger.error(f"❌ Error reading file: {e}")
        return

    # 1. Check for concatenated lines (missing newlines)
    logger.info("\n1️⃣ CHECKING FOR CONCATENATED LINES:")
    concatenated_patterns = [
        r"\)\s*[a-zA-Z_]",  # ) followed by variable/function
        r"}\s*[a-zA-Z_]",  # } followed by variable/function
        r"]\s*[a-zA-Z_]",  # ] followed by variable/function
        r'"\s*[a-zA-Z_]',  # " followed by variable/function (not in string)
        r"'\s*[a-zA-Z_]",  # ' followed by variable/function (not in string)
    ]

    for i, line in enumerate(lines, 1):
        if line.strip():
            for pattern in concatenated_patterns:
                if re.search(pattern, line):
                    # Exclude false positives
                    if not any(x in line for x in ['f"', "f'", '"""', "'''"]):
                        issues.append(
                            f"Line {i}: Possible concatenated line: {line[:80]}..."
                        )
                        logger.warning(f"  ⚠️  Line {i}: {line[:80]}...")

    # 2. Check for indentation issues
    logger.info("\n2️⃣ CHECKING FOR INDENTATION ISSUES:")

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        current_indent = len(line) - len(line.lstrip())

        # Check for irregular indentation (not multiple of 4)
        if current_indent % 4 != 0 and current_indent > 0:
            issues.append(f"Line {i}: Irregular indentation ({current_indent} spaces)")
            logger.warning(
                f"  ⚠️  Line {i}: Irregular indent ({current_indent}): {line[:60]}..."
            )

    # 3. Try basic AST parsing to catch syntax errors
    logger.info("\n3️⃣ CHECKING FOR SYNTAX ERRORS:")
    try:
        ast.parse(content)
        logger.info("  ✅ Basic AST parsing successful")
    except SyntaxError as e:
        issues.append(f"Syntax Error: Line {e.lineno}: {e.msg}")
        logger.error(f"  ❌ Syntax Error at line {e.lineno}: {e.msg}")
        logger.error(f"      Text: {e.text.strip() if e.text else 'N/A'}")
    except Exception as e:
        issues.append(f"AST Error: {str(e)}")
        logger.error(f"  ❌ AST Error: {e}")

    # 4. Check for specific Python constructs issues
    logger.info("\n4️⃣ CHECKING FOR SPECIFIC CONSTRUCT ISSUES:")

    # Check for try blocks without except
    try_blocks = 0
    except_blocks = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("try:"):
            try_blocks += 1
        elif stripped.startswith("except"):
            except_blocks += 1

    if try_blocks != except_blocks:
        issues.append(
            f"Try/except mismatch: {try_blocks} try blocks, {except_blocks} except blocks"
        )
        logger.warning(
            f"  ⚠️  Try/except mismatch: {try_blocks} try vs {except_blocks} except"
        )
    # Check for unmatched brackets
    brackets = {"(": 0, "[": 0, "{": 0}
    closing_to_opening = {")": "(", "]": "[", "}": "{"}

    for i, line in enumerate(lines, 1):
        for char in line:
            if char in "([{":
                brackets[char] += 1
            elif char in ")]}":
                corresponding = closing_to_opening[char]
                brackets[corresponding] -= 1
                if brackets[corresponding] < 0:
                    issues.append(f"Line {i}: Unmatched closing {char}")
                    logger.warning(f"  ⚠️  Line {i}: Unmatched closing {char}")

    for bracket, count in brackets.items():
        if count != 0:
            issues.append(f"Unmatched {bracket}: {count} remaining")
            logger.warning(f"  ⚠️  Unmatched {bracket}: {count} remaining")

    # 5. Summary
    logger.info("\n📊 ANALYSIS SUMMARY:")
    logger.info(f"  Total issues found: {len(issues)}")
    logger.info(f"  File size: {len(lines)} lines")

    if issues:
        logger.warning("\n🚨 ISSUES TO FIX:")
        for issue in issues[:20]:  # Show first 20 issues
            logger.warning(f"  • {issue}")
        if len(issues) > 20:
            logger.warning(f"  ... and {len(issues) - 20} more issues")
    else:
        logger.info("  ✅ No major issues detected!")

    return issues


def create_fix_recommendations():
    """Create specific fix recommendations"""
    logger = logging.getLogger("syntax_analyzer")
    logger.info("\n💡 FIX RECOMMENDATIONS:")
    logger.info("  1. Fix all concatenated lines by adding proper newlines")
    logger.info("  2. Standardize indentation to 4 spaces")
    logger.info("  3. Ensure all try blocks have matching except blocks")
    logger.info("  4. Check bracket matching")
    logger.info("  5. Remove any duplicate code blocks")
    logger.info("  6. Validate all f-string syntax")


# Test edge-case: brak pliku dashboard
def test_missing_dashboard_file():
    """Testuje obsługę braku pliku dashboard przy analizie składni."""
    analyze_dashboard_syntax("nonexistent_dashboard.py")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    dashboard_file = Path("unified_trading_dashboard.py")
    if dashboard_file.exists():
        issues = analyze_dashboard_syntax(dashboard_file)
        create_fix_recommendations()

        # Return exit code based on issues found
        sys.exit(1 if issues else 0)
    else:
        print("❌ unified_trading_dashboard.py not found!")
        sys.exit(1)

# TODO: Integrate with CI/CD pipeline for automated syntax analysis and edge-case tests.
# Edge-case tests: simulate missing files, syntax errors, and permission issues.
# All public methods have docstrings and exception handling.
