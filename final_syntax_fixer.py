#!/usr/bin/env python3
"""
Final comprehensive syntax fix for unified_trading_dashboard.py
Addresses all remaining concatenated lines and indentation issues.
"""

import re
import sys
from pathlib import Path


def fix_concatenated_lines(content):
    """Fix all concatenated lines that should be on separate lines"""

    # Fix concatenated if statements with logical operators
    content = re.sub(
        r"if (.+?)\nand (.+?):", r"if \1 and \2:", content, flags=re.MULTILINE
    )
    content = re.sub(
        r"if (.+?)\nor (.+?):", r"if \1 or \2:", content, flags=re.MULTILINE
    )

    # Fix concatenated ternary operators
    content = re.sub(
        r"(.+?)\nif (.+?) else (.+)", r"\1 if \2 else \3", content, flags=re.MULTILINE
    )

    # Fix concatenated method calls (but keep proper ones)
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        # Skip if this is already a properly formatted line
        if line.strip() and not line.strip().endswith("\\"):
            # Check if next line starts with logical operators that should be on same line
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith(("and ", "or ", "if ", "else ")):
                    # Combine lines
                    combined = line.rstrip() + " " + next_line
                    fixed_lines.append(combined)
                    lines[i + 1] = ""  # Mark next line as processed
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        elif line.strip() == "":  # Keep empty lines but skip already processed ones
            if i == 0 or lines[i - 1].strip():  # Don't add if previous was processed
                fixed_lines.append(line)

    return "\n".join(fixed_lines)


def fix_indentation(content):
    """Fix indentation issues to use consistent 4-space indentation"""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        if line.strip():  # Non-empty line
            # Count leading whitespace
            leading_spaces = len(line) - len(line.lstrip())
            indent_level = leading_spaces // 4

            # Handle odd indentations by rounding down
            if leading_spaces % 4 != 0 and leading_spaces > 0:
                # Fix inconsistent indentation
                if leading_spaces % 4 == 2:  # 2 extra spaces
                    indent_level = leading_spaces // 4
                else:
                    indent_level = (leading_spaces + 2) // 4

            # Reconstruct line with proper indentation
            fixed_line = "    " * indent_level + line.lstrip()
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)  # Keep empty lines as-is

    return "\n".join(fixed_lines)


def fix_specific_syntax_issues(content):
    """Fix specific known syntax issues"""

    # Fix f-string issues that span multiple lines
    content = re.sub(
        r"f\"([^\"]*)\n([^\"]*?)\"", r'f"\1 \2"', content, flags=re.MULTILINE
    )

    # Fix dictionary and list continuations
    content = re.sub(r",\s*\n\s*}", ",\n}", content)
    content = re.sub(r",\s*\n\s*]", ",\n]", content)

    # Fix common concatenated patterns
    fixes = [
        # Fix function calls split across lines
        (r"(\w+\([^)]*)\n([^)]*\))", r"\1\2"),
        # Fix dictionary definitions
        (r'({\s*)\n(\s*["\'][^"\']+["\'])', r"\1\2"),
        # Fix list definitions
        (r'(\[\s*)\n(\s*["\'][^"\']+["\'])', r"\1\2"),
        # Fix method chaining
        (r"(\.[a-zA-Z_]\w*\([^)]*)\n([^)]*\))", r"\1\2"),
    ]

    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return content


def main():
    """Main execution function"""
    file_path = Path("unified_trading_dashboard.py")

    if not file_path.exists():
        print("❌ File not found: unified_trading_dashboard.py")
        return False

    print("🔧 FINAL SYNTAX FIXER")
    print("=" * 50)

    # Create backup
    backup_path = file_path.with_suffix(".py.backup_final_fix")
    content = file_path.read_text(encoding="utf-8")
    backup_path.write_text(content, encoding="utf-8")
    print(f"✅ Backup created: {backup_path}")

    print(f"📋 Original file: {len(content.splitlines())} lines")

    # Apply fixes
    print("🔧 Fixing concatenated lines...")
    content = fix_concatenated_lines(content)

    print("🔧 Fixing indentation...")
    content = fix_indentation(content)

    print("🔧 Fixing specific syntax issues...")
    content = fix_specific_syntax_issues(content)

    # Write fixed content
    file_path.write_text(content, encoding="utf-8")
    print(f"📊 Final file: {len(content.splitlines())} lines")

    # Test compilation
    print("🧪 Testing compilation...")
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(file_path)],
            capture_output=True,
            text=True,
            cwd=file_path.parent,
        )

        if result.returncode == 0:
            print("✅ COMPILATION SUCCESSFUL!")
            return True
        else:
            print("❌ Compilation failed:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error testing compilation: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n📝 Manual fixes may be required for remaining issues.")
        sys.exit(1)
    else:
        print("\n🎉 All syntax issues have been resolved!")
