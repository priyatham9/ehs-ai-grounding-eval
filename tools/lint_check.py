#!/usr/bin/env python3
"""Simple linter for unused imports and undefined names using ast."""
import ast
import sys
from pathlib import Path


class UnusedImportChecker(ast.NodeVisitor):
    """Check for unused imports and undefined names."""

    def __init__(self, filename):
        self.filename = filename
        self.imported_names = {}
        self.used_names = set()
        self.issues = []

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.imported_names[name] = node.lineno
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == '__future__':
            return
        for alias in node.names:
            if alias.name == '*':
                return
            name = alias.asname if alias.asname else alias.name
            self.imported_names[name] = node.lineno
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            self.used_names.add(node.value)
        self.generic_visit(node)

    def check_issues(self):
        for name, lineno in self.imported_names.items():
            if name not in self.used_names:
                self.issues.append((lineno, f"unused import '{name}'"))
        return self.issues


def check_file(filepath):
    """Check a single Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content, filepath)
        checker = UnusedImportChecker(filepath)
        checker.visit(tree)
        return checker.check_issues()
    except (SyntaxError, UnicodeDecodeError) as e:
        return [(0, f"error parsing: {e}")]


def main():
    """Run linter on Python files."""
    exclude_dirs = {'docs', 'synthetic', 'results', '__pycache__', '.git'}
    root = Path('.')

    all_issues = {}
    for pyfile in sorted(root.rglob('*.py')):
        if any(part in exclude_dirs for part in pyfile.parts):
            continue
        issues = check_file(str(pyfile))
        if issues:
            all_issues[str(pyfile)] = issues

    if all_issues:
        found_issues = False
        for filepath in sorted(all_issues.keys()):
            issues = all_issues[filepath]
            for lineno, msg in issues:
                print(f"{filepath}:{lineno}: {msg}")
                found_issues = True
        return 1 if found_issues else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
