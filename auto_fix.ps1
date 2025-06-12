# auto_fix.ps1
# Automatyczne formatowanie, sortowanie importów i sprawdzanie typów dla całego repozytorium ZoL0

Write-Host "[auto_fix] Running Black..."
black .

Write-Host "[auto_fix] Running isort..."
isort .

Write-Host "[auto_fix] Running Ruff..."
ruff check . --fix

Write-Host "[auto_fix] Running mypy..."
mypy .

Write-Host "[auto_fix] All checks complete. Review output above."
