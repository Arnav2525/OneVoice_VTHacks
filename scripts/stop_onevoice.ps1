$ErrorActionPreference = 'SilentlyContinue'
foreach ($port in 8771, 4319) {
    Get-NetTCPConnection -LocalPort $port -State Listen |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
}
