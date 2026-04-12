Param(
  [double]$Cash,
  [double]$MarketValue,
  [double]$Equity,
  [string]$SourceCsv
)

$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$venv = Join-Path $root ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venv)) { Write-Error "Missing venv: $venv"; exit 1 }
. $venv

$script = Join-Path $root "scripts\append_equity_snapshot.py"

if ($PSBoundParameters.ContainsKey('SourceCsv')) {
  python $script --source-csv $SourceCsv
} elseif ($PSBoundParameters.ContainsKey('Equity')) {
  python $script --equity $Equity
} elseif ($PSBoundParameters.ContainsKey('Cash') -and $PSBoundParameters.ContainsKey('MarketValue')) {
  python $script --cash $Cash --market-value $MarketValue
} else {
  Write-Error "Provide --Equity OR both --Cash and --MarketValue OR --SourceCsv"
  exit 1
}
