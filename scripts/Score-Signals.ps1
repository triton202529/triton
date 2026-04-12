function Score-Signals {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory)] [string]$SignalsCsv,
    [Parameter(Mandatory)] [string]$WeightsMapJson,
    [Parameter(Mandatory)] [string]$OutCsv,
    [string]$LearnedTickerMapJson,
    # Canonicalization
    [hashtable]$Aliases = @{
      'buffett'='Buffett'; 'warren buffett'='Buffett'; 'brk'='Buffett'; 'berkshire'='Buffett';
      'cohen'='Cohen'; 'steve cohen'='Cohen'; 'sac'='Cohen'; 'point72'='Cohen';
      'fink'='Fink'; 'larry fink'='Fink'; 'blackrock'='Fink'; 'aladdin'='Fink'
    },
    # Only these columns will be trusted for an advisor value
    [string[]]$AdvisorCols   = @('advisor','advisor_name','source','model','strategy','generator','engine'),
    # Soft-text cols are ignored by default (were causing the bug)
    [string[]]$SoftTextCols  = @('rationale','explanation','notes','note','comment','comments','description'),
    # Ticker fallback(s)
    [hashtable]$StaticTickerMap = @{},      # e.g. @{ AAPL='Buffett'; MSFT='Fink'; GOOGL='Cohen' }
    [hashtable]$ManualWeights   = @{},      # e.g. @{ Fink=0.05 }
    [Nullable[Double]]$ZeroWeightOverride,  # e.g. -ZeroWeightOverride 0.01
    [switch]$DropZeroWeights,
    [double]$MinWeight = 0,
    [switch]$Quiet
  )

  function Normalize-Text([string]$s) {
    if ([string]::IsNullOrWhiteSpace($s)) { return "" }
    ($s -replace '[\u2000-\u206F\u2E00-\u2E7F\p{P}]',' ' -replace '\s+',' ').Trim().ToLower()
  }
  function Canonicalize-Advisor([string]$s, [hashtable]$aliases) {
    $norm = Normalize-Text $s
    if ([string]::IsNullOrWhiteSpace($norm)) { return 'Unlabeled' }
    if ($aliases -and $aliases.ContainsKey($norm)) { return $aliases[$norm] }
    switch ($norm) {
      'buffett' {'Buffett'} 'warren buffett' {'Buffett'} 'brk' {'Buffett'} 'berkshire' {'Buffett'}
      'cohen' {'Cohen'} 'sac' {'Cohen'} 'point72' {'Cohen'}
      'fink' {'Fink'} 'larry fink' {'Fink'} 'blackrock' {'Fink'} 'aladdin' {'Fink'}
      default { $s.Trim() }
    }
  }

  # Load weights
  if (!(Test-Path $WeightsMapJson)) { throw "Weights map not found: $WeightsMapJson" }
  $mapJson = Get-Content $WeightsMapJson -Raw | ConvertFrom-Json
  $weightsHT = @{}
  ($mapJson | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name) | ForEach-Object {
    $k = $_.ToLower()
    $v = $mapJson.$_
    $weightsHT[$k] = ([double]($v -as [double]))
  }

  # Learned ticker map (optional)
  $tickerMapHT = @{}
  if ($LearnedTickerMapJson -and (Test-Path $LearnedTickerMapJson)) {
    $tm = Get-Content $LearnedTickerMapJson -Raw | ConvertFrom-Json
    ($tm | Get-Member -MemberType NoteProperty | Select-Object -ExpandProperty Name) | ForEach-Object {
      $tickerMapHT[$_.ToUpper()] = [string]$tm.$_
    }
  }

  function Resolve-FromAdvisorCols($row, [ref]$fromCol, [ref]$rule) {
    foreach ($c in $AdvisorCols) {
      if ($row.PSObject.Properties.Name -contains $c) {
        $val = [string]$row.$c
        if (-not [string]::IsNullOrWhiteSpace($val)) {
          $canon = Canonicalize-Advisor $val $Aliases
          # Only accept if it canonicalizes to a non-Unlabeled advisor
          if ($canon -and $canon -ne 'Unlabeled') {
            $fromCol.Value = $c; $rule.Value = 'advisor-col'
            return $canon
          }
        }
      }
    }
    return $null
  }

  function Resolve-Advisor($row, [ref]$fromCol, [ref]$rule) {
    # 1) Try advisor-ish columns ONLY
    $tryAdvisor = Resolve-FromAdvisorCols $row ([ref]$fromCol) ([ref]$rule)
    if ($tryAdvisor) { return $tryAdvisor }

    # 2) Fallback: Ticker mapping (learned -> static)
    if ($row.PSObject.Properties.Name -contains 'ticker') {
      $t = [string]$row.ticker
      if ($t) {
        $tu = $t.Trim().ToUpper()
        if ($tickerMapHT.ContainsKey($tu)) { $fromCol.Value='ticker'; $rule.Value='learnedTickerMap'; return $tickerMapHT[$tu] }
        if ($StaticTickerMap.ContainsKey($tu)) { $fromCol.Value='ticker'; $rule.Value='staticTickerMap'; return $StaticTickerMap[$tu] }
        $fromCol.Value='ticker'; $rule.Value='ticker-unmapped'
        return 'Unlabeled'
      }
    }

    # 3) As a last resort, ignore soft-text columns by default (they caused the bug)
    $fromCol.Value = $null; $rule.Value = 'none'
    return 'Unlabeled'
  }

  # Score
  $rows = Import-Csv $SignalsCsv | ForEach-Object {
    $src=$null; $rule=$null
    $adv = Resolve-Advisor $_ ([ref]$src) ([ref]$rule)

    # Weight resolution: Manual > map > ZeroWeightOverride
    $w = 0.0
    if ($ManualWeights.ContainsKey($adv)) {
      $w = [double]$ManualWeights[$adv]
    } else {
      $k = $adv.ToLower()
      if ($weightsHT.ContainsKey($k)) { $w = [double]$weightsHT[$k] }
      if ($w -eq 0.0 -and $ZeroWeightOverride.HasValue) { $w = [double]$ZeroWeightOverride.Value }
    }
    if ($w -lt $MinWeight) { $w = 0.0 }

    $_ | Add-Member -NotePropertyName advisor_resolved   -NotePropertyValue $adv  -Force
    $_ | Add-Member -NotePropertyName advisor_match_from -NotePropertyValue $src  -Force
    $_ | Add-Member -NotePropertyName advisor_match_rule -NotePropertyValue $rule -Force
    $_ | Add-Member -NotePropertyName weight             -NotePropertyValue $w    -Force
    $_
  }

  if ($DropZeroWeights) { $rows = $rows | Where-Object { [double]$_.weight -gt 0 } }
  $rows | Export-Csv -NoTypeInformation -Encoding UTF8 $OutCsv

  if (-not $Quiet) {
    $kept  = ($rows | Where-Object { [double]$_.weight -gt 0 }).Count
    $total = $rows.Count
    Write-Host ""
    Write-Host "== Score-Signals summary ==" -ForegroundColor Cyan
    Write-Host ("Kept: {0} / {1} rows (weight > 0)" -f $kept, $total)
    if ($total -gt 0) {
      $rows |
        Group-Object advisor_resolved |
        Sort-Object Count -Descending |
        Select-Object -First 8 @{n='advisor';e={$_.Name}}, @{n='rows';e={$_.Count}} |
        Format-Table -AutoSize
    }
  }
}
