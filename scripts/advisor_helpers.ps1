function Normalize-Text([string]$s) {
  if ([string]::IsNullOrWhiteSpace($s)) { return "" }
  ($s -replace '[\u2000-\u206F\u2E00-\u2E7F\p{P}]',' ' -replace '\s+',' ').Trim().ToLower()
}

function Canonicalize-Advisor {
  param([string]$s, [hashtable]$aliases)
  $norm = Normalize-Text $s
  if ([string]::IsNullOrWhiteSpace($norm)) { return 'Unlabeled' }
  if ($aliases -and $aliases.ContainsKey($norm)) { return $aliases[$norm] }
  switch ($norm) {
    'buffett' {'Buffett'}
    'warren buffett' {'Buffett'}
    'brk' {'Buffett'}
    'berkshire' {'Buffett'}
    'cohen' {'Cohen'}
    'sac' {'Cohen'}
    'point72' {'Cohen'}
    'fink' {'Fink'}
    'larry fink' {'Fink'}
    'blackrock' {'Fink'}
    'aladdin' {'Fink'}
    default { $s.Trim() }
  }
}

if (-not (Get-Command Resolve-Advisor -ErrorAction SilentlyContinue)) {
  function Resolve-Advisor($row, [ref]$fromCol, [ref]$matchRule) {
    $fromCol.Value = $null; $matchRule.Value = $null
    foreach ($c in @('advisor','advisor_name','source','model','strategy','generator','engine')) {
      if ($row.PSObject.Properties.Name -contains $c) {
        $val = [string]$row.$c
        if (-not [string]::IsNullOrWhiteSpace($val)) { $fromCol.Value=$c; $matchRule.Value='column'; return $val }
      }
    }
    if ($row.PSObject.Properties.Name -contains 'ticker') {
      $fromCol.Value='ticker'; $matchRule.Value='fallback'; return [string]$row.ticker
    }
    'Unlabeled'
  }
}
