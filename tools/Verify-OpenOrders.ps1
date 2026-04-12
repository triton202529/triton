# tools/Verify-OpenOrders.ps1
function Verify-OpenOrders {
  [CmdletBinding(SupportsShouldProcess = $true)]
  param(
    [string]$Base = 'https://paper-api.alpaca.markets',
    [switch]$CancelDupes,
    [switch]$CancelNonGTC,
    [switch]$CancelStandalone,
    [switch]$ShowLegs
  )

  function HasProp {
    param($Obj, [string]$Name)
    return ($null -ne $Obj.PSObject.Properties[$Name])
  }

  function GetProp {
    param($Obj, [string[]]$Names, $Default = $null)
    foreach ($n in $Names) {
      if (HasProp $Obj $n) {
        $v = $Obj.$n
        if ($null -ne $v -and [string]$v -ne '') { return $v }
      }
    }
    return $Default
  }

  function GetOrderClassNorm {
    param($Obj)
    $cls = GetProp $Obj @('order_class') ''
    if ([string]::IsNullOrWhiteSpace([string]$cls)) { return '-' }
    return [string]$cls
  }

  function GetParentId {
    param($Obj)
    return (GetProp $Obj @('parent_order_id','parent_id') '')
  }

  function IsParent {
    param($Obj)
    $parentId = GetParentId $Obj
    return [string]::IsNullOrEmpty([string]$parentId)
  }

  # Label bracket legs as TP/SL when obvious
  function GetLegLabel {
    param($Leg)
    $side = [string](GetProp $Leg @('side') '')
    $type = [string](GetProp $Leg @('type') '')
    $stop = GetProp $Leg @('stop_price','stop') $null
    $lim  = GetProp $Leg @('limit_price','limit') $null

    # Typical Alpaca bracket legs:
    # - take profit: sell limit
    # - stop loss: sell stop (or stop_limit) with stop_price
    if ($side -eq 'sell' -and $type -eq 'limit') { return 'TP' }
    if ($side -eq 'sell' -and ($type -eq 'stop' -or $type -eq 'stop_limit') -and $null -ne $stop -and [string]$stop -ne '') { return 'SL' }

    # If it's a buy bracket (short cover etc.), still label similarly
    if ($side -eq 'buy' -and $type -eq 'limit') { return 'TP' }
    if ($side -eq 'buy' -and ($type -eq 'stop' -or $type -eq 'stop_limit') -and $null -ne $stop -and [string]$stop -ne '') { return 'SL' }

    return 'LEG'
  }

  $H = @{
    'APCA-API-KEY-ID'     = $env:APCA_API_KEY_ID
    'APCA-API-SECRET-KEY' = $env:APCA_API_SECRET_KEY
  }

  function LoadOpen {
    return @(Invoke-RestMethod -Method GET -Uri ($Base.TrimEnd('/') + "/v2/orders?status=open&nested=true&limit=500") -Headers $H)
  }

  try {
    $open = LoadOpen
  } catch {
    Write-Warning ("Failed to load open orders: " + $_.Exception.Message)
    return
  }

  # Cancel NON-GTC
  if ($CancelNonGTC) {
    $nonGtc = $open | Where-Object {
      $tif = GetProp $_ @('time_in_force','tif') ''
      ($tif -ne '') -and ($tif -ne 'gtc')
    }

    foreach ($o in $nonGtc) {
      $oid = GetProp $o @('id') ''
      $sym = GetProp $o @('symbol') '?'
      $tif = GetProp $o @('time_in_force','tif') ''
      if ($oid -and $PSCmdlet.ShouldProcess("order $oid [$sym]", "Cancel non-GTC (tif=$tif)")) {
        try { Invoke-RestMethod -Method DELETE -Uri ($Base.TrimEnd('/') + "/v2/orders/$oid") -Headers $H | Out-Null }
        catch { Write-Warning ("Cancel non-GTC failed $sym id=${oid}: " + $_.Exception.Message) }
      }
    }

    try { $open = LoadOpen } catch {}
  }

  # Cancel standalone BUY LIMIT parents (class '-' / missing)
  if ($CancelStandalone) {
    $standalone = $open | Where-Object {
      (IsParent $_) -and
      ((GetProp $_ @('side') '') -eq 'buy') -and
      ((GetProp $_ @('type') '') -eq 'limit') -and
      ((GetOrderClassNorm $_) -eq '-')
    }

    foreach ($o in $standalone) {
      $oid = GetProp $o @('id') ''
      $sym = GetProp $o @('symbol') '?'
      $lim = GetProp $o @('limit_price','limit') ''
      $tif = GetProp $o @('time_in_force','tif') ''
      if ($oid -and $PSCmdlet.ShouldProcess("order $oid [$sym]", "Cancel standalone buy limit (lim=$lim tif=$tif)")) {
        try { Invoke-RestMethod -Method DELETE -Uri ($Base.TrimEnd('/') + "/v2/orders/$oid") -Headers $H | Out-Null }
        catch { Write-Warning ("Cancel standalone failed $sym id=${oid}: " + $_.Exception.Message) }
      }
    }

    try { $open = LoadOpen } catch {}
  }

  # NON-GTC LIST
  '--- NON-GTC OPEN ORDERS ---'
  $ng = $open | Where-Object {
    $tif = GetProp $_ @('time_in_force','tif') ''
    ($tif -ne '') -and ($tif -ne 'gtc')
  }

  if ($ng -and $ng.Count -gt 0) {
    foreach ($x in $ng) {
      $cls = GetOrderClassNorm $x
      $lim = GetProp $x @('limit_price','limit') ''
      $tif = GetProp $x @('time_in_force','tif') ''
      ("{0} side={1} tif={2} class={3} lim={4}" -f (GetProp $x @('symbol') '?'), (GetProp $x @('side') '?'), $tif, $cls, $lim)
    }
  } else {
    'None'
  }

  # DUPES: build a single key string, then Group-Object on that
  $parentsOnly = $open | Where-Object { IsParent $_ }

  foreach ($o in $parentsOnly) {
    $sym = GetProp $o @('symbol') ''
    $side = GetProp $o @('side') ''
    $type = GetProp $o @('type') ''
    $lim = GetProp $o @('limit_price','limit') ''
    $tif = GetProp $o @('time_in_force','tif') ''
    $o | Add-Member -NotePropertyName '__dupekey' -NotePropertyValue ("$sym|$side|$type|$lim|$tif") -Force
  }

  $dupes = $parentsOnly | Group-Object -Property '__dupekey' | Where-Object { $_.Count -gt 1 }

  if ($dupes) {
    '--- POSSIBLE DUPLICATES ---'
    foreach ($g in $dupes) {
      $sorted = $g.Group | Sort-Object -Property @{
        Expression = {
          $s = GetProp $_ @('submitted_at','created_at','createdAt') ''
          if ($s) { [datetime]$s } else { [datetime]'1970-01-01' }
        }
        Ascending = $true
      }

      foreach ($o in $sorted) {
        $cls  = GetOrderClassNorm $o
        $lim  = GetProp $o @('limit_price','limit') ''
        $tif  = GetProp $o @('time_in_force','tif') ''
        $sub  = GetProp $o @('submitted_at','created_at','createdAt') ''
        $coid = GetProp $o @('client_order_id','clientOrderId','client_orderid') ''
        ("{0} id={1} class={2} lim={3} tif={4} submitted={5} coid={6}" -f (GetProp $o @('symbol') '?'), (GetProp $o @('id') ''), $cls, $lim, $tif, $sub, $coid)
      }

      if ($CancelDupes -and $sorted.Count -gt 1) {
        $toCancel = $sorted[0..($sorted.Count - 2)]
        foreach ($o in $toCancel) {
          $oid = GetProp $o @('id') ''
          $sym = GetProp $o @('symbol') '?'
          if ($oid -and $PSCmdlet.ShouldProcess("order $oid [$sym]", "Cancel duplicate")) {
            try {
              Invoke-RestMethod -Method DELETE -Uri ($Base.TrimEnd('/') + "/v2/orders/$oid") -Headers $H | Out-Null
              ("Canceled duplicate: $sym id=$oid")
            } catch {
              Write-Warning ("Cancel dup failed $sym id=${oid}: " + $_.Exception.Message)
            }
          }
        }
        try { $open = LoadOpen } catch {}
      }
    }
  } else {
    'No duplicates found.'
  }

  # PARENTS + LEGS
  '--- PARENT ORDERS ---'
  $parents = $open | Where-Object { IsParent $_ }

  foreach ($p in $parents) {
    $cls  = GetOrderClassNorm $p
    $sym  = GetProp $p @('symbol') '?'
    $side = GetProp $p @('side') '?'
    $type = GetProp $p @('type') '?'
    $tif  = GetProp $p @('time_in_force','tif') ''
    $lim  = GetProp $p @('limit_price','limit') ''
    $coid = GetProp $p @('client_order_id','clientOrderId','client_orderid') ''

    ("{0} side={1} type={2} class={3} tif={4} lim={5} coid={6}" -f $sym,$side,$type,$cls,$tif,$lim,$coid)

    if ($ShowLegs) {
      $legs = @()

      # A) If the parent object already includes legs, use them
      if (HasProp $p 'legs' -and $null -ne $p.legs) {
        $legs = @($p.legs)
      }

      # B) Otherwise, try to find child orders in the open list that reference this parent
      if (-not $legs -or $legs.Count -eq 0) {
        $parentId = GetProp $p @('id') ''
        if ($parentId) {
          $legs = @($open | Where-Object { (GetParentId $_) -eq $parentId })
        }
      }

      # C) Final fallback: fetch the parent directly (sometimes list endpoint omits legs)
      if (-not $legs -or $legs.Count -eq 0) {
        $parentId = GetProp $p @('id') ''
        if ($parentId) {
          try {
            $one = Invoke-RestMethod -Method GET -Uri ($Base.TrimEnd('/') + "/v2/orders/$parentId") -Headers $H
            if ($one -and (HasProp $one 'legs') -and $null -ne $one.legs) {
              $legs = @($one.legs)
            }
          } catch {
            # ignore; we'll just print "no open legs found"
          }
        }
      }

      if (-not $legs -or $legs.Count -eq 0) {
        "  - (no open legs found)"
        continue
      }

      # Print legs with TP/SL labels
      foreach ($l in $legs) {
        $label = GetLegLabel $l
        $lcls  = GetOrderClassNorm $l
        $lsym  = GetProp $l @('symbol') $sym
        $llim  = GetProp $l @('limit_price','limit') ''
        $lstop = GetProp $l @('stop_price','stop') ''
        $ltif  = GetProp $l @('time_in_force','tif') ''
        $lside = GetProp $l @('side') '?'
        $ltype = GetProp $l @('type') '?'
        $lid   = GetProp $l @('id') ''
        ("  - {0}: sym={1} side={2} type={3} class={4} lim={5} stop={6} tif={7} id={8}" -f $label,$lsym,$lside,$ltype,$lcls,$llim,$lstop,$ltif,$lid)
      }
    }
  }
}
