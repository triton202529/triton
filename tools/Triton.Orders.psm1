# tools/Triton.Orders.psm1

function GPP {
  param(
    [Parameter(Mandatory)] $Obj,
    [Parameter(Mandatory)] [string[]] $Keys
  )
  foreach ($k in $Keys) {
    if ($null -ne $Obj.PSObject.Properties[$k]) { return $Obj.$k }
  }
  return ''
}

$GLOBAL:ORDER_CLASS  = @('order_class')
$GLOBAL:PARENT_ID    = @('parent_order_id','parent_id')
$GLOBAL:LIMIT_PRICE  = @('limit_price','limit')
$GLOBAL:STOP_PRICE   = @('stop_price','stop')
$GLOBAL:TIF          = @('time_in_force','tif')
$GLOBAL:SUBMITTED_AT = @('submitted_at','created_at','createdAt')
$GLOBAL:CLIENT_OID   = @('client_order_id','client_orderid','clientOrderId')
$GLOBAL:FILLED_QTY   = @('filled_qty','filledQty','filled')

function Get-AlpacaOpenOrders {
  param([string]$Base = 'https://paper-api.alpaca.markets')
  $H = @{
    'APCA-API-KEY-ID'     = $env:APCA_API_KEY_ID
    'APCA-API-SECRET-KEY' = $env:APCA_API_SECRET_KEY
  }
  Invoke-RestMethod -Method GET -Uri "$Base/v2/orders?status=open&nested=true&limit=500" -Headers $H
}

function Verify-OpenOrders {
  [CmdletBinding(SupportsShouldProcess=$true)]
  param(
    [string]$Base = 'https://paper-api.alpaca.markets',
    [switch]$CancelDupes,
    [switch]$CancelNonGTC,
    [switch]$ShowLegs
  )

  $H = @{
    'APCA-API-KEY-ID'     = $env:APCA_API_KEY_ID
    'APCA-API-SECRET-KEY' = $env:APCA_API_SECRET_KEY
  }

  $open = Get-AlpacaOpenOrders -Base $Base

  if ($CancelNonGTC) {
    $nonGtc = $open | Where-Object { (GPP $_ $TIF) -ne 'gtc' -and (GPP $_ $TIF) -ne '' }
    foreach ($o in $nonGtc) {
      $oid = GPP $o @('id'); $sym = GPP $o @('symbol')
      if ($PSCmdlet.ShouldProcess("order $oid [$sym]", "Cancel non-GTC")) {
        Invoke-RestMethod -Method DELETE -Uri "$Base/v2/orders/$oid" -Headers $H | Out-Null
        "Canceled non-GTC: {0} id={1}" -f $sym,$oid
      }
    }
    $open = Get-AlpacaOpenOrders -Base $Base
  }

  '--- NON-GTC OPEN ORDERS ---'
  $ng = $open | Where-Object { (GPP $_ $TIF) -ne 'gtc' -and (GPP $_ $TIF) -ne '' }
  if ($ng) {
    $ng | ForEach-Object {
      '{0} side={1} tif={2} class={3} lim={4}' -f `
        (GPP $_ @('symbol')), (GPP $_ @('side')), (GPP $_ $TIF), (GPP $_ $ORDER_CLASS), (GPP $_ $LIMIT_PRICE)
    }
  } else { 'None' }

  # --- POSSIBLE DUPLICATES ---
  $dupes = $open |
    Group-Object `
      { GPP $_ @('symbol') },
      { GPP $_ @('side')   },
      { GPP $_ @('type')   },
      { GPP $_ $LIMIT_PRICE },
      { GPP $_ $TIF } |
    Where-Object { $_.Count -gt 1 }

  if ($dupes) {
    '--- POSSIBLE DUPLICATES ---'
    foreach ($g in $dupes) {
      $sorted = $g.Group | Sort-Object @{Expression={ [datetime](GPP $_ $SUBMITTED_AT) }; Ascending=$true}
      $sorted | ForEach-Object {
        '{0} id={1} class={2} lim={3} tif={4} submitted={5} coid={6}' -f `
          (GPP $_ @('symbol')), (GPP $_ @('id')), (GPP $_ $ORDER_CLASS), (GPP $_ $LIMIT_PRICE),
          (GPP $_ $TIF), (GPP $_ $SUBMITTED_AT), (GPP $_ $CLIENT_OID)
      }
      if ($CancelDupes -and $sorted.Count -gt 1) {
        $toCancel = $sorted[0..($sorted.Count - 2)]
        foreach ($o in $toCancel) {
          $oid = GPP $o @('id'); $sym = GPP $o @('symbol')
          if ($PSCmdlet.ShouldProcess("order $oid [$sym]", "Cancel duplicate")) {
            Invoke-RestMethod -Method DELETE -Uri "$Base/v2/orders/$oid" -Headers $H | Out-Null
            'Canceled duplicate: {0} id={1}' -f $sym,$oid
          }
        }
      }
    }
  } else { 'No duplicates found.' }

  '--- PARENT ORDERS ---'
  $parents = $open | Where-Object { [string]::IsNullOrEmpty((GPP $_ $PARENT_ID)) }
  foreach ($p in $parents) {
    $oc = (GPP $p $ORDER_CLASS); if (-not $oc) { $oc = '-' }
    '{0} side={1} type={2} class={3} tif={4} lim={5} coid={6}' -f `
      (GPP $p @('symbol')), (GPP $p @('side')), (GPP $p @('type')), $oc,
      (GPP $p $TIF), (GPP $p $LIMIT_PRICE), (GPP $p $CLIENT_OID)

    if ($ShowLegs -and ($oc -in @('bracket','oco'))) {
      $pid  = GPP $p @('id')
      $legs = $open | Where-Object { (GPP $_ $PARENT_ID) -eq $pid }
      foreach ($l in ($legs | Sort-Object @{Expression={ [datetime](GPP $_ $SUBMITTED_AT) }})) {
        "  leg: {0} {1} lim={2} stop={3} tif={4}" -f `
          (GPP $l @('side')), (GPP $l @('type')), (GPP $l $LIMIT_PRICE),
          (GPP $l $STOP_PRICE), (GPP $l $TIF)
      }
    }
  }
}

function Audit-OrderIntegrity {
  [CmdletBinding()]
  param([string]$Base = 'https://paper-api.alpaca.markets')

  $open = Get-AlpacaOpenOrders -Base $Base

  $byId     = @{}
  $byParent = @{}
  foreach ($o in $open) {
    $oid = GPP $o @('id'); $byId[$oid]=$o
    $pid = GPP $o $PARENT_ID
    if ($pid) {
      if (-not $byParent.ContainsKey($pid)) { $byParent[$pid]=@() }
      $byParent[$pid] += $o
    }
  }

  $parents = $open | Where-Object { [string]::IsNullOrEmpty((GPP $_ $PARENT_ID)) }

  '--- ORDER INTEGRITY ---'
  foreach ($p in $parents) {
    $legs = @()
    $pid  = GPP $p @('id')
    if ($byParent.ContainsKey($pid)) { $legs = $byParent[$pid] }

    $cls  = (GPP $p $ORDER_CLASS); if (-not $cls) { $cls = '-' }
    "{0} parent class={1} tif={2} lim={3} legs={4}" -f `
      (GPP $p @('symbol')), $cls, (GPP $p $TIF), (GPP $p $LIMIT_PRICE), $legs.Count

    if ($cls -eq 'bracket') {
      $filled = ([decimal]0 + ((GPP $p $FILLED_QTY) -as [decimal]))
      if ($filled -eq 0) {
        "  (awaiting parent fill → children not created yet)"
      } else {
        $tp = $legs | Where-Object { (GPP $_ @('side')) -eq 'sell' -and (GPP $_ @('type')) -eq 'limit' }
        $sl = $legs | Where-Object { (GPP $_ $STOP_PRICE) -ne '' }
        if ( ($tp.Count -ne 1) -or ($sl.Count -ne 1) ) {
          "  ? Missing/extra legs after fill: take-profit={0} stop={1}" -f $tp.Count, $sl.Count
        }
      }
    } elseif ($cls -eq '-' -and (GPP $p @('type')) -eq 'limit' -and (GPP $p @('side')) -eq 'buy') {
      "  ? Standalone buy limit without bracket protection"
    }
  }

  $orphans = $open | Where-Object {
    $pid = GPP $_ $PARENT_ID
    $pid -and -not $byId.ContainsKey($pid)
  }
  if ($orphans -and $orphans.Count -gt 0) {
    '--- ORPHAN LEGS ---'
    $orphans | ForEach-Object { "  {0} id={1} parent={2}" -f (GPP $_ @('symbol')), (GPP $_ @('id')), (GPP $_ $PARENT_ID) }
  } else { 'No orphan legs detected.' }
}

function Report-OrderAges {
  [CmdletBinding()]
  param([string]$Base='https://paper-api.alpaca.markets',[int]$WarnDays=7)

  $open = Get-AlpacaOpenOrders -Base $Base

  '--- ORDER AGES (days) ---'
  foreach ($o in ($open | Sort-Object @{Expression={ [datetime](GPP $_ $SUBMITTED_AT) }})) {
    $age  = [math]::Round((New-TimeSpan -Start ([datetime](GPP $o $SUBMITTED_AT)) -End (Get-Date)).TotalDays,1)
    $mark = if ($age -ge $WarnDays) { '?' } else { '' }
    "{0,-5} {1,4} {2,6} tif={3} age={4}d {5} lim={6}" -f `
      (GPP $o @('symbol')), (GPP $o @('side')), (GPP $o @('type')),
      (GPP $o $TIF), $age, $mark, (GPP $o $LIMIT_PRICE)
  }
}

function Test-OpenOrders {
  [CmdletBinding(SupportsShouldProcess=$true)]
  param(
    [string]$Base = 'https://paper-api.alpaca.markets',
    [switch]$CancelDupes,
    [switch]$CancelNonGTC,
    [switch]$ShowLegs
  )
  Verify-OpenOrders -Base $Base -CancelDupes:$CancelDupes -CancelNonGTC:$CancelNonGTC -ShowLegs:$ShowLegs
}

function Test-OrderIntegrity {
  [CmdletBinding()]
  param([string]$Base = 'https://paper-api.alpaca.markets')
  Audit-OrderIntegrity -Base $Base
}

function Get-OrderAges {
  [CmdletBinding()]
  param([string]$Base='https://paper-api.alpaca.markets',[int]$WarnDays=7)
  Report-OrderAges -Base $Base -WarnDays $WarnDays
}

Export-ModuleMember -Function `
  Verify-OpenOrders, Audit-OrderIntegrity, Report-OrderAges, `
  Test-OpenOrders, Test-OrderIntegrity, Get-OrderAges
