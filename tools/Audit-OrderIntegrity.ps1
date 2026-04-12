function Audit-OrderIntegrity {
  [CmdletBinding()]
  param(
    [string]$Base = 'https://paper-api.alpaca.markets'
  )
  $H = @{
    'APCA-API-KEY-ID'     = $env:APCA_API_KEY_ID
    'APCA-API-SECRET-KEY' = $env:APCA_API_SECRET_KEY
  }
  $open = Invoke-RestMethod -Method GET -Uri "$Base/v2/orders?status=open&nested=true&limit=500" -Headers $H
  $byId=@{}; $byParent=@{}
  foreach ($o in $open) {
    $byId[$o.id]=$o
    if ($o.parent_order_id) {
      if (-not $byParent.ContainsKey($o.parent_order_id)) { $byParent[$o.parent_order_id]=@() }
      $byParent[$o.parent_order_id]+= $o
    }
  }
  $parents = $open | Where-Object { -not $_.parent_order_id }
  '--- ORDER INTEGRITY ---'
  foreach ($p in $parents) {
    $legs = if ($byParent.ContainsKey($p.id)) { $byParent[$p.id] } else { @() }
    $cls  = if ($p.order_class) { $p.order_class } else { '-' }
    "{0} parent class={1} tif={2} lim={3} legs={4}" -f $p.symbol, $cls, $p.time_in_force, $p.limit_price, $legs.Count
    if ($cls -eq 'bracket') {
      if ([decimal]($p.filled_qty) -eq 0) {
        "  (awaiting parent fill → children not created yet)"
      } else {
        $tp = $legs | Where-Object { $_.side -eq 'sell' -and $_.type -eq 'limit' }
        $sl = $legs | Where-Object { $_.stop_price -ne $null }
        if ($tp.Count -ne 1 -or $sl.Count -ne 1) {
          "  ⚠ Missing/extra legs after fill: take-profit={0} stop={1}" -f $tp.Count, $sl.Count
        }
      }
    } elseif ($cls -eq '-' -and $p.type -eq 'limit' -and $p.side -eq 'buy') {
      "  ⚠ Standalone buy limit without bracket protection"
    }
  }
  $orphans = $open | Where-Object { $_.parent_order_id -and -not $byId.ContainsKey($_.parent_order_id) }
  if ($orphans -and $orphans.Count -gt 0) {
    '--- ORPHAN LEGS ---'
    $orphans | ForEach-Object { "  {0} id={1} parent={2}" -f $_.symbol, $_.id, $_.parent_order_id }
  } else { 'No orphan legs detected.' }
}
