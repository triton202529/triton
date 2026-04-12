# tools/Verify-OpenOrders.ps1
function Verify-OpenOrders {
  [CmdletBinding()]
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

  try {
    $open = Invoke-RestMethod -Method GET -Uri "$Base/v2/orders?status=open&nested=true&limit=500" -Headers $H
  } catch {
    Write-Warning "Failed to load open orders: $($_.Exception.Message)"
    return
  }

  # Optional: cancel any lingering non-GTCs
  if ($CancelNonGTC) {
    $nonGtc = $open | Where-Object { $_.time_in_force -ne 'gtc' }
    if ($nonGtc) {
      '--- CANCELING NON-GTC ORDERS ---'
      foreach ($o in $nonGtc) {
        try {
          Invoke-RestMethod -Method DELETE -Uri "$Base/v2/orders/$($o.id)" -Headers $H | Out-Null
          'Canceled DAY: {0} id={1} lim={2}' -f $o.symbol,$o.id,$o.limit_price
        } catch {
          Write-Warning ("Failed cancel {0} id={1}: {2}" -f $o.symbol,$o.id,$_.Exception.Message)
        }
      }
      # refresh after cancels
      $open = Invoke-RestMethod -Method GET -Uri "$Base/v2/orders?status=open&nested=true&limit=500" -Headers $H
    }
  }

  '--- NON-GTC OPEN ORDERS ---'
  $ng = $open | Where-Object { $_.time_in_force -ne 'gtc' }
  if ($ng) {
    $ng | ForEach-Object {
      '{0} side={1} tif={2} class={3} lim={4}' -f $_.symbol,$_.side,$_.time_in_force,$_.order_class,$_.limit_price
    }
  } else { 'None' }

  # Detect likely duplicates of parents (ignore client_order_id so idempotent runs still show dupes)
  $dupes = $open |
           Where-Object { -not $_.parent_order_id } |
           Group-Object symbol,side,type,limit_price,time_in_force |
           Where-Object { $_.Count -gt 1 }

  if ($dupes) {
    '--- POSSIBLE DUPLICATES ---'
    foreach ($g in $dupes) {
      $sorted = $g.Group | Sort-Object submitted_at
      $sorted | ForEach-Object {
        '{0} id={1} class={2} lim={3} tif={4} submitted={5} coid={6}' -f `
          $_.symbol,$_.id,$_.order_class,$_.limit_price,$_.time_in_force,$_.submitted_at,$_.client_order_id
      }
      if ($CancelDupes -and $sorted.Count -gt 1) {
        $toCancel = $sorted[0..($sorted.Count - 2)]
        foreach ($o in $toCancel) {
          try {
            Invoke-RestMethod -Method DELETE -Uri "$Base/v2/orders/$($o.id)" -Headers $H | Out-Null
            'Canceled duplicate: {0} id={1}' -f $o.symbol,$o.id
          } catch {
            Write-Warning ("Failed cancel dup {0} id={1}: {2}" -f $o.symbol,$o.id,$_.Exception.Message)
          }
        }
      }
    }
  } else { 'No duplicates found.' }

  '--- PARENT ORDERS ---'
  $parents = $open | Where-Object { -not $_.parent_order_id }
  foreach ($p in $parents) {
    $oc = if ($p.order_class) { $p.order_class } else { '-' }
    '{0} side={1} type={2} class={3} tif={4} lim={5} coid={6}' -f `
      $p.symbol,$p.side,$p.type,$oc,$p.time_in_force,$p.limit_price,$p.client_order_id

    if ($ShowLegs) {
      $legs = $open | Where-Object { $_.parent_order_id -eq $p.id }
      foreach ($l in $legs) {
        "  └─ leg: side=$($l.side) type=$($l.type) class=$($l.order_class) lim=$($l.limit_price) stop=$($l.stop_price) tif=$($l.time_in_force) id=$($l.id)"
      }
    }
  }
}
