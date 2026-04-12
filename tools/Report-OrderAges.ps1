function Report-OrderAges {
  [CmdletBinding()]
  param(
    [string]$Base='https://paper-api.alpaca.markets',
    [int]$WarnDays=7
  )

  $H=@{
    'APCA-API-KEY-ID'     = $env:APCA_API_KEY_ID
    'APCA-API-SECRET-KEY' = $env:APCA_API_SECRET_KEY
  }

  $open = Invoke-RestMethod -Method GET -Uri "$Base/v2/orders?status=open&nested=true&limit=500" -Headers $H

  '--- ORDER AGES (days) ---'
  foreach ($o in ($open | Sort-Object submitted_at)) {
    $age  = [math]::Round((New-TimeSpan -Start ([datetime]$o.submitted_at) -End (Get-Date)).TotalDays,1)
    $mark = if ($age -ge $WarnDays) { '⚠' } else { '' }
    "{0,-5} {1,4} {2,6} tif={3} age={4}d {5} lim={6}" -f $o.symbol,$o.side,$o.type,$o.time_in_force,$age,$mark,$o.limit_price
  }
}
