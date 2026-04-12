param(
  [string]$Csv = 'data\results\institutional\orders_naive.csv',
  [int]$TopN = 30
)
$argsList = @(
  '--csv', $Csv,
  '--order-type','limit','--limit-pad-bps','10',
  '--tif','gtc',
  '--sl-pct','0.05','--tp-pct','0.08',
  '--idempotent','--client-id-salt','TRITON',
  '--top-n', "$TopN",
  '--preflight-cancel',
  '--reduce-only-sells',
  '--really-place'
)
python .\place_orders_from_csv.py @argsList
