param(
    [Parameter(Mandatory)]
    [ValidateSet('SUPABASE_URL','SUPABASE_ANON_KEY','KD_WORKER_TOKEN_0','KD_WORKER_TOKEN_1','KD_WORKER_TOKEN_2','KD_WORKER_TOKEN_3')]
    [string]$Name,
    [switch]$Update
)
$ErrorActionPreference = 'Stop'
$file = Join-Path (Join-Path $PSScriptRoot 'private') ($Name + '.dpapi')
if ($Update) {
    $value = Read-Host "New value for $Name" -AsSecureString
    if ($value.Length -eq 0) { throw 'Empty value. The saved entry was preserved.' }
    ConvertFrom-SecureString $value | Set-Content -LiteralPath $file -Encoding ascii
    Write-Host "Updated $Name."
} else {
    $value = Get-Content -LiteralPath $file -Raw
    $secure = ConvertTo-SecureString $value.Trim()
    $credential = [System.Net.NetworkCredential]::new('', $secure)
    Set-Clipboard -Value $credential.Password
    Write-Host "Copied the value for $Name. Paste it in Kaggle Secrets."
}
