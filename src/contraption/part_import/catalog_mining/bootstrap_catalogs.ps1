[CmdletBinding()]
param(
    [Parameter()]
    [string]$Root = 'D:\WorldModelCatalogs',

    [Parameter()]
    [ValidateSet('Core', 'CoreAndAssets')]
    [string]$Profile = 'CoreAndAssets',

    [Parameter()]
    [int]$MinimumFreeGiB = 12,

    [Parameter()]
    [string]$UserAgent = 'WorldModelCatalogBootstrap/0.1 (research corpus; local provenance snapshot)',

    [Parameter()]
    [string]$IncludeIds = '',

    [Parameter()]
    [string]$LogName = 'bootstrap.log',

    [Parameter()]
    [switch]$AllowOfflineRevocationFallback,

    [Parameter()]
    [switch]$AdoptExistingCatalogRoot
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version 2.0
[string[]]$script:CurlTlsArgs = if ($AllowOfflineRevocationFallback) { @('--ssl-no-revoke') } else { @() }
$script:FailureReportWritten = $false

trap {
    if (-not $script:FailureReportWritten) {
        try {
            $rootVariable = Get-Variable -Name CatalogRoot -Scope Script -ErrorAction SilentlyContinue
            if ($rootVariable -and (Test-Path -LiteralPath $rootVariable.Value -PathType Container)) {
                $failureDir = Join-Path $rootVariable.Value 'reports'
                New-Item -ItemType Directory -Path $failureDir -Force -ErrorAction SilentlyContinue | Out-Null
                $failureStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
                [ordered]@{
                    succeeded = $false
                    failure_kind = 'bootstrap-preflight-or-reporting'
                    exception_type = $_.Exception.GetType().FullName
                    exception_message = $_.Exception.Message
                    failed_at_utc = [DateTime]::UtcNow.ToString('o')
                } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $failureDir "bootstrap-infrastructure-failure-$failureStamp.json") -Encoding UTF8 -ErrorAction SilentlyContinue
            }
        }
        catch { }
    }
    [Console]::Error.WriteLine("Bootstrap failed: $($_.Exception.Message)")
    exit 1
}

function Write-Log {
    param([string]$Message)
    $stamp = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    $line = "[$stamp] $Message"
    Write-Host $line
    if ($script:LogPath) {
        Add-Content -LiteralPath $script:LogPath -Value $line -Encoding UTF8
    }
}

function Get-SafeRoot {
    param([string]$Candidate)
    $full = [IO.Path]::GetFullPath($Candidate)
    $volume = [IO.Path]::GetPathRoot($full)
    if ($volume -notin @('D:\', 'F:\')) {
        throw "Catalog root must be on D:\ or F:\. Resolved value: $full"
    }
    if ($full.TrimEnd('\') -eq $volume.TrimEnd('\')) {
        throw "Refusing to use a drive root directly. Choose a dedicated directory such as D:\WorldModelCatalogs."
    }
    return $full.TrimEnd('\')
}

function Assert-FreeSpace {
    param(
        [string]$Path,
        [long]$AdditionalBytes = 0
    )
    $driveRoot = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($Path))
    $driveName = $driveRoot.Substring(0, 1)
    $drive = Get-PSDrive -Name $driveName
    $reserve = [long]$MinimumFreeGiB * 1GB
    if (($drive.Free - $AdditionalBytes) -lt $reserve) {
        $needGiB = [math]::Round(($AdditionalBytes + $reserve) / 1GB, 2)
        $freeGiB = [math]::Round($drive.Free / 1GB, 2)
        throw "Storage guard stopped the run: $driveRoot has $freeGiB GiB free; this step plus reserve requires $needGiB GiB. Re-run against F:\WorldModelCatalogsOverflow or reduce the profile."
    }
}

function Convert-ToNativePath {
    param([string]$RelativePath)
    return ($RelativePath -replace '/', '\')
}

function Resolve-CatalogRelativePath {
    param([string]$RelativePath)
    if ([IO.Path]::IsPathRooted($RelativePath)) {
        throw "Manifest path must be relative to the catalog root: $RelativePath"
    }
    $candidate = [IO.Path]::GetFullPath((Join-Path $script:CatalogRoot (Convert-ToNativePath $RelativePath)))
    $prefix = $script:CatalogRoot.TrimEnd('\') + '\'
    if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest path escapes the catalog root: $RelativePath -> $candidate"
    }
    Assert-NoCatalogReparsePoint -Candidate $candidate
    return $candidate
}

function Assert-NoCatalogReparsePoint {
    param([string]$Candidate)
    $root = $script:CatalogRoot.TrimEnd('\')
    $prefix = $root + '\'
    $current = [IO.Path]::GetFullPath($Candidate).TrimEnd('\')
    while ($current.Equals($root, [StringComparison]::OrdinalIgnoreCase) -or $current.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Catalog paths may not traverse a junction or symbolic link: $current"
            }
        }
        if ($current.Equals($root, [StringComparison]::OrdinalIgnoreCase)) { break }
        $parent = Split-Path -Parent $current
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $current) { break }
        $current = $parent.TrimEnd('\')
    }
}

function Get-RelativePathCompat {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )
    $baseUri = New-Object System.Uri(($BasePath.TrimEnd('\') + '\'))
    $targetUri = New-Object System.Uri($TargetPath)
    return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString()).Replace('/', '\')
}

function Get-SafeFileStem {
    param([string]$Name)
    $safe = $Name
    foreach ($char in [IO.Path]::GetInvalidFileNameChars()) {
        $safe = $safe.Replace([string]$char, '_')
    }
    $safe = $safe.Trim().TrimEnd('.')
    if ($safe.Length -gt 150) {
        $safe = $safe.Substring(0, 150)
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($Name)
        $suffix = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').Substring(0, 10).ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
    return "${safe}__${suffix}"
}

function Get-StringSha256 {
    param([string]$Value)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Value)))).Replace('-', '').ToLowerInvariant()
    }
    finally { $sha.Dispose() }
}

function Get-StructuredFileDigest {
    param([string]$Path, [ValidateSet('json', 'csv', 'fuel_catalog', 'fuel_excluded')][string]$Kind)
    if ($Kind -in @('json', 'fuel_catalog', 'fuel_excluded')) {
        $parsed = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
        if ($Kind -in @('fuel_catalog', 'fuel_excluded')) {
            $stableRows = @($parsed | Select-Object name, owner, filesize, modify_date, license_id, license_name, license_url | Sort-Object owner, name)
            $normalized = ConvertTo-Json -InputObject $stableRows -Depth 10 -Compress
        }
        else {
            $normalized = ConvertTo-Json -InputObject @($parsed) -Depth 30 -Compress
        }
    }
    else {
        $normalized = ConvertTo-Json -InputObject @(Import-Csv -LiteralPath $Path) -Depth 10 -Compress
    }
    return Get-StringSha256 -Value $normalized
}

function Assert-OrPromoteGeneratedFile {
    param(
        [string]$Candidate,
        [string]$Target,
        [ValidateSet('json', 'csv', 'fuel_catalog', 'fuel_excluded')][string]$Kind
    )
    if (Test-Path -LiteralPath $Target -PathType Leaf) {
        $candidateDigest = Get-StructuredFileDigest -Path $Candidate -Kind $Kind
        $targetDigest = Get-StructuredFileDigest -Path $Target -Kind $Kind
        if ($candidateDigest -ne $targetDigest) {
            throw "Dated metadata snapshot drifted and was not overwritten: $Target"
        }
        return 'reused'
    }
    Copy-Item -LiteralPath $Candidate -Destination $Target
    return 'downloaded'
}

function Get-ExpectedLengthFromHeaders {
    param([string]$HeaderPath)
    if (-not (Test-Path -LiteralPath $HeaderPath -PathType Leaf)) { return $null }
    $text = Get-Content -LiteralPath $HeaderPath -Raw
    $contentRanges = [regex]::Matches($text, '(?im)^Content-Range:\s*bytes\s+\d+-\d+/(\d+)\s*$')
    if ($contentRanges.Count -gt 0) {
        return [long]$contentRanges[$contentRanges.Count - 1].Groups[1].Value
    }
    $contentLengths = [regex]::Matches($text, '(?im)^Content-Length:\s*(\d+)\s*$')
    if ($contentLengths.Count -gt 0) {
        return [long]$contentLengths[$contentLengths.Count - 1].Groups[1].Value
    }
    return $null
}

function Get-ResumeValidatorFromHeaders {
    param([string]$HeaderPath)
    if (-not (Test-Path -LiteralPath $HeaderPath -PathType Leaf)) { return $null }
    $text = Get-Content -LiteralPath $HeaderPath -Raw
    # Prefer Last-Modified for If-Range. Some otherwise standards-compliant
    # archive servers advertise an ETag but ignore Range when that ETag is sent
    # in If-Range; COD was observed doing so on 2026-08-01. Its HTTP-date form
    # correctly returned 206. ETag remains a fallback when no date is supplied.
    $modified = [regex]::Matches($text, '(?im)^Last-Modified:\s*(.+?)\s*$')
    if ($modified.Count -gt 0) { return $modified[$modified.Count - 1].Groups[1].Value.Trim() }
    $etags = [regex]::Matches($text, '(?im)^ETag:\s*(.+?)\s*$')
    if ($etags.Count -gt 0) { return $etags[$etags.Count - 1].Groups[1].Value.Trim() }
    return $null
}

function Get-RemoteContentLength {
    param([string]$Url)
    [string[]]$tlsArgs = @($script:CurlTlsArgs)
    $head = @(& curl.exe @tlsArgs --head --location --fail --silent --show-error --retry 3 --connect-timeout 30 --max-time 60 --user-agent $UserAgent $Url 2>$null)
    if ($LASTEXITCODE -ne 0) { return $null }
    $text = $head -join "`n"
    $ranges = [regex]::Matches($text, '(?im)^Content-Range:\s*bytes\s+\d+-\d+/(\d+)\s*$')
    if ($ranges.Count -gt 0) { return [long]$ranges[$ranges.Count - 1].Groups[1].Value }
    $lengths = [regex]::Matches($text, '(?im)^Content-Length:\s*(\d+)\s*$')
    if ($lengths.Count -gt 0) { return [long]$lengths[$lengths.Count - 1].Groups[1].Value }
    return $null
}

function Test-UpstreamChecksum {
    param([string]$Path, [string]$Checksum)
    if ([string]::IsNullOrWhiteSpace($Checksum)) { return $true }
    if ($Checksum -notmatch '^(md5|sha256):([0-9a-fA-F]+)$') {
        throw "Unsupported checksum expression: $Checksum"
    }
    $algorithm = if ($Matches[1] -eq 'md5') { 'MD5' } else { 'SHA256' }
    $expected = $Matches[2].ToLowerInvariant()
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm $algorithm).Hash.ToLowerInvariant()
    return ($actual -eq $expected)
}

function Add-Record {
    param(
        [string]$Id,
        [string]$Method,
        [string]$Path,
        [string]$Status,
        [string]$SourceUrl,
        [string]$Version = '',
        [Nullable[long]]$ExpectedBytes = $null,
        [string]$UpstreamChecksum = '',
        [string]$Note = ''
    )
    $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    $length = if ($item -and -not $item.PSIsContainer) { [long]$item.Length } else { $null }
    $localSha256 = if ($item -and -not $item.PSIsContainer) { (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant() } else { '' }
    $script:Records.Add([pscustomobject]@{
        id = $Id
        method = $Method
        status = $Status
        relative_path = if ($item) { Get-RelativePathCompat -BasePath $script:CatalogRoot -TargetPath $item.FullName } else { Get-RelativePathCompat -BasePath $script:CatalogRoot -TargetPath ([IO.Path]::GetFullPath($Path)) }
        bytes = $length
        local_sha256 = $localSha256
        expected_bytes = $ExpectedBytes
        source_url = $SourceUrl
        source_version = $Version
        upstream_checksum = $UpstreamChecksum
        manifest_sha256 = $script:ManifestSha256
        retrieved_at_utc = [DateTime]::UtcNow.ToString('o')
        note = $Note
    }) | Out-Null
}

function Invoke-CurlDownload {
    param(
        [string]$Id,
        [string]$Url,
        [string]$Destination,
        [string]$Version = '',
        [Nullable[long]]$ExpectedBytes = $null,
        [string]$UpstreamChecksum = ''
    )

    $script:CurrentEntryId = $Id
    $script:CurrentSourceUrl = $Url
    $script:LastCurlExitCode = $null
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $headers = "$Destination.headers.txt"
    if (-not $ExpectedBytes) {
        $recordedLength = Get-ExpectedLengthFromHeaders -HeaderPath $headers
        if ($recordedLength) { $ExpectedBytes = [Nullable[long]]$recordedLength }
    }
    if (-not $ExpectedBytes) {
        $remoteLength = Get-RemoteContentLength -Url $Url
        if ($remoteLength) { $ExpectedBytes = [Nullable[long]]$remoteLength }
    }
    if (-not $ExpectedBytes) {
        Write-Log "WARNING $Id has no discoverable upstream size; only the reserve floor can be enforced before transfer."
    }
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $existing = Get-Item -LiteralPath $Destination
        if ($ExpectedBytes -and $existing.Length -ne [long]$ExpectedBytes) {
            throw "Existing completed file has the wrong size and was not overwritten: $Destination"
        }
        if ($UpstreamChecksum -and -not (Test-UpstreamChecksum -Path $Destination -Checksum $UpstreamChecksum)) {
            throw "Existing completed file failed its upstream checksum and was not overwritten: $Destination"
        }
        Write-Log "REUSE $Id ($([math]::Round($existing.Length / 1MB, 2)) MiB)"
        Add-Record -Id $Id -Method 'http' -Path $Destination -Status 'reused' -SourceUrl $Url -Version $Version -ExpectedBytes $ExpectedBytes -UpstreamChecksum $UpstreamChecksum
        return
    }

    $partial = "$Destination.part"
    $partialBytes = if (Test-Path -LiteralPath $partial -PathType Leaf) { [long](Get-Item -LiteralPath $partial).Length } else { 0L }
    if ($ExpectedBytes -and $partialBytes -gt [long]$ExpectedBytes) {
        throw "Partial file is larger than the expected upstream artifact: $partial"
    }
    $remainingBytes = if ($ExpectedBytes) { [long]$ExpectedBytes - $partialBytes } else { 0L }
    Assert-FreeSpace -Path $Destination -AdditionalBytes $remainingBytes
    Write-Log "DOWNLOAD $Id -> $Destination"
    $arguments = @($script:CurlTlsArgs) + @(
        '--location',
        '--fail',
        '--show-error',
        '--retry', '5',
        '--retry-delay', '3',
        '--retry-all-errors',
        '--connect-timeout', '30',
        '--speed-limit', '1024',
        '--speed-time', '120',
        '--max-time', '14400',
        '--continue-at', '-',
        '--remote-time',
        '--user-agent', $UserAgent,
        '--dump-header', $headers,
        '--output', $partial,
        $Url
    )
    if ((Test-Path -LiteralPath $partial -PathType Leaf) -and (Get-Item -LiteralPath $partial).Length -gt 0) {
        $validator = Get-ResumeValidatorFromHeaders -HeaderPath $headers
        if (-not $validator) {
            throw "Refusing to resume $partial without a recorded ETag or Last-Modified validator; this prevents a hybrid artifact if mutable upstream content changed."
        }
        $arguments = @('--header', "If-Range: $validator") + $arguments
    }
    & curl.exe @arguments
    $script:LastCurlExitCode = $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) {
        throw "curl failed for $Id with exit code $LASTEXITCODE. Partial data was retained at $partial for resumption."
    }
    if (-not (Test-Path -LiteralPath $partial -PathType Leaf)) {
        throw "curl reported success but no partial file exists for $Id"
    }
    $downloaded = Get-Item -LiteralPath $partial
    $downloadedLength = [long]$downloaded.Length
    if ($ExpectedBytes -and $downloadedLength -ne [long]$ExpectedBytes) {
        throw "Size mismatch for ${Id}: expected $([long]$ExpectedBytes), got $downloadedLength. Partial file retained."
    }
    if ($UpstreamChecksum -and -not (Test-UpstreamChecksum -Path $partial -Checksum $UpstreamChecksum)) {
        throw "Checksum mismatch for $Id. Partial file retained and was not promoted."
    }
    Move-Item -LiteralPath $partial -Destination $Destination
    Write-Log "DONE $Id ($([math]::Round($downloadedLength / 1MB, 2)) MiB)"
    Add-Record -Id $Id -Method 'http' -Path $Destination -Status 'downloaded' -SourceUrl $Url -Version $Version -ExpectedBytes $ExpectedBytes -UpstreamChecksum $UpstreamChecksum
}

function Invoke-GitSnapshot {
    param(
        [pscustomobject]$Entry,
        [string]$Destination
    )
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Assert-FreeSpace -Path $Destination

    if (Test-Path -LiteralPath (Join-Path $Destination '.git')) {
        $origin = (& git -C $Destination config --get remote.origin.url).Trim()
        if ($LASTEXITCODE -ne 0 -or $origin -ne $Entry.url) {
            throw "Existing Git snapshot origin mismatch for $($Entry.id): '$origin' versus '$($Entry.url)'"
        }
        $commit = (& git -C $Destination rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0) { throw "Could not read existing Git snapshot: $Destination" }
        $status = @(& git -C $Destination status --porcelain)
        if ($status.Count -ne 0) { throw "Existing Git snapshot is not clean: $Destination" }
        $lfsCheck = @(& git -C $Destination lfs fsck 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "Git LFS integrity failed for $Destination : $($lfsCheck -join ' ')" }
        if ($Entry.version -match '^\d+\.\d+\.\d+$') {
            $tags = @(& git -C $Destination tag --points-at HEAD)
            if ($Entry.version -notin $tags) { throw "Existing Git snapshot for $($Entry.id) is not at requested tag $($Entry.version)" }
        }
        if ($Entry.id -eq 'freecad_library') {
            $shallow = (& git -C $Destination rev-parse --is-shallow-repository).Trim()
            if ($shallow -eq 'true') { throw "FreeCAD snapshot is shallow; full history is required to retain contributor attribution." }
        }
        Write-Log "REUSE $($Entry.id) at commit $commit"
        Add-Record -Id $Entry.id -Method 'git' -Path $Destination -Status 'reused' -SourceUrl $Entry.url -Version $commit -Note 'Existing checkout was intentionally not updated.'
        return
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "Destination exists but is not a Git checkout; refusing to overwrite: $Destination"
    }

    $branchArgs = @()
    if ($Entry.version -match '^\d+\.\d+\.\d+$') {
        $branchArgs = @('--branch', $Entry.version)
    }
    Write-Log "GIT CLONE $($Entry.id)"
    $depthArgs = if ($Entry.id -eq 'freecad_library') { @() } else { @('--depth', '1') }
    & git -c advice.detachedHead=false clone @depthArgs @branchArgs -- $Entry.url $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed for $($Entry.id). The incomplete checkout was retained for inspection: $Destination"
    }
    $commit = (& git -C $Destination rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve commit for $Destination" }
    Write-Log "DONE $($Entry.id) at commit $commit"
    Add-Record -Id $Entry.id -Method 'git' -Path $Destination -Status 'downloaded' -SourceUrl $Entry.url -Version $commit
}

function Get-FuelCollection {
    param([pscustomobject]$Entry)
    $base = Join-Path $script:CatalogRoot (Convert-ToNativePath $Entry.relative_path)
    $modelsDir = Join-Path $base 'models'
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null

    $all = New-Object System.Collections.ArrayList
    for ($page = 1; $page -le 50; $page++) {
        $query = "https://fuel.gazebosim.org/1.0/models?page=$page&per_page=100&q=collections%3AScanned%20Objects%20by%20Google%20Research"
        Write-Log "FUEL catalog page $page"
        [string[]]$tlsArgs = @($script:CurlTlsArgs)
        $json = & curl.exe @tlsArgs --location --fail --silent --show-error --retry 5 --connect-timeout 30 --max-time 60 --user-agent $UserAgent $query
        if ($LASTEXITCODE -ne 0) { throw "Fuel API enumeration failed on page $page" }
        # Windows PowerShell 5.1 can preserve a JSON array as one pipeline object
        # when ConvertFrom-Json is wrapped directly in @(...). Assign first so
        # array elements are counted and added individually.
        $parsed = ConvertFrom-Json -InputObject $json
        $batch = @($parsed)
        if ($batch.Count -eq 0) { break }
        [void]$all.AddRange([object[]]$batch)
    }
    $queryUnique = @($all | Group-Object -Property owner, name | ForEach-Object { $_.Group | Select-Object -First 1 } | Sort-Object -Property owner, name)
    $excluded = @($queryUnique | Where-Object { $_.owner -ne 'GoogleResearch' })
    $unique = @($queryUnique | Where-Object { $_.owner -eq 'GoogleResearch' } | Sort-Object -Property name)
    $advertisedBytes = [long](($unique | Measure-Object -Property filesize -Sum).Sum)
    if ($Entry.integrity -notmatch 'fuel-count-([0-9]+)-bytes-([0-9]+)') {
        throw "Fuel manifest entry lacks an exact dated count/byte contract."
    }
    $expectedCount = [int]$Matches[1]
    $expectedTotalBytes = [long]$Matches[2]
    $wrongLicense = @($unique | Where-Object { [int]$_.license_id -ne 2 -or $_.license_name -notmatch 'Attribution 4\.0' })
    if ($unique.Count -ne $expectedCount -or $advertisedBytes -ne $expectedTotalBytes -or $wrongLicense.Count -gt 0) {
        throw "Fuel snapshot contract failed: GoogleResearch count=$($unique.Count)/$expectedCount bytes=$advertisedBytes/$expectedTotalBytes wrong_license=$($wrongLicense.Count)."
    }
    Write-Log "FUEL collection contains $($unique.Count) unique models, advertised $([math]::Round($advertisedBytes / 1GB, 2)) GiB"

    $collectionRemainingBytes = 0L
    foreach ($model in $unique) {
        $fileName = "$(Get-SafeFileStem -Name ([string]$model.name)).zip"
        $destination = Join-Path $modelsDir $fileName
        if (Test-Path -LiteralPath $destination -PathType Leaf) { continue }
        $partial = "$destination.part"
        $partialBytes = if (Test-Path -LiteralPath $partial -PathType Leaf) { [long](Get-Item -LiteralPath $partial).Length } else { 0L }
        $remaining = [long]$model.filesize - $partialBytes
        if ($remaining -lt 0) { throw "Fuel partial file exceeds its advertised size: $partial" }
        $collectionRemainingBytes += $remaining
    }
    Assert-FreeSpace -Path $base -AdditionalBytes $collectionRemainingBytes

    $catalogPath = Join-Path $base 'fuel-catalog.json'
    $excludedPath = Join-Path $base 'fuel-query-excluded-non-google-records.json'
    $indexPath = Join-Path $base 'model-index.csv'
    $candidateDir = Join-Path $script:CatalogRoot (Join-Path 'reports' "fuel-metadata-$($script:RunId)")
    New-Item -ItemType Directory -Path $candidateDir -Force | Out-Null
    $catalogCandidate = Join-Path $candidateDir 'fuel-catalog.json'
    $excludedCandidate = Join-Path $candidateDir 'fuel-query-excluded-non-google-records.json'
    $indexCandidate = Join-Path $candidateDir 'model-index.csv'
    ConvertTo-Json -InputObject @($unique) -Depth 20 | Set-Content -LiteralPath $catalogCandidate -Encoding UTF8
    ConvertTo-Json -InputObject @($excluded) -Depth 20 | Set-Content -LiteralPath $excludedCandidate -Encoding UTF8
    $index = New-Object System.Collections.Generic.List[object]
    $position = 0
    foreach ($model in $unique) {
        $position++
        $escaped = [Uri]::EscapeDataString([string]$model.name)
        $downloadUrl = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/$escaped.zip"
        $fileName = "$(Get-SafeFileStem -Name ([string]$model.name)).zip"
        $destination = Join-Path $modelsDir $fileName
        Write-Log "FUEL model $position/$($unique.Count): $($model.name)"
        Invoke-CurlDownload -Id "google_scanned_objects:$($model.name)" -Url $downloadUrl -Destination $destination -Version $Entry.version -ExpectedBytes ([long]$model.filesize)
        $index.Add([pscustomobject]@{
            source_name = $model.name
            local_file = "models/$fileName"
            source_owner = $model.owner
            source_modified_at = $model.modify_date
            advertised_bytes = [long]$model.filesize
            license = $model.license_name
            source_url = $downloadUrl
        }) | Out-Null
    }
    $index | Export-Csv -LiteralPath $indexCandidate -NoTypeInformation -Encoding UTF8
    $catalogStatus = Assert-OrPromoteGeneratedFile -Candidate $catalogCandidate -Target $catalogPath -Kind fuel_catalog
    [void](Assert-OrPromoteGeneratedFile -Candidate $excludedCandidate -Target $excludedPath -Kind fuel_excluded)
    [void](Assert-OrPromoteGeneratedFile -Candidate $indexCandidate -Target $indexPath -Kind csv)

    $metadataMarkerPath = Join-Path $base '.fuel-metadata-complete.json'
    $metadataMarkerCandidate = Join-Path $candidateDir '.fuel-metadata-complete.json'
    [ordered]@{
        schema = 'world-model-fuel-metadata/v1'
        manifest_id = $Entry.id
        version = $Entry.version
        owner = 'GoogleResearch'
        archive_count = $unique.Count
        advertised_bytes = $advertisedBytes
        catalog_sha256 = (Get-FileHash -LiteralPath $catalogPath -Algorithm SHA256).Hash.ToLowerInvariant()
        excluded_sha256 = (Get-FileHash -LiteralPath $excludedPath -Algorithm SHA256).Hash.ToLowerInvariant()
        index_sha256 = (Get-FileHash -LiteralPath $indexPath -Algorithm SHA256).Hash.ToLowerInvariant()
        completed_at_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $metadataMarkerCandidate -Encoding UTF8
    if (Test-Path -LiteralPath $metadataMarkerPath -PathType Leaf) {
        $existingMarker = Get-Content -LiteralPath $metadataMarkerPath -Raw | ConvertFrom-Json
        $candidateMarker = Get-Content -LiteralPath $metadataMarkerCandidate -Raw | ConvertFrom-Json
        foreach ($field in @('schema', 'manifest_id', 'version', 'owner', 'archive_count', 'advertised_bytes', 'catalog_sha256', 'excluded_sha256', 'index_sha256')) {
            if ([string]$existingMarker.$field -ne [string]$candidateMarker.$field) {
                throw "Fuel metadata completion marker disagrees with the immutable snapshot at field '$field'."
            }
        }
    }
    else {
        Copy-Item -LiteralPath $metadataMarkerCandidate -Destination $metadataMarkerPath
    }
    Add-Record -Id $Entry.id -Method 'fuel_collection' -Path $catalogPath -Status $catalogStatus -SourceUrl $Entry.url -Version $Entry.version -ExpectedBytes $advertisedBytes -Note "$($unique.Count) model archives; per-file records also appear in the report."
}

function Write-Reports {
    $reportDir = Join-Path $script:CatalogRoot 'reports'
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
    $csvPath = Join-Path $reportDir "download-report-$($script:RunId).csv"
    $jsonPath = Join-Path $reportDir "download-report-$($script:RunId).json"
    $contextPath = Join-Path $reportDir "download-run-$($script:RunId).json"
    $script:Records | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8
    if ($script:Records.Count -eq 0) {
        Set-Content -LiteralPath $jsonPath -Value '[]' -Encoding UTF8
    }
    else {
        ConvertTo-Json -InputObject @($script:Records.ToArray()) -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding UTF8
    }
    $csvSha256 = (Get-FileHash -LiteralPath $csvPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $jsonSha256 = (Get-FileHash -LiteralPath $jsonPath -Algorithm SHA256).Hash.ToLowerInvariant()
    [ordered]@{
        run_id = $script:RunId
        succeeded = $script:RunSucceeded
        root = $script:CatalogRoot
        profile = $Profile
        explicit_ids = $IncludeIds
        manifest_sha256 = $script:ManifestSha256
        manifest_snapshot = $script:ManifestSnapshotPath
        report_csv = $csvPath
        report_csv_sha256 = $csvSha256
        report_json = $jsonPath
        report_json_sha256 = $jsonSha256
        tls_offline_revocation_fallback = [bool]$AllowOfflineRevocationFallback
        record_count = $script:Records.Count
        failure = $script:Failure
        finished_at_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $contextPath -Encoding UTF8
    $script:FailureReportWritten = $true
    if ($script:RunSucceeded -and [string]::IsNullOrWhiteSpace($IncludeIds)) {
        # Immutable snapshots remain in manifests/. This root-level file is only
        # the convenience copy corresponding to the latest successful full run.
        Copy-Item -LiteralPath $script:ManifestSnapshotPath -Destination (Join-Path $script:CatalogRoot 'catalog_manifest.csv') -Force
        Copy-Item -LiteralPath $csvPath -Destination (Join-Path $reportDir 'download-report-latest.csv') -Force
        Copy-Item -LiteralPath $jsonPath -Destination (Join-Path $reportDir 'download-report-latest.json') -Force
        Copy-Item -LiteralPath $contextPath -Destination (Join-Path $reportDir 'download-run-latest.json') -Force
    }
    Write-Log "Download report: $csvPath"
}

$script:CatalogRoot = Get-SafeRoot -Candidate $Root
$rootExisted = Test-Path -LiteralPath $script:CatalogRoot -PathType Container
if (-not $rootExisted) {
    New-Item -ItemType Directory -Path $script:CatalogRoot | Out-Null
}
Assert-NoCatalogReparsePoint -Candidate $script:CatalogRoot
$markerPath = Join-Path $script:CatalogRoot '.world-model-catalog-root.json'
if ($rootExisted -and -not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
    $children = @(Get-ChildItem -LiteralPath $script:CatalogRoot -Force)
    if ($children.Count -gt 0) {
        $looksLikeExistingCatalog = (Test-Path -LiteralPath (Join-Path $script:CatalogRoot 'raw') -PathType Container) -and (Test-Path -LiteralPath (Join-Path $script:CatalogRoot 'reports') -PathType Container) -and (Test-Path -LiteralPath (Join-Path $script:CatalogRoot 'catalog_manifest.csv') -PathType Leaf)
        if (-not $AdoptExistingCatalogRoot -or -not $looksLikeExistingCatalog) {
            throw "Refusing to use a non-empty unmarked directory. Choose a new root, or pass -AdoptExistingCatalogRoot only for the previously created D:\WorldModelCatalogs corpus."
        }
    }
}
if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
    [ordered]@{
        schema = 'world-model-catalog-root/v1'
        catalog_id = [guid]::NewGuid().ToString()
        canonical_root = $script:CatalogRoot
        created_or_adopted_at_utc = [DateTime]::UtcNow.ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $markerPath -Encoding UTF8
}
try {
    $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
}
catch {
    throw "Catalog root marker is not valid JSON: $markerPath ($($_.Exception.Message))"
}
if ($marker.schema -ne 'world-model-catalog-root/v1') {
    throw "Catalog root marker has an unsupported schema: $($marker.schema)"
}
$markedRoot = [IO.Path]::GetFullPath([string]$marker.canonical_root).TrimEnd('\')
if (-not $markedRoot.Equals($script:CatalogRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Catalog root marker names '$markedRoot', not '$script:CatalogRoot'."
}
New-Item -ItemType Directory -Path (Join-Path $script:CatalogRoot 'reports') -Force | Out-Null
$script:RunId = "$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))-pid$PID-$([guid]::NewGuid().ToString('N').Substring(0, 8))"
$safeLogName = [IO.Path]::GetFileName($LogName)
if ([string]::IsNullOrWhiteSpace($safeLogName)) { throw 'LogName must name a file.' }
$script:LogPath = Join-Path $script:CatalogRoot (Join-Path 'reports' $safeLogName)
$script:Records = New-Object System.Collections.Generic.List[object]
$script:RunSucceeded = $false
$script:Failure = $null
$script:CurrentEntryId = ''
$script:CurrentSourceUrl = ''
$script:LastCurlExitCode = $null

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifestPath = [IO.Path]::GetFullPath((Join-Path $scriptDir 'catalog_manifest.csv'))
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Manifest not found: $manifestPath"
}
$manifest = @(Import-Csv -LiteralPath $manifestPath)
$script:ManifestSha256 = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestDir = Join-Path $script:CatalogRoot 'manifests'
New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
$script:ManifestSnapshotPath = Join-Path $manifestDir "catalog-manifest-$($script:ManifestSha256).csv"
if (-not (Test-Path -LiteralPath $script:ManifestSnapshotPath -PathType Leaf)) {
    Copy-Item -LiteralPath $manifestPath -Destination $script:ManifestSnapshotPath
}
elseif ((Get-FileHash -LiteralPath $script:ManifestSnapshotPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $script:ManifestSha256) {
    throw "Existing immutable manifest snapshot does not match its content-addressed name: $script:ManifestSnapshotPath"
}
$selectedProfiles = if ($Profile -eq 'Core') { @('core') } else { @('core', 'assets') }
$selected = @($manifest | Where-Object { $_.profile -in $selectedProfiles })
if (-not [string]::IsNullOrWhiteSpace($IncludeIds)) {
    $requestedIds = @($IncludeIds.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    $selected = @($selected | Where-Object { $_.id -in $requestedIds })
    $missingIds = @($requestedIds | Where-Object { $_ -notin @($selected.id) })
    if ($missingIds.Count -gt 0) {
        throw "Requested IDs are absent from the selected profile: $($missingIds -join ', ')"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $script:CatalogRoot 'catalog_manifest.csv') -PathType Leaf)) {
    Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $script:CatalogRoot 'catalog_manifest.csv')
}
Write-Log "Catalog root: $script:CatalogRoot"
Write-Log "Profile: $Profile; selected manifest entries: $($selected.Count)"
if ($IncludeIds) { Write-Log "Explicit IDs: $IncludeIds" }
Write-Log "The script never deletes or updates an existing completed artifact. Partial HTTP downloads use .part files and resume."

try {
    foreach ($entry in $selected) {
        $script:CurrentEntryId = [string]$entry.id
        $script:CurrentSourceUrl = [string]$entry.url
        $destination = Resolve-CatalogRelativePath -RelativePath $entry.relative_path
        switch ($entry.method) {
            'http' {
                $expectedBytes = $null
                $checksum = ''
                if ($entry.integrity -match 'http-content-length-([0-9]+)') {
                    $expectedBytes = [Nullable[long]]([long]$Matches[1])
                }
                if ($entry.integrity -match 'md5-([0-9a-fA-F]{32})') {
                    $checksum = "md5:$($Matches[1])"
                }
                Invoke-CurlDownload -Id $entry.id -Url $entry.url -Destination $destination -Version $entry.version -ExpectedBytes $expectedBytes -UpstreamChecksum $checksum
            }
            'git' {
                Invoke-GitSnapshot -Entry $entry -Destination $destination
            }
            'fuel_collection' {
                Get-FuelCollection -Entry $entry
            }
            default {
                throw "Unsupported selected method '$($entry.method)' for $($entry.id)"
            }
        }
    }
    $script:RunSucceeded = $true
}
catch {
    $script:Failure = [ordered]@{
        failed_id = $script:CurrentEntryId
        failed_url = $script:CurrentSourceUrl
        exception_type = $_.Exception.GetType().FullName
        exception_message = $_.Exception.Message
        curl_exit_code = $script:LastCurlExitCode
        failed_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-Log "FAILED $($script:CurrentEntryId): $($_.Exception.Message)"
    throw
}
finally {
    Write-Reports
}

$driveRoot = [IO.Path]::GetPathRoot($script:CatalogRoot)
$drive = Get-PSDrive -Name $driveRoot.Substring(0, 1)
Write-Log "BOOTSTRAP COMPLETE. Remaining free space on ${driveRoot}: $([math]::Round($drive.Free / 1GB, 2)) GiB"
Write-Host "Next: run verify_catalogs.ps1 -Root '$script:CatalogRoot'"
