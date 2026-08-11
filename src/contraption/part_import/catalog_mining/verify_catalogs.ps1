[CmdletBinding()]
param(
    [Parameter()]
    [string]$Root = 'D:\WorldModelCatalogs',

    [Parameter()]
    [switch]$IncludeGitInternals,

    [Parameter()]
    [ValidateSet('Core', 'CoreAndAssets')]
    [string]$Profile = 'CoreAndAssets',

    [Parameter()]
    [string]$ManifestPath = '',

    [Parameter()]
    [switch]$DeepArchiveValidation
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

trap {
    $failure = [ordered]@{
        succeeded = $false
        failure_kind = 'verification-infrastructure'
        exception_type = $_.Exception.GetType().FullName
        exception_message = $_.Exception.Message
        failed_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    try {
        if ($fullRoot -and (Test-Path -LiteralPath $fullRoot -PathType Container)) {
            $failureDir = Join-Path $fullRoot 'reports'
            New-Item -ItemType Directory -Path $failureDir -Force -ErrorAction SilentlyContinue | Out-Null
            $failureStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
            $failure | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $failureDir "verification-infrastructure-failure-$failureStamp.json") -Encoding UTF8 -ErrorAction SilentlyContinue
        }
    }
    catch { }
    [Console]::Error.WriteLine("Verification infrastructure failed: $($_.Exception.Message)")
    exit 3
}

$fullRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\')
$volume = [IO.Path]::GetPathRoot($fullRoot)
if ($volume -notin @('D:\', 'F:\') -or $fullRoot -eq $volume.TrimEnd('\')) {
    throw "Root must be a dedicated directory on D:\ or F:\"
}
if (-not (Test-Path -LiteralPath $fullRoot -PathType Container)) {
    throw "Catalog root does not exist: $fullRoot"
}
$markerPath = Join-Path $fullRoot '.world-model-catalog-root.json'
if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
    throw "Catalog root marker is missing; refusing to verify an unmarked directory: $fullRoot"
}
try {
    $rootMarker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
}
catch {
    throw "Catalog root marker is not valid JSON: $markerPath ($($_.Exception.Message))"
}
if ($rootMarker.schema -ne 'world-model-catalog-root/v1') {
    throw "Catalog root marker has an unsupported schema: $($rootMarker.schema)"
}
$markedRoot = [IO.Path]::GetFullPath([string]$rootMarker.canonical_root).TrimEnd('\')
if (-not $markedRoot.Equals($fullRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Catalog root marker names '$markedRoot', not '$fullRoot'."
}
$catalogGuid = [guid]::Empty
if (-not [guid]::TryParse([string]$rootMarker.catalog_id, [ref]$catalogGuid)) {
    throw "Catalog root marker has an invalid catalog_id: $($rootMarker.catalog_id)"
}
if (-not (Test-Path -LiteralPath (Join-Path $fullRoot 'raw') -PathType Container)) {
    throw "Catalog raw directory is missing: $fullRoot\raw"
}

function Assert-NoCatalogReparsePoint {
    param([string]$Candidate)
    $root = $fullRoot.TrimEnd('\')
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

Assert-NoCatalogReparsePoint -Candidate $fullRoot

function Resolve-CatalogRelativePath {
    param([string]$RelativePath)
    if ([IO.Path]::IsPathRooted($RelativePath)) { throw "Manifest path must be relative: $RelativePath" }
    $candidate = [IO.Path]::GetFullPath((Join-Path $fullRoot ($RelativePath.Replace('/', '\'))))
    $prefix = $fullRoot.TrimEnd('\') + '\'
    if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest path escapes the catalog root: $RelativePath"
    }
    Assert-NoCatalogReparsePoint -Candidate $candidate
    return $candidate
}

function Resolve-ContainedChildPath {
    param([string]$Parent, [string]$RelativePath)
    if ([IO.Path]::IsPathRooted($RelativePath)) { throw "Child path must be relative: $RelativePath" }
    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd('\')
    $candidate = [IO.Path]::GetFullPath((Join-Path $parentFull ($RelativePath.Replace('/', '\'))))
    $prefix = $parentFull + '\'
    if (-not $candidate.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Child path escapes its parent: $RelativePath"
    }
    Assert-NoCatalogReparsePoint -Candidate $candidate
    return $candidate
}

function Test-FilePrefix {
    param([string]$Path, [byte[]]$Prefix)
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        foreach ($expected in $Prefix) {
            $actual = $stream.ReadByte()
            if ($actual -ne [int]$expected) { return $false }
        }
        return $true
    }
    finally {
        $stream.Dispose()
    }
}

function Test-ZipCentralDirectory {
    param([string]$Path, [switch]$ReadAllEntries)
    try {
        Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null
        $archive = [IO.Compression.ZipFile]::OpenRead($Path)
        try {
            if ($archive.Entries.Count -eq 0) { return $false }
            if ($ReadAllEntries) {
                $buffer = New-Object byte[] (1MB)
                foreach ($entry in $archive.Entries) {
                    if ($entry.FullName.EndsWith('/')) { continue }
                    $entryStream = $entry.Open()
                    try {
                        while ($entryStream.Read($buffer, 0, $buffer.Length) -gt 0) { }
                    }
                    finally { $entryStream.Dispose() }
                }
            }
            return $true
        }
        finally { $archive.Dispose() }
    }
    catch { return $false }
}

function Test-GzipStream {
    param([string]$Path)
    try {
        $fileStream = [IO.File]::OpenRead($Path)
        try {
            $gzipStream = New-Object IO.Compression.GZipStream -ArgumentList $fileStream, ([IO.Compression.CompressionMode]::Decompress)
            try {
                $buffer = New-Object byte[] (1MB)
                while ($gzipStream.Read($buffer, 0, $buffer.Length) -gt 0) { }
            }
            finally { $gzipStream.Dispose() }
        }
        finally { $fileStream.Dispose() }
        return $true
    }
    catch { return $false }
}

function Test-PythonCompressedStream {
    param([string]$Path, [ValidateSet('bz2', 'lzma')][string]$Module)
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $python) { return $false }
    $code = "import $Module,sys;f=$Module.open(sys.argv[1],'rb');any(iter(lambda:not f.read(1048576),True));f.close()"
    & $python.Source -c $code $Path 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Get-ArchiveFormatResult {
    param([string]$Path)
    $lower = $Path.ToLowerInvariant()
    if ($lower.EndsWith('.zip')) {
        $magic = (Test-FilePrefix -Path $Path -Prefix ([byte[]](0x50, 0x4b)))
        return [pscustomobject]@{ checked = $true; pass = ($magic -and (Test-ZipCentralDirectory -Path $Path -ReadAllEntries:$DeepArchiveValidation)); expected = $(if ($DeepArchiveValidation) { 'ZIP signature plus every entry readable with CRC validation' } else { 'ZIP signature and readable central directory' }) }
    }
    if ($lower.EndsWith('.tar.gz') -or $lower.EndsWith('.tgz')) {
        return [pscustomobject]@{ checked = $true; pass = (Test-FilePrefix -Path $Path -Prefix ([byte[]](0x1f, 0x8b))); expected = 'gzip signature; full tar listing when deep validation is enabled' }
    }
    if ($lower.EndsWith('.gz')) {
        $magic = Test-FilePrefix -Path $Path -Prefix ([byte[]](0x1f, 0x8b))
        $pass = $magic -and (-not $DeepArchiveValidation -or (Test-GzipStream -Path $Path))
        return [pscustomobject]@{ checked = $true; pass = $pass; expected = $(if ($DeepArchiveValidation) { 'complete readable gzip stream with trailer validation' } else { 'gzip signature 1f8b' }) }
    }
    if ($lower.EndsWith('.tar.bz2')) {
        return [pscustomobject]@{ checked = $true; pass = (Test-FilePrefix -Path $Path -Prefix ([byte[]](0x42, 0x5a, 0x68))); expected = 'bzip2 signature; full tar listing when deep validation is enabled' }
    }
    if ($lower.EndsWith('.bz2')) {
        $magic = Test-FilePrefix -Path $Path -Prefix ([byte[]](0x42, 0x5a, 0x68))
        $pass = $magic -and (-not $DeepArchiveValidation -or (Test-PythonCompressedStream -Path $Path -Module bz2))
        return [pscustomobject]@{ checked = $true; pass = $pass; expected = $(if ($DeepArchiveValidation) { 'complete readable bzip2 stream' } else { 'bzip2 signature BZh' }) }
    }
    if ($lower.EndsWith('.tar.xz') -or $lower.EndsWith('.txz')) {
        return [pscustomobject]@{ checked = $true; pass = (Test-FilePrefix -Path $Path -Prefix ([byte[]](0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00))); expected = 'xz signature; full tar listing when deep validation is enabled' }
    }
    if ($lower.EndsWith('.xz')) {
        $magic = Test-FilePrefix -Path $Path -Prefix ([byte[]](0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00))
        $pass = $magic -and (-not $DeepArchiveValidation -or (Test-PythonCompressedStream -Path $Path -Module lzma))
        return [pscustomobject]@{ checked = $true; pass = $pass; expected = $(if ($DeepArchiveValidation) { 'complete readable xz stream' } else { 'xz signature fd377a585a00' }) }
    }
    return [pscustomobject]@{ checked = $false; pass = $true; expected = '' }
}

$runContextPath = Join-Path $fullRoot 'reports\download-run-latest.json'
if (-not (Test-Path -LiteralPath $runContextPath -PathType Leaf)) {
    throw "A latest successful full bootstrap context is required before verification: $runContextPath"
}
$runContext = Get-Content -LiteralPath $runContextPath -Raw | ConvertFrom-Json
if (-not $runContext.succeeded -or -not [string]::IsNullOrWhiteSpace([string]$runContext.explicit_ids)) {
    throw "The latest bootstrap context is not a successful full-profile run."
}
$contextRoot = [IO.Path]::GetFullPath([string]$runContext.root).TrimEnd('\')
if (-not $contextRoot.Equals($fullRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Bootstrap context root '$contextRoot' does not match verifier root '$fullRoot'."
}
if ($Profile -eq 'CoreAndAssets' -and $runContext.profile -ne 'CoreAndAssets') {
    throw "CoreAndAssets verification requires a successful CoreAndAssets bootstrap context."
}
if ([string]::IsNullOrWhiteSpace($ManifestPath)) { $ManifestPath = [string]$runContext.manifest_snapshot }
$ManifestPath = [IO.Path]::GetFullPath($ManifestPath)
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Verification manifest does not exist: $ManifestPath"
}
Assert-NoCatalogReparsePoint -Candidate $ManifestPath
$manifestDirectory = [IO.Path]::GetFullPath((Join-Path $fullRoot 'manifests')).TrimEnd('\') + '\'
if (-not $ManifestPath.StartsWith($manifestDirectory, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Verification manifest must be an immutable snapshot under $manifestDirectory"
}
$manifestHash = (Get-FileHash -LiteralPath $ManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifestHash -ne ([string]$runContext.manifest_sha256).ToLowerInvariant()) {
    throw "Verification manifest hash does not match the successful bootstrap context."
}
if ([IO.Path]::GetFileName($ManifestPath) -ne "catalog-manifest-$manifestHash.csv") {
    throw "Verification manifest filename is not its content address: $ManifestPath"
}
$manifest = @(Import-Csv -LiteralPath $ManifestPath)
$selectedProfiles = if ($Profile -eq 'Core') { @('core') } else { @('core', 'assets') }

$acquisitionReportPath = [IO.Path]::GetFullPath([string]$runContext.report_csv)
$reportsPrefix = [IO.Path]::GetFullPath((Join-Path $fullRoot 'reports')).TrimEnd('\') + '\'
if (-not $acquisitionReportPath.StartsWith($reportsPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Acquisition report path escapes the catalog reports directory: $acquisitionReportPath"
}
Assert-NoCatalogReparsePoint -Candidate $acquisitionReportPath
if (-not (Test-Path -LiteralPath $acquisitionReportPath -PathType Leaf)) {
    throw "Acquisition report from the bootstrap context is missing: $acquisitionReportPath"
}
$acquisitionReportHash = (Get-FileHash -LiteralPath $acquisitionReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($acquisitionReportHash -ne ([string]$runContext.report_csv_sha256).ToLowerInvariant()) {
    throw "Acquisition report hash does not match the successful bootstrap context."
}
$acquisitionRows = @(Import-Csv -LiteralPath $acquisitionReportPath)
if ($acquisitionRows.Count -ne [int]$runContext.record_count) {
    throw "Acquisition report row count does not match the bootstrap context."
}
if (@($acquisitionRows | Where-Object { $_.manifest_sha256 -ne $manifestHash }).Count -gt 0) {
    throw "One or more acquisition records refer to a different manifest hash."
}
$acquisitionById = @{}
foreach ($recordGroup in @($acquisitionRows | Group-Object -Property id)) {
    $acquisitionById[[string]$recordGroup.Name] = @($recordGroup.Group)
}

$reportDir = Join-Path $fullRoot 'reports'
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$hashPath = Join-Path $reportDir "sha256-$stamp.csv"
$summaryPath = Join-Path $reportDir "verification-summary-$stamp.json"

$reparseItems = @(Get-ChildItem -LiteralPath (Join-Path $fullRoot 'raw') -Recurse -Force | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 })
if ($reparseItems.Count -gt 0) {
    throw "Raw corpus contains junctions or symbolic links, which are forbidden: $(@($reparseItems | Select-Object -First 10 -ExpandProperty FullName) -join '; ')"
}
$files = @(Get-ChildItem -LiteralPath (Join-Path $fullRoot 'raw') -File -Recurse -Force | Where-Object {
    $_.Name -notlike '*.part' -and ($IncludeGitInternals -or $_.FullName -notmatch '[\\/]\.git[\\/]')
})
$rows = New-Object System.Collections.Generic.List[object]
$hashByRelativePath = @{}
$i = 0
foreach ($file in $files) {
    $i++
    if (($i % 100) -eq 0 -or $i -eq 1 -or $i -eq $files.Count) {
        Write-Host "Hashing $i/$($files.Count): $($file.Name)"
    }
    $hash = Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256
    $baseUri = New-Object System.Uri(($fullRoot.TrimEnd('\') + '\'))
    $targetUri = New-Object System.Uri($file.FullName)
    $relative = [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($targetUri).ToString())
    $hashByRelativePath[$relative.Replace('\', '/')] = $hash.Hash.ToLowerInvariant()
    $rows.Add([pscustomobject]@{
        relative_path = $relative
        bytes = [long]$file.Length
        sha256 = $hash.Hash.ToLowerInvariant()
        last_write_time_utc = $file.LastWriteTimeUtc.ToString('o')
    }) | Out-Null
}
$rows | Export-Csv -LiteralPath $hashPath -NoTypeInformation -Encoding UTF8
Copy-Item -LiteralPath $hashPath -Destination (Join-Path $reportDir 'sha256-latest.csv') -Force

$gitRepos = @(Get-ChildItem -LiteralPath (Join-Path $fullRoot 'raw') -Directory -Recurse -Force -Filter '.git' | ForEach-Object { $_.Parent.FullName })
$gitRows = @(foreach ($repo in $gitRepos) {
    $commit = (& git -C $repo rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not resolve Git HEAD: $repo" }
    $statusLines = @(& git -C $repo status --porcelain)
    if ($LASTEXITCODE -ne 0) { throw "Could not read Git status: $repo" }
    $lfsOutput = @(& git -C $repo lfs fsck 2>&1)
    $lfsExit = $LASTEXITCODE
    $objectFsckOutput = @(& git -C $repo fsck --full 2>&1)
    $objectFsckExit = $LASTEXITCODE
    $shallow = (& git -C $repo rev-parse --is-shallow-repository).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Could not determine Git shallow state: $repo" }
    $tags = @(& git -C $repo tag --points-at HEAD)
    [pscustomobject]@{
        path = $repo.Substring($fullRoot.Length + 1).Replace('\', '/')
        commit = $commit
        origin = (& git -C $repo config --get remote.origin.url).Trim()
        tags_at_head = ($tags -join ';')
        shallow = $shallow
        worktree_clean = ($statusLines.Count -eq 0)
        lfs_fsck_pass = ($lfsExit -eq 0)
        lfs_fsck_note = ($lfsOutput -join ' ')
        object_fsck_pass = ($objectFsckExit -eq 0)
        object_fsck_note = ($objectFsckOutput -join ' ')
    }
})
$gitPath = Join-Path $reportDir "git-snapshots-$stamp.csv"
$gitRows | Export-Csv -LiteralPath $gitPath -NoTypeInformation -Encoding UTF8
Copy-Item -LiteralPath $gitPath -Destination (Join-Path $reportDir 'git-snapshots-latest.csv') -Force

$integrityChecks = New-Object System.Collections.Generic.List[object]
function Add-IntegrityCheck {
    param([string]$Id, [string]$Check, [bool]$Pass, [string]$Expected, [string]$Actual, [string]$Path)
    $integrityChecks.Add([pscustomobject]@{
        id = $Id
        check = $Check
        pass = $Pass
        expected = $Expected
        actual = $Actual
        relative_path = $Path
    }) | Out-Null
}

$manifestPathUsed = $ManifestPath
if (Test-Path -LiteralPath $manifestPathUsed) {
    foreach ($entry in @($manifest | Where-Object { $_.profile -in $selectedProfiles })) {
        $candidate = Resolve-CatalogRelativePath -RelativePath $entry.relative_path
        switch ($entry.method) {
            'http' {
                $exists = Test-Path -LiteralPath $candidate -PathType Leaf
                $actualLength = if ($exists) { [long](Get-Item -LiteralPath $candidate).Length } else { 0L }
                $actual = if ($exists) { "present:$actualLength bytes" } else { 'missing' }
                Add-IntegrityCheck -Id $entry.id -Check 'manifest-completeness' -Pass ($exists -and $actualLength -gt 0) -Expected 'non-empty completed file' -Actual $actual -Path $entry.relative_path
                if ($exists) {
                    # Windows PowerShell 5.1 unwraps a one-element array emitted from an
                    # if-expression. Wrap the whole expression so .Count and [0] remain
                    # well-defined for the overwhelmingly common one-record case.
                    $baselineRows = @(if ($acquisitionById.ContainsKey([string]$entry.id)) { $acquisitionById[[string]$entry.id] })
                    $normalizedPath = $entry.relative_path.Replace('\', '/')
                    $baselineOk = $baselineRows.Count -eq 1 -and
                        $baselineRows[0].method -eq 'http' -and
                        $baselineRows[0].source_url -eq $entry.url -and
                        $baselineRows[0].source_version -eq $entry.version -and
                        $baselineRows[0].relative_path.Replace('\', '/') -eq $normalizedPath -and
                        [long]$baselineRows[0].bytes -eq $actualLength -and
                        -not [string]::IsNullOrWhiteSpace($baselineRows[0].local_sha256) -and
                        $baselineRows[0].local_sha256 -eq $hashByRelativePath[$normalizedPath]
                    $baselineActual = if ($baselineRows.Count -eq 1) { "url=$($baselineRows[0].source_url) version=$($baselineRows[0].source_version) bytes=$($baselineRows[0].bytes) sha256=$($baselineRows[0].local_sha256)" } else { "acquisition rows=$($baselineRows.Count)" }
                    Add-IntegrityCheck -Id $entry.id -Check 'acquisition-content-address' -Pass $baselineOk -Expected "one matching HTTPS acquisition record with local SHA-256 $($hashByRelativePath[$normalizedPath])" -Actual $baselineActual -Path $entry.relative_path

                    $archiveResult = Get-ArchiveFormatResult -Path $candidate
                    if ($archiveResult.checked) {
                        Add-IntegrityCheck -Id $entry.id -Check 'archive-format' -Pass $archiveResult.pass -Expected $archiveResult.expected -Actual $(if ($archiveResult.pass) { 'readable format signature/structure' } else { 'format validation failed' }) -Path $entry.relative_path
                    }
                    if ($DeepArchiveValidation -and ($candidate.ToLowerInvariant() -match '\.(tar\.gz|tgz|tar\.bz2|tar\.xz|txz|tar)$')) {
                        & tar.exe -tf $candidate *> $null
                        $tarPass = $LASTEXITCODE -eq 0
                        Add-IntegrityCheck -Id $entry.id -Check 'deep-archive-listing' -Pass $tarPass -Expected 'tar can list the complete archive' -Actual "tar exit code $LASTEXITCODE" -Path $entry.relative_path
                    }
                    if ($candidate.ToLowerInvariant().EndsWith('.json')) {
                        try {
                            $null = Get-Content -LiteralPath $candidate -Raw | ConvertFrom-Json
                            $jsonPass = $true
                            $jsonActual = 'parseable JSON'
                        }
                        catch {
                            $jsonPass = $false
                            $jsonActual = $_.Exception.Message
                        }
                        Add-IntegrityCheck -Id $entry.id -Check 'json-parse' -Pass $jsonPass -Expected 'parseable JSON' -Actual $jsonActual -Path $entry.relative_path
                    }
                }
            }
            'git' {
                $gitDir = Join-Path $candidate '.git'
                $exists = Test-Path -LiteralPath $gitDir
                $normalizedGitPath = $entry.relative_path.Replace('\', '/').TrimEnd('/')
                $gitRow = @($gitRows | Where-Object { $_.path.Replace('\', '/').TrimEnd('/') -eq $normalizedGitPath } | Select-Object -First 1)
                $originOk = $gitRow.Count -eq 1 -and $gitRow[0].origin -eq $entry.url
                $tagOk = $true
                if ($entry.version -match '^\d+\.\d+\.\d+$') {
                    $tagOk = $gitRow.Count -eq 1 -and $entry.version -in @($gitRow[0].tags_at_head.Split(';'))
                }
                $historyOk = $entry.id -ne 'freecad_library' -or ($gitRow.Count -eq 1 -and $gitRow[0].shallow -eq 'false')
                $baselineRows = @(if ($acquisitionById.ContainsKey([string]$entry.id)) { $acquisitionById[[string]$entry.id] })
                $commitOk = $baselineRows.Count -eq 1 -and
                    $gitRow.Count -eq 1 -and
                    $baselineRows[0].method -eq 'git' -and
                    $baselineRows[0].source_url -eq $entry.url -and
                    $baselineRows[0].relative_path.Replace('\', '/').TrimEnd('/') -eq $entry.relative_path.Replace('\', '/').TrimEnd('/') -and
                    $baselineRows[0].source_version -eq $gitRow[0].commit
                $valid = $exists -and $gitRow.Count -eq 1 -and $originOk -and $tagOk -and $historyOk -and $commitOk -and $gitRow[0].worktree_clean -and $gitRow[0].lfs_fsck_pass -and $gitRow[0].object_fsck_pass
                $actual = if ($gitRow.Count -eq 1) { "commit=$($gitRow[0].commit) acquisition_commit=$(if($baselineRows.Count -eq 1){$baselineRows[0].source_version}else{'missing'}) origin=$($gitRow[0].origin) tags=$($gitRow[0].tags_at_head) shallow=$($gitRow[0].shallow) clean=$($gitRow[0].worktree_clean) lfs=$($gitRow[0].lfs_fsck_pass) objects=$($gitRow[0].object_fsck_pass)" } else { 'missing Git verification row' }
                Add-IntegrityCheck -Id $entry.id -Check 'manifest-git-contract' -Pass $valid -Expected "acquisition-pinned commit; origin=$($entry.url); tag=$($entry.version); FreeCAD full history; Git/LFS fsck" -Actual $actual -Path $entry.relative_path
            }
            'fuel_collection' {
                $indexPath = Join-Path $candidate 'model-index.csv'
                $catalogPath = Join-Path $candidate 'fuel-catalog.json'
                $metadataMarkerPath = Join-Path $candidate '.fuel-metadata-complete.json'
                $indexCount = if (Test-Path -LiteralPath $indexPath -PathType Leaf) { @(Import-Csv -LiteralPath $indexPath).Count } else { 0 }
                $indexRows = @(if ($indexCount -gt 0) { Import-Csv -LiteralPath $indexPath })
                $archiveCount = if (Test-Path -LiteralPath (Join-Path $candidate 'models') -PathType Container) { @(Get-ChildItem -LiteralPath (Join-Path $candidate 'models') -File -Filter '*.zip').Count } else { 0 }
                $expectedCount = 0
                $expectedBytes = 0L
                if ($entry.integrity -match 'fuel-count-([0-9]+)-bytes-([0-9]+)') {
                    $expectedCount = [int]$Matches[1]
                    $expectedBytes = [long]$Matches[2]
                }
                $indexBytes = if ($indexRows.Count -gt 0) { [long](($indexRows | Measure-Object -Property advertised_bytes -Sum).Sum) } else { 0L }
                $identityOk = @($indexRows | Where-Object { $_.source_owner -ne 'GoogleResearch' -or $_.license -notmatch 'Attribution 4\.0' }).Count -eq 0
                $exists = (Test-Path -LiteralPath $catalogPath -PathType Leaf) -and (Test-Path -LiteralPath $metadataMarkerPath -PathType Leaf) -and $expectedCount -gt 0 -and $indexCount -eq $expectedCount -and $archiveCount -eq $expectedCount -and $indexBytes -eq $expectedBytes -and $identityOk
                Add-IntegrityCheck -Id $entry.id -Check 'manifest-completeness' -Pass $exists -Expected "completed immutable metadata marker plus catalog and $expectedCount GoogleResearch CC-BY archives totaling $expectedBytes advertised bytes" -Actual "index=$indexCount archives=$archiveCount bytes=$indexBytes identity_ok=$identityOk catalog=$([bool](Test-Path -LiteralPath $catalogPath -PathType Leaf)) marker=$([bool](Test-Path -LiteralPath $metadataMarkerPath -PathType Leaf))" -Path $entry.relative_path
            }
            default {
                Add-IntegrityCheck -Id $entry.id -Check 'manifest-completeness' -Pass $false -Expected 'supported selected acquisition method' -Actual $entry.method -Path $entry.relative_path
            }
        }
    }
    foreach ($entry in $manifest) {
        if ($entry.profile -notin $selectedProfiles -or $entry.method -ne 'http') { continue }
        $candidate = Resolve-CatalogRelativePath -RelativePath $entry.relative_path
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        if ($entry.integrity -match 'md5-([0-9a-fA-F]{32})') {
            $expectedMd5 = $Matches[1].ToLowerInvariant()
            $actualMd5 = (Get-FileHash -LiteralPath $candidate -Algorithm MD5).Hash.ToLowerInvariant()
            Add-IntegrityCheck -Id $entry.id -Check 'manifest-md5' -Pass ($actualMd5 -eq $expectedMd5) -Expected $expectedMd5 -Actual $actualMd5 -Path $entry.relative_path
        }
        if ($entry.integrity -match 'http-content-length-([0-9]+)') {
            $expectedLength = [long]$Matches[1]
            $actualLength = (Get-Item -LiteralPath $candidate).Length
            Add-IntegrityCheck -Id $entry.id -Check 'manifest-content-length' -Pass ($actualLength -eq $expectedLength) -Expected ([string]$expectedLength) -Actual ([string]$actualLength) -Path $entry.relative_path
        }
    }
}

$ncbiEntry = @($manifest | Where-Object { $_.id -eq 'ncbi_taxonomy' -and $_.profile -in $selectedProfiles } | Select-Object -First 1)
$ncbiMd5Entry = @($manifest | Where-Object { $_.id -eq 'ncbi_taxonomy_md5' -and $_.profile -in $selectedProfiles } | Select-Object -First 1)
if ($ncbiEntry.Count -eq 1 -and $ncbiMd5Entry.Count -eq 1) {
    $ncbiArchive = Resolve-CatalogRelativePath -RelativePath $ncbiEntry[0].relative_path
    $ncbiMd5File = Resolve-CatalogRelativePath -RelativePath $ncbiMd5Entry[0].relative_path
    $expected = ''
    $actual = ''
    $pass = $false
    try {
        if (-not (Test-Path -LiteralPath $ncbiArchive -PathType Leaf) -or -not (Test-Path -LiteralPath $ncbiMd5File -PathType Leaf)) { throw 'archive or MD5 companion is missing' }
        $md5Text = Get-Content -LiteralPath $ncbiMd5File -Raw
        if ($md5Text -notmatch '(?im)^\s*([0-9a-fA-F]{32})\s+\*?new_taxdump\.tar\.gz\s*$') { throw 'MD5 companion lacks the expected filename-bound digest' }
        $expected = $Matches[1].ToLowerInvariant()
        $actual = (Get-FileHash -LiteralPath $ncbiArchive -Algorithm MD5).Hash.ToLowerInvariant()
        $pass = $actual -eq $expected
    }
    catch { $actual = $_.Exception.Message }
    Add-IntegrityCheck -Id 'ncbi_taxonomy' -Check 'upstream-md5-file' -Pass $pass -Expected $(if ($expected) { $expected } else { 'valid filename-bound upstream MD5' }) -Actual $actual -Path $ncbiEntry[0].relative_path
}

$uniprotMetadataEntry = @($manifest | Where-Object { $_.id -eq 'uniprot_release_metadata' -and $_.profile -in $selectedProfiles } | Select-Object -First 1)
$uniprotDataEntry = @($manifest | Where-Object { $_.id -eq 'uniprot_sprot' -and $_.profile -in $selectedProfiles } | Select-Object -First 1)
if ($uniprotMetadataEntry.Count -eq 1 -and $uniprotDataEntry.Count -eq 1) {
    $uniprotMetalink = Resolve-CatalogRelativePath -RelativePath $uniprotMetadataEntry[0].relative_path
    $uniprotData = Resolve-CatalogRelativePath -RelativePath $uniprotDataEntry[0].relative_path
    $expected = ''
    $actual = ''
    $pass = $false
    try {
        if (-not (Test-Path -LiteralPath $uniprotMetalink -PathType Leaf) -or -not (Test-Path -LiteralPath $uniprotData -PathType Leaf)) { throw 'data or RELEASE.metalink is missing' }
        [xml]$xml = Get-Content -LiteralPath $uniprotMetalink -Raw
        $namespaceUri = [string]$xml.DocumentElement.NamespaceURI
        if ($namespaceUri -ne 'http://www.metalinker.org/') { throw "unsupported Metalink namespace: $namespaceUri" }
        $ns = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
        $ns.AddNamespace('m', $namespaceUri)
        $fileNode = $xml.SelectSingleNode("//m:file[@name='uniprot_sprot.dat.gz']", $ns)
        $hashNode = if ($fileNode) { $fileNode.SelectSingleNode("m:verification/m:hash[@type='md5']", $ns) } else { $null }
        $sizeNode = if ($fileNode) { $fileNode.SelectSingleNode('m:size', $ns) } else { $null }
        if (-not $hashNode -or $hashNode.InnerText.Trim() -notmatch '^[0-9a-fA-F]{32}$') { throw 'Metalink lacks the expected MD5 node' }
        if (-not $sizeNode -or $sizeNode.InnerText.Trim() -notmatch '^\d+$') { throw 'Metalink lacks the expected size node' }
        $expected = $hashNode.InnerText.Trim().ToLowerInvariant()
        $expectedSize = [long]$sizeNode.InnerText.Trim()
        $actualSize = [long](Get-Item -LiteralPath $uniprotData).Length
        $actual = (Get-FileHash -LiteralPath $uniprotData -Algorithm MD5).Hash.ToLowerInvariant()
        $pass = $actual -eq $expected -and $actualSize -eq $expectedSize
        $actual = "$actual; bytes=$actualSize"
        $expected = "$expected; bytes=$expectedSize"
    }
    catch { $actual = $_.Exception.Message }
    Add-IntegrityCheck -Id 'uniprot_sprot' -Check 'upstream-metalink-v3-md5-and-size' -Pass $pass -Expected $(if ($expected) { $expected } else { 'valid Metalink v3 MD5 and size' }) -Actual $actual -Path $uniprotDataEntry[0].relative_path
}

$gsoManifestEntry = @($manifest | Where-Object { $_.id -eq 'google_scanned_objects' -and $_.profile -in $selectedProfiles } | Select-Object -First 1)
if ($gsoManifestEntry.Count -eq 1) {
    $gsoBase = Resolve-CatalogRelativePath -RelativePath $gsoManifestEntry[0].relative_path
    $gsoIndex = Join-Path $gsoBase 'model-index.csv'
    $gsoCatalog = Join-Path $gsoBase 'fuel-catalog.json'
    $gsoExcluded = Join-Path $gsoBase 'fuel-query-excluded-non-google-records.json'
    $gsoMarker = Join-Path $gsoBase '.fuel-metadata-complete.json'
    $bad = New-Object System.Collections.Generic.List[string]
    try {
        if (-not (Test-Path -LiteralPath $gsoIndex -PathType Leaf)) { throw 'model-index.csv is missing' }
        if (-not (Test-Path -LiteralPath $gsoCatalog -PathType Leaf)) { throw 'fuel-catalog.json is missing' }
        if (-not (Test-Path -LiteralPath $gsoExcluded -PathType Leaf)) { throw 'excluded-query metadata is missing' }
        if (-not (Test-Path -LiteralPath $gsoMarker -PathType Leaf)) { throw 'metadata completion marker is missing' }
        $gsoRows = @(Import-Csv -LiteralPath $gsoIndex)
        $catalogParsed = Get-Content -LiteralPath $gsoCatalog -Raw | ConvertFrom-Json
        $catalogRows = @($catalogParsed)
        $marker = Get-Content -LiteralPath $gsoMarker -Raw | ConvertFrom-Json
    }
    catch {
        $bad.Add("metadata:$($_.Exception.Message)") | Out-Null
        $gsoRows = @()
        $catalogRows = @()
        $marker = $null
    }

    $duplicateNames = @($gsoRows | Group-Object -Property source_name | Where-Object Count -ne 1)
    $duplicateLocalFiles = @($gsoRows | Group-Object -Property local_file | Where-Object Count -ne 1)
    $duplicateCatalogNames = @($catalogRows | Group-Object -Property name | Where-Object Count -ne 1)
    if ($duplicateNames.Count -gt 0) { $bad.Add("duplicate-index-names:$($duplicateNames.Count)") | Out-Null }
    if ($duplicateLocalFiles.Count -gt 0) { $bad.Add("duplicate-index-paths:$($duplicateLocalFiles.Count)") | Out-Null }
    if ($duplicateCatalogNames.Count -gt 0) { $bad.Add("duplicate-catalog-names:$($duplicateCatalogNames.Count)") | Out-Null }

    $catalogByName = @{}
    foreach ($catalogRow in $catalogRows) { $catalogByName[[string]$catalogRow.name] = $catalogRow }
    $indexedPathSet = @{}
    foreach ($row in $gsoRows) {
        try {
            $asset = Resolve-ContainedChildPath -Parent $gsoBase -RelativePath $row.local_file
        }
        catch {
            $bad.Add("path:$($row.source_name):$($row.local_file)") | Out-Null
            continue
        }
        $normalizedLocalFile = $row.local_file.Replace('\', '/')
        $indexedPathSet[$normalizedLocalFile] = $true
        if (-not (Test-Path -LiteralPath $asset -PathType Leaf)) {
            $bad.Add("missing:$($row.source_name)") | Out-Null
            continue
        }
        $actualLength = (Get-Item -LiteralPath $asset).Length
        if ($actualLength -ne [long]$row.advertised_bytes) {
            $bad.Add("size:$($row.source_name):$actualLength") | Out-Null
        }
        if ($row.source_owner -ne 'GoogleResearch' -or $row.license -notmatch 'Attribution 4\.0') {
            $bad.Add("identity-license:$($row.source_name)") | Out-Null
        }
        if (-not $catalogByName.ContainsKey([string]$row.source_name)) {
            $bad.Add("catalog-missing:$($row.source_name)") | Out-Null
        }
        else {
            $catalogRow = $catalogByName[[string]$row.source_name]
            if ($catalogRow.owner -ne 'GoogleResearch' -or [long]$catalogRow.filesize -ne [long]$row.advertised_bytes -or [int]$catalogRow.license_id -ne 2 -or $catalogRow.license_name -notmatch 'Attribution 4\.0') {
                $bad.Add("catalog-index-mismatch:$($row.source_name)") | Out-Null
            }
        }
        if (-not (Test-ZipCentralDirectory -Path $asset -ReadAllEntries:$DeepArchiveValidation)) {
            $bad.Add("zip:$($row.source_name)") | Out-Null
        }
        $baselineId = "google_scanned_objects:$($row.source_name)"
        $baselineRows = @(if ($acquisitionById.ContainsKey($baselineId)) { $acquisitionById[$baselineId] })
        $rootRelative = $asset.Substring($fullRoot.Length + 1).Replace('\', '/')
        $expectedUrl = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/$([Uri]::EscapeDataString([string]$row.source_name)).zip"
        $baselineOk = $baselineRows.Count -eq 1 -and
            $baselineRows[0].method -eq 'http' -and
            $baselineRows[0].source_url -eq $expectedUrl -and
            $baselineRows[0].source_version -eq $gsoManifestEntry[0].version -and
            $baselineRows[0].relative_path.Replace('\', '/') -eq $rootRelative -and
            [long]$baselineRows[0].bytes -eq [long]$row.advertised_bytes -and
            $baselineRows[0].local_sha256 -eq $hashByRelativePath[$rootRelative]
        if (-not $baselineOk) { $bad.Add("acquisition-baseline:$($row.source_name)") | Out-Null }
    }
    $expectedGsoCount = 0
    $expectedGsoBytes = 0L
    if ($gsoManifestEntry[0].integrity -match 'fuel-count-([0-9]+)-bytes-([0-9]+)') {
        $expectedGsoCount = [int]$Matches[1]
        $expectedGsoBytes = [long]$Matches[2]
    }
    $actualGsoBytes = [long](($gsoRows | Measure-Object -Property advertised_bytes -Sum).Sum)
    $actualArchives = if (Test-Path -LiteralPath (Join-Path $gsoBase 'models') -PathType Container) { @(Get-ChildItem -LiteralPath (Join-Path $gsoBase 'models') -File -Filter '*.zip') } else { @() }
    $actualPathSet = @{}
    foreach ($archive in $actualArchives) { $actualPathSet["models/$($archive.Name)"] = $true }
    $missingIndexedPaths = @($indexedPathSet.Keys | Where-Object { -not $actualPathSet.ContainsKey($_) })
    $extraArchivePaths = @($actualPathSet.Keys | Where-Object { -not $indexedPathSet.ContainsKey($_) })
    if ($missingIndexedPaths.Count -gt 0) { $bad.Add("missing-indexed-paths:$($missingIndexedPaths.Count)") | Out-Null }
    if ($extraArchivePaths.Count -gt 0) { $bad.Add("unindexed-archives:$($extraArchivePaths.Count)") | Out-Null }
    if ($catalogRows.Count -ne $gsoRows.Count) { $bad.Add("catalog-index-count:$($catalogRows.Count)/$($gsoRows.Count)") | Out-Null }

    if ($marker) {
        $markerOk = $marker.schema -eq 'world-model-fuel-metadata/v1' -and
            $marker.manifest_id -eq 'google_scanned_objects' -and
            $marker.version -eq $gsoManifestEntry[0].version -and
            $marker.owner -eq 'GoogleResearch' -and
            [int]$marker.archive_count -eq $expectedGsoCount -and
            [long]$marker.advertised_bytes -eq $expectedGsoBytes -and
            $marker.catalog_sha256 -eq (Get-FileHash -LiteralPath $gsoCatalog -Algorithm SHA256).Hash.ToLowerInvariant() -and
            $marker.excluded_sha256 -eq (Get-FileHash -LiteralPath $gsoExcluded -Algorithm SHA256).Hash.ToLowerInvariant() -and
            $marker.index_sha256 -eq (Get-FileHash -LiteralPath $gsoIndex -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $markerOk) { $bad.Add('metadata-marker-contract') | Out-Null }
    }

    $collectionBaseline = @(if ($acquisitionById.ContainsKey('google_scanned_objects')) { $acquisitionById['google_scanned_objects'] })
    $catalogRootRelative = $gsoCatalog.Substring($fullRoot.Length + 1).Replace('\', '/')
    $collectionBaselineOk = $collectionBaseline.Count -eq 1 -and
        $collectionBaseline[0].method -eq 'fuel_collection' -and
        $collectionBaseline[0].source_url -eq $gsoManifestEntry[0].url -and
        $collectionBaseline[0].source_version -eq $gsoManifestEntry[0].version -and
        $collectionBaseline[0].relative_path.Replace('\', '/') -eq $catalogRootRelative -and
        $collectionBaseline[0].local_sha256 -eq $hashByRelativePath[$catalogRootRelative]
    if (-not $collectionBaselineOk) { $bad.Add('collection-acquisition-baseline') | Out-Null }

    $gsoPass = $bad.Count -eq 0 -and $gsoRows.Count -eq $expectedGsoCount -and $catalogRows.Count -eq $expectedGsoCount -and $actualGsoBytes -eq $expectedGsoBytes -and $actualArchives.Count -eq $expectedGsoCount
    Add-IntegrityCheck -Id 'google_scanned_objects' -Check 'fuel-catalog-index-archive-content-address' -Pass $gsoPass -Expected "$expectedGsoCount unique catalog/index/acquisition-bound GoogleResearch CC-BY archives totaling $expectedGsoBytes advertised bytes, with exact path-set equality" -Actual $(if ($bad.Count -eq 0) { "$($gsoRows.Count) catalog/index/archives and $actualGsoBytes bytes verified" } else { "$($bad.Count) mismatches: $(@($bad | Select-Object -First 15) -join '; ')" }) -Path ($gsoManifestEntry[0].relative_path.TrimEnd('/') + '/model-index.csv')
}

$integrityPath = Join-Path $reportDir "upstream-integrity-$stamp.csv"
$integrityChecks | Export-Csv -LiteralPath $integrityPath -NoTypeInformation -Encoding UTF8
Copy-Item -LiteralPath $integrityPath -Destination (Join-Path $reportDir 'upstream-integrity-latest.csv') -Force

$partialFiles = @(Get-ChildItem -LiteralPath $fullRoot -File -Recurse -Force -Filter '*.part')
$totalBytes = [long](($files | Measure-Object -Property Length -Sum).Sum)
$summary = [ordered]@{
    root = $fullRoot
    profile = $Profile
    verification_manifest = $ManifestPath
    verification_manifest_sha256 = $manifestHash
    acquisition_context = $runContextPath
    acquisition_report = $acquisitionReportPath
    acquisition_report_sha256 = $acquisitionReportHash
    verified_at_utc = [DateTime]::UtcNow.ToString('o')
    deep_archive_validation = [bool]$DeepArchiveValidation
    file_count = $files.Count
    total_bytes = $totalBytes
    total_gib = [math]::Round($totalBytes / 1GB, 3)
    git_repository_count = $gitRows.Count
    git_worktree_lfs_or_object_failure_count = @($gitRows | Where-Object { -not $_.worktree_clean -or -not $_.lfs_fsck_pass -or -not $_.object_fsck_pass }).Count
    incomplete_part_file_count = $partialFiles.Count
    incomplete_part_files = @($partialFiles | ForEach-Object { $_.FullName.Substring($fullRoot.Length + 1).Replace('\', '/') })
    sha256_manifest = $hashPath
    git_manifest = $gitPath
    upstream_integrity_check_count = $integrityChecks.Count
    upstream_integrity_failure_count = @($integrityChecks | Where-Object { -not $_.pass }).Count
    upstream_integrity_manifest = $integrityPath
}
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
Copy-Item -LiteralPath $summaryPath -Destination (Join-Path $reportDir 'verification-summary-latest.json') -Force
$summary | ConvertTo-Json -Depth 10

if ($summary.incomplete_part_file_count -gt 0 -or $summary.git_worktree_lfs_or_object_failure_count -gt 0 -or $summary.upstream_integrity_failure_count -gt 0) {
    [Console]::Error.WriteLine("Verification failed. Inspect $summaryPath and the latest integrity/Git manifests.")
    exit 2
}
