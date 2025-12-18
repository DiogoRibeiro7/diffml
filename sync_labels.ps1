param(
    [string]$LabelFile = "labels.txt"
)

Write-Host "Using label file: $LabelFile"

if (-not (Test-Path $LabelFile)) {
    Write-Error "Label file not found: $LabelFile"
    exit 1
}

# Check that gh is available
$ghVersion = & gh --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "GitHub CLI (gh) is not installed or not in PATH."
    exit 1
}

Write-Host "Syncing labels with the current GitHub repository..."
Write-Host ""

# Make sure repo context is valid (cwd or GH_REPO env)
$repoView = & gh repo view 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Cannot determine GitHub repository. Run this script inside the repo directory or set GH_REPO."
    exit 1
}

Get-Content $LabelFile | ForEach-Object {
    $line = $_.Trim()

    # Skip empty lines and comments
    if ([string]::IsNullOrWhiteSpace($line)) { return }
    if ($line.StartsWith("#")) { return }

    # Split into name|color|description
    $parts = $line.Split("|", 3)
    if ($parts.Count -lt 3) {
        Write-Warning "Skipping malformed line (expected name|color|description): $line"
        return
    }

    $name = $parts[0].Trim()
    $color = $parts[1].Trim()
    $description = $parts[2].Trim()

    Write-Host "Syncing label: '$name'"

    # Try create
    & gh label create "$name" --color "$color" --description "$description" 2>$null
    if ($LASTEXITCODE -ne 0) {
        # If create fails (likely exists), try edit
        & gh label edit "$name" --color "$color" --description "$description"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "  Failed to create or update label '$name'."
        } else {
            Write-Host "  updated"
        }
    } else {
        Write-Host "  created"
    }
}

Write-Host ""
Write-Host "Done syncing labels."
exit 0
