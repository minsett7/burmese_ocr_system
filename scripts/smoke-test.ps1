param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$WebUrl = "http://localhost:3000"
)

$ErrorActionPreference = "Stop"
$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("insurance-smoke-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tempDir | Out-Null

try {
    $pngPath = Join-Path $tempDir "form.png"
    $png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z7N0AAAAASUVORK5CYII="
    [System.IO.File]::WriteAllBytes($pngPath, [Convert]::FromBase64String($png))

    Invoke-RestMethod "$BaseUrl/health" | Out-Null
    Invoke-RestMethod "$WebUrl/healthz" | Out-Null
    Invoke-RestMethod "$WebUrl/api/form-types" | Out-Null

    $registrationResponse = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/template-registrations?form_type_id=motor" -Form @{ files = Get-Item $pngPath }
    $registrationId = $registrationResponse.items[0].id
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $registration = Invoke-RestMethod "$BaseUrl/api/v1/template-registrations/$registrationId"
        if ($registration.status -eq "needs_approval") { break }
        if ($registration.status -eq "failed") { throw ($registration.failure | ConvertTo-Json -Depth 10) }
        Start-Sleep -Seconds 1
    }
    if ($registration.status -ne "needs_approval") { throw "Registration did not reach human review" }

    $approved = Invoke-RestMethod -Method Post "$BaseUrl/api/template-registrations/$registrationId/approve"
    $templateId = $approved.template.id
    $documentResponse = Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/documents?template_id=$templateId&process_immediately=true" -Form @{ files = Get-Item $pngPath }
    $documentId = $documentResponse.items[0].id
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        $document = Invoke-RestMethod "$BaseUrl/api/v1/documents/$documentId"
        if ($document.status -eq "needs_review") { break }
        if ($document.status -eq "failed") { throw ($document.failure | ConvertTo-Json -Depth 10) }
        Start-Sleep -Seconds 1
    }
    if ($document.status -ne "needs_review") { throw "Document did not reach review" }

    $review = @{ reviewer = "smoke-test"; fields = @(@{ field_id = "field_policy"; corrected_value = "MTR002"; reason = "smoke correction" }) } | ConvertTo-Json -Depth 5
    Invoke-RestMethod -Method Put -Uri "$BaseUrl/api/v1/documents/$documentId/review" -ContentType "application/json" -Body $review | Out-Null
    Invoke-RestMethod -Method Post "$BaseUrl/api/v1/documents/$documentId/approve" | Out-Null
    Invoke-WebRequest "$BaseUrl/api/v1/documents/$documentId/export/json" -OutFile (Join-Path $tempDir "export.json")
    Write-Host "Mock template and completed-document workflows passed."
}
finally {
    Remove-Item -Recurse -Force $tempDir -ErrorAction SilentlyContinue
}
