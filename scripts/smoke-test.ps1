param(
    [string]$BaseUrl = "http://localhost:8000",
    [string]$WebUrl = "http://localhost:3000"
)

$ErrorActionPreference = "Stop"
$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("insurance-smoke-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tempDir | Out-Null

# Windows PowerShell 5.1 does not provide Invoke-RestMethod -Form. Keep the
# smoke test compatible with both Windows PowerShell and PowerShell 7.
function Send-MultipartFile {
    param([string]$Uri, [string]$Path)
    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    $form = [System.Net.Http.MultipartFormDataContent]::new()
    $stream = [System.IO.File]::OpenRead($Path)
    $fileContent = [System.Net.Http.StreamContent]::new($stream)
    $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse("image/png")
    $form.Add($fileContent, "files", [System.IO.Path]::GetFileName($Path))
    try {
        $response = $client.PostAsync($Uri, $form).GetAwaiter().GetResult()
        $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        if (-not $response.IsSuccessStatusCode) { throw "HTTP $([int]$response.StatusCode): $body" }
        return $body | ConvertFrom-Json
    }
    finally {
        $fileContent.Dispose()
        $stream.Dispose()
        $form.Dispose()
        $client.Dispose()
    }
}

try {
    $pngPath = Join-Path $tempDir "form.png"
    $png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z7N0AAAAASUVORK5CYII="
    [System.IO.File]::WriteAllBytes($pngPath, [Convert]::FromBase64String($png))

    Invoke-RestMethod "$BaseUrl/health" | Out-Null
    Invoke-RestMethod "$WebUrl/healthz" | Out-Null
    Invoke-RestMethod "$WebUrl/api/form-types" | Out-Null

    $registrationResponse = Send-MultipartFile -Uri "$BaseUrl/api/template-registrations?form_type_id=motor" -Path $pngPath
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
    $documentResponse = Send-MultipartFile -Uri "$BaseUrl/api/documents?template_id=$templateId&process_immediately=true" -Path $pngPath
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
