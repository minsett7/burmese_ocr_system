#!/usr/bin/env sh
set -eu

BASE_URL=${BASE_URL:-http://localhost:8000}
WEB_URL=${WEB_URL:-http://localhost:3000}
SMOKE_TMP=$(mktemp -d)
trap 'rm -rf "$SMOKE_TMP"' EXIT INT TERM

printf '%s' 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Z7N0AAAAASUVORK5CYII=' | base64 -d > "$SMOKE_TMP/form.png"

curl --fail --silent --show-error "$BASE_URL/health" >/dev/null
curl --fail --silent --show-error "$WEB_URL/healthz" >/dev/null
curl --fail --silent --show-error "$WEB_URL/api/form-types" >/dev/null

registration_json=$(curl --fail --silent --show-error -F "files=@$SMOKE_TMP/form.png;type=image/png" "$BASE_URL/api/template-registrations?form_type_id=motor")
registration_id=$(printf '%s' "$registration_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["items"][0]["id"])')

attempt=0
status=analyzing
while [ "$attempt" -lt 30 ]; do
  registration=$(curl --fail --silent --show-error "$BASE_URL/api/v1/template-registrations/$registration_id")
  status=$(printf '%s' "$registration" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  [ "$status" = "needs_approval" ] && break
  [ "$status" = "failed" ] && { printf '%s\n' "$registration"; exit 1; }
  attempt=$((attempt + 1))
  sleep 1
done
[ "$status" = "needs_approval" ]

approved=$(curl --fail --silent --show-error -X POST "$BASE_URL/api/template-registrations/$registration_id/approve")
template_id=$(printf '%s' "$approved" | python3 -c 'import json,sys; print(json.load(sys.stdin)["template"]["id"])')

document_json=$(curl --fail --silent --show-error -F "files=@$SMOKE_TMP/form.png;type=image/png" "$BASE_URL/api/documents?template_id=$template_id&process_immediately=true")
document_id=$(printf '%s' "$document_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["items"][0]["id"])')

attempt=0
status=uploaded
while [ "$attempt" -lt 30 ]; do
  document=$(curl --fail --silent --show-error "$BASE_URL/api/v1/documents/$document_id")
  status=$(printf '%s' "$document" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  [ "$status" = "needs_review" ] && break
  [ "$status" = "failed" ] && { printf '%s\n' "$document"; exit 1; }
  attempt=$((attempt + 1))
  sleep 1
done
[ "$status" = "needs_review" ]

curl --fail --silent --show-error -X PUT -H 'Content-Type: application/json' -d '{"reviewer":"smoke-test","fields":[{"field_id":"field_policy","corrected_value":"MTR002","reason":"smoke correction"}]}' "$BASE_URL/api/v1/documents/$document_id/review" >/dev/null
curl --fail --silent --show-error -X POST "$BASE_URL/api/v1/documents/$document_id/approve" >/dev/null
curl --fail --silent --show-error "$BASE_URL/api/v1/documents/$document_id/export/json" -o "$SMOKE_TMP/export.json"

printf 'Mock template and completed-document workflows passed.\n'
