# Unified Burmese Insurance Platform

This umbrella repository runs five independently owned Git repositories as one Burmese insurance document-processing platform. The React/Vite UI talks only to the FastAPI orchestrator; the orchestrator coordinates layout detection, bilingual OCR, Insurance-VLM semantic registration, human approval, and completed-document extraction.

The service source remains in Git submodules. No microservice history is copied or rewritten here, and no service has to share the umbrella repository's release cycle.

## Architecture at a glance

```text
Browser -> Nginx/React -> Orchestrator -> Visual field detection
                                  |----> OCR FastAPI
                                  |----> Insurance VLM
                                  `----> Document processing layer
                         |
                         `----> PostgreSQL + private artifact volume
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for component boundaries, contract rules, persistence, and sequence diagrams.

| Component | Host URL | Compose URL | Health | Responsibility |
|---|---|---|---|---|
| Unified web app | `http://localhost:3000` | `http://frontend` | `/healthz` | React/Vite review UI served by Nginx |
| Orchestrator | `http://localhost:8000` | `http://orchestrator:8000` | `/health`, `/ready` | Workflows, adapters, review state, audit, exports |
| Document processing | `http://localhost:8001` | `http://document-processing-layer:8000` | `/health` | Approved-template field extraction and JSON/CSV/Excel exports |
| Visual field detection | `http://localhost:8002` | `http://visual-field-detection:8000` | `/health`, `/ready` | Canonical preprocessing, page artifacts, layout and table cells |
| OCR | `http://localhost:8003` | `http://ocr-fastapi-service:8000` | `/health` | Tesseract Myanmar/English tokens |
| Insurance VLM | `http://localhost:8004` | `http://insurance-vlm:8000` | `/health/live`, `/health/ready` | Semantic template draft; always requires human review |
| PostgreSQL | not published | `postgres:5432` | `pg_isready` | Orchestrator records and visual-field metadata |

Swagger is available at `http://localhost:8000/docs` and at ports 8001 through 8004 for the individual APIs.

## Repository layout

```text
services/                         Git submodules; independently versioned
  document-processing-layer/
  visual-field-detection/
  ocr-fastapi-service/
  insurance-vlm/
  insurance-claim-ui/
orchestrator/                     Thin FastAPI gateway and durable review API
adapters/                         Explicit OCR/layout/VLM/template transformations
docker/                           Orchestration-owned Dockerfiles and Nginx config
integration-tests/                Model-free downstream service doubles
scripts/                          Shell and PowerShell smoke tests
tests/                            Umbrella adapter and API tests
compose.yaml                      Full development stack
compose.mock.yaml                 Lightweight model-free integration stack
```

## Prerequisites

- Git 2.30 or newer
- Docker Engine with Docker Compose v2
- At least 8 GB RAM for the full CPU stack; layout builds and inference may need more
- Optional NVIDIA Container Toolkit and a CUDA-capable GPU for Qwen
- Python 3.12 only when running umbrella tests directly on the host
- Node.js 18 or newer only when developing the frontend outside Docker

The mock integration stack does not download models and does not require a GPU.

## Clone with submodules

For a new clone:

```bash
git clone --recurse-submodules <umbrella-repository-url> unified-insurance-platform
cd unified-insurance-platform
git submodule status
```

If the repository was cloned without recursion:

```bash
git submodule sync --recursive
git submodule update --init --recursive
git submodule status
```

A leading `-` in `git submodule status` means that submodule has not been initialized. A leading `+` means its checked-out commit differs from the umbrella pin.

## Environment setup

Copy the safe example and replace the local-only credentials:

```bash
cp .env.example .env
```

At minimum, change `POSTGRES_PASSWORD` and `VLM_API_KEY` outside throwaway local development. Keep internal URLs as Compose service names. `localhost` inside a container points back to that same container, not another service.

Important controls include:

- `MAX_UPLOAD_MB` and `ALLOWED_UPLOAD_EXTENSIONS`
- `REQUEST_TIMEOUT_SECONDS`, `RETRY_ATTEMPTS`, and `RETRY_BACKOFF_SECONDS`
- `VLM_POLL_INTERVAL_SECONDS` and `VLM_POLL_TIMEOUT_SECONDS`
- `DOCUMENT_PROCESSING_URL`, `VISUAL_FIELD_URL`, `OCR_URL`, and `INSURANCE_VLM_URL`
- `VLM_ENGINE`, `VLM_API_KEY`, `VLM_MODEL_ID`, and `HF_TOKEN`
- `CORS_ORIGINS`

The default `VLM_ENGINE=mock` exercises validation, jobs, and review without generating semantic field mappings or downloading Qwen.

## Models and persistent storage

Large models are excluded from Git and from the root Docker build context. Runtime documents, page images, crops, job files, artifacts, exports, and secrets are also excluded.

### Layout models

The visual-field service expects these directories inside its read-only `/models` mount:

```text
/models/pp_doclayout_v3_epoch29_deployment_v1/
/models/RT-DETR-L_wired_table_cell_det/
```

Place both directories under a host folder such as `./models`, then populate the named volume:

```bash
LAYOUT_MODEL_SOURCE=./models docker compose --profile tools run --rm model-loader
```

The service health endpoint can start without inference, but extraction needs the exported PP-DocLayoutV3 package. `/ready` is the stronger model/database readiness check in the service itself.

### Document-processing TrOCR weights

Optional fine-tuned printed-text weights belong at:

```text
/app/models_weights/trocr-small-printed/
```

They are mounted through the `document_models` named volume. Seed an existing local model directory without putting it in Git:

```bash
docker volume create unified-insurance-platform_document_models
docker run --rm \
  -v unified-insurance-platform_document_models:/target \
  -v "$PWD/models/document-processing:/source:ro" \
  alpine:3.21 sh -c 'cp -a /source/. /target/'
```

Without complete weights the current service deliberately uses its fallback extractor. That is useful for pipeline development, not an accuracy test.

### Qwen model cache

The optional GPU container mounts `vlm_models` at `/models` and uses `/models/huggingface` as `HF_HOME`. To use a preloaded offline model, seed that named volume and set `VLM_MODEL_ID` to its container path. If model download is allowed, set `HF_TOKEN` as needed and let the GPU profile populate the cache. Never add the cache or token to Git.

## Start and stop

### Full development stack

After mounting the layout models:

```bash
docker compose config --quiet
docker compose up --build --detach --wait
docker compose ps
```

Open `http://localhost:3000` for the UI and `http://localhost:8000/docs` for the unified API.

View logs and stop containers without deleting data:

```bash
docker compose logs -f orchestrator
docker compose down
```

`docker compose down --volumes` permanently removes PostgreSQL and all named runtime/model volumes. Use it only when a full local reset is intended.

### Model-free mock integration stack

```bash
docker compose -f compose.mock.yaml up --build --detach --wait
./scripts/smoke-test.sh
# Windows PowerShell:
# .\scripts\smoke-test.ps1
docker compose -f compose.mock.yaml down
```

The smoke scripts exercise health, same-origin frontend proxying, blank-template registration, VLM polling, explicit approval, completed-form processing, correction, approval, and downstream export proxying.

### Frontend development

Run Vite at port 5173 while the unified backend is running:

```bash
docker compose --profile frontend-dev up frontend-dev
```

For direct development inside the submodule:

```bash
cd services/insurance-claim-ui/frontend
npm ci
cp .env.example .env
npm run dev -- --host 0.0.0.0
```

Use `VITE_API_BASE_URL=http://localhost:8000`. The production Nginx image uses a same-origin base and proxies `/api` to `orchestrator:8000` with SPA fallback to `index.html`.

The prototype FastAPI backend is intentionally absent from the default stack. It can be started only for independent legacy UI testing:

```bash
docker compose --profile frontend-standalone up frontend-prototype-backend
```

It is exposed at `http://localhost:8005`; its `backend/runtime_data/store.json` is not production storage.

### Optional Qwen GPU profile

Build and start the Qwen service with the profile:

```bash
# In .env:
# VLM_ENGINE=qwen
# INSURANCE_VLM_URL=http://insurance-vlm-gpu:8000
# VLM_PRELOAD_MODEL=true
# HF_TOKEN=...
docker compose --profile gpu up --build --detach insurance-vlm-gpu orchestrator frontend
```

The GPU service uses one worker and is published on host port 18004 for diagnostics. The default CPU/mock VLM may also be present because it is the orchestrator's baseline health dependency; with `INSURANCE_VLM_URL` set as above, workflow requests go to the GPU service. GPU/model quality validation must be performed separately on suitable hardware.

### Desktop llama.cpp VLM

For a low-VRAM demo workstation, run the Qwen3-VL 2B Q4_K_M GGUF with llama.cpp on the GPU desktop and keep the Insurance-VLM API on VM port 8004. Configure the VM's ignored `.env` with an address reachable from Docker:

```env
VLM_ENGINE=llama_cpp
VLM_LLAMA_URL=http://DESKTOP_LAN_OR_VPN_IP:8081
VLM_LLAMA_MODEL=Qwen3VL-2B-Instruct-Q4_K_M.gguf
VLM_LLAMA_TIMEOUT_SECONDS=900
VLM_MAX_NEW_TOKENS=900
VLM_IMAGE_MIN_PIXELS=65536
VLM_IMAGE_MAX_PIXELS=131072
VLM_POLL_TIMEOUT_SECONDS=1200
```

The model value must match a model ID accepted by the desktop server; the basename above is accepted by the verified launcher. Before starting a demo, verify the VM can reach `GET /health` and `GET /v1/models` on desktop port 8081, then run:

```bash
docker compose up --build --detach insurance-vlm orchestrator frontend
```

The desktop firewall should restrict TCP 8081 to the VM address. Do not publish the llama.cpp endpoint broadly.

## Template registration and approval

### Categories and form metadata

Form categories are database records, not a fixed enum. The first startup seeds Health Claim,
Life Claim, Motor Claim, and Fire Claim for compatibility, but users can create any additional
category through the Templates screen or the API:

```bash
curl -X POST http://localhost:8000/api/v1/form-categories \
  -H 'Content-Type: application/json' \
  -d '{"name":"Travel Claim","description":"Travel and emergency insurance forms"}'

curl http://localhost:8000/api/v1/form-categories
```

Category IDs are stable. Renaming a category changes its display name without rewriting every
registration/template reference. A category cannot be removed while a non-archived draft or
approved template still uses it; move or remove those forms first.

Every canonical registration upload supplies `name`, optional `description`, and
`form_type_id` alongside the file:

```bash
curl -F 'file=@data/Vehicle Damage Claim Form.pdf' \
  -F 'name=Vehicle damage claim form' \
  -F 'description=Blank two-page form used for motor damage claims' \
  -F 'form_type_id=motor' \
  http://localhost:8000/api/v1/template-registrations
```

Drafts and approved forms can be renamed, described, or moved to another category with
`PATCH /api/v1/template-registrations/{id}`. Approved templates also support
`PATCH /api/v1/templates/{id}`. Removing a form uses `DELETE` on the same resource. Removal is a
soft archive: list/get endpoints stop returning the form and it cannot process new documents,
but database history, immutable versions, audit records, source files, and prior document records
are retained.

Canonical API flow:

1. `POST /api/v1/template-registrations` (or legacy `POST /api/v1/templates/register`) uploads one blank form file. The file can be a multi-page PDF.
2. Poll `GET /api/v1/template-registrations/{id}` or `GET /api/v1/templates/jobs/{id}`.
3. Read the draft at `GET /api/v1/templates/jobs/{id}/result`.
4. Optionally save the complete editable region set with `PUT /api/v1/template-registrations/{id}/draft`; stale revisions return 409.
5. Validate with `POST /api/v1/template-registrations/{id}/validate`.
6. A human explicitly approves with `POST /api/v1/template-registrations/{id}/approve` or `POST /api/v1/templates/jobs/{id}/approve`.

The orchestrator never auto-publishes a VLM result. Approval converts supported types to `printed_text`, `handwriting`, `checkbox`, `table`, or `signature`, rejects unresolved mappings, creates an immutable version record, and registers that exact definition with the document-processing service.

The production frontend at `http://localhost:3000/#/templates` uses this canonical flow. It polls progress, shows retake instructions and failures, renders each canonical page with authoritative detector boxes, saves the complete draft with optimistic revisions, and calls server validation before approval. Page buttons switch the visible image and regions without dropping edits made on other pages. Detector regions can be disabled but cannot be deleted or duplicated. Model `review_flags` must be explicitly marked reviewed before approval.

Internally, registration uses each visual-field canonical page's exact bytes for OCR and the matching VLM job. The OCR and layout contracts share the same SHA-256, dimensions, document ID, page ID, and page number. Normalized OCR boxes become pixel `xyxy`; approved normalized `xywh` regions become page-specific document-processing pixel `xywh`.

### Register and inspect a multi-page form

Upload the PDF as one file and poll the returned registration ID:

```bash
curl -F 'file=@data/Vehicle Damage Claim Form.pdf' \
  -F 'name=Vehicle damage claim form' \
  -F 'description=Blank two-page motor claim form' \
  -F 'form_type_id=motor' \
  http://localhost:8000/api/v1/template-registrations

curl http://localhost:8000/api/v1/template-registrations/REGISTRATION_ID
curl http://localhost:8000/api/v1/template-registrations/REGISTRATION_ID/pages/1
curl http://localhost:8000/api/v1/template-registrations/REGISTRATION_ID/pages/2
```

The result has one entry per canonical page in `image_identities` and `draft.pages`. Every editable region has a positive `page` number. `downstream_ids.vlm_jobs` records the independent VLM job created for each page. Page inference is sequential so the same deployment works with a small external GPU host; the final draft is still one form and is approved once.

For image uploads, the default `preprocessing_policy=auto` treats JPEG/JPG as likely camera captures and applies document-boundary/glare/background checks. Lossless PNG, BMP, and TIFF inputs are treated as scanner or rendered-digital pages and skip those camera-only rejection checks. Use `preprocessing_policy=force` when a PNG is actually a photographed page and needs forced perspective correction; the orchestrator maps it to the visual service's internal `correction_mode=standard`.

## Completed-document workflow

1. Upload one or more filled forms with an approved umbrella `template_id` using `POST /api/v1/document-jobs` or the UI-compatible `POST /api/documents`.
2. Poll `GET /api/v1/documents/{id}` or `GET /api/v1/documents/jobs/{id}`.
3. Save corrections with `PUT /api/v1/documents/{id}/review`.
4. Approve with `POST /api/v1/documents/{id}/approve`; blocking validation errors prevent approval.
5. Optionally mark synchronized with `POST /api/v1/documents/{id}/sync`.
6. Retrieve the document-processing service's exact exports from `GET /api/v1/documents/{id}/export/{json|csv|excel}`.

Each workflow has a correlation ID, forwarded as `X-Correlation-ID` where supported. Temporary network/5xx/429 failures use bounded exponential retries; permanent 4xx payload failures are not retried. Errors use a service-specific schema without including document contents or API keys.

The current document-processing service keeps templates and jobs in memory. The orchestrator therefore re-registers the immutable approved template definition before every completed-document request, making processing safe after a downstream restart. The orchestrator's review records and audit events remain in PostgreSQL.

## Frontend compatibility API

The unified orchestrator implements the prototype paths expected by `frontend/src/api.js`; they delegate to the same canonical records and workflows rather than running mock OCR:

- `GET /api/form-types`
- `GET /api/templates` and `GET /api/templates/{id}`
- `GET|POST /api/template-registrations`
- `PATCH /api/template-registrations/{id}/fields`
- `POST /api/template-registrations/{id}/approve`
- `GET|POST /api/documents` and `GET|DELETE /api/documents/{id}`
- `PATCH /api/documents/{id}/fields`
- `POST /api/documents/{id}/status`
- `GET /api/audit-events`
- `GET /api/export/{json|csv|excel}`

## Verified interface differences

The umbrella follows service code when repository documentation is stale:

- OCR returns canonical `schema_version`, `document_id`, `model`, and `pages[].tokens[]`. The adapter consumes this page-aware response.
- OCR `bounding_box` is currently pixel `[x_min,y_min,x_max,y_max]`; the adapter also accepts unambiguous legacy normalized boxes and never scales pixel boxes twice.
- Insurance-VLM v1 strict registration accepts exactly one image page plus OCR/layout JSON and always sets `review_required`; the orchestrator fans a multi-page form out into sequential page jobs and merges them. Mock mode intentionally emits no semantic fields.
- Visual-field detection owns canonical page pixels and authoritative layout geometry. Table cells retain `parent_region_id` links to table regions.
- Document processing is synchronous at `POST /api/v1/documents/process`, renders multi-page PDFs with `pdftoppm`, applies template fields by page, has an in-memory template/job registry, and writes exports to its storage volume. Its PostgreSQL scaffold is not part of the active pipeline.
- The frontend repository's FastAPI service is a prototype with mock OCR. Production traffic goes through the umbrella orchestrator only.

## Develop an individual service

Enter the submodule, branch, test, commit, and push to that service's own remote:

```bash
cd services/visual-field-detection
git switch -c feature/my-change
# edit and run this service's documented tests
git add <files>
git commit -m "Describe the service change"
git push -u origin feature/my-change
```

After the service change is merged or the intended commit is available, update the umbrella pin in a separate umbrella commit:

```bash
cd services/visual-field-detection
git switch main
git pull --ff-only
cd ../..
git add services/visual-field-detection
git commit -m "Update visual-field-detection submodule"
```

The first commit belongs to the service repository; the second records only its new commit ID in the umbrella. Do not commit a detached local-only submodule commit that teammates cannot fetch. To update all pins deliberately:

```bash
git submodule update --remote --merge
git status
git add services/
git commit -m "Update service submodule pins"
```

Do not run that command casually: review each service commit before updating the umbrella.

## Tests and validation

Run umbrella tests on a Python host:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r orchestrator/requirements.txt pytest
python -m pytest -q tests
```

These tests cover box conversion, exact image identity, stable IDs, table parents, out-of-bounds geometry, unsupported field types, stale revisions, API compatibility, human approval, downstream restart recovery, corrections, and exports.

Useful non-destructive checks:

```bash
git submodule status
git diff --check
docker compose config --quiet
docker compose -f compose.mock.yaml config --quiet
docker build -f docker/orchestrator.Dockerfile -t unified-orchestrator:test .
docker build -f docker/frontend.Dockerfile -t unified-frontend:test .
```

Each submodule has its own dependencies and test instructions. Run those suites from inside that submodule so pytest does not accidentally collect all independent repositories with one environment.

## Troubleshooting

- **Port already allocated:** stop the process using 3000 or 8000-8004, or use a local Compose override. Do not replace internal service URLs with host ports.
- **Visual extraction says model is unavailable:** verify both layout directories in the `layout_models` volume and use the visual service's `/ready` endpoint.
- **Orchestrator is not ready:** check `docker compose logs postgres orchestrator`; the orchestrator accepts conventional `postgresql://` URLs and selects the installed Psycopg 3 driver.
- **VLM job never completes:** inspect port 8004 health, `VLM_API_KEY`, polling timeout, the `vlm_jobs` volume, and GPU memory when using Qwen.
- **Approval returns 422:** resolve every unsupported/ambiguous extraction mode, missing geometry, empty key, or duplicate field key in the draft.
- **Frontend loads but API calls fail:** production uses the Nginx `/api` proxy; Vite development needs `VITE_API_BASE_URL=http://localhost:8000` and a matching `CORS_ORIGINS` entry.
- **Submodule appears empty:** run `git submodule sync --recursive` followed by `git submodule update --init --recursive`.
- **Submodule shows local changes:** commit them in that service or preserve them; never discard another developer's work from the umbrella root.

## Privacy and security

Insurance forms, crops, OCR text, VLM inputs/results, and corrections may contain sensitive personal and financial data. Use private networks, authenticated ingress, TLS, least-privilege volume access, encryption and backups appropriate to the deployment, explicit retention/deletion policies, and secret management outside `.env` in production. Do not log raw payloads, publish model/job volumes, commit runtime artifacts, or send real documents to public notebooks or endpoints.
