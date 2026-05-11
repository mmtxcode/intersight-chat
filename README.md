# Intersight Chat

A self-contained Streamlit chat app that lets you talk to **Cisco Intersight**
in natural language, powered by a local LLM (via Ollama, GPU-accelerated)
and connected to Intersight through a Node.js **MCP server**.

```
┌────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  Streamlit UI  │ ───▶ │ Ollama (GPU)     │      │  Intersight REST    │
│  + Orchestrator│ ◀─── │  /v1/chat/...    │      │  api.intersight.com │
└────────┬───────┘      └──────────────────┘      └──────────▲──────────┘
         │ stdio MCP                                          │ HTTP-Sig
         ▼                                                    │ ECDSA/RSA
┌────────────────────────┐                                    │
│  Intersight MCP server │ ───────────────────────────────────┘
│  (Node.js / TypeScript)│
└────────────────────────┘
```

All credentials stay **in memory only** — nothing is written to disk.

The whole stack runs from a single `docker compose` command. Two containers:
the Streamlit app + MCP server in one image, Ollama in another with the GPU
passed through.

---

## Prerequisites

This app is intended to run on a Linux server with an NVIDIA GPU.

- **Ubuntu 22.04 / 24.04** (or another modern Linux distro)
- **NVIDIA driver** matching your GPU (`nvidia-smi` should work)
- **Docker Engine 24+** with the Compose plugin
- **NVIDIA Container Toolkit** so Docker can hand the GPU to Ollama
- A Cisco Intersight **v3 API key** (Key ID + PEM private key) for the
  account you want to query

### Verify GPU passthrough works

Before starting the app, confirm Docker can see the GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If both succeed, you're good. If the second one fails with
`could not select device driver "nvidia"`, install the NVIDIA Container
Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Quickstart

```bash
git clone <this repo> intersight-chat
cd intersight-chat

# Bring the stack up with GPU passthrough
make up-gpu
# (equivalent: docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build)

# Pull a tool-calling-capable model into the Ollama container (one-time).
# qwen2.5:14b is the recommended default — strong tool-call discipline,
# fits comfortably on a single L40S / A100 / RTX 6000 Ada / H100.
make pull-model MODEL=qwen2.5:14b

# Open the UI
xdg-open http://<server>:8501
```

In the sidebar:

1. Pick the model you just pulled.
2. Paste your **Intersight Key ID**.
3. Upload your **PEM** private key (held in browser/server memory only).
4. Click **Test Connection** to confirm credentials work.
5. Start chatting.

---

## Recommended models

All must support tool calling. Bigger = more reliable for multi-step
questions but slower and more VRAM.

| Model | VRAM (approx) | When to pick |
|---|---|---|
| `qwen2.5:7b` | ~6 GB | Quick experiments, smaller GPUs |
| `qwen2.5:14b` | ~10 GB | **Default for most demos** — best price/perf |
| `qwen2.5:32b` | ~22 GB | When you want noticeably stronger reasoning |
| `qwen2.5:72b` | ~48 GB | Top tier, needs a beefy GPU (L40S/H100) |
| `llama3.1:70b` | ~42 GB | Alternative to qwen2.5:72b |

Avoid `llama3.1:8b` — it's notably weak at multi-step tool use and tends
to print tool calls as text instead of issuing them properly.

Pull additional models any time:

```bash
make pull-model MODEL=qwen2.5:32b
```

---

## Project layout

```
intersight-chat/
├── app.py                       # Streamlit UI (sidebar, chat, state)
├── mcp_client.py                # Sync wrapper around the async MCP Python SDK
├── orchestrator.py              # Tool-calling loop (Ollama OpenAI-compat)
├── requirements.txt
├── .env.example
├── Dockerfile                   # multi-stage: mcp-builder + python+node runtime
├── docker-compose.yml           # app + ollama (CPU default, no GPU required)
├── docker-compose.gpu.yml       # opt-in GPU override (use via `make up-gpu`)
├── Makefile                     # `make up-gpu`, `make pull-model`, ...
└── mcp-server/
    ├── package.json
    ├── tsconfig.json
    └── src/
        ├── index.ts             # MCP stdio server
        ├── tools.ts             # Tool definitions (17 read-only tools)
        ├── intersight-api.ts    # Signed HTTP client
        └── auth.ts              # HTTP-signature signing (ECDSA / RSA)
```

---

## Available MCP tools

The MCP server exposes these tools to the model:

| Tool | Endpoint |
|---|---|
| `test_connection` | `GET /api/v1/iam/Accounts?$top=1` |
| `get_server_profiles` | `GET /api/v1/server/Profiles` |
| `get_server_profile_by_name` | `GET /api/v1/server/Profiles?$filter=Name eq '…'` |
| `get_physical_servers` | `GET /api/v1/compute/PhysicalSummaries` |
| `get_chassis` | `GET /api/v1/equipment/Chasses` |
| `get_compute_blades` | `GET /api/v1/compute/Blades` |
| `get_compute_rack_units` | `GET /api/v1/compute/RackUnits` |
| `get_pci_nodes` | `GET /api/v1/pci/Nodes` |
| `get_fabric_interconnects` | `GET /api/v1/network/Elements` |
| `get_alarms` | `GET /api/v1/cond/Alarms` |
| `get_alarm_summary` | `GET /api/v1/cond/Alarms?$apply=groupby((Severity)…)` |
| `get_hcl_status` | `GET /api/v1/cond/HclStatuses` |
| `get_running_firmware` | `GET /api/v1/firmware/RunningFirmwares` |
| `get_organizations` | `GET /api/v1/organization/Organizations` |
| `get_advisories` | `GET /api/v1/tam/AdvisoryInstances` |
| `get_contracts` | `GET /api/v1/asset/DeviceContractInformations` |
| `generic_api_call` | any path under `/api/v1/` (escape hatch) |

`configure_credentials` is a 17th tool but it's hidden from the model —
the host app sets credentials directly from the sidebar.

All "list" tools accept OData params (`filter`, `select`, `top`, `skip`,
`orderby`) and return a curated default field set to keep token usage
manageable. Pass `select` explicitly to override.

---

## Authentication

The MCP server signs every Intersight request using HTTP Signatures.
Key type is auto-detected from the PEM:

- **v3 keys (EC)** → `hs2019` (ECDSA-SHA256)
- **v2 keys (RSA)** → `rsa-sha256`

Signed headers: `(request-target) host date digest content-type`.

Your Key ID and PEM are pushed to the MCP server only at chat time, held
in memory, and re-pushed on every turn so a container restart doesn't
silently desync state. Nothing is written to disk.

---

## Common operations

| Action | Command |
|---|---|
| Bring stack up (GPU) | `make up-gpu` |
| Tail logs | `make logs` |
| Stop containers | `make down` |
| Stop and wipe model volume | `make clean` |
| Rebuild app image | `make build` |
| Shell into app container | `make shell-app` |
| Shell into Ollama container | `make shell-ollama` |
| Pull a different model | `make pull-model MODEL=qwen2.5:32b` |

If you don't have `make`, every target is a thin wrapper around a
`docker compose …` command — see the `Makefile` for the equivalents.

---

## How the stack runs

The app container is built from a multi-stage Dockerfile:

1. **`mcp-builder`** — `node:20-bookworm-slim`, compiles the TypeScript
   MCP server to `dist/`.
2. **`runtime`** — `python:3.12-slim` + Node.js 20 (the Python app
   spawns the MCP server as a stdio child process). Final image is
   ~600 MB.

Models live in the `ollama_models` named Docker volume, so a
`make down` / `make up-gpu` cycle is instant after the first model pull.
`make clean` wipes the volume.

---

## Troubleshooting

**"could not select device driver 'nvidia'"** when running `make up-gpu`
→ NVIDIA Container Toolkit isn't installed or Docker hasn't been
restarted after configuring the runtime. Re-run the toolkit setup steps
above and `sudo systemctl restart docker`.

**Model dropdown is empty** → the Ollama container is up but no models
have been pulled. Run `make pull-model MODEL=qwen2.5:14b`, then refresh
the browser tab.

**Test Connection fails with 401 `iam_api_key_is_invalid`** → the API
key isn't authenticating against your Intersight account. Verify in the
Intersight UI under **Settings → API Keys** that the key is Active and
matches the Key ID you pasted; download a fresh PEM if it was rotated.

**Test Connection fails with 403 `InvalidUrl`** → this is a
request-validation error, not a permissions error. Usually means a
fabricated `$select` field. The orchestrator now surfaces the real
`code` and `message` so the model can self-correct.

**"Reached the maximum number of tool-calling rounds"** or **"called
the same tool 3 times in a row"** → small models can spin on multi-step
questions. Try a bigger model (`qwen2.5:32b` or `:72b`) or break the
question into smaller steps.

**MCP server fails to start** → make sure the build stage ran cleanly:
`make build` then `make logs` and look for `intersight-mcp-server: ready
on stdio`.

---

## Security notes

- Credentials (Intersight Key ID + PEM) are never persisted to disk —
  they live only in browser session state and the MCP server process
  memory.
- The model never sees `configure_credentials` — that tool is reserved
  for the host application.
- All Intersight tool calls are read-only (`GET`). The `generic_api_call`
  escape hatch *can* issue `POST`/`PATCH`/`DELETE` if the model decides
  to, but our system prompt steers it toward read-only operations. If
  you want to lock this down hard, add an allowlist check in
  `mcp-server/src/intersight-api.ts`.
- For **on-prem Intersight Appliance**, set `INTERSIGHT_BASE_URL` in
  `.env` (or in the compose `environment:` block) to your appliance
  hostname.
