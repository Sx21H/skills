# Ollama Dashboard

A lightweight local web UI for managing and chatting with [Ollama](https://ollama.com).

## Features

- Connection status and Ollama version
- List local models (size, family, quantization, modified time)
- Pull models with live progress
- Delete models
- Inspect model details (`/api/show`)
- View models currently loaded in memory (`/api/ps`)
- Streaming chat with temperature and max-token controls
- Optional system prompt
- Local reverse proxy so the browser is not blocked by CORS

## Prerequisites

1. [Install Ollama](https://ollama.com/download) and start the daemon:

   ```bash
   ollama serve
   ```

2. Python 3.10+ (standard library only — no pip packages required)

## Run

From this directory:

```bash
python3 server.py
```

Then open [http://127.0.0.1:8080](http://127.0.0.1:8080).

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8080` | Bind port |
| `--ollama` | `http://127.0.0.1:11434` or `$OLLAMA_HOST` | Upstream Ollama base URL |

Examples:

```bash
# Custom UI port
python3 server.py --port 3000

# Point at a remote/LAN Ollama host
python3 server.py --ollama http://192.168.1.50:11434

# Listen on all interfaces (use only on trusted networks)
python3 server.py --host 0.0.0.0 --port 8080
```

## Health check

```bash
curl -s http://127.0.0.1:8080/healthz
```

## Layout

```text
ollama-dashboard/
├── README.md
├── server.py          # Static file server + /api and /v1 proxy
└── static/
    └── index.html     # Single-page dashboard
```

## Notes

- The UI calls same-origin `/api/*` paths. `server.py` proxies those to your Ollama instance.
- Chat uses Ollama's streaming `/api/chat` endpoint (NDJSON).
- Do not expose `--host 0.0.0.0` to untrusted networks without additional access controls.
