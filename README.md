<p align="center">
  <img src="assets/banner.png" alt="Kinqhi" width="100%">
</p>

# Kinqhi (Lean)

**Personal AI agent — lean distribution.** Stripped of 50%+ of upstream. Core agent runs across CLI, TUI, Discord, API server, and Zed (ACP). 6 model providers, holographic memory (local SQLite), curated skills.

<table>
<tr><td><b>Terminal interface</b></td><td>TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, streaming tool output.</td></tr>
<tr><td><b>Multi-surface</b></td><td>Discord + CLI + TUI + API server — all from a single gateway. Cross-platform continuity.</td></tr>
<tr><td><b>Learning loop</b></td><td>Holographic memory (local SQLite). Autonomous skill creation. Skills self-improve during use. FTS5 session search.</td></tr>
<tr><td><b>Cron</b></td><td>Built-in scheduler. Natural language job scheduling with Discord delivery.</td></tr>
<tr><td><b>Delegation</b></td><td>Spawn isolated subagents for parallel workstreams. Hierarchical orchestrator/leaf roles.</td></tr>
<tr><td><b>Terminal sandboxes</b></td><td>Local, Docker, SSH. Run commands in containers or on remote machines.</td></tr>
<tr><td><b>IDE integration</b></td><td>ACP adapter for Zed. Highlight code, agent refactors with full tool access.</td></tr>
<tr><td><b>Computer use</b></td><td>macOS desktop control via cua-driver — mouse, keyboard, screenshots, background.</td></tr>
<tr><td><b>Kanban</b></td><td>Multi-agent work queue. SQLite-backed. Parallel task execution with auto-blocking on failures.</td></tr>
</table>

---

## Quick Start

```bash
# Create venv and install
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,cli,pty,mcp,acp]"

# Configure
mkdir -p ~/.kinqhi
cp .env.example ~/.kinqhi/.env
cp cli-config.yaml.example ~/.kinqhi/config.yaml

# Add your API key to ~/.kinqhi/.env
#   DEEPSEEK_API_KEY=sk-...
#   or OPENROUTER_API_KEY=sk-...

# Run
kinqhi              # interactive CLI
kinqhi --tui        # terminal UI
kinqhi gateway      # start gateway (Discord + API server on localhost:8642)
kinqhi-acp          # start ACP server for Zed integration
```

**For Discord:** add `DISCORD_BOT_TOKEN=...` to `.env`, then `kinqhi gateway`.

**For browser:** install `agent-browser` + Chromium:
```bash
npx agent-browser install --with-deps
```

**For computer use (macOS):** install `cua-driver`:
```bash
kinqhi computer-use install
```

State lives at `~/.kinqhi/` — sessions, memory (SQLite), skills, config. Profile-aware.

---

## Model Providers

| Provider | Notes |
|----------|-------|
| **DeepSeek** | Primary. DeepSeek V4. |
| **OpenRouter** | 200+ models through one API key. |
| **Anthropic** | Claude. Top-tier coding model. |
| **OpenAI Codex** | GPT + Codex for coding delegation. |
| **Custom** | Your own endpoint. Any URL + key. |
| **Ollama** | Local models. Free, offline-capable. |

Switch with `/model <provider:model>` or `kinqhi model`.

---

## CLI Reference

```
kinqhi              # Interactive CLI
kinqhi --tui        # Terminal UI
kinqhi model        # Choose provider and model
kinqhi tools        # Configure enabled tools
kinqhi gateway      # Start gateway (Discord + API)
kinqhi cron         # Manage scheduled jobs
kinqhi kanban       # Multi-agent work queue
kinqhi skills       # Browse and install skills
kinqhi doctor       # Diagnose issues
```

| Action | CLI | Messaging |
|--------|-----|-----------|
| Start | `kinqhi` | `kinqhi gateway`, then DM the bot |
| New session | `/new` or `/reset` | `/new` or `/reset` |
| Change model | `/model` | `/model` |
| Retry/undo | `/retry`, `/undo` | `/retry`, `/undo` |
| Compress context | `/compress` | `/compress` |
| Skills | `/skills` or `/<name>` | `/<name>` |
| Interrupt | `Ctrl+C` | `/stop` |

---

## License

MIT — see [LICENSE](LICENSE). Built by [Nous Research](https://nousresearch.com).
