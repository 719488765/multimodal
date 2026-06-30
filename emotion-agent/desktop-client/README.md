# Phase 2 Desktop Client Placeholder

This directory is reserved for the desktop client implementation (Tauri or
Electron) in phase 2.

## Scope

- Reuse backend HTTP and WebSocket APIs from `emotion-agent/backend`.
- Capture local camera/microphone on desktop app.
- Send chunks to `/api/v1/ingest/chunk` and consume stream updates from
  `/api/v1/session/{id}/stream`.

## Why Placeholder Now

The current milestone prioritizes browser-first delivery for zero-install demo.
Desktop client will be implemented without changing backend contracts.
