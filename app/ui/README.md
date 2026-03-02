# Desktop / Web UI

This directory is the home for the user-facing application.

## Options

| Option | Description |
|--------|-------------|
| **Web frontend** | React / Vue SPA that calls the FastAPI backend (`app/api`). |
| **Desktop app** | Tkinter or PyQt6 GUI for local, offline use. |

Place your chosen implementation here.  The REST API is always available via
`uvicorn app.main:app`.
