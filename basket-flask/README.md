# Basket

A lightweight grocery shopping-list app built with Flask, SQLite, HTML, CSS and vanilla JavaScript. Tesco-inspired colours with original Basket branding. No accounts or build tools.

## Run locally

Requires Python 3.10 or newer. In a terminal inside this folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py
```

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python`.

Open http://127.0.0.1:5000. Stop the server with Ctrl+C.

## Use

Name a list, add items and quantities (for example `2`, `500 g`, or `1 pack`), then select **Save list**. Open saved lists from the sidebar to edit them or tick off purchases. Save again to persist changes. Deleting a list requires confirmation. Leaving with unsaved changes triggers a browser warning.

The database is created automatically at `instance/shopping.sqlite` and survives restarts. Back up this file while the server is stopped. Lists are shared across everyone who can access this app; it is intended for local use, without authentication. The server binds only to your machine by default. Optional `SECRET_KEY` environment variable keeps session tokens stable across server restarts.

## Structure

```
app.py                 Flask routes, validation and SQLite storage
templates/index.html   List editor and saved-list navigation
templates/error.html   Friendly error page
static/style.css       Responsive styling
static/app.js          Item controls and unsaved-change protection
tests/test_app.py      Database and request workflow tests
instance/              Local database (created automatically, ignored)
requirements.txt       Flask dependency
```

## Test

```powershell
python -m unittest discover -s tests -v
```

Tests use temporary databases and do not modify your saved lists.
