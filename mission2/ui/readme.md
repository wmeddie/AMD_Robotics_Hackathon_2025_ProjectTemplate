# Zen Garden Planner

A minimal Flask + p5.js demo for visualizing Zen garden plans with interactive rake patterns and rocks.

## Prerequisites
- Python 3.9+
- `pip` for installing dependencies

## Setup and Run
1. (Optional) Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
   ```
2. Install dependencies from `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```
3. Start the development server from the project root.
   ```bash
   python app.py
   ```
4. Open the app in your browser at `http://127.0.0.1:5000/` to use the planner UI.

## Project Structure
- `app.py` — Flask app with simple in-memory plan storage and API endpoints.
- `templates/index.html` — UI layout, buttons for adding steps, and plan save/load hooks.
- `static/sketch.js` — p5.js sketch that renders sand shading and applies rake patterns from the plan.
- `static/assets/` — optional local-only storage for image assets (e.g., rock or logo files) you do not want to commit; pair with the environment variables below or update the bundled base64 text sources.
  The committed base64 sources keep binaries out of version control while still allowing you to bundle the latest assets.

## Local images (rock and karesansui logo)
The app loads images from committed base64 text files so you can keep binaries out of the repository.

* Rock: served from the bundled base64 rock asset (`static/assets/rock_base64.txt`). You can also provide a base64 string directly via `ROCK_IMAGE_BASE64` (with or without a `data:` prefix):
  ```bash
  export ROCK_IMAGE_BASE64="<base64-string>"
  python app.py
  ```
  Another option is to drop your base64 string into `static/assets/rock_base64.txt`; the server will read that file automatically. The committed file is expected to hold the latest shared base64 rock; update it locally if you prefer a different stone.

* Karesansui logo: served from the committed base64 payload at `static/assets/karesansui_base64.txt`. Update that file locally to swap the logo without shipping binaries. You may also override the payload via `KARESANSUI_IMAGE_BASE64` (with or without a `data:image/png;base64,` prefix) to test alternate logos without editing the file. The `/karesansui-image` endpoint reads whichever payload is available and displays it beneath the preview title.

You can also drop updated base64 text files into `static/assets/` if you prefer keeping assets next to the project without committing binaries.

