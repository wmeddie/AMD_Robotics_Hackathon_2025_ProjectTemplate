# Static Assets Placeholder

This directory is reserved for local image assets (e.g., rock or karesansui logo files) that you do not want to commit to the repository. The files `rock_base64.txt` and `karesansui_base64.txt` are the default text-only sources for the rock image and karesansui logo. Populate them with base64-encoded PNG content to ensure the preview uses your preferred assets without storing binaries in git. You may also override the rock payload with the `ROCK_IMAGE_BASE64` environment variable if you prefer not to edit the file.
