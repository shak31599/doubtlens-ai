# Deploying to Hugging Face Spaces

Create a public Gradio Space at https://huggingface.co/new-space with CPU Basic. Upload the contents of the project directory to the root of the Space repository. The YAML README header selects Gradio, `app.py`, and a Gradio version. `requirements.txt` declares runtime dependencies. Keep the bundled `artifacts/` files: the Space performs inference and must not retrain on startup.

After the build is healthy, try a Dijkstra question in Classification, a CSS query in Similar questions, and open the Model performance confusion matrix. If a model file is missing, the app shows a recovery hint. Do not substitute a guessed public URL for a confirmed deployment. User account authentication is required for publishing. Verify project license and public visibility before sharing.
