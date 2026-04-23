FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Create a non-root user for security (Hugging Face recommendation)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the application code
COPY --chown=user . .

# Start the application using Gunicorn. 
# We increase the timeout to 300s to account for the model training in app.py's init.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "app:app"]