FROM python:3.11-slim

WORKDIR /app

# Install PyTorch CPU first to save ~2.5GB of image size! 
# (By default it installs huge CUDA/GPU packages which you don't need on free tiers)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the standard UI port (Hugging Face / Render)
EXPOSE 7860

# Start FastAPI server on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
