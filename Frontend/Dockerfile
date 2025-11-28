FROM python:3.12-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files
COPY streamlit_ui.py .
COPY rag_methods.py .
COPY .streamlit/ .streamlit/

# Expose port 80
EXPOSE 80

# Run Streamlit on port 80
CMD ["streamlit", "run", "streamlit_ui.py", "--server.port", "80", "--server.address", "0.0.0.0"]