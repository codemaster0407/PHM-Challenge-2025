FROM python:3.13-slim
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from the project/ subfolder
COPY lib/ lib/
COPY model/ model/
COPY main.py .
COPY run.sh .

# Ensure Unix line endings and executable bit
RUN sed -i 's/\r$//' run.sh && chmod +x run.sh

CMD ["sh", "run.sh"]