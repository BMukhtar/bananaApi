FROM python:3.10

# Set the working directory
WORKDIR /code

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y gcc libgomp1 \
    && apt-get clean

# Install Python dependencies
COPY requirements.txt /code/
RUN pip install -r requirements.txt

# Download the PyTorch model file (.pt) and save it as 'mode_full_download.pt'
ARG MODEL_URL=https://media.githubusercontent.com/media/BMukhtar/bananaApi/main/mode_full.pt
RUN curl -o mode_full_download.pt ${MODEL_URL}

# Copy the rest of your application
COPY . /code/

# Specify the command to run your application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Expose the port the app runs on
EXPOSE 8000
