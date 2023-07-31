# Set base image in Python 3.9
FROM python:3.9-slim

# Set Working Directory
WORKDIR /app

# Copy all files from local project folder to Docker image
COPY . .

# Run command line instructions
RUN pip3 install -r requirements.txt

# Expose Port 8501 for app to be run on
EXPOSE 8501

# Command to run streamlit application
CMD streamlit run app.py