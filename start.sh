#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Start Gunicorn with our configuration
exec gunicorn -c gunicorn_config.py app:app