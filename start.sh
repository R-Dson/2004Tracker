#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if database exists in instance directory
if [ ! -f "instance/database.db" ]; then
    echo "Database not found. Initializing database..."
    python init_db.py
fi

# Start Gunicorn with our configuration
exec gunicorn -c gunicorn_config.py app:app