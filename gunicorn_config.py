import multiprocessing
import os

# Gunicorn configuration for production
bind = "127.0.0.1:5002"  # Only listen locally, if Cloudflare tunnel will proxy
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
worker_class = "sync"
timeout = 120
keepalive = 5

# Access logging
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Restart workers after this many requests
max_requests = 1000
max_requests_jitter = 50

# Process naming
proc_name = "2004tracker"

# SSL is handled by Cloudflare
forwarded_allow_ips = "127.0.0.1"  # Only trust local forwarded headers