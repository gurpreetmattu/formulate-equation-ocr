"""Shared Flask extension instances.

Kept separate from app/__init__.py (the factory) and routes.py so both can
import the same instance without a circular import: routes.py needs
`limiter` for the @limiter.limit(...) decorator, and app/__init__.py needs
it to call init_app().
"""
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# In-memory storage matches this app's deployment model (see Dockerfile):
# a single gunicorn worker process, so there's no need for a shared backend
# like Redis -- all threads in that one process see the same counters.
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
