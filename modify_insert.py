import re
p = 'frontend/streamlit_app.py'
s = open(p, 'r', encoding='utf-8').read()
old = 'API_PREFIX = BACKEND_URL\nbackend_status = "configured"\n'
if old in s:
    add = '''

def http_get_with_retry(url: str, retries: int = 3, delay: float = 1.0, timeout: int = 5):
    """Simple GET with retries for transient 5xx/connection errors."""
    import time
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if 500 <= resp.status_code < 600 and attempt < retries:
                time.sleep(delay * attempt)
                continue
            return resp
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay * attempt)
                continue
            raise
'''
    s = s.replace(old, old + add)
    open(p, 'w', encoding='utf-8').write(s)
    print('patched')
else:
    print('pattern-not-found')
