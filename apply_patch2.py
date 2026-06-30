from pathlib import Path
p = Path('frontend/streamlit_app.py')
s = p.read_text(encoding='utf-8')
old_block = '''def http_get_with_retry(url: str, retries: int = 3, delay: float = 1.0, timeout: int = 5):
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
new_block = '''def http_get_with_retry(url: str, retries: int = 3, delay: float = 1.0, timeout: int = 5):
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


def http_post_with_retry(url: str, json=None, files=None, retries: int = 3, delay: float = 2.0, timeout: int = 60):
    """POST with retries on exceptions or 5xx responses."""
    import time
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=json, files=files, timeout=timeout)
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
if old_block in s:
    s = s.replace(old_block, new_block)
    s = s.replace('health = requests.get(f"{API_PREFIX}/api/v1/health", timeout=5)', 'health = http_get_with_retry(f"{API_PREFIX}/api/v1/health", retries=2, timeout=5)')
    s = s.replace('response = requests.post(f"{API_PREFIX}/api/v1/upload", files=files, timeout=600)', "response = http_post_with_retry(f\"{API_PREFIX}/api/v1/upload\", files=files, timeout=600)")
    s = s.replace("response = requests.post(\n                f\"{API_PREFIX}/api/v1/ask\", json=payload, timeout=300\n            )", "response = http_post_with_retry(f\"{API_PREFIX}/api/v1/ask\", json=payload, retries=3, timeout=300)")
    p.write_text(s, encoding='utf-8')
    print('patched')
else:
    print('pattern-not-found')
