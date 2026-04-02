import json
import httpx
from pathlib import Path

RAG = 'http://127.0.0.1:8010'
VISION = 'http://127.0.0.1:8020'
img_path = '/workspace/test/img/do-anh-tuan2.png'

results = {}

# 1) Health checks
for name, url in [
    ('rag_health', f'{RAG}/health'),
    ('vision_docs', f'{VISION}/docs'),
]:
    try:
        r = httpx.get(url, timeout=15.0)
        results[name] = {'status': r.status_code, 'ok': r.status_code < 400}
    except Exception as e:
        results[name] = {'status': None, 'ok': False, 'error': repr(e)}

# 2) Vision identify
try:
    with open(img_path, 'rb') as f:
        r = httpx.post(
            f'{VISION}/vision/identify',
            data={'session_id': 'smoke_sys_vision_001', 'source': 'camera'},
            files={'image': ('do-anh-tuan2.png', f, 'image/png')},
            timeout=90.0,
        )
    payload = r.json() if r.headers.get('content-type', '').startswith('application/json') else {'raw': r.text[:500]}
    results['vision_identify'] = {
        'status': r.status_code,
        'ok': r.status_code < 400,
        'customer_id': payload.get('customer_id') if isinstance(payload, dict) else None,
        'is_existing_customer': payload.get('is_existing_customer') if isinstance(payload, dict) else None,
        'gender_estimate': payload.get('gender_estimate') if isinstance(payload, dict) else None,
        'age_group_estimate': payload.get('age_group_estimate') if isinstance(payload, dict) else None,
    }
except Exception as e:
    results['vision_identify'] = {'status': None, 'ok': False, 'error': repr(e)}

# 3) RAG query stream
try:
    with httpx.stream(
        'POST',
        f'{RAG}/query/stream',
        json={'query': 'Xin chao, toi muon tim du an phu hop de o', 'session_id': 'smoke_sys_rag_001'},
        timeout=120.0,
    ) as r:
        lines = []
        for line in r.iter_lines():
            if line:
                lines.append(line)
            if len(lines) >= 5:
                break
        results['rag_query_stream'] = {
            'status': r.status_code,
            'ok': r.status_code < 400,
            'sample_lines': lines,
        }
except Exception as e:
    results['rag_query_stream'] = {'status': None, 'ok': False, 'error': repr(e)}

print(json.dumps(results, ensure_ascii=False, indent=2))
