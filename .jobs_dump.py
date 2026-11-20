import json, pathlib, glob

rows = []
for p in glob.glob('projects/*/commits/*.json'):
    try:
        d = json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
    except Exception:
        continue
    snap = d.get('snapshot') or {}
    for j in (snap.get('jobs') or []):
        rows.append({
            'at': str(d.get('createdAt') or ''),
            'job': j.get('id'),
            'status': j.get('status'),
            'ext': j.get('externalTaskId'),
            'stage': j.get('stage'),
            'model': j.get('modelConfigId') or j.get('model'),
        })

rows.sort(key=lambda r: r['at'], reverse=True)
for r in rows[:12]:
    print(r['at'], '|', r['job'], '|', r['status'], '|', r['stage'], '|', r['ext'])
