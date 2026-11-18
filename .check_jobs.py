import datetime
import json
import pathlib

files = sorted(pathlib.Path('projects').glob('*/commits/*.json'),
               key=lambda p: p.stat().st_mtime, reverse=True)[:2]
for file in files:
    ts = datetime.datetime.fromtimestamp(file.stat().st_mtime).strftime('%H:%M:%S')
    data = json.loads(file.read_bytes())
    command = data.get('command') or {}
    snap = data.get('snapshot') or {}
    jobs = snap.get('jobs') or []
    print('---', ts, file.parent.parent.name[:12], 'cmd=', command.get('type'), 'jobs=', len(jobs))
    for job in jobs[-5:]:
        print('    job', str(job.get('id'))[:8], 'kind=', job.get('kind'), 'status=', job.get('status'),
              'phase=', job.get('phase'), 'created=', str(job.get('createdAt'))[11:19],
              'updated=', str(job.get('updatedAt') or job.get('createdAt'))[11:19],
              'err=', (job.get('errorDetails') or {}).get('category'))
