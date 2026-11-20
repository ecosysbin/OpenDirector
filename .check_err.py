import json
import pathlib

files = sorted(pathlib.Path('projects').glob('*/commits/*.json'),
               key=lambda p: p.stat().st_mtime, reverse=True)[:1]
data = json.loads(files[0].read_bytes())
for job in (data.get('snapshot') or {}).get('jobs') or []:
    if str(job.get('id')).startswith('844e5faa'):
        print(json.dumps(job.get('errorDetails'), ensure_ascii=False, indent=2))
        print('--- execution ---')
        print(json.dumps(job.get('execution'), ensure_ascii=False, indent=2)[:800])
        print('--- remote ids ---')
        print('externalTaskId=', job.get('externalTaskId'))
