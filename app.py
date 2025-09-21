from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uuid, os
from generate import generate_rule

app = FastAPI()

@app.post('/generate')
async def generate(file: UploadFile = File(...)):
content = (await file.read()).decode()
xml = generate_rule(content)
fname = f"/tmp/{uuid.uuid4()}.xml"
with open(fname, 'w') as f:
f.write(xml)
return FileResponse(fname, media_type='application/xml', filename='wazuh_rule.xml')

@app.get('/health')
def health():
return {'status':'ok'}