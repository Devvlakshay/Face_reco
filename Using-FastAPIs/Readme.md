Uvicorn Run 
```bash 
   uvicorn main:app --host 127.0.0.1 --port 8100 --reload
```

for request using curl
```bash
   curl -X POST http://localhost:8100/check-duplicate/ \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg"}'
```