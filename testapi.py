import requests

url = "http://localhost:8000/detect"

with open("test.jpg", "rb") as f:
    r = requests.post(url, files={"file": f})

with open("result.jpg", "wb") as out:
    out.write(r.content)
