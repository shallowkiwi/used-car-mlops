FROM python:3.10-slim

WORKDIR /app

<<<<<<< HEAD
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
=======
COPY . .

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn pandas numpy

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> c49679dfd936bc738dd1b4c49795b7daa9a40fd4
