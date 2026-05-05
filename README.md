# Titanic ML Pipeline — MLOps con GitHub Actions

Pipeline de Machine Learning automatizado con CI/CD para predecir la supervivencia en el Titanic.

## Requisitos
- Python 3.10+
- Git

## Instalación y ejecución local

# Clonar repositorio
git clone https://github.com/dianamps2486/titanic-ml-pipeline.git
cd titanic-ml-pipeline

# Crear entorno virtual
python -m venv venv
source venv/Scripts/activate

# Instalar dependencias
make install

# Ejecutar pipeline completo
make train

# Ver resultados en MLflow UI
mlflow ui
# Abrir en el navegador: http://localhost:5000

# Ejecutar pruebas
make test

## Estructura del proyecto

titanic-ml-pipeline/
├── src/
│   └── train.py
├── data/
│   └── titanic.csv
├── tests/
│   └── test_train.py
├── .github/
│   └── workflows/
│       └── ml.yml
├── config.yaml
├── Makefile
├── requirements.txt
└── README.md

## CI/CD
Cada push a main ejecuta automáticamente: instalación, lint, tests, entrenamiento y guardado del modelo.# titanic-ml-pipeline
Guia Final
