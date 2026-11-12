# Deep Retention

Red neuronal que estima la probabilidad de fuga de clientes bancarios y app que recomienda retener a un cliente solo si la ganancia neta esperada es positiva.

**Informe:** [informe ejecutivo (PDF)](https://drive.google.com/file/d/1mwi6zsandPU638z_GQz3wF3qWPacmpXc/view?usp=sharing) · **Diseño:** [docs/DESIGN.md](docs/DESIGN.md) · **Uso de IA:** [AI_USAGE.md](AI_USAGE.md)

Margen, costos y efectividades son supuestos editables, no datos reales.

## Instalación
Requiere Python 3.12. Descarga cerca de 600 MB (TensorFlow). Todos los comandos se ejecutan desde la raíz del repositorio.

```bash
git clone https://github.com/manu-neg/deep-retention.git
cd deep-retention
python3.12 -m venv .venv
.venv/bin/pip install --require-hashes -r requirements.txt
```

## App
```bash
.venv/bin/streamlit run app/Inicio.py
```
El repositorio incluye el modelo entrenado, así que no hace falta entrenar para usarla. Se abre en `http://localhost:8501`.

## Entrenamiento (Google Colab, opcional)
Colab debe usar Python 3.12 o 3.13.

1. Crear el paquete: `.venv/bin/python -m scripts.empaquetar_colab`
2. Abrir `notebooks/entrenar_colab.ipynb` en Colab, subir `colab_bundle.zip` y ejecutar todas las celdas. El navegador descarga `artefactos.zip`; moverlo a la raíz del repositorio.
3. Importar el resultado: `.venv/bin/python -m scripts.importar_artefactos artefactos.zip`

## Pruebas
```bash
.venv/bin/python -m pytest tests -q
```

## Datos
`data/Customer-Churn-Records.csv`: 10.000 clientes de un banco europeo, 14 variables de entrada y la etiqueta `Exited`. El SHA-256 se verifica antes de interpretar el archivo. Los identificadores personales (CustomerId, Surname) se descartan al cargar.
