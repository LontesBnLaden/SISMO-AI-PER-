# SISMO AI | Laboratorio de Huanuco

Aplicacion educativa para explorar como un clasificador KNN relaciona
caracteristicas de un sismo con una **clase didactica de intensidad**.
Interfaz y materiales originales, inspirados en los requisitos de la guia
SENATI del archivo ZIP.

> **Limite importante:** el CSV incluido contiene datos totalmente sinteticos.
> No son registros del IGP, CENSIS ni USGS, ni observaciones de intensidad.
> El modelo no predice terremotos y no sirve como alerta, evaluacion estructural
> ni recomendacion de seguridad.

## Inicio rapido (Windows)

1. Instala Python 3.10 o posterior.
2. Abre PowerShell en esta carpeta y ejecuta:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   streamlit run app.py
   ```

3. Abre la direccion local que muestra Streamlit (normalmente
   `http://localhost:8501`).

Si PowerShell impide activar el entorno, ejecuta el interprete directamente:
`.venv\Scripts\python.exe -m pip install -r requirements.txt` y luego
`.venv\Scripts\python.exe -m streamlit run app.py`.

## Estructura

- `app.py`: dashboard, mapa, evaluacion y simulador interactivo.
- `sismo_model.py`: pipeline de escalado y KNN, particion agrupada y metricas.
- `data/sismos_sinteticos.csv`: 648 filas educativas; procedencia indicada en
  cada registro.
- `scripts/generar_datos.py`: generador determinista del dataset.
- `scripts/generar_informe.py`: crea el informe PDF con metricas y graficos.
- `tests/test_model.py`: pruebas del dataset, separacion y clasificador.
- `informe/`: informe y recursos visuales generados.
- `referencias/guia_docente/`: HTML original descomprimido desde el ZIP.

La barra superior abre cinco secciones: El reto, Datos, Prototipo, Desarrollo y
Evaluacion. **Imprimir guia** descarga el PDF. La vista de Evaluacion tambien
ejecuta tres escenarios de ejemplo contra el modelo entrenado.

## Datos y metodo

El dataset incluye 36 eventos simulados y 18 localidades peruanas representadas
con coordenadas aproximadas. Cada fila es una pareja sintetica evento-localidad.
La intensidad se agrupa en I-IV, V-VI y VII-XII exclusivamente para la practica;
las etiquetas derivan de una regla generativa artificial, no de la escala
Mercalli observada. La columna `procedencia` marca todos los registros como
`SINTETICO_EDUCATIVO`.

El 20 % de los eventos se reserva para prueba mediante particion por `evento_id`.
La busqueda de K (3, 5, 7 y 9) usa validacion cruzada `GroupKFold` sobre el
entrenamiento. Un `Pipeline` ajusta `StandardScaler` dentro de cada pliegue para
evitar fuga de informacion. Se muestran exactitud, precision, recall, F1 y
matriz de confusion. La evaluacion solo mide el comportamiento frente a este
generador simulado.

## Entregables

Este proyecto incluye codigo fuente, dataset CSV, dependencias, instrucciones,
pruebas e informe PDF con figuras. Para regenerar los datos e informe:

```powershell
python scripts\generar_datos.py
python scripts\generar_informe.py
```

Para validar: `python -m unittest discover -s tests -v`.

Las fuentes reales recomendadas por la guia son [IGP](https://www.gob.pe/igp),
[CENSIS](https://ultimosismo.igp.gob.pe/) y
[USGS](https://earthquake.usgs.gov/earthquakes/search/). Antes de sustituir el
CSV, comprueba los permisos, unidades, escalas y disponibilidad de intensidad
observada y conserva la procedencia de cada dato.
