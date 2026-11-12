# Registro de uso de IA

Cada unidad de trabajo realizada con IA: agente que la hizo y en qué consiste.
Si la unidad responde a un cambio pedido por el usuario, se añade el motivo.

## 2026-11-05

- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** escanear todos los archivos en busca de malware y problemas de seguridad; sin hallazgos maliciosos (revisión heurística, sin antivirus).
  - **Por qué:** el usuario pidió analizar todo el directorio antes de continuar.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** descargar el dataset de OpenML, verificar su checksum y borrar el escáner y el trabajo anterior.
  - **Por qué:** el usuario pidió empezar de nuevo, no entrenar aquí y usar solo fuentes fiables, sin confiar en las descargas.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** crear las reglas del proyecto, el diseño y los 6 agentes propios del proyecto.
  - **Por qué:** el usuario pidió agentes solo para este proyecto, con un orquestador, un desarrollador y un control de calidad que verifique dependencias.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** definir el informe como académico, con portada para el docente asignado y el grupo.
  - **Por qué:** el usuario pidió ese formato; el docente se asigna al equipo más adelante.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** comparar el enunciado y la rúbrica con las instrucciones, añadir la regla de avisar de diferencias y el indicador de efectividad proyectada al diseño.
  - **Por qué:** el usuario pidió que se le avise de lo que se pida y no esté en sus instrucciones.
- **Agente:** general-purpose con las instrucciones de `developer`
  - **Unidad:** crear dependencias fijadas con hash, entorno, configuración, carga de datos, motor financiero y sus pruebas, sin entrenar.
- **Agente:** general-purpose con las instrucciones de `developer`
  - **Unidad:** crear preprocesamiento, modelo, inferencia, entrenamiento, evaluación, SHAP, empaquetado e importación para Colab, cuaderno y pruebas, sin entrenar.
- **Agente:** general-purpose con las instrucciones de `developer`
  - **Unidad:** crear la app Streamlit con sus 3 vistas y pruebas, y añadir los indicadores de efectividad proyectada y «retener a todos».
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** escribir el README y desactivar la telemetría de Streamlit para que no salga nada ni se escriba fuera del proyecto.
- **Agente:** general-purpose con las instrucciones de `quality-control`
  - **Unidad:** revisar todo el proyecto, las dependencias y la calidad del código; sin vulnerabilidades conocidas.
- **Agente:** general-purpose con las instrucciones de `ml-auditor`
  - **Unidad:** auditar el pipeline y simular la ejecución en Colab; sin fuga de datos, y detectó lotes repetidos en cada época.
- **Agente:** general-purpose con las instrucciones de `finance-verifier`
  - **Unidad:** recalcular decisiones e indicadores de forma independiente; sin diferencias en los cálculos.
- **Agente:** general-purpose con las instrucciones de `repro-checker`
  - **Unidad:** instalar y ejecutar el proyecto desde una copia limpia; pruebas y app correctas.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** ajustar reglas, diseño, README y configuración según las revisiones.
- **Agente:** general-purpose con las instrucciones de `developer`
  - **Unidad:** aplicar las correcciones de ML: lotes reordenados, salida del cuaderno, SHAP agrupado, lista exacta de clases Keras y limpieza.
- **Agente:** general-purpose con las instrucciones de `developer`
  - **Unidad:** aplicar las correcciones de la app y finanzas: capital de intervenidos, división de `comun.py`, formatos y limpieza.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** corregir la etiqueta de variables sin categoría en el gráfico SHAP y añadir su prueba.
- **Agente:** general-purpose con las instrucciones de `report-writer`
  - **Unidad:** redactar el borrador del informe académico, con cifras enlazadas a los resultados y conclusiones pendientes.
  - **Por qué:** el usuario pidió un informe académico con portada para el docente y el grupo.
- **Agente:** general-purpose con las instrucciones de `ml-auditor`
  - **Unidad:** volver a revisar el entrenamiento y el cuaderno tras los cambios; sin problemas graves.
- **Agente:** general-purpose con las instrucciones de `developer`
  - **Unidad:** crear el generador del PDF del informe y su portada editable, y añadir `reportlab`.
- **Agente:** general-purpose con las instrucciones de `developer`
  - **Unidad:** añadir cifras por segmento y mejor época, y reforzar el cuaderno de Colab.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** escribir las instrucciones para completar el informe y completar los identificadores de celdas del cuaderno.
  - **Por qué:** el usuario pidió instrucciones precisas en inglés y español.
- **Agente:** `quality-control`
  - **Unidad:** revisión final de todo el proyecto y sus dependencias antes de subirlo; aprobada, sin vulnerabilidades conocidas.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** corregir las instrucciones según esa revisión, proteger el ARFF de cambios de fin de línea y subir el proyecto a una rama nueva.
  - **Por qué:** el usuario pidió subirlo cuando pasara el control de calidad.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** publicar en el panel la lista de lo que falta y quién debe hacerlo.
  - **Por qué:** el usuario pidió ver lo pendiente en un archivo del panel.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** crear la carpeta `AI/project` en Google Drive con el cuaderno y el paquete de Colab.
  - **Por qué:** el usuario pidió subir a Google Drive solo lo necesario para ejecutar en Colab.

## 2026-11-06

- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** importar `artefactos.zip`, recalcular las métricas con el modelo importado y comprobar las referencias.
  - **Por qué:** el usuario terminó la ejecución en Colab y pidió importar los resultados.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** redactar las 12 secciones pendientes del informe a partir de los resultados reales.
  - **Por qué:** el usuario pidió no dejar nada por completar salvo la portada.
- **Agente:** `developer`
  - **Unidad:** crear el generador del informe en Word y aceptar Python 3.13 en el cuaderno.
  - **Por qué:** el usuario pidió un documento editable; Colab ya usa Python 3.13.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** probar la app con el modelo real, subir el informe a Google Drive como documento y actualizar las instrucciones.
  - **Por qué:** el usuario pidió el informe en un documento en Google Drive para editar la portada.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** traducir al español las variables del informe y explicar las etiquetas en inglés de la Figura 2.
  - **Por qué:** el usuario pidió el documento en español.
- **Agente:** `quality-control`
  - **Unidad:** revisar el proyecto, dependencias, modelo importado y cifras del informe; sin vulnerabilidades conocidas y con las cifras correctas.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** matizar afirmaciones del informe e instrucciones según esa revisión, y regenerar el documento.
- **Agente:** `quality-control`
  - **Unidad:** volver a comprobar el documento y los últimos cambios; aprobado.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** subir el modelo y el informe a la rama, y borrar el entorno y los archivos temporales del proyecto.
  - **Por qué:** el usuario pidió subirlo al pasar el control de calidad, limpiar y ejecutarlo en otro equipo.
- **Agente:** Claude Code (Opus 5.5), orquestador
  - **Unidad:** cambiar el conjunto de datos a `Customer-Churn-Records.csv` (retirar el de OpenML), ampliar la red a 21 columnas de entrada, adaptar la app, las pruebas, el paquete de Colab y el texto del informe.
  - **Por qué:** el usuario cambió el conjunto de datos por requisito del curso y decidió incluir todas sus variables, también `Complain`.
