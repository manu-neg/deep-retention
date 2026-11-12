# Diseño de Deep Retention

Basado en la propuesta aprobada, con los cambios del dueño: entrenamiento en Colab, conjunto de datos entregado por el dueño y sin pickle.

## Datos
- Origen: `Customer-Churn-Records.csv` (entregado por el dueño en `archive (2).zip`), en `data/Customer-Churn-Records.csv`. Sustituye al conjunto de OpenML anterior, que se retiró del proyecto.
- SHA-256: `2abdc051d9170540777bf35652ccb02a6ce9c9645892b078762cb729039e44d2`. Verificar antes de leer.
- 10.000 filas. Columnas de entrada: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Complain, Satisfaction Score, Card Type, Point Earned. Objetivo: Exited (1 = fuga, tasa 20,38 %).
- Se descartan al cargar CustomerId y Surname (datos personales). RowNumber da `cliente_id` = `CLI-00001`…
- **Complain:** coincide con Exited en el 99,86 % de los clientes (2.034 de 2.038 fugados se quejaron; 10 de 7.962 retenidos). Es casi una copia de la etiqueta. El dueño decidió incluirla; el informe lo declara como limitación y compara su peso con SHAP.
- Rangos válidos: CreditScore 350–850, Age 18–92, Tenure 0–10, Balance ≥ 0, NumOfProducts 1–4, binarias 0/1, EstimatedSalary ≥ 0, Satisfaction Score 1–5, Point Earned 0–1000. Geography: France, Germany, Spain. Gender: Female, Male. Card Type: SILVER, GOLD, PLATINUM, DIAMOND.
- Partición estratificada 70/15/15, semilla 42. Validación: early stopping y calibración. Prueba: solo al final.

## Modelo (solo en Colab)
- Preprocesamiento ajustado solo con entrenamiento: StandardScaler (CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary, SatisfactionScore, PointEarned), passthrough (HasCrCard, IsActiveMember, Complain, `saldo_cero`) y OneHotEncoder (Geography, Gender, CardType). 21 columnas. Sus parámetros se guardan en JSON.
- MLP: Dense 128 ReLU, BatchNorm, Dropout 0.3, Dense 64 ReLU, Dropout 0.2, Dense 1 sigmoide. Adam 1e-3, entropía cruzada binaria, lote 256, máx. 200 épocas.
- EarlyStopping `val_loss` paciencia 15 con `restore_best_weights`; ReduceLROnPlateau ×0,5 paciencia 5.
- Pesos de clase N/(2·N_k), calculados con entrenamiento y aplicados como pesos por muestra en un `tf.data` con `reshuffle_each_iteration=True`: fijar la semilla con `shuffle=True` repetiría los mismos lotes en cada época. Semilla 42.
- Calibración de Platt: regresión logística sobre el logit, ajustada en validación. Sus coeficientes se guardan en JSON.
- Línea base: regresión logística `class_weight="balanced"`, solo para comparar.
- Métricas en prueba: ROC-AUC, PR-AUC, Brier, log-loss y ECE (10 bins), con la red cruda, la calibrada y la línea base. La línea base se compara solo en AUC, porque sus pesos de clase degradan su calibración por construcción; el diagrama de fiabilidad muestra la red cruda y la calibrada.
- SHAP: KernelExplainer sobre la predicción calibrada, fondo k-means 100, beeswarm con ~500 clientes de prueba, 3 factores principales. La importancia global suma por cliente los valores con signo de las columnas one-hot de cada variable, para no repartirla entre niveles. Limitación a declarar: KernelExplainer trata cada columna por separado y puede evaluar combinaciones imposibles. Comparar AUC por género y país.

## Margen (supuesto, editable)
`M = horizonte × (2,5 % × saldo + $60 × productos + $90 × tarjeta − $120)`, horizonte 1 año. `M ≤ 0` = deficitario.

## Catálogo (supuesto, editable)
| id | Incentivo | Elegible | Costo | Efectividad |
|---|---|---|---|---|
| vip | Gestor VIP + tarifa plana | saldo ≥ 100.000 | 400 | 0,45 |
| exo | Exoneración de comisiones (12 meses) | 25.000 ≤ saldo < 100.000 | 150 | 0,28 |
| bon | Bonificación transaccional | cualquiera | 50 | 0,20 |

## Decisión
`G_i = p·e_i·M − c_i`, con p calibrada. Se toma el mayor G_i entre los elegibles y solo se actúa si `M > 0`, `G* > 0` y `p ≥ p_min` (0,30, editable).
- Se actúa con VIP: **A · Retención prioritaria**. Con otro incentivo: **B · Retención automatizada**.
- No se actúa (gasto $0), según p: `p ≥ 0,65` **C · Salida Permitida**; `0,30 ≤ p < 0,65` **V · Vigilancia**; `p < 0,30` **D · Estable**.
- Un deficitario de riesgo bajo es D, no C: la matriz del enunciado define C solo para p ≥ 0,65 y D para «cualquier saldo». La propuesta lo trataba distinto; manda el enunciado.

Ejemplos que deben probarse (p, saldo, productos, tarjeta → M, G*, segmento):
1. 0,72, 120.000, 2, sí → 3.090, +601,16, A
2. 0,70, 60.000, 1, sí → 1.530, +164,20, B (bonificación, no exoneración)
3. 0,81, 0, 1, no → −60, sin opción, C
4. 0,78, 8.000, 1, no → 140, −28,16, C
5. 0,12, 150.000, 3, sí → 3.900, +43,60, D

## Indicadores
Sobre intervenidos salvo indicación; todo es valor esperado según los supuestos.
- Capital en riesgo crítico: Σ saldo con p ≥ 0,65 (todos los clientes).
- Capital de intervenidos: Σ saldo, el «monto de clientes rentables a intervenir» de la plantilla.
- Capital rescatable: Σ saldo·p·e, la parte de ese monto que se espera conservar. Incluye riesgo intermedio, así que no es subconjunto del capital crítico.
- Inversión: Σ c. Retorno esperado: Σ p·e·M. ROI: (retorno − inversión) / inversión; nulo si la inversión es 0.
- Efectividad proyectada (la pide el enunciado para el panel): Σ p·e / Σ p, la parte de la fuga esperada de los intervenidos que se evitaría; nula si no hay intervenidos.
- Ahorro por salida permitida: costo evitado en clientes con p ≥ 0,65 sin intervención, con el incentivo que les correspondería por saldo.
- Retener a todos (comparación del simulador): intervenir a todos los clientes con p ≥ 0,65, incluidos los deficitarios, con el incentivo que les corresponde por saldo. Inversión Σ c, retorno Σ p·e·M (un margen negativo resta), ROI y ganancia neta, junto a los de la retención selectiva.
- Se informan por separado la cartera completa y el conjunto de prueba. La cartera completa incluye clientes con los que entrenó la red, así que puede verse mejor: la app y el informe lo indican, y el ROI principal cita también el de prueba.

## Flujo con Colab
1. `python -m scripts.empaquetar_colab` crea `colab_bundle.zip` con `src/`, datos y versiones fijadas.
2. El dueño abre `notebooks/entrenar_colab.ipynb` en Colab y sube el zip. El cuaderno verifica el SHA-256 de los datos, instala, entrena, evalúa, calcula SHAP y descarga `artefactos.zip`.
3. `python -m scripts.importar_artefactos artefactos.zip` valida lista de nombres, rutas y tamaños, y copia a `models/` y `reports/figuras/`.

Artefactos: `modelo.keras`, `preprocesador.json`, `calibrador.json`, `metricas.json`, `shap_global.json`, `fondo_shap.json` (fondo k-means para SHAP en vivo) y figuras PNG.

## App Streamlit
`app/Inicio.py` y `app/pages/`: `1_Buscador.py`, `2_Cartera.py`, `3_Simulador.py`.
- Sin artefactos, la app muestra un aviso claro con los pasos a seguir; no falla ni inventa cifras.
- **Buscador:** por `cliente_id` o formulario validado de cliente nuevo. Muestra p, margen, G*, decisión e incentivo, y SHAP individual en vivo con fondo reducido.
- **Cartera:** KPI, histograma de p, clientes y dinero por segmento, tabla de intervenidos ordenada por G* y exportación CSV.
- **Simulador:** costos, efectividades, márgenes y p_min. Solo recalcula la parte financiera, sin volver a ejecutar la red. ROI frente al escenario base y comparación «retener a todos» frente a selectiva.

## Estructura
`src/` (config, data, preprocessing, model, train, evaluate, explain, finance, inference), `scripts/` (paquete e importación de Colab), `notebooks/`, `app/`, `tests/`, `models/`, `reports/figuras/`, `data/`. El informe ejecutivo se publica en Google Drive y el README enlaza su PDF.
