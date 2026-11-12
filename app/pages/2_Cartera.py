"""Panel de cartera con los supuestos base."""
import streamlit as st

import comun
import graficos
import tablas
from src import finance

comun.configurar("Panel de cartera")
clientes, p = comun.elegir_conjunto(*comun.requerir_cartera())
decisiones = finance.decidir_cartera(clientes, p)

comun.mostrar_kpis(finance.resumir_kpis(decisiones))
st.caption(
    "Capital de intervenidos: saldo total de los clientes a intervenir. "
    "Capital rescatable esperado: la parte de ese saldo que se espera conservar (Σ saldo·p·e)."
)
st.caption(comun.NOTA_SUPUESTOS)
st.plotly_chart(graficos.histograma_p(p))

st.subheader("Clientes y dinero por segmento")
segmentos = tablas.tabla_segmentos(decisiones)
izquierda, derecha = st.columns(2)
izquierda.plotly_chart(graficos.barras_segmento(segmentos, "clientes", "Clientes por segmento", "Clientes"))
derecha.plotly_chart(graficos.barras_segmento(segmentos, "saldo", "Saldo por segmento", "Dólares"))
tablas.mostrar_tabla(segmentos)

st.subheader("Clientes a intervenir, de mayor a menor G*")
intervenidos = tablas.tabla_intervenidos(clientes, decisiones)
if intervenidos.empty:
    st.info("Con estos supuestos no se recomienda intervenir a ningún cliente.")
else:
    tablas.mostrar_tabla(intervenidos)
    st.download_button(
        "Exportar a CSV",
        intervenidos.to_csv(index=False).encode("utf-8-sig"),
        file_name="clientes_a_intervenir.csv",
        mime="text/csv",
        key="exportar_csv",
    )
