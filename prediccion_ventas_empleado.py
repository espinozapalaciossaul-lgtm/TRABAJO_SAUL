import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==================== 1. DATOS DE EJEMPLO (Historial de ventas mensuales) ====================
# Meses (1 a 12)
meses = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)

# Ventas del empleado por mes (datos simulados con tendencia creciente)
# Ejemplo: ventas empiezan en ~8000 y aumentan ~500 por mes + algo de variación
np.random.seed(123)
ventas = 8000 + 450 * meses.flatten() + np.random.normal(0, 800, size=12)
ventas = np.round(ventas, 2)

df = pd.DataFrame({
    'Mes': meses.flatten(),
    'Ventas': ventas
})
print("=== HISTORIAL DE VENTAS DEL EMPLEADO ===")
print(df)
print()

# ==================== 2. ENTRENAR EL MODELO DE REGRESIÓN ====================
modelo = LinearRegression()
modelo.fit(meses, ventas)

pendiente = modelo.coef_[0]
intercepcion = modelo.intercept_

print("=== MODELO DE REGRESIÓN ===")
print(f"Ecuación: Ventas = {pendiente:.2f} * Mes + {intercepcion:.2f}")
print(f"Tendencia mensual (pendiente): {pendiente:.2f}")
print(f"Ventas base (intercepción): {intercepcion:.2f}")
print()

# ==================== 3. EVALUACIÓN DEL MODELO ====================
ventas_predichas = modelo.predict(meses)
mse = mean_squared_error(ventas, ventas_predichas)
r2 = r2_score(ventas, ventas_predichas)

print("=== CALIDAD DEL MODELO ===")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"R² (coeficiente de determinación): {r2:.4f}")
print(f"(R² cercano a 1 = buen ajuste)")
print()

# ==================== 4. PREDICCIONES FUTURAS ====================
print("=== PREDICCIÓN DE VENTAS FUTURAS ===")
meses_futuros = [13, 14, 15, 18, 24]
for mes in meses_futuros:
    prediccion = modelo.predict([[mes]])[0]
    print(f"Mes {mes} → Ventas estimadas: {prediccion:,.2f}")

# ==================== 5. VISUALIZACIÓN ====================
plt.figure(figsize=(10, 6))

# Datos históricos
plt.scatter(meses, ventas, color='darkblue', s=90, label='Ventas reales', zorder=5)

# Línea de regresión
meses_linea = np.linspace(1, 24, 100).reshape(-1, 1)
ventas_linea = modelo.predict(meses_linea)
plt.plot(meses_linea, ventas_linea, color='crimson', linewidth=2.5, label='Tendencia (Regresión Lineal)')

# Marcar predicciones futuras
for mes in meses_futuros[:3]:
    pred = modelo.predict([[mes]])[0]
    plt.scatter(mes, pred, color='green', s=120, marker='*', zorder=6)

plt.title('Predicción de Ventas del Empleado por Meses', fontsize=14, fontweight='bold')
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Ventas', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 25, 2))
plt.tight_layout()

# Guardar gráfico
plt.savefig('grafico_ventas.png', dpi=150)
print("\nGráfico guardado como: grafico_ventas.png")
plt.show()

print("\n¡Listo! Puedes reemplazar los datos de 'meses' y 'ventas' con el historial real del empleado.")