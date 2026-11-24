import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==================== 1. DATOS DE EJEMPLO ====================
# Área del terreno en metros cuadrados (m²)
areas = np.array([80, 120, 150, 180, 200, 250, 300, 350, 400, 450, 500]).reshape(-1, 1)

# Costo del terreno (en soles, dólares, etc.) - datos simulados con tendencia lineal + ruido
# Fórmula aproximada: costo ≈ 800 * área + 20000 + ruido
np.random.seed(42)
costos = 800 * areas.flatten() + 20000 + np.random.normal(0, 15000, size=len(areas))
costos = np.round(costos, 2)  # Redondear a 2 decimales

# Crear DataFrame para mejor visualización
df = pd.DataFrame({
    'Area_m2': areas.flatten(),
    'Costo': costos
})
print("=== DATOS DE ENTRENAMIENTO ===")
print(df)
print()

# ==================== 2. DIVIDIR DATOS (opcional, para evaluación) ====================
X_train, X_test, y_train, y_test = train_test_split(areas, costos, test_size=0.2, random_state=42)

# ==================== 3. CREAR Y ENTRENAR EL MODELO ====================
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Parámetros del modelo
pendiente = modelo.coef_[0]
intercepcion = modelo.intercept_

print("=== MODELO ENTRENADO ===")
print(f"Ecuación: Costo = {pendiente:.2f} * Área + {intercepcion:.2f}")
print(f"Pendiente (costo por m²): {pendiente:.2f}")
print(f"Intercepción: {intercepcion:.2f}")
print()

# ==================== 4. PREDICCIONES ====================
# Predecir en los datos de prueba
y_pred = modelo.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== EVALUACIÓN DEL MODELO ===")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")
print()

# ==================== 5. PREDICCIÓN PARA NUEVOS TERRENOS ====================
print("=== PREDICCIONES PARA NUEVOS TERRENOS ===")
nuevas_areas = [100, 220, 375, 600]
for area in nuevas_areas:
    prediccion = modelo.predict([[area]])[0]
    print(f"Terreno de {area} m² → Costo estimado: {prediccion:,.2f}")

# ==================== 6. VISUALIZACIÓN ====================
plt.figure(figsize=(10, 6))

# Datos reales
plt.scatter(areas, costos, color='blue', label='Datos reales', s=80)

# Línea de regresión
areas_linea = np.linspace(areas.min(), areas.max(), 100).reshape(-1, 1)
costos_linea = modelo.predict(areas_linea)
plt.plot(areas_linea, costos_linea, color='red', linewidth=2, label='Regresión Lineal')

# Predicciones de prueba
plt.scatter(X_test, y_pred, color='green', marker='x', s=100, label='Predicciones (test)')

plt.title('Regresión Lineal: Costo del Terreno vs Área', fontsize=14, fontweight='bold')
plt.xlabel('Área del terreno (m²)', fontsize=12)
plt.ylabel('Costo', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Guardar el gráfico
plt.savefig('grafico_terreno.png', dpi=150)
print("\nGráfico guardado como: grafico_terreno.png")
plt.show()

print("\n¡Modelo listo! Puedes cambiar los datos de 'areas' y 'costos' por datos reales.")