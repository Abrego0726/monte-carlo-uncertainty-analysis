import numpy as np
import matplotlib.pyplot as plt

# ============================================
# CASO PRÁCTICO 4: ESTABILIDAD Y NÚMERO DE CONDICIÓN
# f(T) = (e^T - 1) / T
# ============================================

# Valores nominales
T_nom = 0.05
sigma_T = 0.005       # Incertidumbre de T (1 sigma)
N = 200000
rng = np.random.default_rng(42)

# ============================================
# Funciones
# ============================================
def f(T):
    """Calcula f(T) evitando el problema cuando T está cerca de cero."""
    return np.where(np.abs(T) < 1e-10, 1.0, np.expm1(T) / T)


def derivada_f(T):
    """Derivada analítica de f(T)."""
    T = np.asarray(T, dtype=float)
    resultado = np.empty_like(T)
    cerca_de_cero = np.abs(T) < 1e-10
    resultado[cerca_de_cero] = 0.5
    resultado[~cerca_de_cero] = (
        T[~cerca_de_cero] * np.exp(T[~cerca_de_cero])
        - np.expm1(T[~cerca_de_cero])
    ) / T[~cerca_de_cero]**2
    return resultado.item() if resultado.ndim == 0 else resultado


def numero_condicion(T):
    """Número de condición relativo: kappa = |T*f'(T)/f(T)|."""
    return np.abs(T * derivada_f(T) / f(T))

# ============================================
# MÉTODO ANALÍTICO (Taylor de 1er orden)
# ============================================
f_nom = float(f(T_nom))
df_nom = float(derivada_f(T_nom))
kappa_nom = float(numero_condicion(T_nom))

# Incertidumbre absoluta de f por Taylor
sigma_f_taylor = np.abs(df_nom) * sigma_T
error_rel_taylor = sigma_f_taylor / np.abs(f_nom)

print("=" * 60)
print("MÉTODO ANALÍTICO (Taylor de 1er orden)")
print("f(T) = (e^T - 1) / T")
print("=" * 60)
print(f"f(T_nom) = {f_nom:.10f}")
print(f"f'(T_nom) = {df_nom:.10f}")
print(f"Número de condición = {kappa_nom:.10f}")
print(f"Incertidumbre analítica = {sigma_f_taylor:.10f}")
print(f"Error relativo = {error_rel_taylor:.10f} ({error_rel_taylor * 100:.4f} %)")
print()

# ============================================
# MÉTODO MONTE CARLO
# ============================================
T_samples = rng.normal(T_nom, sigma_T, N)
f_samples = f(T_samples)

f_media_MC = np.mean(f_samples)
sigma_f_MC = np.std(f_samples, ddof=1)
error_rel_MC = sigma_f_MC / np.abs(f_media_MC)

print("=" * 60)
print(f"MÉTODO MONTE CARLO (N = {N:,} muestras)")
print("=" * 60)
print(f"Media Monte Carlo = {f_media_MC:.10f}")
print(f"Desviación estándar Monte Carlo = {sigma_f_MC:.10f}")
print(f"Error relativo Monte Carlo = {error_rel_MC:.10f} ({error_rel_MC * 100:.4f} %)")
print()

# ============================================
# COMPARACIÓN
# ============================================
diferencia_media = f_media_MC - f_nom
diferencia_sigma = sigma_f_MC - sigma_f_taylor

print("=" * 60)
print("COMPARACIÓN")
print("=" * 60)
print(f"f(T_nom) [Taylor] = {f_nom:.10f}")
print(f"Media Monte Carlo = {f_media_MC:.10f}")
print(f"Diferencia (media) = {diferencia_media:.10e} ({diferencia_media / f_nom * 100:.4f} %)")
print()
print(f"Sigma Taylor = {sigma_f_taylor:.10f}")
print(f"Sigma Monte Carlo = {sigma_f_MC:.10f}")
print(f"Diferencia (sigma) = {diferencia_sigma:.10e} ({diferencia_sigma / sigma_f_taylor * 100:.4f} %)")
print("=" * 60)
print()

# ============================================
# SENSIBILIDAD PARA DIFERENTES VALORES DE T
# ============================================
T_values = np.linspace(0.001, 1.0, 300)
kappa_values = numero_condicion(T_values)

print("=" * 60)
print("SENSIBILIDAD PARA DIFERENTES VALORES DE T")
print("=" * 60)
print(f"{'T':>10} {'f(T)':>15} {'f\'(T)':>15} {'kappa(T)':>15}")
print("-" * 60)

for T in [0.001, 0.01, 0.05, 0.10, 0.50, 1.00]:
    print(f"{T:10.3f} {float(f(T)):15.8f} {float(derivada_f(T)):15.8f} {float(numero_condicion(T)):15.8f}")

print("=" * 60)
print()

# ============================================
# GRÁFICAS
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Histograma de f(T)
ax1.hist(f_samples, bins=60, density=True, alpha=0.7,
         color="steelblue", edgecolor="black", label="Histograma Monte Carlo")
ax1.axvline(f_nom, color="green", linestyle="--", linewidth=2,
            label=f"f(T_nom) = {f_nom:.5f}")
ax1.axvline(f_media_MC, color="red", linestyle="-", linewidth=2,
            label=f"Media MC = {f_media_MC:.5f}")
ax1.set_title("Distribución de f(T): Monte Carlo vs Taylor")
ax1.set_xlabel("f(T)")
ax1.set_ylabel("Densidad")
ax1.grid(alpha=0.3)
ax1.legend()

# Número de condición para distintos T
ax2.plot(T_values, kappa_values, color="purple", linewidth=2,
         label=r"$\kappa(T)=|T f'(T)/f(T)|$")
ax2.axvline(T_nom, color="red", linestyle="--", linewidth=2,
            label=f"T nominal = {T_nom}")
ax2.scatter(T_nom, kappa_nom, color="black", zorder=5,
            label=f"kappa nominal = {kappa_nom:.4f}")
ax2.set_title("Variación del número de condición")
ax2.set_xlabel("T")
ax2.set_ylabel("Número de condición kappa(T)")
ax2.grid(alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("caso_4_estabilidad.png", dpi=150)
plt.show()

print("Gráfica guardada como 'caso_4_estabilidad.png'")
