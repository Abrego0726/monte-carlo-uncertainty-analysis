import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound

# ============================================
# CASO PRÁCTICO 4: ESTABILIDAD Y NÚMERO DE CONDICIÓN
# Coeficiente de expansión volumétrica: f(T) = (e^T - 1) / T
# ============================================

# Función principal
def f(T):
    """Coeficiente de expansión volumétrica aproximado"""
    if np.abs(T) < 1e-10:
        return 1.0  # Límite cuando T → 0
    return (np.exp(T) - 1) / T

# Función vectorizada
def f_vec(T):
    """Versión vectorizada de f"""
    result = np.zeros_like(T, dtype=float)
    mask = np.abs(T) > 1e-10
    result[mask] = (np.exp(T[mask]) - 1) / T[mask]
    result[~mask] = 1.0
    return result

# Derivada analítica de f(T)
def f_prime(T):
    """Derivada de f(T) = (e^T - 1) / T"""
    if np.abs(T) < 1e-10:
        return 0.5  # Límite de f'(T) cuando T → 0
    numerator = T * np.exp(T) - (np.exp(T) - 1)
    denominator = T**2
    return numerator / denominator

# Función vectorizada de la derivada
def f_prime_vec(T):
    """Versión vectorizada de f'"""
    result = np.zeros_like(T, dtype=float)
    mask = np.abs(T) > 1e-10
    numerator = T[mask] * np.exp(T[mask]) - (np.exp(T[mask]) - 1)
    denominator = T[mask]**2
    result[mask] = numerator / denominator
    result[~mask] = 0.5
    return result

# Número de condición: κ(T) = |T * f'(T) / f(T)|
def condition_number(T):
    """Número de condición relativo"""
    f_val = f(T)
    if np.abs(f_val) < 1e-15:
        return np.inf
    f_p = f_prime(T)
    return np.abs(T * f_p / f_val)

# Función vectorizada del número de condición
def condition_number_vec(T):
    """Versión vectorizada del número de condición"""
    f_val = f_vec(T)
    f_p = f_prime_vec(T)
    result = np.zeros_like(T, dtype=float)
    mask = np.abs(f_val) > 1e-15
    result[mask] = np.abs(T[mask] * f_p[mask] / f_val[mask])
    result[~mask] = np.inf
    return result

# ============================================
# PARTE 1: ANÁLISIS EN TORNO A UN VALOR NOMINAL
# ============================================
T_nom = 0.05  # Temperatura nominal (en unidades adimensionales)
sigma_T = 0.005  # Incertidumbre en T (1 sigma)
N = 200000
rng = np.random.default_rng(42)

print("=" * 70)
print("CASO PRÁCTICO 4: ESTABILIDAD Y NÚMERO DE CONDICIÓN")
print("f(T) = (e^T - 1) / T")
print("=" * 70)
print()

# ============================================
# MÉTODO ANALÍTICO (Taylor de 1er orden)
# ============================================
f_nom = f(T_nom)
f_prime_nom = f_prime(T_nom)
kappa_nom = condition_number(T_nom)

# Propagación lineal de incertidumbre
sigma_f_taylor = np.abs(f_prime_nom) * sigma_T

print("=" * 70)
print("ANÁLISIS EN TORNO A T_nom = {:.6f}".format(T_nom))
print("=" * 70)
print(f"f(T_nom) = {f_nom:.10f}")
print(f"f'(T_nom) = {f_prime_nom:.10f}")
print(f"Número de condición κ(T_nom) = {kappa_nom:.10f}")
print()
print("PROPAGACIÓN DE INCERTIDUMBRE (Taylor 1er orden):")
print(f"σ_T = {sigma_T:.6f}")
print(f"σ_f = |f'(T_nom)| × σ_T = {sigma_f_taylor:.10f}")
print(f"Error relativo (σ_f / f) = {(sigma_f_taylor / f_nom):.10f}")
print("=" * 70)
print()

# ============================================
# MÉTODO MONTE CARLO
# ============================================
T_samples = rng.normal(loc=T_nom, scale=sigma_T, size=N)
f_samples = f_vec(T_samples)

f_media_MC = np.mean(f_samples)
sigma_f_MC = np.std(f_samples, ddof=1)
error_rel_MC = sigma_f_MC / np.abs(f_media_MC)

print("=" * 70)
print(f"MÉTODO MONTE CARLO (N = {N:,} muestras)")
print("=" * 70)
print(f"Media f Monte Carlo = {f_media_MC:.10f}")
print(f"Desv. estándar f Monte Carlo = {sigma_f_MC:.10f}")
print(f"Error relativo (σ_f / f) MC = {error_rel_MC:.10f}")
print("=" * 70)
print()

# ============================================
# COMPARACIÓN
# ============================================
diferencia_media = f_media_MC - f_nom
error_pct_media = (diferencia_media / f_nom) * 100
diferencia_sigma = sigma_f_MC - sigma_f_taylor
error_pct_sigma = (diferencia_sigma / sigma_f_taylor) * 100

print("=" * 70)
print("COMPARACIÓN: ANALÍTICO vs MONTE CARLO")
print("=" * 70)
print(f"f(T_nom) [Analítico] = {f_nom:.10f}")
print(f"f media [MC]         = {f_media_MC:.10f}")
print(f"Diferencia           = {diferencia_media:.10e} ({error_pct_media:.4f} %)")
print()
print(f"σ_f [Taylor]         = {sigma_f_taylor:.10f}")
print(f"σ_f [MC]             = {sigma_f_MC:.10f}")
print(f"Diferencia           = {diferencia_sigma:.10e} ({error_pct_sigma:.4f} %)")
print("=" * 70)
print()

# ============================================
# PARTE 2: ANÁLISIS DE SENSIBILIDAD PARA DIFERENTES VALORES DE T
# ============================================
print("=" * 70)
print("ANÁLISIS DE SENSIBILIDAD PARA DIFERENTES VALORES DE T")
print("=" * 70)
print()

# Rango de temperaturas
T_range = np.logspace(-3, 0.5, 100)  # De 0.001 a ~3.16

# Calcular función, derivada y número de condición
f_range = f_vec(T_range)
f_prime_range = f_prime_vec(T_range)
kappa_range = condition_number_vec(T_range)

# Tabla de valores representativos
T_table = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0])
print(f"{'T':>10} {'f(T)':>15} {'f\'(T)':>15} {'κ(T)':>15} {'σ_f/f (%)':>15}")
print("-" * 70)
for t_val in T_table:
    f_val = f(t_val)
    f_p_val = f_prime(t_val)
    kappa_val = condition_number(t_val)
    rel_error = np.abs(f_p_val) * sigma_T / np.abs(f_val) * 100
    print(f"{t_val:10.3f} {f_val:15.10f} {f_p_val:15.10f} {kappa_val:15.10f} {rel_error:15.6f}")
print("=" * 70)
print()

# ============================================
# VISUALIZACIONES
# ============================================
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Histograma de f(T) para T_nom
ax1 = plt.subplot(3, 3, 1)
ax1.hist(f_samples, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='black')
x_hist = np.linspace(f_samples.min(), f_samples.max(), 500)
normal_hist = (1 / (sigma_f_taylor * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_hist - f_nom) / sigma_f_taylor) ** 2)
ax1.plot(x_hist, normal_hist, 'r--', lw=2, label='Aprox. Normal (Taylor)')
ax1.axvline(f_nom, color='green', linestyle='-', lw=2, label=f"f(T_nom) = {f_nom:.4f}")
ax1.axvline(f_media_MC, color='purple', linestyle='-', lw=2, label=f"Media MC = {f_media_MC:.4f}")
ax1.set_title(f"Histograma f(T) en T = {T_nom}", fontweight='bold')
ax1.set_xlabel("f(T)")
ax1.set_ylabel("Densidad")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

# Subplot 2: Función f(T)
ax2 = plt.subplot(3, 3, 2)
ax2.plot(T_range, f_range, 'b-', lw=2)
ax2.axvline(T_nom, color='red', linestyle='--', lw=1.5, label=f"T_nom = {T_nom}")
ax2.axhline(f_nom, color='red', linestyle='--', lw=1.5, alpha=0.5)
ax2.scatter([T_nom], [f_nom], color='red', s=100, zorder=5)
ax2.set_xlabel("T")
ax2.set_ylabel("f(T)")
ax2.set_title("Función f(T) = (e^T - 1) / T", fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Subplot 3: Derivada f'(T)
ax3 = plt.subplot(3, 3, 3)
ax3.plot(T_range, f_prime_range, 'g-', lw=2)
ax3.axvline(T_nom, color='red', linestyle='--', lw=1.5, label=f"T_nom = {T_nom}")
ax3.axhline(f_prime_nom, color='red', linestyle='--', lw=1.5, alpha=0.5)
ax3.scatter([T_nom], [f_prime_nom], color='red', s=100, zorder=5)
ax3.set_xlabel("T")
ax3.set_ylabel("f'(T)")
ax3.set_title("Derivada f'(T)", fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Subplot 4: Número de condición κ(T)
ax4 = plt.subplot(3, 3, 4)
ax4.semilogy(T_range, kappa_range, 'r-', lw=2)
ax4.axvline(T_nom, color='blue', linestyle='--', lw=1.5, label=f"T_nom = {T_nom}")
ax4.axhline(kappa_nom, color='blue', linestyle='--', lw=1.5, alpha=0.5)
ax4.scatter([T_nom], [kappa_nom], color='blue', s=100, zorder=5)
ax4.set_xlabel("T")
ax4.set_ylabel("κ(T) (log scale)")
ax4.set_title("Número de Condición κ(T) = |T f'(T) / f(T)|", fontweight='bold')
ax4.grid(True, alpha=0.3, which='both')
ax4.legend()

# Subplot 5: Error relativo vs T
ax5 = plt.subplot(3, 3, 5)
error_rel_range = np.abs(f_prime_range) * sigma_T / np.abs(f_range)
ax5.semilogy(T_range, error_rel_range * 100, 'purple', lw=2)
ax5.axvline(T_nom, color='red', linestyle='--', lw=1.5, label=f"T_nom = {T_nom}")
nom_error = (sigma_f_taylor / f_nom) * 100
ax5.axhline(nom_error, color='red', linestyle='--', lw=1.5, alpha=0.5)
ax5.scatter([T_nom], [nom_error], color='red', s=100, zorder=5)
ax5.set_xlabel("T")
ax5.set_ylabel("Error relativo (%)")
ax5.set_title("Error relativo σ_f / f vs T", fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.legend()

# Subplot 6: Estabilidad numérica en T cercano a 0
ax6 = plt.subplot(3, 3, 6)
T_small = np.logspace(-8, -1, 100)
f_small = f_vec(T_small)
ax6.loglog(T_small, f_small, 'b-', lw=2)
ax6.axvline(T_nom, color='red', linestyle='--', lw=1.5, label=f"T_nom = {T_nom}")
ax6.scatter([T_nom], [f_nom], color='red', s=100, zorder=5)
ax6.set_xlabel("T (log scale)")
ax6.set_ylabel("f(T) (log scale)")
ax6.set_title("Análisis de estabilidad cerca de T ≈ 0", fontweight='bold')
ax6.grid(True, alpha=0.3, which='both')
ax6.legend()

# Subplot 7: Comparación Taylor vs MC (barras)
ax7 = plt.subplot(3, 3, 7)
metodos = ['Taylor\n(Analítico)', 'Monte Carlo']
sigmas = [sigma_f_taylor, sigma_f_MC]
colores = ['green', 'purple']
bars = ax7.bar(metodos, sigmas, color=colores, alpha=0.7, edgecolor='black', linewidth=2)
ax7.set_ylabel("σ_f")
ax7.set_title("Comparación: Incertidumbre en f", fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(sigmas):
    ax7.text(i, v + 0.00001, f'{v:.6e}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# Subplot 8: Sensibilidad κ(T) en escala lineal
ax8 = plt.subplot(3, 3, 8)
ax8.plot(T_range, kappa_range, 'r-', lw=2)
ax8.axvline(T_nom, color='blue', linestyle='--', lw=1.5, label=f"T_nom = {T_nom}")
ax8.scatter([T_nom], [kappa_nom], color='blue', s=100, zorder=5)
ax8.fill_between(T_range, 0, kappa_range, alpha=0.3, color='red')
ax8.set_xlabel("T")
ax8.set_ylabel("κ(T)")
ax8.set_title("Número de Condición (escala lineal)", fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.legend()

# Subplot 9: Resumen de texto
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
RESUMEN DE RESULTADOS

Valor nominal T = {T_nom:.6f}
Incertidumbre σ_T = {sigma_T:.6f}

f(T_nom) = {f_nom:.10f}
f'(T_nom) = {f_prime_nom:.10f}
κ(T_nom) = {kappa_nom:.10f}

Incertidumbre analítica:
σ_f [Taylor] = {sigma_f_taylor:.6e}
Error relativo = {(sigma_f_taylor/f_nom)*100:.6f} %

Monte Carlo ({N:,} muestras):
σ_f [MC] = {sigma_f_MC:.6e}
Error relativo = {(sigma_f_MC/f_media_MC)*100:.6f} %

Diferencia (σ_f):
{diferencia_sigma:.6e} ({error_pct_sigma:.4f} %)

Interpretación:
κ << 1: Problema bien condicionado
κ ≈ 1: Acondicionamiento medio
κ >> 1: Problema mal condicionado
"""
ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig("caso_4_estabilidad.png", dpi=150, bbox_inches='tight')
plt.show()

print("Gráfico guardado como 'caso_4_estabilidad.png'")
print()

# ============================================
# PARTE 3: ANÁLISIS DE PUNTOS CRÍTICOS
# ============================================
print("=" * 70)
print("ANÁLISIS DE PUNTOS CRÍTICOS")
print("=" * 70)
print()

# Encontrar mínimo del número de condición
min_idx = np.argmin(kappa_range)
T_min_kappa = T_range[min_idx]
kappa_min = kappa_range[min_idx]

print(f"Mínimo número de condición:")
print(f"  T = {T_min_kappa:.6f}")
print(f"  κ_min = {kappa_min:.10f}")
print()

# Comportamiento cerca de T = 0
print(f"Comportamiento cerca de T = 0 (desarrollo de Taylor):")
print(f"  f(T) ≈ 1 + T/2 + T²/6 + ...")
print(f"  f'(0) = 1/2")
print(f"  κ(0) = 0")
print()

# Análisis de monotonía
T_test = np.linspace(0.001, 5, 1000)
kappa_test = condition_number_vec(T_test)
idx_increasing = np.where(np.diff(kappa_test) > 0)[0]

if len(idx_increasing) > 0:
    print(f"κ(T) es creciente para T > {T_test[idx_increasing[0]]:.6f}")
else:
    print(f"κ(T) es decreciente en todo el rango")
print()
print("=" * 70)
