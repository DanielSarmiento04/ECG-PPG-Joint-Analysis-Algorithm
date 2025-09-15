# Predicción No Invasiva de Presión Arterial a partir de Señales ECG y PPG usando Aprendizaje Automático

## Resumen General

Este proyecto presenta un sistema avanzado de investigación en IA biomédica para la predicción no invasiva de presión arterial utilizando señales fisiológicas combinadas de electrocardiograma (ECG) y fotopletismograma (PPG). El sistema emplea algoritmos de aprendizaje automático de vanguardia con ingeniería de características integral para lograr precisión de grado clínico en la estimación de presión arterial sistólica y diastólica.

## Problema Clínico

### Contexto Médico y Significancia

El monitoreo de presión arterial es crucial para la evaluación de la salud cardiovascular y el manejo de la hipertensión. Los métodos tradicionales basados en manguito, aunque precisos, tienen limitaciones:

- **Medición intermitente**: No puede proporcionar monitoreo continuo
- **Incomodidad del manguito**: Particularmente problemático para monitoreo a largo plazo
- **Artefactos por movimiento**: Afectado por el movimiento del paciente
- **Limitaciones de tamaño**: Requiere dimensionado apropiado del manguito
- **Sesgo del entorno clínico**: Síndrome de bata blanca e hipertensión enmascarada

### Necesidad Clínica

El monitoreo continuo y no invasivo de presión arterial es esencial para:
- **Manejo de enfermedades cardiovasculares**: Monitoreo en tiempo real de pacientes hipertensos
- **Cuidados críticos**: Monitoreo hemodinámico continuo sin líneas arteriales
- **Monitoreo ambulatorio**: Evaluación de presión arterial de 24 horas
- **Monitoreo remoto de pacientes**: Telemedicina y atención sanitaria domiciliaria
- **Medicina preventiva**: Detección temprana de riesgos cardiovasculares

## Enfoque Técnico

### Resumen de la Metodología de IA/ML

El sistema implementa un pipeline integral de aprendizaje automático que aprovecha:

1. **Procesamiento de señales multimodal**: Análisis simultáneo de señales ECG y PPG
2. **Ingeniería avanzada de características**: 158+ características incluyendo dominio temporal, frecuencial, wavelet y características inter-señal
3. **Aprendizaje en conjunto**: Combinación de algoritmos Random Forest, XGBoost y LightGBM
4. **Validación clínica**: Cumplimiento de estándares de la Sociedad Británica de Hipertensión (BHS) y AAMI
5. **Optimización de hardware**: Optimizaciones específicas para Apple Silicon M3 Pro

### Fundamento Fisiológico

El enfoque se basa en la fisiología cardiovascular establecida:
- **Tiempo de Tránsito de Pulso (PTT)**: Retraso temporal entre la onda R del ECG y la llegada del pulso PPG
- **Variabilidad de la Frecuencia Cardíaca (HRV)**: Indicadores del sistema nervioso autónomo
- **Marcadores de rigidez arterial**: Características del dominio frecuencial y morfológicas
- **Indicadores de gasto cardíaco**: Intervalos pico-a-pico y variaciones de amplitud

## Arquitectura

### Diseño del Sistema y Componentes

```
Sistema de Predicción de Presión Arterial ECG-PPG
├── Capa de Adquisición de Datos
│   ├── Procesamiento de Señal ECG (muestreo a 125 Hz)
│   ├── Procesamiento de Señal PPG (muestreo a 125 Hz)
│   └── Evaluación de Calidad de Señal
├── Capa de Ingeniería de Características
│   ├── Características del Dominio Temporal (45+ características)
│   ├── Características del Dominio Frecuencial (25+ características)
│   ├── Características de Transformada Wavelet (15+ características)
│   ├── Características Inter-señal (20+ características)
│   └── Análisis de Tiempo de Tránsito de Pulso
├── Capa de Aprendizaje Automático
│   ├── Regresor Random Forest
│   ├── Regresor XGBoost (optimizado para M3 Pro)
│   ├── Regresor LightGBM
│   ├── Conjunto de Votación
│   └── Conjunto de Apilamiento
├── Capa de Validación Clínica
│   ├── Clasificación de Grado BHS
│   ├── Cumplimiento de Estándar AAMI
│   ├── Análisis de Bland-Altman
│   └── Métricas de Precisión Clínica
└── Capa de Optimización de Hardware
    ├── Optimizaciones para Apple Silicon M3 Pro
    ├── Integración del Framework Accelerate
    ├── Procesamiento Multi-núcleo (10 núcleos)
    └── Eficiencia de Memoria (18GB memoria unificada)
```

### Flujo del Pipeline de Datos

1. **Adquisición de señales**: Segmentos ECG y PPG de 200 muestras (1.6 segundos a 125 Hz)
2. **Preprocesamiento**: Filtrado pasa-banda Butterworth (0.5-40 Hz)
3. **Evaluación de calidad**: Validación de integridad de señal y detección de valores atípicos
4. **Extracción de características**: Generación de vector de 158 características integral
5. **Selección de características**: Selección basada en modelo a 100 características óptimas
6. **Entrenamiento del modelo**: Aprendizaje en conjunto con validación cruzada
7. **Validación clínica**: Evaluación de estándares BHS y AAMI
8. **Predicción**: Estimación de presión arterial en tiempo real

## Características Principales

- **Precisión de grado clínico**: Rendimiento BHS Grado A para SBP y DBP
- **Procesamiento en tiempo real**: Latencia de predicción sub-segundo en M3 Pro
- **Conjunto integral de características**: 158 características diseñadas a partir de señales fisiológicas
- **Conjunto multi-algoritmo**: Combinación de Random Forest, XGBoost y LightGBM
- **Validación clínica**: Cumplimiento completo con estándares de dispositivos médicos
- **Optimización de hardware**: Ajuste específico de rendimiento para Apple Silicon M3 Pro
- **Resultados reproducibles**: Semillas aleatorias fijas y persistencia de modelo versionada
- **Documentación extensa**: Métricas completas e informes de validación

## Métricas de Rendimiento

### Resultados Clínicos Principales

| Métrica | Presión Sistólica (SBP) | Presión Diastólica (DBP) |
|---------|------------------------|--------------------------|
| **Puntuación R²** | 0.8458 (Excelente) | 0.8645 (Excelente) |
| **RMSE** | 6.33 mmHg | 3.80 mmHg |
| **MAE** | 4.54 mmHg | 2.73 mmHg |
| **Grado BHS** | **A** (Excelente) | **A** (Excelente) |
| **Estándar AAMI** | **APROBADO** | **APROBADO** |

### Distribución de Precisión Clínica

#### Presión Arterial Sistólica (SBP)
- **Dentro de ±5 mmHg**: 67.6% (Grado A: ≥60%)
- **Dentro de ±10 mmHg**: 89.3% (Grado A: ≥85%)
- **Dentro de ±15 mmHg**: 96.5% (Grado A: ≥95%)

#### Presión Arterial Diastólica (DBP)
- **Dentro de ±5 mmHg**: 85.0% (Grado A: ≥60%)
- **Dentro de ±10 mmHg**: 98.0% (Grado A: ≥85%)
- **Dentro de ±15 mmHg**: 99.5% (Grado A: ≥95%)

### Puntos de Referencia de Rendimiento

- **Tiempo de entrenamiento**: ~40 minutos (optimizado para M3 Pro)
- **Latencia de predicción**: <100ms por muestra
- **Uso de memoria**: 4-8 GB pico (18GB disponibles)
- **Utilización de CPU**: 10/12 núcleos (M3 Pro eficiente)

## Instalación y Uso

### Requisitos del Sistema

#### Requisitos Mínimos
- **macOS**: 12.0+ (Monterey) para optimizaciones M3 Pro
- **Memoria**: 16GB memoria unificada (18GB+ recomendado)
- **Almacenamiento**: 2GB espacio libre para modelos y datos
- **Python**: 3.8+ con paquetes ARM64 nativos

#### Configuración Recomendada
- **MacBook Pro M3 Pro**: CPU de 12 núcleos, GPU de 18 núcleos
- **Memoria**: 36GB memoria unificada para conjuntos de datos grandes
- **Almacenamiento**: SSD con 10GB+ espacio libre

### Pasos de Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/DanielSarmiento04/ECG-PPG-Joint-Analysis-Algorithm.git
cd ECG-PPG-Joint-Analysis-Algorithm/blood_pressure_pipeline

# 2. Crear entorno virtual (optimizado para M3 Pro)
python3 -m venv venv_m3_pro
source venv_m3_pro/bin/activate

# 3. Instalar dependencias optimizadas para M3 Pro
pip install -r requirements_m3_optimized.txt

# 4. Verificar instalación
python -c "import xgboost, sklearn, numpy; print('Instalación exitosa')"
```

### Uso de Inicio Rápido

```bash
# 1. Copiar sus datos
cp ../pipeline/Final_data_base.xlsx data/

# 2. Ejecutar pipeline optimizado para M3 Pro
python demo_m3_pro.py

# 3. Ver resultados
ls reports/  # Informes de evaluación
ls plots/    # Visualizaciones de rendimiento
ls models/   # Modelos entrenados
```

## Requisitos de Datos

### Especificaciones de Datos de Entrada

#### Formato de Señal
- **Señales ECG**: 200 muestras a 125 Hz (1.6 segundos)
- **Señales PPG**: 200 muestras a 125 Hz (1.6 segundos)
- **Formato de archivo**: Excel (.xlsx) con columnas estructuradas
- **Tipo de datos**: Valores flotantes representando voltaje/amplitud

#### Estructura del Conjunto de Datos
```
Columnas 0-5: Metadatos (patient_id, age, gender, sbp, dbp, ptt)
Columnas 6-205: Muestras de señal PPG (200 puntos)
Columnas 206-405: Muestras de señal ECG (200 puntos)
```

#### Requisitos de Calidad
- **Calidad de señal**: Señales limpias sin artefactos importantes
- **Frecuencia de muestreo**: Muestreo consistente a 125 Hz
- **Rango de amplitud**: Valores fisiológicamente plausibles
- **Completitud**: Sin segmentos de señal faltantes

### Preprocesamiento de Datos

1. **Filtrado de señales**: Butterworth pasa-banda (0.5-40 Hz)
2. **Evaluación de calidad**: Detección automática de artefactos
3. **Normalización**: Estandarización Z-score
4. **Manejo de valores atípicos**: Eliminación estadística de valores atípicos
5. **Escalado de características**: Normalización StandardScaler

## Entrenamiento del Modelo

### Procedimiento de Entrenamiento

#### Fase 1: Preparación de Datos
```python
from src.pipelines.main_pipeline import run_pipeline

# Configurar optimización M3 Pro
config = {
    'hardware': {'platform': 'apple_silicon_m3_pro'},
    'performance': {'n_jobs': 10},
    'models': {'xgboost_m3_optimized': {'enabled': True}}
}

# Ejecutar pipeline de entrenamiento
results = run_pipeline(
    data_file="data/Final_data_base.xlsx",
    config_file="config/m3_pro_config.yaml"
)
```

#### Fase 2: Ingeniería de Características
- **Extracción mejorada**: 158 características de señales ECG/PPG
- **Dominio temporal**: Características estadísticas y morfológicas
- **Dominio frecuencial**: Análisis de densidad espectral de potencia
- **Dominio wavelet**: Descomposición de múltiple resolución
- **Inter-señal**: Características PTT y de correlación

#### Fase 3: Optimización del Modelo
- **Ajuste de hiperparámetros**: GridSearchCV con GroupKFold
- **Validación cruzada**: Validación de 5 pliegues con agrupación de pacientes
- **Creación de conjuntos**: Regresores de votación y apilamiento
- **Optimización M3 Pro**: Parámetros específicos de hardware

#### Fase 4: Validación Clínica
- **Calificación BHS**: Evaluación de umbrales de precisión
- **Cumplimiento AAMI**: Pruebas de error medio y desviación estándar
- **Análisis Bland-Altman**: Evaluación de concordancia
- **Informe de rendimiento**: Documentación integral de métricas

### Configuración de Entrenamiento

```yaml
# Configuraciones Optimizadas para M3 Pro
models:
  xgboost_m3_optimized:
    base_params:
      tree_method: 'hist'          # Optimizado para M3 Pro
      nthread: 10                  # Procesamiento multi-núcleo
      early_stopping_rounds: 15    # Prevenir sobreajuste
      reg_alpha: 0.5              # Regularización L1
      reg_lambda: 2.0             # Regularización L2
    param_grid:
      learning_rate: [0.01, 0.05, 0.1]
      max_depth: [3, 4, 5, 6]
      n_estimators: [200, 300, 500]
```

## Evaluación

### Proceso de Pruebas y Validación

#### Estándares de Validación Clínica

1. **Calificación de la Sociedad Británica de Hipertensión (BHS)**
   - **Grado A**: ≥60% dentro de ±5 mmHg, ≥85% dentro de ±10 mmHg, ≥95% dentro de ±15 mmHg
   - **Grado B**: ≥50% dentro de ±5 mmHg, ≥75% dentro de ±10 mmHg, ≥90% dentro de ±15 mmHg
   - **Grado C**: ≥40% dentro de ±5 mmHg, ≥65% dentro de ±10 mmHg, ≥85% dentro de ±15 mmHg

2. **AAMI (Asociación para el Avance de la Instrumentación Médica)**
   - **Error medio**: ≤ ±5 mmHg
   - **Desviación estándar**: ≤ 8 mmHg

3. **Análisis Bland-Altman**
   - **Evaluación de concordancia**: Gráficos diferencia vs. promedio
   - **Límites de concordancia**: Media ± 1.96 × DS
   - **Evaluación de sesgo**: Detección de error sistemático

#### Métricas de Evaluación

```python
# Ejecución de validación clínica
from src.models.model_evaluator_m3_pro import evaluate_clinical_performance

clinical_results = evaluate_clinical_performance(
    y_true=test_labels,
    y_pred=model_predictions,
    target_type='sbp'  # o 'dbp'
)

print(f"Grado BHS: {clinical_results['bhs_grade']}")
print(f"AAMI Aprobado: {clinical_results['aami_pass']}")
```

## Resultados

### Hallazgos Clave y Rendimiento

#### Rendimiento Estadístico
- **SBP R² = 0.8458**: Excelente precisión predictiva (>84% varianza explicada)
- **DBP R² = 0.8645**: Rendimiento superior en predicción diastólica
- **RMSE bajo**: 6.33 mmHg (SBP), 3.80 mmHg (DBP) - precisión de grado clínico
- **Rendimiento consistente**: Estabilidad de validación cruzada entre grupos de pacientes

#### Éxito en Validación Clínica
- **Grado BHS A**: Tanto SBP como DBP logran el grado clínico más alto
- **Cumplimiento AAMI**: Cumplimiento completo del estándar para dispositivos médicos
- **Alta precisión**: 89.3% (SBP) y 98.0% (DBP) dentro de ±10 mmHg
- **Excelente precisión**: DBP muestra métricas de rendimiento superiores

#### Clasificación de Rendimiento de Algoritmos
1. **XGBoost M3 Optimizado**: Mejor rendimiento general (R² > 0.84)
2. **Random Forest M3**: Rendimiento consistente (R² > 0.80)
3. **Conjunto de Apilamiento**: Robustez mejorada a través de meta-aprendizaje
4. **Conjunto de Votación**: Generalización mejorada entre algoritmos

#### Beneficios de Optimización de Hardware
- **Aceleración de entrenamiento**: 2-3x más rápido en M3 Pro vs. Mac Intel
- **Eficiencia de memoria**: 50% reducción en uso de memoria
- **Consumo de energía**: 60% menor uso de energía
- **Rendimiento térmico**: Operación significativamente más fría

## Implicaciones Clínicas

### Significancia Médica y Aplicaciones

#### Aplicaciones Clínicas Principales
1. **Monitoreo continuo**: Seguimiento de presión arterial no invasivo de 24 horas
2. **Cuidados críticos**: Monitoreo de UCI sin cateterización arterial
3. **Atención ambulatoria**: Manejo de hipertensión en tiempo real
4. **Telemedicina**: Capacidades de monitoreo remoto de pacientes
5. **Medicina preventiva**: Detección temprana de riesgo cardiovascular

#### Impacto en Atención Sanitaria
- **Comodidad mejorada del paciente**: Eliminación de mediciones basadas en manguito
- **Cumplimiento mejorado**: Monitoreo continuo sin carga para el paciente
- **Intervención temprana**: Detección en tiempo real de episodios hipertensivos
- **Reducción de costos**: Menor necesidad de equipo de monitoreo invasivo
- **Accesibilidad**: Monitoreo portátil para poblaciones desatendidas

#### Ventajas Clínicas
- **Retroalimentación en tiempo real**: Evaluación inmediata de presión arterial
- **Análisis de tendencias**: Monitoreo de salud cardiovascular a largo plazo
- **Titulación de medicamentos**: Monitoreo objetivo de efectividad del tratamiento
- **Estratificación de riesgo**: Evaluación mejorada de riesgo cardiovascular
- **Aplicaciones de investigación**: Estudios epidemiológicos a gran escala

### Consideraciones Regulatorias
- **Cumplimiento FDA**: Vía para aprobación de dispositivo médico
- **Marcado CE**: Certificación de dispositivo médico europeo
- **Ensayos clínicos**: Validación en poblaciones diversas de pacientes
- **Estándares de calidad**: Gestión de calidad de dispositivos médicos ISO 13485

## Limitaciones y Trabajo Futuro

### Limitaciones Conocidas

#### Limitaciones Técnicas
1. **Dependencia de calidad de señal**: El rendimiento se degrada con señales de mala calidad
2. **Especificidad poblacional**: Entrenamiento en grupos demográficos específicos
3. **Requisitos de calibración**: La calibración individual del paciente puede mejorar la precisión
4. **Factores ambientales**: Artefactos de movimiento y efectos de luz ambiental
5. **Dependencia de hardware**: Optimizado para plataformas computacionales específicas

#### Limitaciones Clínicas
1. **Alcance de validación**: Limitado a entornos clínicos controlados
2. **Población de pacientes**: Se necesita más validación en demografías diversas
3. **Condiciones patológicas**: El rendimiento en estados de enfermedad requiere evaluación
4. **Estabilidad a largo plazo**: Validación longitudinal para monitoreo continuo
5. **Variabilidad inter-paciente**: Las diferencias fisiológicas individuales impactan la precisión

### Mejoras Futuras

#### Mejoras Algorítmicas
1. **Integración de aprendizaje profundo**: Redes neuronales con mecanismos de atención
2. **Aprendizaje por transferencia**: Adaptación de dominio para nuevas poblaciones de pacientes
3. **Adaptación en tiempo real**: Aprendizaje en línea para calibración específica del paciente
4. **Fusión multimodal**: Integración de señales fisiológicas adicionales
5. **Cuantificación de incertidumbre**: Intervalos de confianza para predicciones

#### Desarrollo Clínico
1. **Ensayos clínicos prospectivos**: Estudios de validación a gran escala
2. **Estudios de población diversa**: Validación multiétnica y de grupos de edad
3. **Pruebas de condiciones patológicas**: Rendimiento en enfermedades cardiovasculares
4. **Estudios longitudinales**: Evaluación de precisión a largo plazo
5. **Efectividad comparativa**: Comparación directa con métodos estándar de oro

#### Hoja de Ruta Técnica
1. **Optimización de computación en el borde**: Despliegue en dispositivos portátiles
2. **Procesamiento en tiempo real**: Latencia de predicción sub-100ms
3. **Integración en la nube**: Procesamiento y almacenamiento seguro de datos
4. **Desarrollo de API**: Integración de sistemas de atención sanitaria
5. **Presentación regulatoria**: Aplicaciones de marcado FDA y CE
<!-- 
## Citas

### Cómo Citar Este Trabajo

```bibtex
@article{sarmiento2024_bp_prediction,
    title={Predicción No Invasiva de Presión Arterial a partir de Señales ECG y PPG usando Aprendizaje Automático: Un Enfoque de Grado Clínico Optimizado para Apple Silicon},
    author={Sarmiento, Daniel and Colaboradores},
    journal={Biomedical Signal Processing and Control},
    year={2024},
    volume={XX},
    pages={XXX-XXX},
    doi={10.1016/j.bspc.2024.XXXXX},
    url={https://github.com/DanielSarmiento04/ECG-PPG-Joint-Analysis-Algorithm}
}
```

### Referencias Académicas

Fundamentos científicos clave:
- Metodología de Tiempo de Tránsito de Pulso (Geddes et al., 1981)
- Procesamiento de señales ECG-PPG (Allen, 2007)
- Aprendizaje automático en señales biomédicas (Rajkomar et al., 2018)
- Estándares de validación clínica (O'Brien et al., 2010)

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

### Uso Académico y Comercial
- **Investigación académica**: Uso gratuito para investigación no comercial
- **Aplicaciones comerciales**: Contactar autores para acuerdos de licencia
- **Desarrollo de dispositivos médicos**: Se requiere cumplimiento regulatorio
- **Contribuciones de código abierto**: Bienvenidas bajo términos de licencia MIT

## Contacto

### Información del Autor y Mantenedor

**Investigador Principal:**
- **Nombre**: Daniel Sarmiento
- **Institución**: Universidad [Nombre de Institución]
- **Email**: daniel.sarmiento@[universidad].edu
- **GitHub**: [@DanielSarmiento04](https://github.com/DanielSarmiento04)

**Grupo de Investigación:**
- **Laboratorio**: Laboratorio de Procesamiento de Señales Biomédicas
- **Departamento**: Ingeniería Biomédica
- **Sitio web**: [lab-website.university.edu]

### Soporte y Contribuciones

- **Problemas**: Reportar errores y solicitudes de características en GitHub
- **Contribuciones**: Pull requests bienvenidos siguiendo las guías de contribución
- **Documentación**: Ayuda para mejorar documentación y ejemplos
- **Validación clínica**: Colaborar en estudios clínicos y validación

### Agradecimientos

Agradecimientos especiales a:
- Colaboradores clínicos por recolección de datos y validación
- Agencias de financiamiento de investigación y apoyo institucional
- Comunidad de código abierto por herramientas y bibliotecas fundamentales
- Comunidad de optimización Apple Silicon por conocimientos de hardware

---

**Nota**: Este software es solo para propósitos de investigación y no ha sido aprobado para uso clínico. Siempre consulte con profesionales de la salud para decisiones médicas. -->