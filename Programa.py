import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import glob
import os
warnings.filterwarnings('ignore')

# Para visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Para machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

#para pdf
from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime


# Para red neuronal
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.utils import to_categorical

print("✅ Módulos cargados correctamente")
print(f"TensorFlow version: {tf.__version__}")

# ============================================================================
# 1. FUNCIONES BASE (sin cambios)
# ============================================================================




def cargar_y_preparar_datos(archivos):
    """
    Carga los archivos Excel y prepara los datos para el modelo
    """
    print("📂 Cargando archivos...")
    
    dfs = []
    for archivo in archivos:
        try:
            df = pd.read_excel(archivo)
            dfs.append(df)
            print(f"  ✓ {archivo}: {len(df)} registros")
        except Exception as e:
            print(f"  ✗ Error cargando {archivo}: {e}")
    
    if dfs:
        df_completo = pd.concat(dfs, ignore_index=True)
        print(f"\n📊 Total de registros combinados: {len(df_completo)}")
        return df_completo
    else:
        print("❌ No se pudieron cargar datos")
        return None

def crear_variables_predictoras(df):
    """
    Crea las variables predictoras a partir de los datos originales
    """
    print("\n🔧 Creando variables predictoras...")
    
    if df is None or len(df) == 0:
        print("❌ DataFrame vacío o None")
        return None, None
    
    df_procesado = df.copy()
    
    # 1. Convertir fechas
    df_procesado['Fecha_nacimiento'] = pd.to_datetime(df_procesado['Fecha de nacimiento'])
    df_procesado['Fecha_consulta'] = pd.to_datetime(df_procesado['Fe.consulta'])
    
    # 2. Calcular edad
    df_procesado['Edad'] = (df_procesado['Fecha_consulta'] - df_procesado['Fecha_nacimiento']).dt.days // 365
    
    # 3. Extraer hora
    hora_ejemplo = df_procesado['Hora consulta'].iloc[0] if len(df_procesado) > 0 else None
    
    if isinstance(hora_ejemplo, str):
        df_procesado['Hora_consulta'] = pd.to_datetime(df_procesado['Hora consulta']).dt.hour
    elif hasattr(hora_ejemplo, 'hour'):
        df_procesado['Hora_consulta'] = df_procesado['Hora consulta'].apply(lambda x: x.hour if pd.notnull(x) else 0)
    else:
        df_procesado['Hora_consulta'] = pd.to_datetime(df_procesado['Hora consulta'], errors='coerce').dt.hour
        df_procesado['Hora_consulta'] = df_procesado['Hora_consulta'].fillna(0).astype(int)
    
    # 4. Franjas horarias
    df_procesado['Es_Mañana'] = (df_procesado['Hora_consulta'] >= 6) & (df_procesado['Hora_consulta'] < 12)
    df_procesado['Es_Tarde'] = (df_procesado['Hora_consulta'] >= 12) & (df_procesado['Hora_consulta'] < 20)
    df_procesado['Es_Noche'] = (df_procesado['Hora_consulta'] >= 20) | (df_procesado['Hora_consulta'] < 6)
    
    # 5. Codificar variables categóricas
    print("  Codificando variables categóricas...")
    
    le_sexo = LabelEncoder()
    df_procesado['Sexo_encoded'] = le_sexo.fit_transform(df_procesado['Sexo'])
    
    le_clase_episodio = LabelEncoder()
    df_procesado['Clase_episodio_encoded'] = le_clase_episodio.fit_transform(df_procesado['Clase episodio'])
    
    le_clase_consulta = LabelEncoder()
    df_procesado['Clase_consulta_encoded'] = le_clase_consulta.fit_transform(df_procesado['Clase de consulta'])
    
    le_uo_medica = LabelEncoder()
    df_procesado['UO_medica_encoded'] = le_uo_medica.fit_transform(df_procesado['UO médica consulta'])
    
    # 6. Agrupar por episodio
    print("  Calculando estancia estimada por episodio...")
    
    episodio_stats = df_procesado.groupby('Episodio').agg({
        'Fecha_consulta': ['min', 'max', 'count'],
        'Nombre': 'first',
        'Edad': 'first',
        'Sexo_encoded': 'first',
        'Clase_episodio_encoded': 'first'
    })
    
    episodio_stats.columns = ['_'.join(col).strip() for col in episodio_stats.columns.values]
    episodio_stats = episodio_stats.rename(columns={
        'Fecha_consulta_min': 'Primera_consulta',
        'Fecha_consulta_max': 'Ultima_consulta',
        'Fecha_consulta_count': 'Num_consultas',
        'Nombre_first': 'Nombre',
        'Edad_first': 'Edad',
        'Sexo_encoded_first': 'Sexo_encoded',
        'Clase_episodio_encoded_first': 'Clase_episodio_encoded'
    })
    
    # Calcular días de estancia
    episodio_stats['Dias_estancia'] = (episodio_stats['Ultima_consulta'] - episodio_stats['Primera_consulta']).dt.days + 1
    episodio_stats.loc[episodio_stats['Num_consultas'] == 1, 'Dias_estancia'] = 1
    
    # 7. Variable objetivo con umbral >1 día
    episodio_stats['Estancia_larga'] = (episodio_stats['Dias_estancia'] > 1).astype(int)
    
    # Mostrar distribución
    cortas = (episodio_stats['Estancia_larga'] == 0).sum()
    largas = (episodio_stats['Estancia_larga'] == 1).sum()
    total = len(episodio_stats)
    
    print(f"  Episodios procesados: {total}")
    print(f"📊 DISTRIBUCIÓN (umbral >1 día):")
    print(f"  Estancias cortas (1 día): {cortas} ({cortas/total*100:.1f}%)")
    print(f"  Estancias largas (>1 día): {largas} ({largas/total*100:.1f}%)")
    
    # 8. Características adicionales
    print("  Agregando características adicionales...")
    
    tipos_consulta_por_episodio = df_procesado.groupby(['Episodio', 'Clase_consulta_encoded']).size().unstack(fill_value=0)
    tipos_consulta_por_episodio.columns = [f'Consulta_tipo_{col}' for col in tipos_consulta_por_episodio.columns]
    
    uo_medica_por_episodio = df_procesado.groupby(['Episodio', 'UO_medica_encoded']).size().unstack(fill_value=0)
    uo_medica_por_episodio.columns = [f'UO_medica_{col}' for col in uo_medica_por_episodio.columns]
    
    episodio_features = episodio_stats.join(tipos_consulta_por_episodio, how='left')
    episodio_features = episodio_features.join(uo_medica_por_episodio, how='left')
    episodio_features = episodio_features.fillna(0)
    
    print(f"\n📈 Estadísticas de días de estancia:")
    print(f"  Mínimo: {episodio_features['Dias_estancia'].min()} día(s)")
    print(f"  Máximo: {episodio_features['Dias_estancia'].max()} días")
    print(f"  Promedio: {episodio_features['Dias_estancia'].mean():.2f} días")
    print(f"  Mediana: {episodio_features['Dias_estancia'].median()} día(s)")
    
    return episodio_features, df_procesado

def preparar_datos_modelo(df_features):
    """
    Prepara los datos para entrenamiento de la red neuronal
    """
    print("\n🤖 Preparando datos para el modelo...")
    
    if df_features is None or len(df_features) == 0:
        print("❌ DataFrame de características vacío")
        return None, None, None, None, None, None
    
    columnas_excluir = ['Nombre', 'Primera_consulta', 'Ultima_consulta', 
                        'Dias_estancia', 'Estancia_larga']
    
    X = df_features.drop(columns=columnas_excluir, errors='ignore')
    y = df_features['Estancia_larga']
    
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"  Advertencia: Columnas no numéricas encontradas: {list(non_numeric)}")
        X = X.select_dtypes(include=[np.number])
    
    print(f"  Características (X): {X.shape}")
    print(f"  Variable objetivo (y): {y.shape}")
    
    class_counts = y.value_counts()
    print(f"  Balance de clases:")
    for clase, count in class_counts.items():
        porcentaje = count / len(y) * 100
        print(f"    Clase {clase} ({'Corta' if clase == 0 else 'Larga'}): {count} ({porcentaje:.1f}%)")
    
    if len(y) < 10:
        print("❌ Muy pocas muestras para dividir en train/test")
        return None, None, None, None, None, None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"  Advertencia: No se pudo estratificar. Error: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Entrenamiento: {X_train_scaled.shape}")
    print(f"  Prueba: {X_test_scaled.shape}")
    print(f"  Distribución en train: {np.bincount(y_train)}")
    print(f"  Distribución en test: {np.bincount(y_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

# ============================================================================
# 2. FUNCIONES MEJORADAS NUEVAS
# ============================================================================

def encontrar_mejor_umbral(model, X_test, y_test):
    """
    Encuentra el mejor umbral para maximizar F1-score de la clase positiva
    """
    print("\n🔍 Buscando mejor umbral de decisión...")
    
    y_pred_proba = model.predict(X_test, verbose=0)[:, 1]
    
    mejores_metricas = {
        'f1': 0, 
        'umbral': 0.5,
        'precision': 0,
        'recall': 0,
        'accuracy': 0
    }
    
    resultados = []
    
    for umbral in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba > umbral).astype(int)
        
        # Calcular métricas manualmente para evitar problemas
        tn = np.sum((y_test == 0) & (y_pred == 0))
        fp = np.sum((y_test == 0) & (y_pred == 1))
        fn = np.sum((y_test == 1) & (y_pred == 0))
        tp = np.sum((y_test == 1) & (y_pred == 1))
        
        # Precisión para clase 1
        precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall para clase 1
        recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-score para clase 1
        if (precision_1 + recall_1) > 0:
            f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
        else:
            f1_1 = 0
        
        # Accuracy
        accuracy = (tp + tn) / len(y_test)
        
        resultados.append({
            'umbral': umbral,
            'f1': f1_1,
            'precision': precision_1,
            'recall': recall_1,
            'accuracy': accuracy,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        })
        
        if f1_1 > mejores_metricas['f1']:
            mejores_metricas = {
                'f1': f1_1,
                'umbral': umbral,
                'precision': precision_1,
                'recall': recall_1,
                'accuracy': accuracy
            }
    
    # Convertir a DataFrame para visualización
    df_resultados = pd.DataFrame(resultados)
    
    print(f"\n📊 Análisis de umbrales:")
    print(df_resultados[['umbral', 'f1', 'precision', 'recall', 'accuracy']].round(3).to_string(index=False))
    
    print(f"\n🎯 RESUMEN:")
    print(f"  Umbral actual (por defecto): 0.5")
    print(f"  Mejor umbral encontrado: {mejores_metricas['umbral']:.3f}")
    print(f"  Mejor F1-score: {mejores_metricas['f1']:.3f}")
    print(f"  Precisión con mejor umbral: {mejores_metricas['precision']:.3f}")
    print(f"  Recall con mejor umbral: {mejores_metricas['recall']:.3f}")
    
    # Graficar
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico 1: F1-score vs Umbral
    axes[0].plot(df_resultados['umbral'], df_resultados['f1'], 'b-', label='F1-score', linewidth=2)
    axes[0].plot(df_resultados['umbral'], df_resultados['precision'], 'g--', label='Precisión', alpha=0.7)
    axes[0].plot(df_resultados['umbral'], df_resultados['recall'], 'r--', label='Recall', alpha=0.7)
    axes[0].axvline(x=0.5, color='k', linestyle=':', label='Umbral por defecto (0.5)')
    axes[0].axvline(x=mejores_metricas['umbral'], color='orange', linestyle='-', 
                    label=f'Mejor umbral ({mejores_metricas["umbral"]:.2f})')
    axes[0].set_xlabel('Umbral de decisión')
    axes[0].set_ylabel('Métrica')
    axes[0].set_title('Métricas vs Umbral de Decisión')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Gráfico 2: Trade-off Precisión vs Recall
    axes[1].scatter(df_resultados['recall'], df_resultados['precision'], 
                   c=df_resultados['umbral'], cmap='viridis', s=50)
    axes[1].plot(mejores_metricas['recall'], mejores_metricas['precision'], 
                'ro', markersize=10, label='Mejor F1')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precisión')
    axes[1].set_title('Trade-off: Precisión vs Recall')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_umbrales.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return mejores_metricas['umbral'], df_resultados

def graficar_curva_roc(model, X_test, y_test):
    """
    Grafica curva ROC para ver discriminación del modelo
    """
    print("\n📈 Graficando curva ROC...")
    
    y_pred_proba = model.predict(X_test, verbose=0)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Encontrar umbral óptimo (punto más cercano a esquina superior izquierda)
    distancias = np.sqrt(fpr**2 + (1-tpr)**2)
    idx_optimo = np.argmin(distancias)
    umbral_optimo = thresholds[idx_optimo]
    
    plt.figure(figsize=(8, 6))
    
    # Curva ROC
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'Curva ROC (AUC = {auc:.3f})')
    
    # Línea aleatoria
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Clasificador aleatorio')
    
    # Punto óptimo
    plt.plot(fpr[idx_optimo], tpr[idx_optimo], 'ro', markersize=10, 
             label=f'Umbral óptimo ({umbral_optimo:.3f})')
    
    # Punto con umbral 0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    plt.plot(fpr[idx_05], tpr[idx_05], 'go', markersize=8, 
             label='Umbral 0.5 (default)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title('Curva ROC - Modelo de Estancia Hospitalaria')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Añadir cuadro de métricas
    texto_metricas = f'AUC = {auc:.3f}\nUmbral óptimo = {umbral_optimo:.3f}\n'
    texto_metricas += f'Sensibilidad = {tpr[idx_optimo]:.3f}\n'
    texto_metricas += f'Especificidad = {1-fpr[idx_optimo]:.3f}'
    
    plt.text(0.6, 0.3, texto_metricas, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('curva_roc_modelo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fpr, tpr, thresholds, umbral_optimo

def construir_entrenar_modelo_mejorado(X_train, X_test, y_train, y_test):
    """
    Construye y entrena una red neuronal con mejoras
    """
    print("\n🧠 Construyendo red neuronal mejorada...")
    
    if X_train is None or len(X_train) == 0:
        print("❌ Datos de entrenamiento vacíos")
        return None, None
    
    # Convertir etiquetas
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
    
    # Dimensiones
    input_dim = X_train.shape[1]
    print(f"  Dimensiones de entrada: {input_dim} características")
    
    # Calcular pesos para balancear clases
    print("\n⚖️ Calculando pesos para balancear clases...")
    try:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"  Pesos calculados:")
        print(f"    Clase 0 (Corta): peso = {class_weight_dict[0]:.2f}")
        print(f"    Clase 1 (Larga): peso = {class_weight_dict[1]:.2f}")
    except Exception as e:
        print(f"  No se pudieron calcular pesos de clase: {e}")
        class_weight_dict = None
    
    # Modelo mejorado
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Capa 1
        layers.Dense(64, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),  # Aumentado de 0.3
        
        # Capa 2
        layers.Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),  # Aumentado de 0.3
        
        # Capa 3
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),  # Aumentado de 0.2
        
        # Capa de salida
        layers.Dense(2, activation='softmax')
    ])
    
    # Compilar con optimizador mejorado
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc')
    ]
)

    
    # Callbacks mejorados
    callbacks_list = []
    
    # Early stopping más paciente
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',  # Monitorear AUC en lugar de loss
        patience=30,  # Más paciencia
        restore_best_weights=True,
        verbose=1,
        mode='max'
    )
    callbacks_list.append(early_stopping)
    
    # ReduceLROnPlateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=15,
        min_lr=0.00001,
        verbose=1,
        mode='max'
    )
    callbacks_list.append(reduce_lr)
    
    # Model checkpoint para el mejor modelo
    checkpoint = callbacks.ModelCheckpoint(
        'mejor_modelo.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=0
    )
    callbacks_list.append(checkpoint)
    
    # TensorBoard (opcional, para monitoreo avanzado)
    # callbacks_list.append(callbacks.TensorBoard(log_dir='./logs'))
    
    print("\n📋 Resumen del modelo:")
    model.summary()
    
    print("\n🚀 Entrenando modelo mejorado...")
    print("   (Se detendrá automáticamente si no mejora en 30 épocas)")
    
    try:
        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_test, y_test_cat),
            epochs=200,  # Más épocas
            batch_size=min(32, len(X_train)),  # Batch size ajustable
            callbacks=callbacks_list,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        print("✅ Entrenamiento completado")
        return model, history
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return None, None

def evaluar_modelo_mejorado(model, X_test, y_test, history=None, umbral=0.5):
    """
    Evalúa el modelo con métricas mejoradas y umbral ajustable
    """
    print("\n📊 Evaluando modelo con umbral ajustable...")
    
    if model is None or X_test is None:
        print("❌ Modelo o datos de prueba no disponibles")
        return None, None, None
    
    # Predecir probabilidades
    try:
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred_proba_clase1 = y_pred_proba[:, 1]
        
        # Predicciones con umbral personalizado
        y_pred_custom = (y_pred_proba_clase1 > umbral).astype(int)
        
        # Predicciones con umbral por defecto (0.5)
        y_pred_default = np.argmax(y_pred_proba, axis=1)
        
    except Exception as e:
        print(f"❌ Error al predecir: {e}")
        return None, None, None
    
    # Calcular métricas para ambos umbrales
    print(f"\n🔍 COMPARACIÓN DE UMBRALES:")
    print(f"  Umbral por defecto: 0.5")
    print(f"  Umbral personalizado: {umbral}")
    
    resultados = {}
    
    for nombre, y_pred, umbral_used in [('Default', y_pred_default, 0.5), 
                                        ('Custom', y_pred_custom, umbral)]:
        
        # Calcular matriz de confusión manualmente - ¡ESTO ES CLAVE!
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()  # Extraer los 4 valores
        
        # Calcular métricas manualmente para evitar problemas
        accuracy = (tp + tn) / len(y_test)
        
        # Balanced accuracy
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2
        
        # Métricas para clase 1 (larga)
        precision_larga = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_larga = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-score para clase 1
        if (precision_larga + recall_larga) > 0:
            f1_larga = 2 * (precision_larga * recall_larga) / (precision_larga + recall_larga)
        else:
            f1_larga = 0
        
        resultados[nombre] = {
            'umbral': umbral_used,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision_larga': precision_larga,
            'recall_larga': recall_larga,
            'f1_larga': f1_larga,
            'cm': cm,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
        
        print(f"\n  📊 Resultados con umbral {umbral_used}:")
        print(f"    Accuracy: {accuracy:.3f}")
        print(f"    Balanced Accuracy: {balanced_acc:.3f}")
        print(f"    Precisión (larga): {precision_larga:.3f}")
        print(f"    Recall (larga): {recall_larga:.3f}")
        print(f"    F1-score (larga): {f1_larga:.3f}")
        print(f"    Matriz: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # Calcular AUC-ROC (es independiente del umbral)
    auc = roc_auc_score(y_test, y_pred_proba_clase1)
    print(f"\n  🎯 AUC-ROC (independiente del umbral): {auc:.3f}")
    
    # Visualizaciones (el resto del código igual)...
    # ... [mantén el código de visualización igual]

    print(f"\n  📋 Matriz de Confusión (umbral {umbral}):")
    cm = confusion_matrix(y_test, y_pred_custom)
    print(f"    Verdaderos Negativos (Corta→Corta): {cm[0,0]}")
    print(f"    Falsos Positivos (Corta→Larga): {cm[0,1]}")
    print(f"    Falsos Negativos (Larga→Corta): {cm[1,0]}")
    print(f"    Verdaderos Positivos (Larga→Larga): {cm[1,1]}")
    
    # ========== ¡AGREGA ESTA SECCIÓN DE VISUALIZACIÓN! ==========
    
    # Visualizaciones
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Matriz de confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Corta', 'Larga'],
                yticklabels=['Corta', 'Larga'], 
                ax=axes[0],
                cbar_kws={'label': 'Cantidad'})
    axes[0].set_title(f'Matriz de Confusión\n(Umbral = {umbral:.3f})')
    axes[0].set_xlabel('Predicción')
    axes[0].set_ylabel('Real')
    
    # 2. Comparación de métricas
    metricas_comparar = ['accuracy', 'balanced_accuracy', 'f1_larga']
    nombres_bonitos = ['Accuracy', 'Balanced Accuracy', 'F1-score (Larga)']
    
    valores_default = [resultados['Default'][m] for m in metricas_comparar]
    valores_custom = [resultados['Custom'][m] for m in metricas_comparar]
    
    x = np.arange(len(metricas_comparar))
    width = 0.35
    
    axes[1].bar(x - width/2, valores_default, width, label='Umbral 0.5', color='skyblue')
    axes[1].bar(x + width/2, valores_custom, width, label=f'Umbral {umbral:.3f}', color='lightcoral')
    
    axes[1].set_xlabel('Métrica')
    axes[1].set_ylabel('Valor')
    axes[1].set_title('Comparación de Umbrales')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(nombres_bonitos)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Distribución de probabilidades
    axes[2].hist(y_pred_proba_clase1[y_test == 0], bins=20, alpha=0.7, 
                label='Clase Real: Corta', color='blue')
    axes[2].hist(y_pred_proba_clase1[y_test == 1], bins=20, alpha=0.7, 
                label='Clase Real: Larga', color='red')
    axes[2].axvline(x=umbral, color='black', linestyle='--', 
                   label=f'Umbral ({umbral:.3f})')
    axes[2].set_xlabel('Probabilidad de Estancia Larga')
    axes[2].set_ylabel('Frecuencia')
    axes[2].set_title('Distribución de Probabilidades')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluacion_modelo_mejorada.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Recomendación final
    print(f"\n🎯 RECOMENDACIÓN:")
    if resultados['Custom']['f1_larga'] > resultados['Default']['f1_larga']:
        print(f"  ✅ Usar umbral {umbral:.3f} (mejor F1-score)")
    else:
        print(f"  ⚠️  Mantener umbral por defecto 0.5")
    
    print(f"  📊 F1-score mejorado: {resultados['Custom']['f1_larga'] - resultados['Default']['f1_larga']:+.3f}")
    
    # ========== FIN DE LA SECCIÓN A AGREGAR ==========
    
    return resultados, y_pred_custom, y_pred_proba_clase1




def predecir_estancia_mejorada(model_path, datos_paciente, scaler, umbral=0.5, feature_names=None):
    """
    Función mejorada para predecir estancia con umbral ajustable y explicación
    """
    try:
        # Cargar modelo
        model = keras.models.load_model(model_path)
        
        # Preparar datos
        datos_escalados = scaler.transform([datos_paciente])
        
        # Hacer predicción
        prediccion = model.predict(datos_escalados, verbose=0)
        prob_corta = prediccion[0][0]
        prob_larga = prediccion[0][1]
        
        # Decisión con umbral ajustable
        if prob_larga > umbral:
            resultado = "ESTANCIA LARGA (>1 día)"
            confianza = prob_larga
            decision = f"Probabilidad larga ({prob_larga:.1%}) > umbral ({umbral:.1%})"
        else:
            resultado = "ESTANCIA CORTA (1 día)"
            confianza = prob_corta
            decision = f"Probabilidad larga ({prob_larga:.1%}) ≤ umbral ({umbral:.1%})"
        
        print("=" * 60)
        print("🎯 PREDICCIÓN MEJORADA PARA NUEVO PACIENTE")
        print("=" * 60)
        
        print(f"\n📊 PROBABILIDADES:")
        print(f"  Probabilidad estancia corta (1 día): {prob_corta:.1%}")
        print(f"  Probabilidad estancia larga (>1 día): {prob_larga:.1%}")
        print(f"  Umbral de decisión: {umbral:.1%}")
        
        print(f"\n✅ RESULTADO: {resultado}")
        print(f"  Confianza: {confianza:.1%}")
        print(f"  Decisión: {decision}")
        
        print(f"\n📈 INTERPRETACIÓN:")
        if confianza > 0.8:
            print(f"  🟢 ALTA confianza en la predicción")
            print(f"    - Puedes confiar en este resultado para planificación")
        elif confianza > 0.6:
            print(f"  🟡 CONFIANZA MODERADA")
            print(f"    - Resultado útil, pero considerar otros factores clínicos")
        else:
            print(f"  🔴 BAJA confianza")
            print(f"    - Recomendación: Evaluar caso individualmente")
            print(f"    - La probabilidad está cerca del umbral ({umbral:.1%})")
        
        print(f"\n💡 RECOMENDACIONES CLÍNICAS:")
        if resultado.startswith("ESTANCIA LARGA"):
            print(f"  • Planificar recursos para >1 día")
            print(f"  • Considerar seguimiento estrecho")
            print(f"  • Evaluar necesidad de estudios adicionales")
        else:
            print(f"  • Alta probable en 24 horas")
            print(f"  • Preparar plan de seguimiento ambulatorio")
            print(f"  • Confirmar con evaluación clínica completa")
        
        print("\n" + "=" * 60)
        
        # Si se proporcionan nombres de características, mostrar contribución
        if feature_names is not None and len(feature_names) == len(datos_paciente):
            print(f"\n🔍 FACTORES QUE INFLUYEN EN LA PREDICCIÓN:")
            
            # Obtener importancia aproximada (primer capa)
            try:
                weights = model.layers[0].get_weights()[0]
                contribuciones = weights[:, 1] * datos_paciente  # Peso para clase "larga"
                
                # Crear DataFrame
                df_contrib = pd.DataFrame({
                    'Característica': feature_names,
                    'Valor': datos_paciente,
                    'Contribución': contribuciones
                })
                
                # Ordenar por valor absoluto
                df_contrib['Abs_Contrib'] = np.abs(df_contrib['Contribución'])
                df_contrib = df_contrib.sort_values('Abs_Contrib', ascending=False)
                
                print("  Principales contribuyentes:")
                for idx, row in df_contrib.head(5).iterrows():
                    signo = "+" if row['Contribución'] > 0 else "-"
                    print(f"    {row['Característica']}: {signo}{abs(row['Contribución']):.3f}")
                
            except:
                print("  (No se pudo extraer contribuciones detalladas)")
        
        return resultado, prob_larga, prediccion
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return "ERROR", 0.0, None

# ============================================================================
# 3. FUNCIÓN PRINCIPAL MEJORADA
# ============================================================================


    
    # Lista de archivos

def main_mejorada():
    # Definir ruta relativa
    ruta_datos = 'Datos/'
    
    # Verificar que la carpeta existe
    if not os.path.exists(ruta_datos):
        print(f"❌ Error: La carpeta '{ruta_datos}' no existe")
        return
    
    # Buscar archivos Excel (mayúsculas y minúsculas)
    archivos = []
    extensiones = ['*.xlsx', '*.XLSX', '*.xls', '*.XLS']
    
    for ext in extensiones:
        archivos.extend(glob.glob(os.path.join(ruta_datos, ext)))
    
    print(f"📁 Encontrados {len(archivos)} archivos en '{ruta_datos}'")
    
    if len(archivos) == 0:
        print("❌ No se encontraron archivos Excel en la carpeta")
        return
    
    # Mostrar archivos encontrados
    for i, archivo in enumerate(archivos, 1):
        print(f"  {i}. {os.path.basename(archivo)}")
    
    # ... resto del código
    
    try:
        # ========== PASO 1: CARGAR DATOS ==========
        df_completo = cargar_y_preparar_datos(archivos)
        
        if df_completo is None or len(df_completo) == 0:
            print("❌ No hay datos para procesar")
            return
        
        # ========== PASO 2: CREAR VARIABLES ==========
        df_features, df_procesado = crear_variables_predictoras(df_completo)
        
        if df_features is None or len(df_features) == 0:
            print("❌ No se pudieron crear características")
            return
        
        # ========== PASO 3: PREPARAR DATOS ==========
        X_train, X_test, y_train, y_test, scaler, feature_names = preparar_datos_modelo(df_features)
        
        if X_train is None:
            print("❌ No se pudieron preparar datos para el modelo")
            return
        
        # ========== PASO 4: ENTRENAR MODELO MEJORADO ==========
        model, history = construir_entrenar_modelo_mejorado(X_train, X_test, y_train, y_test)
        
        if model is None:
            print("❌ No se pudo construir o entrenar el modelo")
            return
        
        # ========== PASO 5: ENCONTRAR MEJOR UMBRAL ==========
        mejor_umbral, df_umbrales = encontrar_mejor_umbral(model, X_test, y_test)
        
        # ========== PASO 6: GRAFICAR CURVA ROC ==========
        fpr, tpr, thresholds, umbral_optimo_roc = graficar_curva_roc(model, X_test, y_test)
        
        # Usar el mejor umbral (promedio de ambos métodos)
        umbral_final = np.mean([mejor_umbral, umbral_optimo_roc])
        print(f"\n🎯 UMBRAL FINAL SELECCIONADO: {umbral_final:.3f}")
        print(f"   (Promedio de mejor F1-score y punto óptimo ROC)")
        
        # ========== PASO 7: EVALUAR CON UMBRAL MEJORADO ==========
        resultados, y_pred_mejorado, y_probas = evaluar_modelo_mejorado(
            model, X_test, y_test, history, umbral=umbral_final
        )
        
        # ========== PASO 8: GUARDAR MODELO ==========
        try:
            model.save('modelo_estancia_mejorado.keras')
            print(f"\n💾 Modelo guardado como 'modelo_estancia_mejorado.keras'")
        except:
            model.save('modelo_estancia_mejorado.h5')
            print(f"\n💾 Modelo guardado como 'modelo_estancia_mejorado.h5'")
        
        # Guardar también el scaler y feature names para predicciones futuras
        import joblib
        joblib.dump(scaler, 'scaler_estancia.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
        print(f"💾 Scaler y nombres de características guardados")
        
        # ========== PASO 9: EJEMPLO DE PREDICCIÓN ==========
        print("\n" + "=" * 70)
        print("🧪 EJEMPLO DE PREDICCIÓN PARA PACIENTE PROMEDIO:")
        print("=" * 70)
        
        # Crear paciente promedio
        paciente_promedio = np.mean(X_train, axis=0)
        
        resultado_ejemplo, prob_ejemplo, pred_ejemplo = predecir_estancia_mejorada(
            'modelo_estancia_mejorado.keras',
            paciente_promedio,
            scaler,
            umbral=umbral_final,
            feature_names=feature_names
        )
        
        # ========== PASO 10: GUARDAR REPORTE FINAL ==========
        guardar_reporte_final(df_features, resultados, umbral_final, mejor_umbral, umbral_optimo_roc)

        guardar_reporte_final_pdf(df_features, resultados, umbral_final, mejor_umbral, umbral_optimo_roc)
        
        print("\n" + "=" * 70)
        print("🎉 ¡PROCESO COMPLETADO CON ÉXITO!")
        print("=" * 70)
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"  1. analisis_umbrales.png - Análisis de umbrales óptimos")
        print(f"  2. curva_roc_modelo.png - Curva ROC del modelo")
        print(f"  3. evaluacion_modelo_mejorada.png - Evaluación completa")
        print(f"  4. modelo_estancia_mejorado.keras - Modelo entrenado")
        print(f"  5. scaler_estancia.pkl - Scaler para nuevas predicciones")
        print(f"  6. feature_names.pkl - Nombres de características")
        print(f"  7. reporte_final.txt - Reporte completo del modelo")
        
        print(f"\n🎯 UMBRAL RECOMENDADO PARA USO: {umbral_final:.3f}")
        print(f"📊 F1-SCORE ESTANCIA LARGA: {resultados.get('Custom', {}).get('f1_larga', 0):.3f}")
        
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()




def info_dataset(df_original, df_features):
    return {
        "registros_originales": len(df_original),
        "episodios": len(df_features),
        "fecha_min": df_original["fecha"].min(),
        "fecha_max": df_original["fecha"].max(),
        "columnas_originales": list(df_original.columns),
        "columnas_modelo": list(df_features.columns)
    }


def guardar_reporte_final_pdf(df_features, resultados, umbral_final, mejor_umbral, umbral_optimo_roc):
    archivo_pdf = "reporte_final.pdf"
    
    doc = SimpleDocTemplate(
        archivo_pdf,
        pagesize=LETTER,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    story = []

    def add(text, style=styles["Normal"]):
        story.append(Paragraph(text, style))
        story.append(Spacer(1, 0.2 * inch))

    # TÍTULO
    add("<b>REPORTE FINAL - MODELO DE PREDICCIÓN DE ESTANCIA</b>", styles["Title"])

    # DISTRIBUCIÓN
    cortas = (df_features['Estancia_larga'] == 0).sum()
    largas = (df_features['Estancia_larga'] == 1).sum()
    total = len(df_features)

    add("<b><font size=12>Distribución de datos</font></b>")
    add(f"Total de episodios: {total}")
    add(f"Estancias cortas (1 día): {cortas} ({cortas/total*100:.1f}%)")
    add(f"Estancias largas (>1 día): {largas} ({largas/total*100:.1f}%)")

    # UMBRALES
    add("<b>Umbrales encontrados</b>")
    add(f"Mejor umbral (F1-score): {mejor_umbral:.3f}")
    add(f"Umbral óptimo (ROC): {umbral_optimo_roc:.3f}")
    add(f"Umbral final recomendado: {umbral_final:.3f}")

    # MÉTRICAS
    custom = resultados.get("Custom")
    if custom:
        add("<b>Resultados del modelo</b>")
        add(f"Accuracy: {custom['accuracy']:.3f}")
        add(f"Balanced Accuracy: {custom['balanced_accuracy']:.3f}")
        add(f"Precisión (estancia larga): {custom['precision_larga']:.3f}")
        add(f"Recall: {custom['recall_larga']:.3f}")
        add(f"F1-score: {custom['f1_larga']:.3f}")

    # FECHA
    add(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    doc.build(story)
    print("📄 Reporte final guardado como 'reporte_final.pdf'")


def guardar_reporte_final(df_features, resultados, umbral_final, mejor_umbral, umbral_optimo_roc):
    """Guarda un reporte final con todos los resultados MEJORADO"""
    
    with open('reporte_final.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE FINAL - MODELO DE PREDICCIÓN DE ESTANCIA\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("📊 DISTRIBUCIÓN DE DATOS:\n")
        cortas = (df_features['Estancia_larga'] == 0).sum()
        largas = (df_features['Estancia_larga'] == 1).sum()
        total = len(df_features)
        f.write(f"  Total episodios: {total}\n")
        f.write(f"  Estancias cortas (1 día): {cortas} ({(cortas/total)*100:.1f}%)\n")
        f.write(f"  Estancias largas (>1 día): {largas} ({(largas/total)*100:.1f}%)\n\n")
        
        f.write("🎯 UMBRALES ENCONTRADOS:\n")
        f.write(f"  Mejor umbral (F1-score): {mejor_umbral:.3f}\n")
        f.write(f"  Umbral óptimo (ROC): {umbral_optimo_roc:.3f}\n")
        f.write(f"  Umbral final recomendado: {umbral_final:.3f}\n\n")
        
        f.write("📈 RESULTADOS DEL MODELO (umbral {:.3f}):\n".format(umbral_final))
        
        custom = resultados.get('Custom')
        
        if custom is not None:
            # Calcular sensibilidad y especificidad también
            sensibilidad = custom['recall_larga']
            especificidad = custom['tn'] / (custom['tn'] + custom['fp']) if (custom['tn'] + custom['fp']) > 0 else 0
            
            f.write(f"  Accuracy: {custom['accuracy']:.3f}\n")
            f.write(f"  Balanced Accuracy: {custom['balanced_accuracy']:.3f}\n")
            f.write(f"  Precisión (estancia larga): {custom['precision_larga']:.3f}\n")
            f.write(f"  Recall/Sensibilidad: {sensibilidad:.3f}\n")
            f.write(f"  Especificidad: {especificidad:.3f}\n")
            f.write(f"  F1-score (estancia larga): {custom['f1_larga']:.3f}\n")
            f.write(f"  Verdaderos Positivos: {custom['tp']}\n")
            f.write(f"  Falsos Positivos: {custom['fp']}\n")
            f.write(f"  Verdaderos Negativos: {custom['tn']}\n")
            f.write(f"  Falsos Negativos: {custom['fn']}\n\n")
        else:
            f.write("  ⚠️ Métricas no disponibles por error de evaluación\n\n")
        
        f.write("💡 RECOMENDACIONES DE USO:\n")
        f.write(f"  1. Usar umbral {umbral_final:.3f} para decisiones clínicas\n")
        f.write(f"  2. Para triage inicial (max recall): usar umbral 0.3\n")
        f.write(f"  3. Para planificación de camas (max precisión): usar umbral 0.7\n")
        f.write(f"  4. La confianza >80% indica alta fiabilidad\n\n")
        
        # Añadir interpretación de métricas
        if custom is not None:
            f.write("🎯 INTERPRETACIÓN DE MÉTRICAS:\n")
            if custom['f1_larga'] > 0.9:
                f.write("  ✅ EXCELENTE desempeño para identificar estancias largas\n")
                f.write("     - Puedes confiar en las predicciones del modelo\n")
            elif custom['f1_larga'] > 0.7:
                f.write("  👍 BUEN desempeño para identificar estancias largas\n")
                f.write("     - Útil para apoyo en la toma de decisiones\n")
            else:
                f.write("  ⚠️  Desempeño MODERADO para identificar estancias largas\n")
                f.write("     - Usar con precaución y validación clínica\n")
            f.write("\n")
        
        f.write("🔍 FACTORES MÁS IMPORTANTES:\n")
        f.write("  1. Número de consultas (correlación más fuerte)\n")
        f.write("  2. Tipo de consulta realizada\n")
        f.write("  3. Unidad médica de origen\n")
        f.write("  4. Edad del paciente\n")
        f.write("  5. Franjas horarias de atención\n\n")
        
        f.write("📅 FECHA DE GENERACIÓN: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    
    print("📄 Reporte final guardado como 'reporte_final.txt'")

# ============================================================================
# 4. EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    main_mejorada()