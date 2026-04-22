import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("Veri yükleniyor...")
df = pd.read_csv("data/student_lifestyle_dataset.csv")

# Gereksiz kolonları düşürme
df = df.drop("Student_ID", axis=1)

print("Veri ön işleme yapılıyor...")
# Kategorik Verilerin Kodlanması
# Stress Level: Ordinal (Sıralı) değişken (Low: 0, Moderate: 1, High: 2)
stress_mapping = {"Low": 0, "Moderate": 1, "High": 2}
df["Stress_Level"] = df["Stress_Level"].map(stress_mapping)

# Gender: Binary kodlama (Male: 0, Female: 1)
gender_mapping = {"Male": 0, "Female": 1}
df["Gender"] = df["Gender"].map(gender_mapping)

# X ve y ayrımı
X = df.drop("Grades", axis=1)
y = df["Grades"].values

# Feature isimlerini kaydetme (ileride Streamlit'te hatasız kullanılabilmesi için)
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print(f"Kullanılan Özellikler: {feature_names}")

X = X.values

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Veri ölçeklendiriliyor (StandardScaler)...")
# StandardScaler (Derin öğrenme modelleri genellikle standartlaştırmada daha iyi sonuç verir)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Scaler'ı kaydet
joblib.dump(scaler, 'scaler.pkl')

print("Derin Öğrenme Modeli oluşturuluyor...")
# Geliştirilmiş Derin Öğrenme Modeli Mimarisi
model = Sequential()

model.add(Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2)) # Aşırı öğrenmeyi (Overfitting) engellemek için

model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(16, activation="relu"))

model.add(Dense(1, activation="linear")) # Regresyon problemi olduğu için çıkışta linear

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001, verbose=1)

print("Model eğitimi başlıyor...")
history = model.fit(
    x=X_train_scaled, 
    y=y_train, 
    epochs=150, 
    batch_size=32,
    validation_data=(X_test_scaled, y_test), 
    verbose=1, 
    callbacks=[early_stop, reduce_lr]
)

print("\nModel Değerlendirmesi:")
# Tahminler
y_pred = model.predict(X_test_scaled)

# Metrikler
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# Eğitilen Modeli Kaydetme
model.save("improved_model.h5")
print("\nModel 'improved_model.h5' olarak ve scaler 'scaler.pkl' olarak kaydedildi.")
