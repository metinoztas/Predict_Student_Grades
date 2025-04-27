#%%
import pandas as pd
import numpy as np
import seaborn as sbn


# %%
dataFrame = pd.read_csv("data/student_lifestyle_dataset.csv",delimiter=",")

# %%
dataFrame.isnull().sum()
# %%
dataFrame.info()
# %%
dataFrame.describe()
# %%
sbn.displot(dataFrame["Grades"])
# dataFrame["Stress_Level"].nunique() # Kaç tane farklı değer olduğunu gösterir
# %%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataFrame["Gender"] = le.fit_transform(dataFrame["Gender"])
# Female : 1 , Male : 0
dataFrame["Stress_Level"] = le.fit_transform(dataFrame["Stress_Level"])
# Moderate : 2 , Low : 1 , High : 0

# %%
dataFrame.corr()["Grades"]
# %%
dataFrame = dataFrame.drop("Gender",axis=1)
dataFrame = dataFrame.drop("Student_ID",axis=1)
dataFrame.describe()
# %%
y = dataFrame["Grades"].values
x = dataFrame.drop("Grades",axis=1).values
# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=42)
# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train

# %%
import joblib

# scaler save
joblib.dump(scaler, 'scaler.pkl')

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation = "relu"))
model.add(Dense(12,activation = "relu"))

model.add(Dense(1))

model.compile(optimizer="adam",loss="mse")
# %%
from tensorflow.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25,restore_best_weights=True)

model.fit(x=x_train,y=y_train,epochs=700,validation_data = (x_test,y_test),verbose=1,callbacks=[earlyStopping])
# %%
modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()

# %%
tahminDizisi = model.predict(x_test)
tahminDizisi
# %%
import matplotlib.pyplot as plt

plt.hist(tahminDizisi,bins=50)
# %%
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,tahminDizisi)
# %%
dataFrame.describe()
# % 19 hata payı var ( grades mean / mean_absolute error  = hata payı)
# %%

model.save("modelV2.h5")

# from tensorflow.keras.models import load_model  
# Kaydedilen modeli kullanmak için gerekli olan kütüphane
# sonradanCagirilanModel = load_model("bisiklet_modeli.h5",custom_objects={'mse': mean_squared_error})
# sonradanCagirilanModel.predict(yeniBisikletOzellikleri)
# %%

# %%
