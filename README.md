# 🎓 Öğrenci Performansı Tahmin Uygulaması

Bu proje, öğrencilerin günlük alışkanlıklarını analiz ederek akademik başarılarını tahmin etmeyi amaçlayan bir **veri bilimi ve makine öğrenmesi projesidir**. Linear Regression (Doğrusal Regresyon) algoritması kullanılarak geliştirilen bu model; öğrencinin ders çalışma süresi, uyku düzeni, sosyal aktiviteler, hobiler, fiziksel aktivite, cinsiyet ve stres seviyesini değerlendirerek not tahmini yapmaktadır.

---

## 📌 Özellikler

- Günlük alışkanlıklara dayalı **öğrenci başarı tahmini**
- **Scikit-learn** kütüphanesi ile **Linear Regression (Doğrusal Regresyon)** Modeli
- **StandardScaler** ile veri normalizasyonu (öncesinde MinMaxScaler vardı)
- Streamlit tabanlı **interaktif web uygulaması**
- Model hata payı **~%5 (±0.41 Not Tahmin Hatası MAE)** değerlerine indirilerek daha kesin sonuçlar elde edilmiştir.
- Linear Regression'a geçiş yapılıp işlemler oldukça hızlandırılmıştır.
- Kullanıcıya kişiselleştirilmiş öneriler sunma

---

## 🛠 Kullanılan Teknolojiler

- Python 3.x  
- Pandas & NumPy  
- Seaborn & Matplotlib  
- Scikit-learn
- Streamlit  
- Joblib  

---

## 📊 Veri Seti

Veri seti: `student_lifestyle_dataset.csv`  

**Özellikler:**
- `Student_ID`: Öğrenci kimlik numarası (Modelde kullanılmadı)
- `Gender`: Cinsiyet (Erkek: 0, Kadın: 1 olarak kodlandı ve modele EKLENDİ)
- `Grades`: Akademik başarı notu (0-10)  
- `Stress_Level`: Stres seviyesi (Düşük: 0, Orta: 1, Yüksek: 2 olarak Ordinal Encoding ile kodlandı)
- Günlük alışkanlıklar: Ders çalışma süresi, hobiler, uyku süresi, sosyal etkinlikler, fiziksel aktivite  


