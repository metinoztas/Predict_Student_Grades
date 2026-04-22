# 🎓 Öğrenci Performansı Tahmin Uygulaması

Bu proje, öğrencilerin günlük alışkanlıklarını analiz ederek akademik başarılarını tahmin etmeyi amaçlayan bir **makine öğrenmesi ve yapay zeka projesidir**. TensorFlow kullanılarak geliştirilen model, öğrencinin ders çalışma süresi, uyku düzeni, sosyal aktiviteler, hobiler, fiziksel aktivite ve stres seviyesini değerlendirerek not tahmini yapmaktadır.

---

## 📌 Özellikler

- Günlük alışkanlıklara dayalı **öğrenci başarı tahmini**
- **TensorFlow Keras** ile geliştirilmiş Derin Öğrenme Modeli (Dropout, BatchNormalization destekli)
- **StandardScaler** ile veri normalizasyonu (öncesinde MinMaxScaler vardı)
- Streamlit tabanlı **interaktif web uygulaması**
- Model hata payı **%19.4'ten ~%6.4 (±0.5 Hata)** değerlerine düşürülmüştür
- Kullanıcıya kişiselleştirilmiş öneriler sunma

---

## 🛠 Kullanılan Teknolojiler

- Python 3.x  
- Pandas & NumPy  
- Seaborn & Matplotlib  
- Scikit-learn  
- TensorFlow Keras  
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


