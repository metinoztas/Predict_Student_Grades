# 🎓 Öğrenci Performansı Tahmin Uygulaması

Bu proje, öğrencilerin günlük alışkanlıklarını analiz ederek akademik başarılarını tahmin etmeyi amaçlayan bir **makine öğrenmesi ve yapay zeka projesidir**. TensorFlow kullanılarak geliştirilen model, öğrencinin ders çalışma süresi, uyku düzeni, sosyal aktiviteler, hobiler, fiziksel aktivite ve stres seviyesini değerlendirerek not tahmini yapmaktadır.

---

## 📌 Özellikler

- Günlük alışkanlıklara dayalı **öğrenci başarı tahmini**
- **TensorFlow Keras** ile derin öğrenme modeli
- **MinMaxScaler** ile veri normalizasyonu
- Streamlit tabanlı **interaktif web uygulaması**
- Modelin %19.4 hata payı ile tahmin yapması
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
- `Student_ID`: Öğrenci kimlik numarası  
- `Gender`: Cinsiyet (Kadın/Erkek)  
- `Grades`: Akademik başarı notu (0-10)  
- `Stress_Level`: Stres seviyesi (Düşük, Orta, Yüksek)  
- Günlük alışkanlıklar: Ders çalışma süresi, hobiler, uyku süresi, sosyal etkinlikler, fiziksel aktivite  


