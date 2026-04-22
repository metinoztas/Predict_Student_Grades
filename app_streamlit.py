import streamlit as st

from tensorflow.keras.models import load_model
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error


st.set_page_config(page_title="Öğrenci Performansı Tahmini", layout="centered")

st.markdown("""
    <style>
    .stApp {
       background: linear-gradient(to right, #002B71, #007389);
    }
    </style>
""", unsafe_allow_html=True)



st.title("🎓 Öğrenci Performansı Tahmin Uygulaması")
st.markdown("""
    Bu uygulama, öğrencilerin günlük alışkanlıklarını analiz ederek başarı tahmini yapmayı amaçlayan bir araçtır. 
    TensorFlow tabanlı bir model, öğrencinin ders çalışma süresi, sosyal aktiviteler, uyku düzeni, 
    fiziksel aktivite düzeyi ve stres seviyesi gibi faktörleri göz önünde bulundurarak gelecekteki akademik performansı hakkında tahminlerde bulunur.
""")



st.markdown("""
    <div class="social-buttons">
        <a href="https://www.linkedin.com/in/metin-oztas/" target="_blank" style="position: fixed; bottom: 20px; right: 40px; margin-right: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="40" height="40" alt="Linkedin"/>
        </a>
       
    </div>
    <a href="https://github.com/metinoztas/Predict_Student_Grades" target="_blank" style="position: fixed; bottom: 20px; right: 100px;">
            <img src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" width="40" height="40" alt="Github"/>
    </a>
    
""", unsafe_allow_html=True)





# Form
with st.form("girdi_formu"):
    st.header("🧾 Bilgi Girişi")

    calisma_suresi = st.slider("Okuldaki Dersler Dahil Günlük Ders Çalışma Süresi (saat)", 0.0, 10.0, 2.0, 0.1)
    hobi_saat = st.slider("Günlük Hobiler'e ayrılan süre (saat)", 0.0, 10.0, 2.0, 0.1)
    uyku_suresi = st.slider("Günlük Uyku Süresi (saat)", 0.0, 12.0, 7.0, 0.1)
    etkinlik_suresi = st.slider("Günlük Kulüp / Sosyal etkinliklere ayrılan süre (saat)", 0.0, 5.0, 1.0, 0.1)
    spor_suresi = st.slider("Günlük Fiziksel Aktivite Süresi (saat)", 0.0, 5.0, 1.0, 0.1)

    stres_secim = st.selectbox("Stres Seviyesi", ["Düşük", "Orta", "Yüksek"])
    cinsiyet_secim = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])

    stres_sayisal = {"Düşük": 0, "Orta": 1, "Yüksek": 2}[stres_secim]
    cinsiyet_sayisal = {"Erkek": 0, "Kadın": 1}[cinsiyet_secim]

    tahmin_buton = st.form_submit_button("🔍 Performansı Tahmin Et")



if tahmin_buton:
    try:
        # Geliştirilmiş Modeli Yükle
        model = load_model("improved_model.h5", custom_objects={'mse': mean_squared_error})

        girdi = np.array([[calisma_suresi, etkinlik_suresi, uyku_suresi, hobi_saat, spor_suresi, stres_sayisal, cinsiyet_sayisal]])

        
        # Tahmin
        scaler = joblib.load("scaler.pkl")
        girdi_transform = scaler.transform(girdi)


        tahmin = model.predict(girdi_transform)[0][0]
        tahmin = round(tahmin, 2)
        
        
        st.success(f"📈 10 üzerinden Tahmin Edilen Başarı Notu: {tahmin}")

        st.write("Bu model yaklaşık ±0.5 not (yaklaşık %6.4 hata payı) hassasiyetinde tahmin yapmaktadır! Kesin bir sonuç veremez.")
        
        if tahmin >= 7:
            st.balloons()
            st.info("✨ Harika! Bu alışkanlıklar başarıyı destekliyor.")
        elif tahmin >= 5:
            st.warning("🔍 Fena değil. Birkaç alışkanlık geliştirilebilir.")
        else:
            st.error("⚠️ Düşük başarı tahmini. Gözden geçirilmesi gereken alışkanlıklar olabilir.")

        with st.expander("Öneriler !"):
            if calisma_suresi < 7:            
                st.write("Ders çalışmaya biraz daha vakit ayırabilirsin.")
            if uyku_suresi > 7:           
                st.write("İhtiyacından fazla uyku alıyorsun.")
            if etkinlik_suresi > 2.7 :
                st.write("Sosyal olmak iyidir fakat herşeyi dengeli yapmalısın.")
            if spor_suresi > 4 :
                st.write("İhtiyacın kadar spor yapmalısın.")                
            if stres_sayisal==0:
                st.write("Stresini kontrol edebilmelisin.")    
                
        

    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")








