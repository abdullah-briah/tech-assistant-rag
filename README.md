# 🤖 Teknik Asistan

RAG (Retrieval-Augmented Generation) mimarisiyle geliştirilmiş, teknik ve yapay zeka konularında Türkçe yanıt veren bir yapay zeka asistanıdır.

[**Web Uygulamasına Erişin →**](https://tech-assistant-rag-gfcxf9ydnkrptkdrxuuvyv.streamlit.app/)
🎥 [Demo Video](demo.mp4)

---

## 1. Projenin Amacı

Bu projenin temel amacı, kullanıcıların **yapay zeka, veri bilimi, makine öğrenimi ve geliştirici araçları** gibi teknik alanlarda sorduğu sorulara **doğru, güncel ve bağlamsal olarak zengin** yanıtlar sunmaktır. Geleneksel dil modellerinin "hayal ürünü" yanıtlar üretme riskini azaltmak için **RAG mimarisi** kullanılmıştır. Bu sayede, tüm yanıtlar önceden hazırlanmış ve doğrulanmış bir bilgi tabanına dayanmaktadır.

---

## 2. Veri Seti Hakkında Bilgi

Proje, **yaklaşık 90 adet teknik soru-cevap çiftinden** oluşan bir veri seti kullanmaktadır. Bu veri seti şu şekilde hazırlanmıştır:

- **Kaynaklar**: StackExchange (AI, Data Science), teknik dokümantasyonlar, akademik makaleler ve güvenilir çevrimiçi kaynaklar.
- **Hazırlama Yöntemi**: 
  - İlk olarak, bu kaynaklardan temel kavramlar ve sık sorulan sorular belirlendi.
  - Ardından, büyük dil modelleri (LLM) kullanılarak bu kavramlar **Türkçe’ye uyarlanmış**, net ve teknik olarak doğru soru-cevap çiftleri üretildi.
  - Son olarak, tüm içerikler **elle gözden geçirilerek** doğrulandı ve projeye entegre edildi.
- **Kapsam**: 
  - Yapay Zeka (YZ) ve Etik  
  - Makine Öğrenimi (Denetimli, Denetimsiz, Pekiştirmeli)  
  - Derin Öğrenme ve Sinir Ağları  
  - Veri Bilimi ve Veri Madenciliği  
  - Geliştirici Araçları (Docker, Git, API, vs.)  
  - Büyük Dil Modelleri (LLM) ve RAG  

Tüm içerikler **Türkçe** olarak hazırlanmış ve teknik doğrulukları kontrol edilmiştir. Veri seti, projenin kaynak kodunun bir parçası olarak `src/data.py` dosyasında yer almaktadır.

---

## 3. Kullanılan Yöntemler

### Teknoloji Yığını
- **Dil Modeli**: Google Gemini 2.0 Flash  
- **Embedding Modeli**: `models/text-embedding-004` (Google)  
- **Vektör Veritabanı**: Chroma (bellek içi)  
- **RAG Çerçevesi**: LangChain  
- **Web Arayüzü**: Streamlit + Özel CSS (`styles.css`)  
- **API Yönetimi**: `google-generativeai` kütüphanesi  

### RAG Mimarisi
1. **Veri Hazırlama**: Soru-cevap çiftleri `Document` formatına dönüştürüldü.  
2. **Embedding**: Google’ın `text-embedding-004` modeliyle metinler vektörlere çevrildi.  
3. **Depolama**: Vektörler ChromaDB’ye yüklendi.  
4. **Retrieval**: Kullanıcı sorduğunda, en ilgili 3 parça bağlam getirildi.  
5. **Generation**: Gemini 2.0 Flash, bağlamı kullanarak **Türkçe ve doğal** bir yanıt oluşturdu.  

### Diğer Optimizasyonlar
- `temperature=0.3` → teknik doğruluk için düşük rastgelelik  
- Özel prompt şablonu → "Bilmiyorum." yanıtı sadece gerekliyse  
- Sosyal mesajlar için akıllı ön işlem (`Merhaba`, `Teşekkürler`, `Seni kim yaptı?`)  
- Modern CSS ile zengin kullanıcı arayüzü (gradient arka plan, cam efekti, animasyonlu butonlar)

---

## 4. Elde Edilen Sonuçlar

- **Doğruluk**: Tüm yanıtlar veri setindeki bilgilere dayanmaktadır; hayal ürünü içerik üretilmemektedir.  
- **Dil Tutarlılığı**: Tüm yanıtlar **Türkçe** ve teknik olarak anlaşılırdır.  
- **Kullanıcı Deneyimi**:  
  - İlk açılışta alan seçimi ile rehberlik  
  - Sosyal etkileşim desteği (`Merhaba`, `Teşekkürler`, `Seni kim yaptı?`)  
  - Hızlı ve akıcı yanıt süresi  
  - **Modern arayüz**: Gradient arka plan, cam efekti (glassmorphism), animasyonlu butonlar  
- **Performans**: RAG zinciri bir kez başlatılır (`@st.cache_resource`), tekrar tekrar yüklenmez.

---

## 5. Web Arayüzü & Product Kılavuzu

### Nasıl Kullanılır?
1. Uygulamayı açın: [https://your-streamlit-app-url.streamlit.app](https://tech-assistant-rag-gfcxf9ydnkrptkdrxuuvyv.streamlit.app/)  
2. **İlk ziyaretinizde**, modern ve etkileyici bir arayüzle karşılaşacaksınız:
   - Mor-mavi gradient arka plan
   - Cam efekti (glassmorphism) ile mesaj kutuları
   - Renkli ve interaktif alan butonları
3. İlgilendiğiniz alanı seçin (örneğin: "🤖 Yapay Zeka").
4. Size önerilen örnek sorulardan birini sorun veya kendi sorunuzu yazın.
5. Asistan, teknik olarak doğru ve Türkçe bir yanıtla size yardımcı olacak.

### Özellikler
- 🎨 **Modern UI**: Özel `styles.css` dosyası ile tasarlandı  
- 🤖 **Akıllı Etkileşim**: 
  - Alan bazlı rehberlik
  - Sosyal mesaj desteği (`Merhaba`, `Teşekkürler`)
  - Kişisel sorulara özel yanıt (`Seni kim yaptı?`)
- ⚡ **Hızlı Performans**: RAG motoru önbelleğe alınmıştır
- 🔒 **Güvenli Bilgi**: Tüm yanıtlar doğrulanmış veri setine dayanır

> **Not**: Uygulama, Streamlit Cloud üzerinde ücretsiz olarak barındırılmaktadır.

---

## Kurulum (Yerel Ortamda Çalıştırmak İçin)

```bash
# 1. Ortamı klonlayın
git clone https://github.com/abdullah-briah/tech-assistant-rag.git
cd tech-assistant-rag

# 2. Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# 4. Google API Anahtarı ekleyin
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 5. Uygulamayı başlatın
streamlit run src/app.py
