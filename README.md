Elbette, bu proje için detaylı bir Türkçe README dosyası oluşturalım:

## YOLO Nesne Takip ve Video Analiz Projesi

**Proje Açıklaması:**

Bu proje, YOLO (You Only Look Once) nesne algılama modeli ile nesne takibi yapmayı ve videolar üzerinde gelişmiş analizler gerçekleştirmeyi amaçlar. Özellikle sinema filmlerinde belirli nesnelerin (örneğin, karakterler, objeler) takibi, etiket verileriyle IoU (Intersection over Union) hesaplaması, düşük IoU değerlerine sahip karelerin tespiti ve görselleştirilmesi gibi işlemleri kapsar.

**Projenin Ana Özellikleri:**

* **YOLO ile Nesne Takibi:** Videoda belirlenen bir nesneyi YOLO modeli kullanarak takip eder. Başlangıç konumu ve tahmini çerçeve boyutları kullanıcı tarafından verilir.
* **IoU Hesaplaması:** YOLO ile tahmin edilen nesnelerin konumu ile etiket verilerinde belirtilen gerçek konumlar arasındaki IoU değerini hesaplar. Bu, modelin performansını değerlendirmek için önemli bir metriktir.
* **Düşük IoU Karelerinin Tespiti:** IoU değeri belirli bir eşik değerin altında olan kareleri otomatik olarak tespit eder. Bu kareler genellikle modelin nesneyi takip etmekte zorlandığı anları temsil eder.
* **Görselleştirme ve Çıktı:** Düşük IoU değerine sahip kareleri görsel olarak işaretler. Ayrıca, nesne takibi sonuçlarını ve IoU değerlerini içeren dosyalar üretir.
* **Sahne Tespiti ve Kesme:** Videoda ani değişimler gösteren sahneleri tespit eder ve bu sahneleri keser. Bu, videoların daha küçük parçalara bölünerek analiz edilmesine yardımcı olur.


**Kurulum:**

1. Gerekli Python paketlerini `pip install -r requirements.txt` komutuyla kurun.
2. `model_path` değişkenini YOLO modelinizin dosya yoluna göre ayarlayın.
3. Etiket verilerinizi içeren `.txt` dosyasını `user_txt` değişkenine göre ayarlayın.

**Kullanım:**

```bash
# API sunucusunu başlat
uvicorn app:app --host 0.0.0.0 --port 6060

# Nesne takibi yapmak için API'ye POST isteği gönderin
curl -X POST http://0.0.0.0:6060/api/track_with_initial_yolo_check \
-H "Content-Type: application/json" \
-d '{"videoPath": "/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/video/1917.mp4", "intime": 0.2, "outtime": 42.459, "xPer": 44.35, "yPer": 21.49, "wPer": 25.02, "hPer": 65.5, "expFrame": 1, "objId": 0}'
```
**Sonuç:**

Bu proje sayesinde, videolarda nesne takibi yapabilir, modelinizin performansını IoU değerleri ile analiz edebilir ve düşük performansa sahip kareleri otomatik olarak tespit edebilirsiniz. Bu bilgiler, modelinizi geliştirmeniz veya video analizi uygulamalarınızda daha iyi sonuçlar elde etmeniz için size yol gösterecektir.


![ IoU tespitine Sahip Bir Kare](https://github.com/AbdulbariSoylemez/YOLO_ile_Nesne_Takibi_ve_Analizi_Projesi/blob/main/debug_frame_zero0.jpg)

![Video üzerinde test edilemsi](https://github.com/AbdulbariSoylemez/YOLO_ile_Nesne_Takibi_ve_Analizi_Projesi/blob/main/output_with_initial_zero.mp4)


