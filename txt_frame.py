import cv2

# Kare bilgisi (koordinatlar) içeren dosyadan, belirtilen kare numarasına ait verileri okuyan fonksiyon.
def read_frame_info(frame_count, file_path):
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            # Dosyadaki kare numarası ile işlenen kare numarası eşleşirse koordinatları al
            if parts[0] == str(frame_count).zfill(4): 
                xmin2, ymin2, xmax2, ymax2 = map(int, parts[1:])
                return xmin2, ymin2, xmax2, ymax2
    return None, None, None, None  # Eşleşme bulunamazsa None değerleri döndür

# İşlenecek video dosyasının yolu
video_path = "/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/video/1917.mp4"
video = cv2.VideoCapture(video_path)

# Çıktı videosu için VideoWriter nesnesi oluşturma ve video özelliklerini ayarlama
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/video/output_video_text.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Etiket bilgisi içeren dosyanın yolu
user_txt = "/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/ground_info_index.txt"

frame_count = 0
while True:
    ret, frame = video.read()
    if not ret: # Video bittiyse döngüden çık
        break

    # Mevcut karenin etiket bilgisini oku
    xmin2, ymin2, xmax2, ymax2 = read_frame_info(frame_count, user_txt)
    if xmin2 is not None:  # Etiket bilgisi mevcutsa
        width2 = xmax2 - xmin2
        height2 = ymax2 - ymin2

        # Kare numarasını gösteren metni oluştur ve ekle
        text = f"frame: {frame_count}"
        org = (frame.shape[1] - 400, 50) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)
        thickness = 2
        cv2.putText(frame, text, org, font, font_scale, color, thickness)

        # Dikdörtgeni çiz
        cv2.rectangle(frame, (xmin2, ymin2), (xmax2, ymax2), (255, 0, 0), 2)

    # İşlenmiş kareyi çıktı videosuna yaz
    out.write(frame)
    
    frame_count += 1

print("Toplam frame sayısı: ", frame_count)
video.release()
out.release()
cv2.destroyAllWindows()
