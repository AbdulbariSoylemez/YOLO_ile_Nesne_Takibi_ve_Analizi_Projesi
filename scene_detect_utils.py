# Standard PySceneDetect imports:
from scenedetect import VideoManager, SceneManager, FrameTimecode

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector
from math import ceil, floor
import cv2
def find_cut(video_path, threshold=30.0, start = 0.0, end = 5.0):
    #print(start, end)
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Set start-end range
    fps = video_manager.get_framerate()
    startFrameTimeCode = FrameTimecode(timecode = start, fps=fps)
    endFrameTimeCode = FrameTimecode(timecode = end, fps=fps)
    video_manager.set_duration(start_time=startFrameTimeCode, end_time=endFrameTimeCode)

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=False)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_cut_list()

def find_first_cut(video_path, start, end):
    start_end_times = []    
    for i in range(0, floor(end - start), 5):
        if start + i + 5 > end:
            break    
        if i > 0:
            start = start - 0.5
        start_end_times.append({
            "start": start + i,
            "end": start + i + 5
        })
        
    
    remainder = (end - start) % 5
    if remainder > 0:
        if len(start_end_times) > 0:
            start_end_times.append({
                "start": start_end_times[-1]["end"] - 0.5,
                "end": start_end_times[-1]["end"] + remainder
            })
        else:
            start_end_times.append({
                "start": start,
                "end": start + remainder
            })


    for t in start_end_times:
        result  = find_cut(video_path, start=t["start"], end=t["end"])
        if result and len(result) > 0:        
            return float("{:.3f}".format(result[0].get_seconds())) 
        

def read_frame_info(frame_count, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if parts[0] == str(frame_count).zfill(4):  # Frame_count değerini dört haneli olarak kontrol et
                    xmin2, ymin2, xmax2, ymax2 = map(int, parts[1:])  # Değerleri integer'a çevir
                    return xmin2, ymin2, xmax2, ymax2
        return None, None, None, None  # Eğer frame_count'a uygun bir satır bulunamazsa None değerleri döndür

def save_frame_with_iou_text(frame, iou, frame_count, output_path):
        """
        Frame üzerine iou ve dframe bilgisini yazan ve frame'i kaydeden fonksiyon.
        
        Args:
        frame (numpy.ndarray): İşlenecek çerçeve.
        iou (float): Intersection over Union (IoU) değeri.
        frame_count (int): Frame sayacı.
        output_path (str): Çerçevenin kaydedileceği yol.
        """
      
        text = f"iou: {iou:.2f}, frame: {frame_count}"
        org = (frame.shape[1] - 400, 50)  # Yazının yerleştirileceği koordinatlar (sağ üst köşe)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 255)  # BGR formatında kırmızı
        thickness = 2
        cv2.putText(frame, text, org, font, font_scale, color, thickness)
        cv2.imwrite(output_path, frame)

def draw_rectangles(frame, box1, box2):
        """
        Frame üzerine iki dikdörtgen çizen fonksiyon.
        
        Args:
        frame (numpy.ndarray): İşlenecek çerçeve.
        box1 (tuple): İlk dikdörtgenin koordinatları (x, y, x+width, y+height).
        box2 (tuple): İkinci dikdörtgenin koordinatları (x, y, x+width, y+height).
        """
        
        cv2.rectangle(frame, (box2[0], box2[1]), (box2[2], box2[3]), (255, 0, 0), 2)
        cv2.rectangle(frame, (box1[0], box1[1]), (box1[2], box1[3]), (0, 255, 0), 2)

def update_results(results, video, video_width, video_height, objId, x, y, width, height):
    x_per = x / video_width * 100
    y_per = y / video_height * 100
    width_per = width / video_width * 100
    height_per = height / video_height * 100

    results.append({
        "x": x_per,
        "y": y_per,
        "width": width_per,
        "height": height_per,
        "time": video.get(cv2.CAP_PROP_POS_MSEC) / 1000,
        "objId": objId
    })
def write_result_line(result_file, objId, x, y, width, height):
    result_line = f"{objId:04d} {x} {y} {x + width} {y + height}\n"
    result_file.write(result_line)


