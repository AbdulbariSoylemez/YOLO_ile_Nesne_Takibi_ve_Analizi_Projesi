import os
import cv2
import torch
from ultralytics import YOLO
from scene_detect_utils import find_first_cut,read_frame_info,draw_rectangles,save_frame_with_iou_text,update_results,write_result_line

class ObjectTracker:
    def __init__(self, model_path, threshold=0.5):
        self.model = YOLO(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.threshold = threshold
        self.iou_low_value_frame=0.6
    


    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (tuple): (xmin, ymin, xmax, ymax) for the first box.
            box2 (tuple): (xmin, ymin, xmax, ymax) for the second box.

        Returns:
            float: IoU value.
        """
        # İki kutunun kesişim alanını hesapla.
        x1_min, y1_min, x1_max, y1_max = box1 #box of the first rectangle
        x2_min, y2_min, x2_max, y2_max = box2 #box of the second rectangle

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # Kesişim alanını hesapla.
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Her iki kutunun alanını hesapla.
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # IoU değerini hesapla.
        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def preprocess_frame(self, frame):
        """
        Preprocess the frame to match the YOLO model's expected input format.
        
        Args:
            frame (np.ndarray): Input frame in HWC format.

        Returns:
            torch.Tensor: Preprocessed frame in BCHW format.
        """
        resized_frame = cv2.resize(frame, (640, 640))
        chw_frame = resized_frame.transpose(2, 0, 1)
        normalized_frame = chw_frame / 255.0
        tensor_frame = torch.from_numpy(normalized_frame).float()
        tensor_frame = tensor_frame.unsqueeze(0).to(self.device)
        return tensor_frame, resized_frame

    def track_with_initial_yolo_check(self, video_path, xPer, yPer, wPer, hPer, intime, outtime, expFrame, objId):
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError("Video could not be opened!")

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        x = int(xPer * frame_width / 100)
        y = int(yPer * frame_height / 100)
        width = int(wPer * frame_width / 100)
        height = int(hPer * frame_height / 100)

        video.set(cv2.CAP_PROP_POS_MSEC, intime * 1000)

        calculated_out_time = find_first_cut(video_path=video_path, start=intime, end=outtime)
        if calculated_out_time and calculated_out_time > intime and calculated_out_time < outtime:
            outtime = calculated_out_time

        results = []
        frame_count = 0 
        prev_box = (x, y, x + width, y + height)

        # Create and open the file for writing
        result_file = open('tracking_results.txt', 'w') 
        frame_iou = open('/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/frame_iou.txt', 'w')
        

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = 'output_with_initial.mp4'
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        ret, frame1 = video.read()
        if not ret:
            raise ValueError("Failed to read the initial frame!")

        tensor_frame, _ = self.preprocess_frame(frame1)
        detection_results = self.model(tensor_frame)
        detections = detection_results[0]  

        # Initialize frame index
        best_iou = 0
        best_box = None
        for detection in detections.boxes:
            # Her bir algılama için koordinatları listeye çevir.
            box = detection.xyxy[0].tolist()  
            if len(box) == 4:
                # Koordinatları ayıkla ve ölçeklendir.
                xmin, ymin, xmax, ymax = box
                xmin, xmax = int(xmin * frame_width / 640), int(xmax * frame_width / 640)
                ymin, ymax = int(ymin * frame_height / 640), int(ymax * frame_height / 640)
                current_box = (xmin, ymin, xmax, ymax)
                
                # Önceki kutu ile mevcut kutu arasındaki IoU değerini hesapla.
                iou = self.calculate_iou(prev_box, current_box)
                
                if iou > best_iou:
                    # Eğer bulunan IoU, şimdiye kadar bulunan en iyi IoU'dan büyükse güncelle.
                    print("BEST IOU SCORE: ", iou)
                    best_iou = iou
                    best_box = current_box

                    # Çerçeveyi kopyala ve kutuları çizmek için kullan.
                    debug_frame = frame1.copy()
                    # Önceki kutuyu mavi renkte çiz.
                    cv2.rectangle(debug_frame, (prev_box[0], prev_box[1]), (prev_box[2], prev_box[3]), (255, 0, 0), 2)  # blue
                    # Mevcut kutuyu kırmızı renkte çiz.
                    cv2.rectangle(debug_frame, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 0, 255), 2)  # red
                    # Çerçeveyi kaydet.
                    cv2.imwrite(f'debug_frame_zero{frame_count}.jpg', debug_frame)
        
        if best_iou < self.threshold:
            tracker = cv2.legacy.TrackerCSRT_create()
            multi_tracker = cv2.legacy.MultiTracker_create()
            multi_tracker.add(tracker, frame1, (x, y, width, height))
            using_yolo = False
            prev_box = (x, y, x + width, y + height)
        else:
            prev_box = best_box
            using_yolo = True 

        samliouframe = "/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/smaliouframe/"
        user_txt="/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/ground_info_index.txt"
        
        video = cv2.VideoCapture(video_path) 
        
       
        frame_count = 0 
        while video.isOpened():
            ret, frame = video.read()

            if not ret or video.get(cv2.CAP_PROP_POS_MSEC) > outtime * 1000:
                break

            if frame_count % expFrame == 0:
                
                if using_yolo:
                    tensor_frame, _ = self.preprocess_frame(frame)
                    detection_results = self.model(tensor_frame)
                    detections = detection_results[0]

                    scale_x = frame_width / 640
                    scale_y = frame_height / 640

                    best_iou = 0
                    best_box = None
                    for detection in detections.boxes:
                        box = detection.xyxy[0].tolist()
                        if len(box) == 4:
                            xmin, ymin, xmax, ymax = box
                            xmin, xmax = int(xmin * scale_x), int(xmax * scale_x)
                            ymin, ymax = int(ymin * scale_y), int(ymax * scale_y)
                            current_box = (xmin, ymin, xmax, ymax)
                            iou = self.calculate_iou(prev_box, current_box)

                            if iou > best_iou:
                                best_iou = iou
                                best_box = current_box

                    if best_box is not None:
                        xmin, ymin, xmax, ymax = best_box
                        x, y, width, height = xmin, ymin, xmax - xmin, ymax - ymin
                        
                        update_results(results, video, frame_width, frame_height, objId, x, y, width, height)
                        write_result_line(result_file, objId, x, y, width, height)

                        xmin2, ymin2, xmax2, ymax2 = read_frame_info(frame_count, user_txt)
                        width2 = xmax2 - xmin2
                        height2 = ymax2 - ymin2

                        box1=(x,y,(x + width),(y + height))
                        box2=(xmin2, ymin2, (xmin2 + width2), (ymin2 + height2))

                        draw_rectangles(frame, box1, box2)


                        iou = self.calculate_iou(box1, box2)
                        frame_iou_text=f"{frame_count} {iou}\n"
                        frame_iou.write(frame_iou_text)
                        if iou < self.iou_low_value_frame:
                                 output_path = os.path.join(samliouframe, f"frame{frame_count}.jpg")
                                 save_frame_with_iou_text(frame, iou, frame_count, output_path)
                        


                        
                    else:
                        success, boxes = multi_tracker.update(frame)
                        if success:
                            for i, new_box in enumerate(boxes):
                                x, y, width, height = [int(v) for v in new_box]
                                
                                update_results(results, video, frame_width, frame_height, objId, x, y, width, height)
                                write_result_line(result_file, objId, x, y, width, height)

                                xmin2, ymin2, xmax2, ymax2 = read_frame_info(frame_count, user_txt)
                                width2 = xmax2 - xmin2
                                height2 = ymax2 - ymin2

                                box1=(x,y,(x + width),(y + height))
                                box2=(xmin2, ymin2, (xmin2 + width2), (ymin2 + height2))

                                draw_rectangles(frame, box1, box2)


                                iou = self.calculate_iou(box1, box2)
                                frame_iou_text=f"{frame_count} {iou}\n"
                                frame_iou.write(frame_iou_text)
                                if iou < self.iou_low_value_frame:
                                    output_path = os.path.join(samliouframe, f"frame{frame_count}.jpg")
                                    save_frame_with_iou_text(frame, iou, frame_count, output_path)
                               

                        else:
                            results.append({
                                "x": None,
                                "y": None,
                                "width": None,
                                "height": None,
                                "time": video.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                                "objId": objId
                            })
                            ### eğer boş index değerleri giriş olarak alsa 
                            result_line = f"{objId:04d} None None None None\n"
                            result_file.write(result_line)
                else:  
                    success, boxes = multi_tracker.update(frame)
                    if success:
                        for i, new_box in enumerate(boxes):
                            x, y, width, height = [int(v) for v in new_box]

                            update_results(results, video, frame_width, frame_height, objId, x, y, width, height)
                            write_result_line(result_file, objId, x, y, width, height)

                            xmin2, ymin2, xmax2, ymax2 = read_frame_info(frame_count, user_txt)
                            width2 = xmax2 - xmin2
                            height2 = ymax2 - ymin2

                            box1=(x,y,(x + width),(y + height))
                            box2=(xmin2, ymin2,(xmin2 + width2), (ymin2 + height2))

                            draw_rectangles(frame, box1, box2)
                           


                            iou = self.calculate_iou(box1, box2)
                            frame_iou_text=f"{frame_count} {iou}\n"
                            frame_iou.write(frame_iou_text)
                            if iou < self.iou_low_value_frame:
                                output_path = os.path.join(samliouframe, f"frame{frame_count}.jpg")
                                save_frame_with_iou_text(frame, iou, frame_count, output_path)
                            

                    else:
                        results.append({
                            "x": None,
                            "y": None,
                            "width": None,
                            "height": None,
                            "time": video.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                            "objId": objId
                        })
                        ##### tespit edilen isim ve indexleri result değişkenine ekle 
                        result_line = f"{objId:04d} None None None None\n"
                        result_file.write(result_line)
                        

            out.write(frame)
            frame_count += 1
            

        video.release()
        out.release()
        print("Toplam frame sayısı bu :",frame_count)
        
        cv2.destroyAllWindows()

        video_info = {
            "fps": float("{:.3f}".format(fps)),
            "frameCount": frame_count,
            "duration": float("{:.3f}".format(duration)),
            "height": frame_height,
            "width": frame_width
        }

        request_info = {
            "videoPath": video_path,
            "intime": intime,
            "outtime": outtime,
            "xPer": xPer,
            "yPer": yPer,
            "wPer": wPer,
            "hPer": hPer,
            "expFrame": expFrame,
            "objId": objId
        }

        return {
            "videoInfo": video_info,
            "requestInfo": request_info,
            "output": results
        }

     