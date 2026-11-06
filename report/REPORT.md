# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  }

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
  Record này nói về cả 1 bức ảnh, vì ta có thể thấy trong record có mục "task" : "image_classification", nghĩa là task đang thực hiện là phân loại ảnh, đồng thời trong record còn có thông số về id, chiều cao và rộng là coco_image_id, image_height và image_width, đây là các thông số của 1 bức ảnh chứ không phải 1 vật thể riêng
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Danh sách tên lớp là đi kèm taxonomy/checkpoint

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
Phải ghi cả vì nhiều taxonomy có thể sở hữu chung 1 tên lớp, ghi taxonomy_name để có thể phân biệt/truy vấn xem model dùng taxonomy nào, còn id và name lần lượt giúp thuận tiện hơn trong việc truy cập và đọc 

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Guideline cần có những quy định như 
  -quy định về main subject như chiếm nhiều % hay vị trí gần trung tâm
  -Quy định class ưu tiên 
6. Vì sao không được lấy `score` của model làm ground truth?
Ground truth là sự thật tuyệt đối do con người đặt, chỉ có thể đúng hoặc sai. CÒn score là giá trị mô hình ước tính, dựa trên xác xuất toán học, có thể không chắc chắn đúng hoặc sai hoàn toàn mà chỉ chắc chắn theo %. Ngoài ra Score cũng có thể sai lệch do model ví dụ như Overconfident
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
"sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
  góc trên bên trái ở tọa độ (x,y) = (385.33px,69.24px)
  góc trên bên trái ở tọa độ (x,y) = (498.92px,348.92px)
  Chiều rộng = 113.58 px (= 498.92 - 385.33)
  chiều cao =  279.68 px (= 348.92 - 69.24)

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
ngưỡng 0.20 có 11 prediction
ngưỡng 0.6 có 6 prediction


4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
  khi hạ ngưỡng model có thể nhầm nhiều hơn, reviewer cần phải 
  -xóa, sửa những khung sai, trùng lặp, loại bỏ khung rác
  -chỉnh sửa kích thước, hình dạng khung
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
  -Không để box bị thừa nền quá nhiều, phải ôm sát vật thể
  -Không thiếu phần nào của vật thể, phải bao trùm toàn bộ vật thể kể cả khi vật thể bị khuất
  -Không bao vật ảo, rác như bóng vật thể, hình ảnh phản chiếu trong gương...

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Vật bị cắt mép là person (0.61) ở "bbox_xyxy": [      0.12,      263.18,      61.55,      310.78],
Việc vật thể bị khuất hoặc cắt mép 1 phần là việc thường gặp, thường sẽ có guideline quy chuẩn như phải khoanh box trùm lên cả phần bị che với vật bị che, ôm sát phần vật thể còn xuất hiện trong khung hình với vật thể bị cắt mép...
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
{
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "traffic-001",
    "class_id": 5,
    "class_name": "bus",
    "score": 0.925745,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      95.3,
      188.72,
      224.08,
      319.96
    ],
    "polygon_point_count": 120,
    "polygon_xy": [
      [
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],]}


2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Polygon bao quanh vật thể theo cả các góc cạnh thay vì 1 hộp 4 cạnh như box, có thể cung cấp hình dạng, diện tích thật của vật thể
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- dùng để phân biệt từng cá thể riêng biệt thuộc cùng một class xuất hiện trong cùng một bức ảnh, như trong kitchen có nhiều vật có class bowl
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- đường polygon phải chạy sát mép ranh giới thực tế của vật thể, không lấn ra ngoài nền hay lấn vào trong 
- phải tách rãnh khuyết, vẽ thêm đường cắt với vật thể có lỗ hổng
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
2 vật chạm nhau/bị che là bowl 0.74 và bowl 0.53. Trường hợp các vật chạm nhau hay bị che ở trong bài này là trường hợp thường gặp thường sẽ có guideline quy chuẩn, không cần phải escalation 
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn duy nhất cho ảnh class_id + class_name |  | chọn 1 nhãn theo Guideline | Kiểm tra nhãn được chọn có đúng hay không|
| Phát hiện vật thể | Các khung hình chữ nhật bbox_xyxy, bbox_width, bbox_height, class_id  |  | dùng khung hình chữ nhật ôm sát vừa đủ vật | Kiểm tra độ ôm sát của khung, loại những phần thuawf, tìm vật bị xót |
| Instance segmentation | Tập hợp các tọa độ đỉnh polygon [x1, y1, x2, y2, ...], class_id, instance_id |  | vẽ đường đa giác chính xác theo ranh giới của từng vật thể | tra lỗi lẹm viền, lấn nền, hoặc gộp nhầm 2 cá thể chạm nhau thành 1 |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Tuyệt đối không chia sẻ, tải lên hoặc phát tán dữ liệu bài lab, ảnh gán nhãn và thông tin nội bộ dự án ra bên ngoài phạm vi hệ thống được quy định
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Báo cho coach và mentor
## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
