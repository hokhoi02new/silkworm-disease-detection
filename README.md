# SILKWORM DISEASES DETECTION USING DEEPLEARNING AND COMPUTER VISION TECHNOLOGY
### PHÁT HIỆN TẰM BỆNH SỬ DỤNG HỌC SÂU VÀ CÔNG NGHỆ THỊ GIÁC MÁY TÍNH
### 1) Giới thiệu

Dự án này là đề tài khóa luận tốt nghiệp của tôi, dự án này thực hiện việc phát hiện các con tằm bệnh trên hình ảnh, thuộc bài toán segmentation.
Bộ dữ liệu được tôi tự thu thập bao gồm 4063 ảnh và gán nhãn cấp độ pixel. Dự án sử dụng các mô hình học sâu (U-Net, DeepLab, YOLO,...) để phân đoạn các con tằm bệnh trên ảnh các con tằm, nhằm phát hiện con tằm bệnh và để có phương pháp xử lý kịp thời.

### 2) Cấu trúc
silkworm_diseases_detection/

├── dataset/ # dataset (images and annotations)

│   ├── images.zip/              # ảnh các con tằm

│   ├── masks.zip/               # ảnh mask label

│   ├── json/              # json data form

│   ├── guildline.docx/               #guilinde gán nhãn

│   ├── demo_annotate_data/               #video hướng dẫn gán nhãn

├── build_training_model/ 

│   ├── build_training_model.ipynb #Jupyter notebooks để build và tranining model

│   ├── data_argumentation.ipynb 

├── model_save/    #file lưu model 

│   ├──model.h5 

├── src/      

│   ├──demo.py #file demo


### 3) Ý tưởng
Những con tằm bệnh trên nong khi đến mua ăn rỗi nếu không nhặt bỏ sẽ lây sang những con tằm khỏe mạnh khác, ảnh hưởng đến sản lượng thu hoạch

Hiện tại, việc phát hiện tằm bệnh phụ thuộc vào việc kiểm tra thủ công của người nông dân. Tốn nhiều thời gian, tiền bạc, công sức, đôi khi nhặt sót thiếu.

<img width="350" height="450" alt="image" src="https://github.com/user-attachments/assets/6d4391f0-4973-49a9-932e-3cc495ad3c16" />

Cần một giải pháp tự động hóa hiệu quả, chính xác, để phát hiện tằm bệnh. Giúp can thiệp và có biện pháp xử lý sớm, giảm thiểu số lượng kén thất thoát mỗi đợt nuôi.

Là một người sống trong gia đình làm nghề trồng dâu nuôi tằm lâu năm ở tỉnh Lâm đồng, tôi mong muốn ứng dụng kiến thức kỹ năng về công nghệ đã học để giúp đỡ cải thiện ngành nghề nông nghiệp then chốt ở vùng này.



### 4) Bộ dữ liệu
Bộ dữ liệu sử dụng trong dự án bao gồm các hình ảnh của con tằm ở các giai đoạn tình trạng sức khỏe khác nhau gồm 4000 tấm ảnh, được chụp bằng Iphone X với kích thước hình ảnh thu được là 1920 x 2560 pixel, tất cả các hình ảnh con tằm đều được thu thập trong môi trường thực tế. 

Mỗi hình ảnh được gán nhãn chi tiết (mức độ pixel) cho các con tằm bệnh bằng công cụ Roboflow. Mỗi hình ảnh đều có độ phân giải cao và được chụp trong nhiều điều kiện ánh sáng và góc chụp khác nhau, chỗ nuôi khác nhau (ở cả nong và sàn) nhằm tăng tính đa dạng và tính đại diện của bộ dữ liệu.

Note:
The full dataset is custom-collected and not publicly available due to its size and proprietary nature. A sample dataset is included for demonstration.

- **Sample Dataset**: Available in 'images.zip' and 'masks.zip' (contains 30 images and 30 masks).
- **Full Dataset Access**: Contact hokhoi02new@gmail.com for access.


### 5) Công nghệ
Keras, tensorflow, Opencv,....

Kỹ thuật data argumentation (xoay ảnh, lật ảnh, crop, tăng độ sáng,...)

Chúng tôi sử dụng một số mô hình DeepLabV3+, encoder-decoder U-net, YOLO, kết hợp U-net+VGG16, Unet+ResNET34,... để giải quyết bài toán image segmentation và đạt được Dice score 0.873, IoU 0.774 với mô hình Unet+Resnet34. 

Input: ảnh gồm các con tằm 

<img width="350" height="450" alt="image" src="https://github.com/user-attachments/assets/49d4a8e5-ff54-48ed-97b3-4864c6924447" />

Output: detect ra các con tằm nào bị bệnh

<img width="350" height="450" alt="image" src="https://github.com/user-attachments/assets/fa1f9aec-becf-4314-868f-a7781110c940" />


### 6) Yêu cầu hệ thống 
Python 3.x 
Thư viện cần cài đặt: pip install -r requirements.txt

### License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
See the [LICENSE](./LICENSE) file for details.

