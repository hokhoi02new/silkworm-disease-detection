# 🐛 Silkworm Disease Detection using Deeplearning and Computer vision technology

## 🚀 Introduction
This project building a system to automatically detect silkworm disease to help people who working in the silkworm/sericulture industry. The goal is to reduce the manual effort required for identifying diseased silkworms, which is labor intensive, time-consuming, and sometimes may miss detecting diseased silkworms.

We propose an enhanced method by fine-tuning U-Net architecture using ResNet34 as a encoder and a BCE Dice loss combined to solve class imbalance problem. This approach achievd 95.59% accuracy, 0.8730 Dice score and 0.7746 IoU. In addition, we provide a dataset called SilkLDP, contains 4,063 pixel-level labeled images for this task.










---

## 📌 Features
- 🧠 **Deep Learning Model**: We build and fine-tune some model for silkworm disease segmentation task.**Unet+Resnet, DeeplabV3+** → optimized for higher accuracy, **YOLOv8-seg** → suitable for speed and real-time segmentation.
- 🐛 **Dataset (SilkLDP)**: a dataset containts 4063 images labeled at pixel level for segmentation task in sericulture industry 
- 🌐 **Backend API (FastAPI)**: Provides a production-ready API for real-time silkworm disease detection.  
- 💻 **Frontend UI (Streamlit)**: A user interface to upload and view predictions.  
- 📦 **Dockerized Deployment**: To deployment on cloud services (Render)
- 🧪 **Integration test**: provieds test with Pytest

Note: Due to the dataset’s size and proprietary nature, only a sample dataset is included in this repository for demonstration. If you want to research about sericulture please contact hokhoi02new@gmail.com for full access dataset.

---

## 📌 Demo



https://github.com/user-attachments/assets/9f0aebe5-1af5-4fd8-b2c3-5a8c0a4f75b7


---

## 📂 Project Structure
```
detect_workers_without_helmets_in_construction_site/
│── app/ 		  				
│   ├── app_api.py             # FastAPI server
│   ├── app_ui.py              # Streamlit UI
│   └── integration_test.py    # Integration test
│── data/                      # dataset 
│── config/                    # Configuration file (path, img_size,…)
│── utils/                     # Utility functions (losses,…)
│── src/ 		
│   ├── data_loader.py         # load and process data  				
│   ├── models/             
│   │ ├── unet_resnet.py       # define UNet + ResNet50 model
│   │ └── deeplabv3plus.py     # define deepLabV3+ model
│   ├── train.py		       # training scripts
│   ├── evaluate.py            # model evaluation  
│   ├── inference_image.py     # image inference 
│   └── inference_video.py     # video inference
│── results/                   # result logs
│── saved_models/              # saved trained model (empty here)
│   ├── unet_resnet.h5           
│   ├── deeplabv3+.h5      
│   └── YOLO.pt    
│── requirements.txt 
│── README.md

```
⚠️ Note: The `save_models/` folders are empty in this repository because the files are too large for GitHub.  
You can download them from the following links: https://drive.google.com/drive/folders/1ZiRRvXXMj06FNmfFI6rvsmggL0KRE7U-?usp=drive_link

---

## ⚙️ Usage
#### Clone repo & install dependencies

```bash
git clone https://github.com/<username>/silkworm-disease-segmentation.git
cd silkworm-disease-segmentation
pip install -r requirements.txt
```

### 🚀 Traning
Training model:
```bash
python src/train.py --model_name unet_resnet --epochs 50 --batch_size 16
```
--model_name: choose model to train

### 📊 Evaluation
Evaluate on test set:
```bash
python src/evaluate.py --model_name unet_resnet --img_dir data/test/images --mask_dir data/test/masks
```

### 🔍 Inference
Inference on image:
```bash
python src/inference_image.py --model unet_resnet --image_path data/test_sample/test_image.jpg
```
Inference on video:
```bash
python src/inference_video.py --video data/test_sample/test_video.mp4 --output results/output.mp4
```

### 🌐 API (FastAPI):
Run API server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Server will be run at: http://127.0.0.1:8000

### 💻 UI (Streamlit):
Run UI:
```bash
streamlit run app/app_ui.py
```
UI app will be run at: http://localhost:8501


### ✅ Testing
Run integration tests:
```bash
pytest app/integration_test.py -v
```

### 🐳 Docker
```bash
# Build Docker image
docker build -t silkworm-api -f Dockerfile .

# Run container locally
docker run -p 8000:8000 silkworm-api
```
The API will be available at: http://localhost:8000
Test health check: curl http://localhost:8000/health


---

## 📜 License
This project is released under the MIT License

