This repository provides a **Hybrid CNN-Transformer Model** to predict the **Strength** and **Ductility** of asphalt binders using fracture surface images. The models were trained as part of the research paper titled:

> "Hybrid CNN-Transformer Model for Predicting Failure Properties of Asphalt Binder from Fracture Surface Images"  
> *(Under review at the Journal of Computer-Aided Civil and Infrastructure Engineering)*

## 🔍 Inference: Predicting Strength and Ductility

The models for inference are available under the **Releases** section:
- [PCNet_Strength.pth](https://github.com/BabakAsadi94/PCNet/releases/download/v1.1/PCNet_Strength.pth) – Predicts **Strength**
- [PCNet_Ductility.pth](https://github.com/BabakAsadi94/PCNet/releases/download/v1.0/PCNet_Ductility.pth) – Predicts **Ductility**

### 1️⃣ **Setup Environment**

First, install the required dependencies:

```bash
pip install torch torchvision timm pandas pillow pyyaml
```


### 2️⃣ **Prepare Input Data**

Place your fracture surface images inside a directory (e.g., `Testing_images`).

Each sample requires two images:

- **Bottom surface**: `<sample_name>-B.png` 
- **Top surface**: `<sample_name>-T.png` 

💡 Training images can be generated using `preprocess.py` to provide standardized images as model input.

📂 **Example file structure:**

```bash
Testing_images/
├── M1-B.png
├── M1-T.png
├── M2-B.png
├── M2-T.png
```



### 3️⃣ ** Run Inference for Ductility**

Run the following command to predict **Ductility**:
```bash
python scripts/inference.py \
  --model_path PCNet_Ductility.pth \
  --image_bottom Testing_images/M1-B.png \
  --image_top Testing_images/M1-T.png
```
