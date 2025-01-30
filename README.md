# PCNet

## Hybrid CNN-Transformer Model for Predicting Failure Properties of Asphalt Binder from Fracture Surface Images

This repository provides a **Hybrid CNN-Transformer Model** to predict the **Strength** and **Ductility** of asphalt binders using fracture surface images. The models were trained as part of the research paper titled:

> "Hybrid CNN-Transformer Model for Predicting Failure Properties of Asphalt Binder from Fracture Surface Images"
> (Under review at the *Journal of Computer-Aided Civil and Infrastructure Engineering*)

## 🔍 Inference: Predicting Strength and Ductility

The models for inference are available under the **Releases** section:
- [PCNet_Strength.pth](https://github.com/BabakAsadi94/PCNet/releases/download/v1.1/PCNet_Strength.pth) – Predicts **Strength**
- [PCNet_Ductility.pth](https://github.com/BabakAsadi94/PCNet/releases/download/v1.0/PCNet_Ductility.pth) – Predicts **Ductility**

### 1️⃣ **Setup Environment**
First, install the required dependencies:

```bash
pip install torch torchvision timm pandas pillow pyyaml
