### 2️⃣ **Prepare Input Data**

Place your **fracture surface images** inside a directory (e.g., `Testing_images`).

Each sample requires **two images**:

- **Bottom surface**: `<sample_name>-B.png` (or `.jpg`, `.jpeg`)
- **Top surface**: `<sample_name>-T.png` (or `.jpg`, `.jpeg`)

💡 **Training images** can be generated using [`preprocess.py`](./preprocess.py) to provide standardized images as model input.

#### 📂 Example file structure:
```bash
Testing_images/
├── M1-B.png
├── M1-T.png
├── M2-B.png
├── M2-T.png
...
