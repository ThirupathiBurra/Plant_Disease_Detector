# 🌿 Plant Disease Detector

An AI-powered web app for detecting plant diseases from leaf images using a trained **TensorFlow Lite model**, **Streamlit**, and **Gemini Pro API** for explanations.

---
## MODEL USED  
  --By combining the robustness of VGG16 CNNs with the contextual capabilities of Gemini AI, this app enables accessible and smart plant disease diagnostics.
## 🚀 Features

- 📸 Upload plant leaf image and detect disease instantly
- 🤖 Lightweight `.tflite` model optimized for performance
- 🧠 Gemini API integration for smart explanations about predictions
- 🖼️ Simple and clean UI built with Streamlit
- ☁️ Easily deployable on Streamlit Cloud
- 📦 Handles large model files using Git LFS

---

## 🔍 Demo

[🔗 Click here to view the live Streamlit app](https://your-streamlit-app-link)

---

## 📁 Project Structure

```
Plant_Disease_Prediction/
│
├── app.py                      # Main Streamlit app entry point
├── requirements.txt            # Python dependencies
├── .gitattributes              # Git LFS tracking file
├── .gitignore                  # Ignored files (venv, cache, etc.)
├── README.md                   # Project documentation
│
├── model/
│   └── plant_disease_model.tflite  # Trained TensorFlow Lite model (Git LFS)
│
├── pages/
│   └── gemini_explainer.py     # Page to show AI-generated explanations
│
└── utils/
    └── image_utils.py          # Helper functions for image preprocessing
```

---

## 🔧 Installation (For Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/ThirupathiBurra/Plant_Disease_Detector.git
cd Plant_Disease_Detector
```

### 2. Install dependencies

Make sure you have Python 3.10 installed. Then install required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🌐 Deployment on Streamlit Cloud

> Streamlit Cloud automatically installs packages from `requirements.txt`.

**Make sure:**

- Your `.tflite` model is tracked via Git LFS
- The `.gitattributes` file includes:

```
*.tflite filter=lfs diff=lfs merge=lfs -text
```

---

## 🤖 Gemini API Integration

This project uses **Google's Gemini Pro API** to provide AI-generated explanations for the predicted plant disease.

To use it:

1. [Get your Gemini API Key](https://aistudio.google.com/app/apikey)
2. Create a file named `.env` and add:

```
GEMINI_API_KEY=your_api_key_here
```

3. Make sure your `gemini_explainer.py` uses `os.getenv("GEMINI_API_KEY")` securely.

---

## 🧠 Model

- Trained on a dataset of plant leaf images
- Converted to `.tflite` for faster inference and smaller size
- Stored and versioned using Git LFS

---

## 📝 Tech Stack

- Python 3.10
- TensorFlow Lite
- Streamlit
- Gemini Pro API
- Git & Git LFS

---

## 🙋‍♂️ Author

**Thirupathi Burra**

- 🔗 [GitHub](https://github.com/ThirupathiBurra)
- 💼 [LinkedIn](https://www.linkedin.com/in/thirupathi-burra-49658b2a6)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 💡 Contributing

We welcome contributions! Check out the [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

