# Personalized Shopping Experience via Facial Recognition

## 📌 Project Overview
This project implements a real-time facial recognition system designed to enhance customer service in retail environments. By identifying registered customers via a webcam feed, the system displays personalized welcome messages and shopping recommendations instantly.

## 👥 Team Members & Roles
* **Bishwa Thakuri:** Data & Encoding Lead (Database creation & management)
* **Dipendra Shrestha:** Live Capture & Detection Lead (Webcam integration)
* **Prabin Shrestha:** Core Recognition & Matching Lead (Algorithm implementation)
* **Richesh Khatri:** Integration & UI/Personalization Lead (Interface & Logic)

## 🛠️ Technologies Used
* Python 3.10+
* OpenCV (Computer Vision)
* Face_Recognition (Dlib-based AI model)
* NumPy

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BishwaThakuri/Face-Recognition-System.git
    cd Face-Recognition-System
    ```

2.  **Install Dependencies:**
    *(Note: CMake and Visual Studio Build Tools C++ are required for dlib)*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data:**
    * Add photos of team members to the `photos/` directory.
    * Run the training script to generate the database:
        ```bash
        python create_encodings.py
        ```

## 🚀 How to Run
Run the main application script:
```bash
python main.py
```

## 📂 Project Structure
- `main.py`: The primary application for real-time recognition.
- `create_encodings.py`: Helper script to train the model on new images.
- `known_encodings.pkl`: The serialized database of face encodings.