# 🌊 Smart Tsunami Mitigation: AI-Driven IoT Barrier System

## 🏆 WFIDC U-DARE 2.0 Competition Entry

### 👥 Team Sunibian from BINUS University

1. Farhan Aulianda (2702471183)
2. Abdullah Ghassan Ragheed Rachmat (2702274835)
3. Ahya Muhammad Abiyu Salam (2702341360)
---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Project Overview

This repository contains the code for our Smart Tsunami Mitigation system, which implements an Artificial Intelligence-driven and Internet of Things-based barrier system. Our project aims to revolutionize disaster preparedness and response, particularly for tsunami threats.

The system consists of two main components:
1. A Machine Learning (ML) model for predicting tsunami occurrences and severity
2. A simulator to generate synthetic sensor data and visualize the tsunami's impact

This innovative approach combines cutting-edge AI algorithms with real-time IoT sensor data to provide early warnings and adaptive response strategies, potentially saving countless lives in coastal regions vulnerable to tsunamis.

---

## 📁 Repository Structure

```
.

├── ML/
│   ├── .gitignore
│   ├── main.py
│   └── requirements.txt
├── simulator/
│   ├── .gitignore
│   ├── main.py
│   └── requirements.txt
└── README.md
```

- `ML/`: Houses the Machine Learning component of our project.
- `simulator/`: Contains the Tsunami Simulator code.

---

## 🚀 Installation

To get started with our Smart Tsunami Mitigation system, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/smart-tsunami-mitigation.git
   cd smart-tsunami-mitigation
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages for the ML component:
   ```
   cd ML
   pip install -r requirements.txt
   ```

4. Install the required packages for the simulator:
   ```
   cd ../simulator
   pip install -r requirements.txt
   ```

> **Note**: This project has been tested on Arch Linux with Python 3.12.

---

## 🖥️ Usage

### Running the Machine Learning Component

1. Navigate to the ML directory:
   ```
   cd ML
   ```

2. Run the main script:
   ```
   python main.py
   ```

This will start the ML model, which will begin processing incoming sensor data and making predictions.

### Running the Simulator

1. Navigate to the simulator directory:
   ```
   cd simulator
   ```

2. Run the main script:
   ```
   python main.py
   ```

This will launch the tsunami simulator, generating synthetic sensor data and providing a visual representation of a potential tsunami event.

---

## 🛠️ Technologies Used

- **Python**: The primary programming language used for both the ML model and the simulator.
- **TensorFlow**: Used for building and training the deep learning models.
- **Pandas & NumPy**: For data manipulation and numerical computations.
- **Matplotlib**: For data visualization in the simulator.
- **Paho MQTT**: For implementing the IoT communication protocol.
- **Tkinter**: For creating the GUI of the simulator.

---

## 🤝 Contributing

We welcome contributions to improve our Smart Tsunami Mitigation system! Here's how you can contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code adheres to our coding standards and include appropriate tests.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by Team Sunibian
</p>

<p align="center">
  <sub>Let's build a safer, more resilient future together! 🌊🛡️</sub>
</p>