# Fruit & Vegetable Classification with Deep Learning

A machine learning project that classifies fruits and vegetables using Convolutional Neural Networks (CNN) with TensorFlow/Keras and provides a user-friendly web interface using Streamlit.

## 🚀 Features

- **Deep Learning Model**: Custom CNN architecture for image classification
- **36 Categories**: Supports classification of 36 different fruits and vegetables
- **Web Interface**: Interactive Streamlit app for easy image classification
- **High Accuracy**: Trained model with validation accuracy tracking
- **Real-time Prediction**: Upload images and get instant predictions with confidence scores

## 📋 Supported Categories

The model can classify the following 36 fruits and vegetables:

```
apple, banana, beetroot, bell pepper, cabbage, capsicum, carrot, cauliflower, 
chilli pepper, corn, cucumber, eggplant, garlic, ginger, grapes, jalepeno, 
kiwi, lemon, lettuce, mango, onion, orange, paprika, pear, peas, pineapple, 
pomegranate, potato, raddish, soy beans, spinach, sweetcorn, sweetpotato, 
tomato, turnip, watermelon
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- pip

### Required Libraries

```bash
pip install tensorflow
pip install streamlit
pip install numpy
pip install pandas
pip install matplotlib
```

Or install all dependencies at once:

```bash
pip install tensorflow streamlit numpy pandas matplotlib
```

## 📁 Project Structure

```
Fruit-Classification/
│
├── app.py                          # Streamlit web application
├── train_model.py                  # Model training script
├── Image_classify.keras            # Trained model file
├── README.md                       # Project documentation
│
├── Fruits_Vegetables/              # Dataset directory
│   ├── train/                      # Training images
│   ├── test/                       # Testing images
│   └── validation/                 # Validation images
│
└── sample_images/                  # Sample images for testing
    ├── Apple.jpg
    ├── corn.jpg
    └── ...
```

## 🎯 Usage

### Running the Web Application

1. Clone the repository:
```bash
git clone https://github.com/PreetishMajumdar/Simple-Fruit-Classification.git
cd Simple-Fruit-Classification
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

4. Enter the image filename (e.g., `Apple.jpg`) in the text input field

5. View the prediction results with confidence score

### Training Your Own Model

If you want to retrain the model with your own dataset:

1. Organize your dataset in the following structure:
```
Fruits_Vegetables/
├── train/
│   ├── apple/
│   ├── banana/
│   └── ...
├── test/
│   ├── apple/
│   ├── banana/
│   └── ...
└── validation/
    ├── apple/
    ├── banana/
    └── ...
```

2. Run the training script:
```bash
python train_model.py
```

## 🧠 Model Architecture

The CNN model consists of:

- **Input Layer**: 180x180x3 RGB images
- **Preprocessing**: Rescaling pixel values to [0,1]
- **Convolutional Layers**: 
  - Conv2D (16 filters, 3x3 kernel)
  - Conv2D (32 filters, 3x3 kernel)  
  - Conv2D (64 filters, 3x3 kernel)
- **Pooling**: MaxPooling2D after each conv layer
- **Regularization**: Dropout (0.2)
- **Dense Layers**: 128 neurons + output layer (36 classes)
- **Activation**: ReLU for hidden layers, Softmax for output

## 📊 Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 25
- **Batch Size**: 32
- **Image Size**: 180x180 pixels

## 📈 Performance

The model tracks both training and validation accuracy/loss during training. Training history is visualized with matplotlib plots showing:

- Training vs Validation Accuracy
- Training vs Validation Loss

## 🔧 Customization

### Adding New Categories

1. Add new fruit/vegetable folders to your dataset
2. Update the `data_cat` list in `app.py`
3. Retrain the model with the updated dataset

### Modifying Model Architecture

Edit the model definition in `train_model.py`:

```python
model = Sequential([
    layers.Rescaling(1./255),
    # Add/modify layers here
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    # ... rest of the architecture
])
```

## 📝 API Reference

### Key Functions

- `tf.keras.utils.load_img()`: Load and preprocess images
- `model.predict()`: Generate predictions
- `tf.nn.softmax()`: Convert logits to probabilities
- `np.argmax()`: Get class with highest probability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- Streamlit for the web interface
- Dataset contributors for providing training data

## 📞 Contact

For questions or suggestions, please open an issue on GitHub or contact the project maintainer.

---

**Note**: Make sure to place your image files in the same directory as the app.py file, or provide the full path to the image when using the Streamlit interface.