# Guide to your own artificial intelligence app in 3 steps

Small example AI application. It includes dataset of dog and cat images I used to train the model,
created model in json file, and model weights that can be loaded and used in an application for predictions.
It uses `keras` library for deep learning, Flask for running the app and React for the frontend.

## 1) Clone repo and install requirements from requirements.txt
`git clone `  
`pip install -r requirements.txt`

## 2) Train a model
Model is already trained and stored in `model_new.json` file with its weights stored in `cats_dogs_model.h5` file.
You can use these, or run `train.py` again to see model training in process. Before doing this, you will need to create your own dataset in data/train and data/validation folders.

## 3) Run the backend for predicitions
`python app.py`

## 4. Run frontend
`cd /frontend`  
`npm install`  
`npm start`  

### 4.1 Visit `http://localhost:3000` to test the trained model by uploading an image


