import gradio as gr
import joblib
import numpy as np

# Load your trained model (Make sure 'model.pkl' is present in the same folder)
model = joblib.load("iris_classifier.pkl")
iris_species = ['Setosa', 'Versicolor', 'Virginica']


# Modify this function based on your model input/output
def predict(sepal_length, sepal_width, petal_length, petal_width):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)[0]
    
    return f"Iris class: {iris_species[prediction]}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs="text"
)

if __name__ == "__main__":
    iface.launch()