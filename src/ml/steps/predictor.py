import numpy as np

def predictor(model, np_img: np.array) -> dict:

    # Run inference
    prediction = model.predict(np_img)

    return prediction