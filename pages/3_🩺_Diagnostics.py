"""Page to make predictions and diagnostics"""

from typing import Union, AnyStr
from pathlib import Path
import io

import librosa
import numpy as np
import streamlit as st
from soundlit import AudioWidget
import plotly.express as px

from utils import GIF_DIR, IMAGES_DIR
from utils import ExtraForm
from pipeline import (
    TabularPreprocessor,
    AudioPreprocessor,
    PredictionSession
)


# --- PAGE CONFIG ---
PAGE_TITLE: str = "Diagnostics"
PAGE_ICON: str = "🩺"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON
)


# --- CONSTANTS ---
# Models
classes = ["artifact", "healthy", "abnormal"]
uno_session = PredictionSession("cardionetv2uno.onnx", classes=classes)
# multimodal_session = PredictionSession("cardionetv2multi.onnx", classes=classes) # TODO: refit or del model

# Data loading and preprocessing
audio_widget = AudioWidget(min_duration=10, max_duration=60)
tabular_preprocessor = TabularPreprocessor()
audio_preprocessor = AudioPreprocessor(duration=20, n_mels=128, n_mfcc=128)


# --- FUNCTIONS ---
def plot_predictions(predict: dict[str, float]) -> None:
    classes, probs = list(predict.keys()), list(predict.values())
    fig = px.bar(x=classes, y=probs)
    fig.update_layout(
        xaxis_title="Classes",
        yaxis_title="Probabilities"
    )
    st.plotly_chart(
        fig, theme="streamlit", use_container_width=True,
        labels=dict(x="Classes", y="Probabilities")
    )


def get_indicates(predicted: str) -> str:
    if predicted.lower() in ["healthy", "normal"]:
        return "deviations from the norm, which can be either symptoms of serious heart disease or a temporary phenomenon"
    return "the absence of deviations from the norm in the cardiovascular system"


def classification_report(predicted: str, prob: float) -> None:
    col1, col2 = st.columns([0.5, 0.5], gap="large")
    indicates = get_indicates(predicted)
    prob = str(np.round(prob, 2))
    col1.write(
        f"""
        <div class="alert alert-block alert-info" style="font-size:20px; background-color: #0b0e22; font-family:verdana; color: #ffffff; border-radius: 10px; border: 0px #533078 solid">
            Currently, a recording of your heartbeat has a probability of <b>{prob}</b> indicates {indicates}.
            <br>In any case, for a more reliable result, you need to take measurements three times a day for at least 3-5 days.<br>
            It is important to remember that we do not have a medical license and cannot give recommendations or make diagnoses.
        </div>
        """,
        unsafe_allow_html=True
    )
    col2.image(f"{GIF_DIR}/status-{predicted}.gif")


def artifact_report() -> None:
    st.warning(
        "How noisy! Heartbeat sounds were not found in the file you uploaded. "
        "Try recording again.", icon="😮"
    )
    st.markdown(
        """
        There may have been a lot of extraneous noise during the recording. 
        ### Try:
        - Find a quiet place
        - Make sure there is no extraneous noise
        - Attach the phone closer to the body as shown in the picture
        - If all else fails, try recording audio from another device
        """
    )
    st.image(f"{IMAGES_DIR}/phone-body-location.png")


def load_audio() -> np.ndarray:
    st.image(f"{GIF_DIR}/circle.gif")
    if (audio := audio_widget.get_audio()) is not None:
        return audio_preprocessor(audio)


def predict(audio: np.ndarray) -> dict[str, float]:
    #is_multimodal = st.radio(
    #    label="Do you want to fill out a medical card?",
    #    options=["Default", "Yes", "No"],
    #    help="More information about your health can significantly improve your prognosis."
    #)

    #if is_multimodal.lower() == "no":
     with st.spinner("Please wait... We examine your heart 🫀"):
        return uno_session(audio)
    
    #if is_multimodal.lower() == "yes":
    #    form = ExtraForm()
    #    if (tabular := form.get_all_form()) is not None:
    #        tabular = tabular_preprocessor(**tabular)
    #        with st.spinner("Please wait... We examine your heart 🫀"):
    #            return multimodal_session(audio, tabular)


if (audio := load_audio()) is not None:
    if (outs := predict(audio)) is not None:
        predict, outputs = outs
        if predict == "artifact":
            artifact_report()
        else:
            classification_report(predict, outputs[predict])
            plot_predictions(outputs)
