import streamlit as st
import numpy as np
import pickle

# LOAD MODEL
model = pickle.load(
    open(
        "knn_classifier/models/knn_classifier.pkl",
        "rb"
    )
)

# LOAD SCALER
scaler = pickle.load(
    open(
        "knn_classifier/models/scaler.pkl",
        "rb"
    )
)
