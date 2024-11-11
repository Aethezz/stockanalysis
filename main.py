import requests
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from datetime import datetime, timedelta, timezone
from flask import Flask, send_file
import io 
from get.data import get_chart
from get.sentiment import get_sentiment

TF_ENABLE_ONEDNN_OPTS = 0
API_KEY = "68221ded8dmsh6c87339388bb8d9p16f55djsn5e0f84453dcb"

def get_prediction(sentiment, data, prediction):
    pass

def main():
    get_chart(symbol=None)
    get_sentiment()


get_chart(API_KEY, symbol="AAPL")
image = cv2.imread()