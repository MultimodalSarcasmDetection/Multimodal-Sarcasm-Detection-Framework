# Core
import os
import re
import cv2
import numpy as np
import pandas as pd

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Transformers
import torch
from numpy import matlib
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    AutoProcessor,
    AutoModelForVision2Seq
)

# Vision
from torchvision import transforms
from PIL import Image

# RegionViT
import timm

# OCR
from torchvision import models

from Global_Vars import Global_Vars


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)


# Text Preprocessing
an = 1
if an == 1:
    text_processor = TextPreprocessor()
    processed_text = text_processor.clean_text("I absolutely LOVE waiting in traffic for hours!!!")
    print(processed_text)

# Caption Generation
an = 1
if an == 1:
    class ImageCaptionGenerator:
        def __init__(self, model_name="Qwen/Qwen-VL-Chat"):
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

        def generate_caption(self, image_path):
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            return caption


    caption_model = ImageCaptionGenerator()
    caption = caption_model.generate_caption("sample.jpg")
    print(caption)


# Ocr Detection
an = 0
if an == 1:
    class YOLOv5_CRNN_OCR:
        def __init__(self):
            self.yolo = torch.hub.load(
                'ultralytics/yolov5', 'custom', path='yolov5_text.pt'
            )
            self.crnn = models.resnet18(pretrained=True)
            self.crnn.fc = torch.nn.Linear(512, 37)

        def extract_text(self, image_path):
            img = Image.open(image_path)
            detections = self.yolo(img)
            ocr_text = []

            for *box, conf, cls in detections.xyxy[0]:
                x1, y1, x2, y2 = map(int, box)
                crop = img.crop((x1, y1, x2, y2))
                crop = transforms.ToTensor()(crop).unsqueeze(0)
                pred = self.crnn(crop)
                ocr_text.append(str(torch.argmax(pred)))

            return " ".join(ocr_text)

    ocr_model = YOLOv5_CRNN_OCR()
    ocr_output = ocr_model.extract_text("sample.jpg")
    print(ocr_output)


# Feature Extraction -DistilBERT
an =0
if an == 1:
    class DistilBERTFeatureExtractor:
        def __init__(self):
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        def extract(self, text):
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)


    bert_extractor = DistilBERTFeatureExtractor()

    F1 = bert_extractor.extract(processed_text)
    F2 = bert_extractor.extract(caption)
    F3 = bert_extractor.extract(ocr_output)

    print(F1.shape, F2.shape, F3.shape)

# Feature Extraction - RegionViT
an = 1
if an == 1:
    class RegionViTFeatureExtractor:
        def __init__(self):
            self.model = timm.create_model(
                "regionvit_small_224", pretrained=True, num_classes=0
            )
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        def extract(self, image_path):
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.model(image)
            return features


    regionvit_extractor = RegionViTFeatureExtractor()
    F4 = regionvit_extractor.extract("sample.jpg")
    print(F4.shape)

no_of_dataset = 2

# Optimization for Classification
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feature = np.load('Final_Features_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Feat = Feature
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 2
        xmin = matlib.repmat([5, 50, 0.01, 50], Npop, 1)
        xmax = matlib.repmat([255, 100, 0.99, 50], Npop, 1)
        fname = objfun_feat()
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 10

        print("PSO...")
        [bestfit, fitness, bestsol1, time] = DHOA(initsol, fname, xmin, xmax, Max_iter)

        print("GWO...")
        [bestfit, fitness, bestsol2, time] = Jaya(initsol, fname, xmin, xmax, Max_iter)

        print("EFO...")
        [bestfit, fitness, bestsol3, time] = BOA(initsol, fname, xmin, xmax, Max_iter)

        print("WSA...")
        [bestfit, fitness, bestsol4, time] = ASBA(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed...")
        [bestfit, fitness, bestsol5, time] = Proposed(initsol, fname, xmin, xmax, Max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol.npy', BestSol)


# Classification
an = 0
if an == 0:
    for n in range(no_of_dataset):
        Selected_Features = np.load('Final_Features_' + str(n + 1) + '.npy', allow_pickle=True)[:100]
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)[:100]
        BestSol = np.load('BestSol.npy', allow_pickle=True)
        Learnper = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        Eval_all = []
        for i in range(len(Learnper)):
            learnperc = round(Selected_Features.shape[0] * Learnper[i])
            Train_Data = Selected_Features[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Selected_Features[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            Eval = np.zeros((11, 14))
            for j in range(BestSol.shape[0]):
                print(i, j)
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred = Model_RNN_DNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5, :], pred = Model_GRU(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred = Model_DNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :], pred = Model_RNN_DNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[10, :], pred = Model_RNN_DNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval_all.append(Eval)
        np.save('Eval_all.npy', Eval_all)

