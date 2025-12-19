import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import requests

# --- 設定 ---
st.set_page_config(page_title="AI画像判定アプリ", layout="centered")
st.title("📸 AI画像判定アプリ")
st.write("画像をアップロードすると、AI（ResNet18）が何が写っているか推論します。")

# --- モデルの読み込み ---
@st.cache_resource # モデルをキャッシュして高速化
def load_model():
    # 学習済みモデルをロード
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval()
    return model

# ImageNetのラベル（クラス名）を取得
@st.cache_data
def get_labels():
    url = "https://raw.githubusercontent.com/prakhar1989/LabelsForImageNet/master/resnet18_labels.txt"
    response = requests.get(url)
    labels = eval(response.text)
    return labels

model = load_model()
labels = get_labels()

# --- 画像の前処理定義 ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- UI部分 ---
uploaded_file = st.file_uploader("画像をアップロードしてください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='アップロードされた画像', use_container_width=True)
    
    st.write("推論中...")
    
    # 予測の実行
    img_t = preprocess(image)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        out = model(batch_t)
    
    # スコアの高い順に並び替え
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    # 結果表示
    st.subheader("結果:")
    for idx in indices[0][:3]: # 上位3つを表示
        st.write(f"**{labels[idx]}**: {percentage[idx].item():.2f}%")