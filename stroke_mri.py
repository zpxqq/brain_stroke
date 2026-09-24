## ============================================================
# 🧠 STROKE MRI ANALYSIS
# GOOGLE COLAB + STREAMLIT + PUBLIC LINK
#
# DWI + ADC + FLAIR
# Только ишемический инсульт
# МОДЕЛЬ ПОКА НЕ ПОДКЛЮЧЕНА
#
# Центральное изображение = FLAIR
# Будущая маска будет отображаться только на FLAIR
# ============================================================


import os
import re
import time
import socket
import subprocess
from pathlib import Path
import requests

!pip -q install -U streamlit pillow
# ============================================================
# НАСТРОЙКИ
# ============================================================
APP = "/content/brain_stroke/app.py"
PORT = 8501

STREAMLIT_LOG = "/content/streamlit.log"
TUNNEL_LOG = "/content/tunnel.log"


# ============================================================
# 1. СОЗДАЁМ APP.PY
# ============================================================

app_code = r'''
import os
import sys
import tempfile



import streamlit as st
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="МРТ Анализ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 50% 15%,
            rgba(0,255,240,0.05),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #061111 0%,
            #0a1b1b 50%,
            #061010 100%
        );
    color: #efffff;
}

.block-container {
    max-width: 1500px;
    padding: 22px 30px 35px 30px;
}

.top-space {
    height: 5px;
}

h1, h2, h3 {
    color: #efffff !important;
}

[data-testid="stFileUploader"] section {
    background: #0b2020 !important;
    border: 1px dashed #2d6664 !important;
    border-radius: 5px !important;
}

[data-testid="stFileUploader"] button {
    background: #102f2f !important;
    color: #72fff6 !important;
    border: 1px solid #367875 !important;
}

.stButton > button {
    background: #102f2f;
    color: #72fff6;
    border: 1px solid #397875;
    border-radius: 4px;
    min-height: 42px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #173c3c;
    color: white;
    border-color: #72fff6;
}

.viewer-box {
    background: #020505;
    border: 1px solid #28504f;
    padding: 10px;
    min-height: 580px;
}

.result-box {
    background: #0d2222;
    border: 1px solid #274d4d;
    padding: 16px;
    margin-bottom: 12px;
}

.result-title {
    color: #72fff6;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
}

.muted {
    color: #718b89;
    font-size: 11px;
    line-height: 1.6;
}

.green-text {
    color: #85fff7;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# моделька
# ============================================================
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.ndimage import label as connected_components

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ResampleToMatchd,
    NormalizeIntensityd,
    ConcatItemsd,
    EnsureTyped,
)

from monai.inferers import sliding_window_inference


sys.path.append("/content/brain_stroke")

from models.unet3d import create_unet3d


MODEL_PATH = (
    "/content/drive/MyDrive/AI_brain_data/"
    "models/best_unet3d.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_model():
    model = create_unet3d()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model
    
inference_transform = Compose([
    LoadImaged(
        keys=["dwi", "adc", "flair"]
    ),

    EnsureChannelFirstd(
        keys=["dwi", "adc", "flair"]
    ),

    Orientationd(
        keys=["dwi", "adc", "flair"],
        axcodes="RAS"
    ),

    ResampleToMatchd(
        keys=["dwi", "adc"],
        key_dst="flair",
        mode=("bilinear", "bilinear")
    ),

    NormalizeIntensityd(
        keys=["dwi", "adc", "flair"],
        nonzero=True,
        channel_wise=True
    ),

    ConcatItemsd(
        keys=["dwi", "adc", "flair"],
        name="image",
        dim=0
    ),

    EnsureTyped(
        keys=["image", "flair"]
    )
])
   def predict(dwi_path, adc_path, flair_path):

    data = {
        "dwi": dwi_path,
        "adc": adc_path,
        "flair": flair_path,
    }

    data = inference_transform(data)

    image = data["image"].unsqueeze(0).to(DEVICE)

    model = load_model()

    with torch.no_grad():

        logits = sliding_window_inference(
            inputs=image,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25
        )

        probability = torch.sigmoid(logits)

        mask = (
            probability > 0.5
        ).float()

    mask = mask[0, 0].cpu().numpy()

    flair_tensor = data["flair"]
    flair_volume = flair_tensor[0].cpu().numpy()

    return flair_volume, mask, probability[0, 0].cpu().numpy()
    
def save_uploaded_file(uploaded_file, folder):

    path = os.path.join(
        folder,
        uploaded_file.name
    )

    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return path
     
# ============================================================
# HEADER
# ============================================================
st.title("🧠 МРТ АНАЛИЗ")

st.caption(
    "DWI + ADC + FLAIR  •  Поиск признаков ишемического инсульта"
)


# ============================================================
# 3 КОЛОНКИ
# ============================================================

left, center, right = st.columns(
    [0.95, 1.65, 1.0],
    gap="medium"
)


# ============================================================
# LEFT — ЗАГРУЗКА
# ============================================================

with left:

    st.subheader("ЗАГРУЗКА МРТ")


    # --------------------------------------------------------
    # DWI
    # --------------------------------------------------------

    st.markdown(
        "**01 / DWI**"
    )

    st.caption(
        "Диффузионно-взвешенное изображение"
    )

    dwi = st.file_uploader(
    "DWI",
    type=["nii", "gz"],
    key="dwi",
    label_visibility="collapsed"
    )


    st.divider()


    # --------------------------------------------------------
    # ADC
    # --------------------------------------------------------

    st.markdown(
        "**02 / ADC**"
    )

    st.caption(
        "Карта коэффициента диффузии"
    )

    adc = st.file_uploader(
    "ADC",
    type=["nii", "gz"],
    key="adc",
    label_visibility="collapsed"
    )


    st.divider()


    # --------------------------------------------------------
    # FLAIR
    # --------------------------------------------------------

    st.markdown(
        "**03 / FLAIR**"
    )

    st.caption(
        "FLAIR изображение"
    )

    flair = st.file_uploader(
    "FLAIR",
    type=["nii", "gz"],
    key="flair",
    label_visibility="collapsed"
    )


    st.divider()


    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    if st.button(
    "ЗАПУСТИТЬ АНАЛИЗ",
    use_container_width=True
):

    if dwi is None or adc is None or flair is None:

        st.error(
            "Необходимо загрузить DWI, ADC и FLAIR."
        )

    else:

        with st.spinner("Модель анализирует МРТ..."):

            with tempfile.TemporaryDirectory() as temp_dir:

                dwi_path = save_uploaded_file(
                    dwi,
                    temp_dir
                )

                adc_path = save_uploaded_file(
                    adc,
                    temp_dir
                )

                flair_path = save_uploaded_file(
                    flair,
                    temp_dir
                )

                flair_volume, mask, probability = predict(
                    dwi_path,
                    adc_path,
                    flair_path
                )

                st.session_state["result"] = {
                    "flair": flair_volume,
                    "mask": mask,
                    "probability": probability
                }

        st.success("Анализ завершён")


            # ------------------------------------------------
            # ОТДЕЛЬНЫЕ СТАТУСЫ
            # ------------------------------------------------

            st.subheader(
                "СТАТУС ЗАГРУЗКИ"
            )


            if flair:

                st.success(
                    "FLAIR загружен"
                )


            if dwi:

                st.success(
                    "DWI загружен"
                )


            if adc:

                st.success(
                    "ADC загружен"
                )


            st.info(
                "Модель пока не подключена."
            )


# ============================================================
# CENTER — FLAIR
# ============================================================

with center:

    st.subheader("МРТ / Показ Изображения")

    st.markdown(
        "### FLAIR"
    )

    if "result" in st.session_state:

    result = st.session_state["result"]

    flair_volume = result["flair"]
    mask = result["mask"]

    # ищем срез, где модель нашла больше всего очага
    lesion_per_slice = mask.sum(axis=(0, 1))

    slice_id = int(
        np.argmax(lesion_per_slice)
    )

    flair_slice = flair_volume[:, :, slice_id]
    mask_slice = mask[:, :, slice_id]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(
        flair_slice.T,
        cmap="gray",
        origin="lower"
    )

    if mask_slice.any():

        ax.contour(
            mask_slice.T,
            levels=[0.5],
            colors="red",
            linewidths=2
        )

    ax.set_title(
        f"Предсказанный очаг • срез {slice_id}"
    )

    ax.axis("off")

    st.pyplot(fig)


    st.info(
        "FLAIR с контуром обнаруженного очага."
    )


    # =================================================================
    # Переменные с результатами (в будущем будут приходить из predict())
    # =================================================================
    # Переменные с результатами
    components, lesion_count = connected_components(
    mask > 0)
    lesion_size = "—"
    lesion_area = "—"
    lesion_volume = "—"
    lesion_shape = "—"
    lesion_intensity = "—"
    lesion_coords = "—"

    # Вывод результатов в интерфейс


    with right:
      st.subheader("РЕЗУЛЬТАТ")
      st.write(f"**Количество очагов:** {lesion_count}")
      st.write(f"**Размер:** {lesion_size}")
      st.write(f"**Площадь:** {lesion_area}")
      st.write(f"**Объём по серии срезов:** {lesion_volume}")
      st.write(f"**Форма:** {lesion_shape}")
      st.write(f"**Интенсивность:** {lesion_intensity}")
      st.write(f"**Координаты:** {lesion_coords}")


    # --------------------------------------------------------
    # ВАЖНЫЙ СТАТУС
    # --------------------------------------------------------

    st.markdown(
        "#### Показ Ишемического Очага"
    )

    st.info(
        "Очаг будет показываться только на FLAIR. "
        "DWI  и ADC и используются как входные исследования."
    )
    # --------------------------------------------------------
    # ЛИНИЯ ТОЧНОСТИ
    # --------------------------------------------------------
    with right:

      st.markdown("#### Качество модели")

      VAL_DICE = 0.325686704377320645
      VAL_IOU = 0.226763211233022
      pro = VAL_DICE * 100

      st.markdown(
        f"""
        <div style="
            width:100%;
            height:22px;
            background:#173737;
            border:1px solid #315b59;
            position:relative;
            overflow:hidden;
            margin-top:8px;
        ">
            <div style="
                width:{pro}%;
                height:100%;
                background:#72fff6;
            "></div>
        </div>
        """,
        unsafe_allow_html=True)

      st.markdown(
    f"**Dice на валидации: {VAL_DICE:.3f} ({VAL_DICE * 100:.1f}%)**")
        
      st.markdown(
    f"**IoU на валидации: {VAL_IOU:.3f} ({VAL_IOU * 100:.1f}%)**")

'''


# ============================================================
# СОХРАНЯЕМ
# ============================================================

Path(APP).write_text(
    app_code,
    encoding="utf-8"
)

print("✅ app.py создан")


# ============================================================
# 2. ОСТАНОВИТЬ СТАРЫЕ ПРОЦЕССЫ
# ============================================================

os.system(
    "pkill -9 -f '[s]treamlit' >/dev/null 2>&1 || true"
)

os.system(
    "pkill -9 -f '[s]sh.*localhost.run' >/dev/null 2>&1 || true"
)

os.system(
    f"fuser -k {PORT}/tcp >/dev/null 2>&1 || true"
)

time.sleep(2)


# ============================================================
# 3. ЗАПУСК STREAMLIT
# ============================================================

streamlit_log = open(
    STREAMLIT_LOG,
    "w",
    encoding="utf-8"
)

streamlit = subprocess.Popen(
    [
        "python",
        "-m",
        "streamlit",
        "run",
        APP,

        "--server.address=0.0.0.0",
        f"--server.port={PORT}",
        "--server.headless=true",

        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",

        "--browser.gatherUsageStats=false"
    ],

    stdout=streamlit_log,
    stderr=subprocess.STDOUT
)


print("🚀 Streamlit запускается...")


# ============================================================
# 4. ПРОВЕРКА STREAMLIT
# ============================================================

streamlit_ok = False

for _ in range(30):

    time.sleep(1)

    try:

        r = requests.get(
            f"http://127.0.0.1:{PORT}",
            timeout=2
        )

        if r.status_code == 200:

            streamlit_ok = True
            break

    except Exception:
        pass


if not streamlit_ok:

    print(
        "❌ Streamlit не запустился."
    )

    print(
        Path(STREAMLIT_LOG).read_text(
            encoding="utf-8",
            errors="ignore"
        )
    )

    raise RuntimeError(
        "Streamlit не отвечает"
    )


print(
    "✅ Streamlit работает"
)


# ============================================================
# 5. SSH TUNNEL LOCALHOST.RUN
# ============================================================

print(
    "🌐 Создаю публичную ссылку..."
)


tunnel_log = open(
    TUNNEL_LOG,
    "w",
    encoding="utf-8"
)


tunnel = subprocess.Popen(

    [
        "ssh",

        "-o",
        "StrictHostKeyChecking=no",

        "-o",
        "ServerAliveInterval=30",

        "-o",
        "ServerAliveCountMax=3",

        "-o",
        "ExitOnForwardFailure=yes",

        "-R",
        f"80:127.0.0.1:{PORT}",

        "nokey@localhost.run"
    ],

    stdout=tunnel_log,
    stderr=subprocess.STDOUT,

    text=True
)


# ============================================================
# 6. ИЩЕМ ПУБЛИЧНЫЙ URL
# ============================================================

public_url = None


for _ in range(60):

    time.sleep(1)

    try:

        text = Path(
            TUNNEL_LOG
        ).read_text(
            encoding="utf-8",
            errors="ignore"
        )

        # современные localhost.run
        matches = re.findall(
            r"https://[A-Za-z0-9._-]+(?:localhost\.run|lhr\.life)",
            text
        )

        if matches:

            public_url = matches[-1]

            break

    except Exception:
        pass


# ============================================================
# 7. РЕЗУЛЬТАТ
# ============================================================

print()

if public_url:

    print(
        "=============================================="
    )

    print(
        "✅ САЙТ ГОТОВ"
    )

    print(
        "=============================================="
    )

    print()

    print(
        "🌐",
        public_url
    )

    print()

    print(
        "=============================================="
    )

else:

    print(
        "❌ Публичная ссылка не получена."
    )

    print()

    print(
        "======== TUNNEL LOG ========"
    )

    print(
        Path(TUNNEL_LOG).read_text(
            encoding="utf-8",
            errors="ignore"
        )
    )
