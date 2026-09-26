import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import tempfile
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from scipy.ndimage import label as connected_components
from scipy.ndimage import center_of_mass
from monai.inferers import sliding_window_inference
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

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from models.unet3d import create_unet3d


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="МРТ Анализ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = str(PROJECT_DIR / "weights" / "best_unet3d.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
VAL_DICE = 0.325686704377320645
VAL_IOU = 0.226763211233022
THRESHOLD = 0.5


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 50% 15%, rgba(0,255,240,0.05), transparent 30%),
            linear-gradient(135deg, #061111 0%, #0a1b1b 50%, #061010 100%);
        color: #efffff;
    }

    .block-container {
        max-width: 1500px;
        padding: 22px 30px 35px 30px;
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL
# ============================================================
@st.cache_resource
def load_model():
    device = torch.device(DEVICE)

    model = create_unet3d()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, device


# ============================================================
# INFERENCE PREPROCESSING
# ============================================================
load_transform = Compose([
    LoadImaged(keys=["dwi", "adc", "flair"]),
    EnsureChannelFirstd(keys=["dwi", "adc", "flair"]),
    Orientationd(
        keys=["dwi", "adc", "flair"],
        axcodes="RAS"
    ),
])

dwi_resample = ResampleToMatchd(
    keys=["dwi"],
    key_dst="flair",
    mode="bilinear",
)

adc_resample = ResampleToMatchd(
    keys=["adc"],
    key_dst="flair",
    mode="bilinear",
)

finish_transform = Compose([
    NormalizeIntensityd(
        keys=["dwi", "adc", "flair"],
        nonzero=True,
        channel_wise=True,
    ),
    ConcatItemsd(
        keys=["dwi", "adc", "flair"],
        name="image",
        dim=0,
    ),
    EnsureTyped(
        keys=["image", "flair", "dwi", "adc"]
    ),
])
def same_grid(image1, image2):
    shape1 = tuple(image1.shape[1:])
    shape2 = tuple(image2.shape[1:])

    affine1 = image1.affine
    affine2 = image2.affine

    if torch.is_tensor(affine1):
        affine1 = affine1.cpu().numpy()

    if torch.is_tensor(affine2):
        affine2 = affine2.cpu().numpy()

    return (
        shape1 == shape2
        and np.allclose(
            affine1,
            affine2,
            atol=1e-4
        )
    )


def save_uploaded_file(uploaded_file, folder):
    path = os.path.join(folder, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _voxel_geometry(flair_tensor):
    """Return voxel volume (mm^3) and in-plane voxel area (mm^2)."""
    affine = flair_tensor.affine
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    affine = np.asarray(affine, dtype=float)

    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)
    inplane_area_mm2 = float(voxel_sizes[0] * voxel_sizes[1])
    return voxel_volume_mm3, inplane_area_mm2


def predict(dwi_path, adc_path, flair_path):

    data = {
        "dwi": dwi_path,
        "adc": adc_path,
        "flair": flair_path,
    }

    # ========================================================
    # 1. ЗАГРУЗКА + ORIENTATION
    # ========================================================

    t0 = time.perf_counter()

    data = load_transform(data)

    t1 = time.perf_counter()


    # ========================================================
    # 2. ПЕРЕНОС PREPROCESSING НА GPU
    # ========================================================

    preprocess_device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    data["dwi"] = data["dwi"].to(preprocess_device)
    data["adc"] = data["adc"].to(preprocess_device)
    data["flair"] = data["flair"].to(preprocess_device)

    if preprocess_device.type == "cuda":
        torch.cuda.synchronize()

    t2 = time.perf_counter()


    # ========================================================
    # 3. ПРОВЕРКА ГЕОМЕТРИИ
    # ========================================================

    dwi_need_resample = not same_grid(
        data["dwi"],
        data["flair"]
    )

    adc_need_resample = not same_grid(
        data["adc"],
        data["flair"]
    )


    # ========================================================
    # 4. RESAMPLE
    # ========================================================

    if dwi_need_resample:
        data = dwi_resample(data)

    if adc_need_resample:
        data = adc_resample(data)

    # Важно для правильного измерения времени CUDA
    if preprocess_device.type == "cuda":
        torch.cuda.synchronize()

    t3 = time.perf_counter()


    # ========================================================
    # 5. NORMALIZATION + CONCAT
    # ========================================================

    data = finish_transform(data)

    if preprocess_device.type == "cuda":
        torch.cuda.synchronize()

    t4 = time.perf_counter()


    # ========================================================
    # 6. MODEL
    # ========================================================

    model, device = load_model()

    image = data["image"].unsqueeze(0)

    # Если preprocessing уже был на CUDA,
    # этот вызов практически ничего не стоит.
    image = image.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t5 = time.perf_counter()


    # ========================================================
    # 7. INFERENCE
    # ========================================================

    with torch.inference_mode():

        if device.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=(96, 96, 96),
                    sw_batch_size=2,
                    predictor=model,
                    overlap=0.25,
                )

        else:

            logits = sliding_window_inference(
                inputs=image,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )


        probability = torch.sigmoid(logits)

        mask = (
            probability > THRESHOLD
        ).float()


    if device.type == "cuda":
        torch.cuda.synchronize()

    t6 = time.perf_counter()


    # ========================================================
    # ВРЕМЯ
    # ========================================================

    st.write(
        f"Загрузка + Orientation: "
        f"{t1 - t0:.1f} сек"
    )

    st.write(
        f"Перенос на GPU: "
        f"{t2 - t1:.1f} сек"
    )

    st.write(
        f"Resample: "
        f"{t3 - t2:.1f} сек"
    )

    st.write(
        f"Normalize + объединение: "
        f"{t4 - t3:.1f} сек"
    )

    st.write(
        f"Подготовка модели: "
        f"{t5 - t4:.1f} сек"
    )

    st.write(
        f"Работа модели: "
        f"{t6 - t5:.1f} сек"
    )

    st.write(
        f"DWI resample: "
        f"{'да' if dwi_need_resample else 'не требуется'}"
    )

    st.write(
        f"ADC resample: "
        f"{'да' if adc_need_resample else 'не требуется'}"
    )


    # ========================================================
    # RESULTS
    # ========================================================

    mask_np = (
        mask[0, 0]
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    probability_np = (
        probability[0, 0]
        .float()
        .cpu()
        .numpy()
    )


    flair_volume = (
        data["flair"][0]
        .float()
        .cpu()
        .numpy()
    )

    dwi_volume = (
        data["dwi"][0]
        .float()
        .cpu()
        .numpy()
    )

    adc_volume = (
        data["adc"][0]
        .float()
        .cpu()
        .numpy()
    )


    voxel_volume_mm3, inplane_area_mm2 = (
        _voxel_geometry(
            data["flair"]
        )
    )


    return (
        flair_volume,
        dwi_volume,
        adc_volume,
        mask_np,
        probability_np,
        voxel_volume_mm3,
        inplane_area_mm2,
    )


def analyze_mask(mask, voxel_volume_mm3, inplane_area_mm2):

    labeled, raw_count = connected_components(mask > 0)

    # Быстро считаем размер всех компонент за один проход
    if raw_count > 0:
        component_sizes = np.bincount(
            labeled.ravel(),
            minlength=raw_count + 1
        )[1:]
    else:
        component_sizes = np.array([], dtype=np.int64)

    lesion_count = raw_count

    total_voxels = int(mask.sum())

    total_volume_cm3 = (
        total_voxels
        * voxel_volume_mm3
        / 1000.0
    )

    if component_sizes.size > 0:
        largest_voxels = int(component_sizes.max())

        largest_volume_cm3 = (
            largest_voxels
            * voxel_volume_mm3
            / 1000.0
        )
    else:
        largest_volume_cm3 = 0.0

    area_per_slice = (
        mask.sum(axis=(0, 1))
        * inplane_area_mm2
    )

    max_area_mm2 = (
        float(area_per_slice.max())
        if area_per_slice.size
        else 0.0
    )

    coords_text = "—"

    if total_voxels > 0:
        coords = center_of_mass(mask > 0)

        coords_text = (
            f"({coords[0]:.1f}, "
            f"{coords[1]:.1f}, "
            f"{coords[2]:.1f}) voxel"
        )

    return {
        "lesion_count": lesion_count,
        "total_volume_cm3": total_volume_cm3,
        "largest_volume_cm3": largest_volume_cm3,
        "max_area_mm2": max_area_mm2,
        "coords": coords_text,
    }


# ============================================================
# HEADER
# ============================================================
st.title("🧠 МРТ АНАЛИЗ")
st.caption("DWI + ADC + FLAIR • Сегментация возможных ишемических очагов")
st.caption(f"Устройство: {DEVICE.upper()}")
if DEVICE == "cuda":
    st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    st.warning(
        "CUDA недоступна в текущем Python-окружении. "
        "Запустите приложение командой: python -m streamlit run app.py"
    )

left, center, right = st.columns([0.95, 1.65, 1.0], gap="medium")


# ============================================================
# LEFT — UPLOAD + RUN
# ============================================================
with left:
    st.subheader("ЗАГРУЗКА МРТ")

    st.markdown("**01 / DWI**")
    st.caption("Диффузионно-взвешенное изображение")
    dwi = st.file_uploader(
        "DWI",
        type=["nii", "gz"],
        key="dwi",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**02 / ADC**")
    st.caption("Карта коэффициента диффузии")
    adc = st.file_uploader(
        "ADC",
        type=["nii", "gz"],
        key="adc",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("**03 / FLAIR**")
    st.caption("FLAIR изображение")
    flair = st.file_uploader(
        "FLAIR",
        type=["nii", "gz"],
        key="flair",
        label_visibility="collapsed",
    )

    st.divider()

    if st.button("ЗАПУСТИТЬ АНАЛИЗ", use_container_width=True):
        if dwi is None or adc is None or flair is None:
            st.error("Необходимо загрузить DWI, ADC и FLAIR.")
        else:
            try:
                with st.spinner("Модель анализирует МРТ..."):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        dwi_path = save_uploaded_file(dwi, temp_dir)
                        adc_path = save_uploaded_file(adc, temp_dir)
                        flair_path = save_uploaded_file(flair, temp_dir)

                        (
                            flair_volume,
                            dwi_volume,
                            adc_volume,
                            mask,
                            probability,
                            voxel_volume_mm3,
                            inplane_area_mm2,
                        ) = predict(dwi_path, adc_path, flair_path)

                        t_stats = time.perf_counter()

                        stats = analyze_mask(
                            mask,
                            voxel_volume_mm3,
                            inplane_area_mm2,
                        )

                        st.write(
                            f"Анализ маски: "
                            f"{time.perf_counter() - t_stats:.1f} сек"
                        )

                        st.session_state["result"] = {
                            "flair": flair_volume,
                            "dwi": dwi_volume,
                            "adc": adc_volume,
                            "mask": mask,
                            "probability": probability,
                            "stats": stats,
                        }

                st.success("Анализ завершён")
            except Exception as e:
                st.exception(e)

    if dwi is not None or adc is not None or flair is not None:
        st.subheader("СТАТУС ЗАГРУЗКИ")
        if dwi is not None:
            st.success("DWI загружен")
        if adc is not None:
            st.success("ADC загружен")
        if flair is not None:
            st.success("FLAIR загружен")


# ============================================================
# CENTER — VISUALIZATION & MODALITY SELECTOR
# ============================================================
with center:
    st.subheader("МРТ / Показ изображения")

    if "result" in st.session_state:
        result = st.session_state["result"]
        mask = result["mask"]

        # Кнопка переключения модальности
        selected_modality = st.radio(
            "Режим отображения:",
            options=["FLAIR", "DWI", "ADC"],
            horizontal=True,
            key="modality_selector"
        )

        selected_volume = result[selected_modality.lower()]

        if mask.any():
            lesion_per_slice = mask.sum(axis=(0, 1))
            slice_id = int(np.argmax(lesion_per_slice))
        else:
            slice_id = int(selected_volume.shape[2] // 2)

        vol_slice = selected_volume[:, :, slice_id]
        mask_slice = mask[:, :, slice_id]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(vol_slice.T, cmap="gray", origin="lower")

        if mask_slice.any():
            ax.contour(
                mask_slice.T,
                levels=[0.5],
                colors="red",
                linewidths=2,
            )
            ax.set_title(f"Режим: {selected_modality} • Очаг на срезе {slice_id}")
        else:
            ax.set_title(f"Режим: {selected_modality} • Маска пуста • Срез {slice_id}")

        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.info(f"Отображается: **{selected_modality}**. Красный контур — предсказанный очаг.")
    else:
        st.info("Загрузите DWI, ADC и FLAIR и нажмите «ЗАПУСТИТЬ АНАЛИЗ».")


# ============================================================
# RIGHT — RESULTS + VALIDATION METRICS
# ============================================================
with right:
    st.subheader("РЕЗУЛЬТАТ")

    if "result" in st.session_state:
        stats = st.session_state["result"]["stats"]

        st.write(f"**Количество очагов:** {stats['lesion_count']}")
        st.write(
            f"**Общий объём маски:** {stats['total_volume_cm3']:.3f} см³"
        )
        st.write(
            f"**Объём крупнейшего очага:** {stats['largest_volume_cm3']:.3f} см³"
        )
        st.write(
            f"**Макс. площадь на срезе:** {stats['max_area_mm2']:.1f} мм²"
        )
        st.write(f"**Центр маски:** {stats['coords']}")
    else:
        st.write("**Количество очагов:** —")
        st.write("**Общий объём маски:** —")
        st.write("**Объём крупнейшего очага:** —")
        st.write("**Макс. площадь на срезе:** —")
        st.write("**Центр маски:** —")

    st.divider()
    st.markdown("#### Качество модели")

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
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Dice на валидации: {VAL_DICE:.3f} ({VAL_DICE * 100:.1f}%)**"
    )
    st.markdown(
        f"**IoU на валидации: {VAL_IOU:.3f} ({VAL_IOU * 100:.1f}%)**"
    )
    st.caption(
        "Это метрики сегментации на валидационном наборе, "
        "а не вероятность диагноза для текущего пациента."
    )

st.caption(
    "Исследовательский прототип. Результат модели не является медицинским заключением."
)
