import os, sys
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.pyplot as plt
import torch

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Insert ROOT first so `from app.visualise import …` resolves to ROOT/app/visualise.py
# and NOT to this script file (app/streamlit/app.py) which Streamlit also adds to sys.path.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DATAROOT = os.path.join(ROOT, "data", "nuscenes", "v1.0-mini")

# ── visualise utilities — loaded by file path to avoid sys.path conflicts ─────
# Streamlit adds app/streamlit/ to sys.path, so `from app.visualise import …`
# would find app/streamlit/app.py as the `app` module instead of the app/ package.
# Loading by absolute path sidesteps this entirely.
import importlib.util as _ilu
_vis_spec = _ilu.spec_from_file_location(
    "urbansense_visualise",
    os.path.join(ROOT, "app", "visualise.py"),
)
_vis_mod = _ilu.module_from_spec(_vis_spec)
_vis_spec.loader.exec_module(_vis_mod)
draw_tracks   = _vis_mod.draw_tracks
draw_anomalies = _vis_mod.draw_anomalies

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UrbanSense – NuScenes Explorer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette (category → RGB) ──────────────────────────────────────────
PALETTE = {
    "human.pedestrian.adult":        (255,  69,   0),
    "human.pedestrian.child":        (255, 140,   0),
    "human.pedestrian.wheelchair":   (255, 215,   0),
    "human.pedestrian.stroller":     (255, 255,   0),
    "human.pedestrian.personal_mobility": (173, 255,  47),
    "human.pedestrian.police_officer":    (  0, 255,   0),
    "human.pedestrian.construction_worker": (  0, 200, 100),
    "vehicle.car":                   ( 30, 144, 255),
    "vehicle.motorcycle":            (  0, 191, 255),
    "vehicle.bicycle":               (135, 206, 235),
    "vehicle.bus.bendy":             (138,  43, 226),
    "vehicle.bus.rigid":             (148,   0, 211),
    "vehicle.truck":                 (199,  21, 133),
    "vehicle.construction":          (255,  20, 147),
    "vehicle.emergency.ambulance":   (220,  20,  60),
    "vehicle.emergency.police":      (178,  34,  34),
    "vehicle.trailer":               (255, 127,  80),
    "movable_object.barrier":        (210, 180, 140),
    "movable_object.debris":         (139,  90,  43),
    "movable_object.pushable_pullable": (205, 133,  63),
    "movable_object.trafficcone":    (255, 165,   0),
    "static_object.bicycle_rack":    (169, 169, 169),
    "animal":                        ( 50, 205,  50),
}
DEFAULT_COLOR = (200, 200, 200)

def cat_color(cat_name: str):
    return PALETTE.get(cat_name, DEFAULT_COLOR)

# ── load NuScenes (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NuScenes v1.0-mini …")
def load_nusc():
    from nuscenes.nuscenes import NuScenes
    return NuScenes("v1.0-mini", dataroot=DATAROOT, verbose=False)

# ── load YOLO (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8n …")
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

# ── helpers ───────────────────────────────────────────────────────────────────
def get_sample_camera_image(nusc, sample_token, cam_channel, show_boxes=True):
    sample = nusc.get("sample", sample_token)
    if cam_channel not in sample["data"]:
        return None, []
    cam_data = nusc.get("sample_data", sample["data"][cam_channel])
    img_path = os.path.join(DATAROOT, cam_data["filename"])
    if not os.path.exists(img_path):
        return None, []
    img  = Image.open(img_path).convert("RGB")
    if not show_boxes:
        return img, []
    draw = ImageDraw.Draw(img)
    annotations = []
    for ann_token in sample["anns"]:
        ann      = nusc.get("sample_annotation", ann_token)
        cat_name = ann["category_name"]
        color    = cat_color(cat_name)
        try:
            from nuscenes.utils.geometry_utils import view_points
            from nuscenes.utils.data_classes import Box
            from pyquaternion import Quaternion
            box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]), name=cat_name)
            sd  = nusc.get("sample_data", sample["data"][cam_channel])
            cs  = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ep  = nusc.get("ego_pose", sd["ego_pose_token"])
            box.translate(-np.array(ep["translation"]))
            box.rotate(Quaternion(ep["rotation"]).inverse)
            box.translate(-np.array(cs["translation"]))
            box.rotate(Quaternion(cs["rotation"]).inverse)
            K = np.array(cs["camera_intrinsic"])
            corners_3d = view_points(box.corners(), K, normalize=True)
            xs, ys = corners_3d[0, :], corners_3d[1, :]
            if (corners_3d[2, :] < 0.1).all():
                continue
            x1, y1 = max(xs.min(), 0), max(ys.min(), 0)
            x2, y2 = min(xs.max(), img.width), min(ys.max(), img.height)
            if x2 > x1 and y2 > y1:
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                draw.text((x1 + 2, y1 + 2), cat_name.split(".")[-1], fill=color)
                annotations.append({
                    "category": cat_name,
                    "short":    cat_name.split(".")[-1],
                    "num_pts":  ann.get("num_lidar_pts", 0),
                })
        except Exception:
            pass
    return img, annotations


def ego_trajectory(nusc, scene_token):
    scene = nusc.get("scene", scene_token)
    token = scene["first_sample_token"]
    xs, ys = [], []
    while token:
        sample = nusc.get("sample", token)
        if "CAM_FRONT" in sample["data"]:
            sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            xs.append(ep["translation"][0])
            ys.append(ep["translation"][1])
        token = sample["next"] if sample["next"] else None
    return np.array(xs), np.array(ys)


def annotation_counts(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    cats = {}
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        top = ann["category_name"].split(".")[1] if "." in ann["category_name"] else ann["category_name"]
        cats[top] = cats.get(top, 0) + 1
    return pd.DataFrame(list(cats.items()), columns=["category", "count"]).sort_values("count", ascending=False)


def run_yolo_on_sample(nusc, sample_token, cam, conf_thresh):
    """Run YOLO on one camera frame and return (image_with_boxes, tracks, anomalies)."""
    from tracking.bytetrack_wrapper import ByteTrackWrapper
    from anomaly.ood_detector import AnomalyPipeline
    # draw_tracks / draw_anomalies already imported at module top level

    sample = nusc.get("sample", sample_token)
    if cam not in sample["data"]:
        return None, [], []

    sd       = nusc.get("sample_data", sample["data"][cam])
    img_path = os.path.join(DATAROOT, sd["filename"])
    if not os.path.exists(img_path):
        return None, [], []

    model   = load_yolo()
    results = model(img_path, conf=conf_thresh, verbose=False)
    res     = results[0]
    img_yolo = Image.fromarray(res.plot()[:, :, ::-1])  # BGR→RGB

    # build detection array for tracker
    if res.boxes and len(res.boxes):
        pred_boxes  = res.boxes.xyxy.cpu().numpy()
        pred_scores = res.boxes.conf.cpu().numpy()
        pred_cls    = res.boxes.cls.cpu().numpy().astype(int)
    else:
        pred_boxes  = np.zeros((0, 4))
        pred_scores = np.array([])
        pred_cls    = np.array([], dtype=int)

    det_arr = np.hstack([
        pred_boxes,
        pred_scores.reshape(-1, 1),
        pred_cls.reshape(-1, 1),
    ]) if len(pred_boxes) else np.zeros((0, 6))

    tracker  = st.session_state.get("tracker")
    if tracker is None:
        tracker = ByteTrackWrapper(frame_rate=2)
        st.session_state["tracker"] = tracker

    tracks = tracker.update(det_arr, img_shape=(900, 1600))
    img_tracked = draw_tracks(img_yolo, tracks, show_id=True, show_score=True)

    # anomaly analysis on confidence scores
    anomaly_results = []
    if len(pred_scores):
        pipe = AnomalyPipeline()
        scores_t = torch.tensor(pred_scores, dtype=torch.float32).unsqueeze(1)
        anomaly_results = pipe.analyse(scores=scores_t, boxes=pred_boxes)
        img_tracked = draw_anomalies(img_tracked, pred_boxes, anomaly_results)

    return img_tracked, tracks, anomaly_results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
nusc = load_nusc()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 UrbanSense")
    st.caption("NuScenes v1.0-mini Explorer")
    st.divider()

    scenes      = nusc.scene
    scene_names = [f"{s['name']} — {s['description'][:40]}" for s in scenes]
    scene_idx   = st.selectbox("Scene", range(len(scenes)), format_func=lambda i: scene_names[i])
    scene       = scenes[scene_idx]

    sample_tokens = []
    tok = scene["first_sample_token"]
    while tok:
        sample_tokens.append(tok)
        tok = nusc.get("sample", tok)["next"] if nusc.get("sample", tok)["next"] else None

    sample_idx = st.slider("Sample frame", 0, len(sample_tokens) - 1, 0)
    sample_tok = sample_tokens[sample_idx]

    st.divider()
    show_boxes = st.toggle("Show GT bounding boxes", value=True)
    cam_layout = st.radio("Camera layout", ["3 x 2 grid", "Front only"], index=0)
    st.divider()

    st.markdown(f"**Scene:** `{scene['name']}`")
    st.markdown(f"**Location:** {nusc.get('log', scene['log_token'])['location']}")
    st.markdown(f"**Frames:** {len(sample_tokens)}")
    st.markdown(f"**Samples:** {scene['nbr_samples']}")

# ── tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📷  Scene Explorer",
    "🎯  Tracking & Anomaly Detection",
    "📊  Benchmark Results",
])

# ════════════════════════ TAB 1: SCENE EXPLORER ═══════════════════════════════
with tab1:
    st.header(f"🏙️ {scene['name']}  —  Frame {sample_idx + 1} / {len(sample_tokens)}")
    sample_obj = nusc.get("sample", sample_tok)
    st.caption(f"Sample token: `{sample_tok}`  |  Timestamp: {sample_obj['timestamp']}")

    CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
               "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]

    all_anns = []
    st.subheader("Camera Views")

    if cam_layout == "Front only":
        img, anns = get_sample_camera_image(nusc, sample_tok, "CAM_FRONT", show_boxes)
        if img:
            st.image(img, caption="CAM_FRONT", use_container_width=True)
        all_anns.extend(anns)
    else:
        rows = [CAMERAS[:3], CAMERAS[3:]]
        for row_cams in rows:
            cols = st.columns(3)
            for col, cam in zip(cols, row_cams):
                with col:
                    img, anns = get_sample_camera_image(nusc, sample_tok, cam, show_boxes)
                    if img:
                        st.image(img, caption=cam.replace("CAM_", "").replace("_", " "),
                                 use_container_width=True)
                    else:
                        st.info(f"{cam} — not available")
                    all_anns.extend(anns)

    st.divider()
    col_chart, col_traj, col_table = st.columns([2, 2, 3])

    with col_chart:
        st.subheader("Annotation Counts")
        df_counts = annotation_counts(nusc, sample_tok)
        if not df_counts.empty:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.barh(df_counts["category"], df_counts["count"], color="#30a5d4")
            ax.set_xlabel("Count")
            ax.invert_yaxis()
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_traj:
        st.subheader("Ego Trajectory")
        xs, ys = ego_trajectory(nusc, scene["token"])
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(xs, ys, "-o", markersize=3, color="#30a5d4", linewidth=1.5)
        if sample_idx < len(xs):
            ax2.plot(xs[sample_idx], ys[sample_idx], "ro", markersize=7, label="current")
            ax2.legend(fontsize=7)
        ax2.set_aspect("equal")
        ax2.tick_params(labelsize=7)
        ax2.set_title("GPS trajectory (XY)", fontsize=9)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_table:
        st.subheader("Visible Annotations")
        if all_anns:
            st.dataframe(pd.DataFrame(all_anns), use_container_width=True, height=220)
        else:
            st.info("No annotations projected into visible cameras.")

    st.divider()
    with st.expander("Full scene metadata"):
        log  = nusc.get("log", scene["log_token"])
        meta = {
            "Scene token":  scene["token"],
            "Description":  scene["description"],
            "Location":     log["location"],
            "Vehicle":      log["vehicle"],
            "Date":         log["date_captured"],
            "Total samples": scene["nbr_samples"],
        }
        for k, v in meta.items():
            st.markdown(f"**{k}:** {v}")


# ════════════════════════ TAB 2: TRACKING & ANOMALY ══════════════════════════
with tab2:
    st.header("🎯 YOLOv8 + ByteTrack + Anomaly Detection")
    st.caption("Run the full Phase 4 pipeline: detection → tracking → OOD anomaly flagging.")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        t2_cam   = st.selectbox("Camera", ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                                            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"])
    with col_ctrl2:
        conf_thr = st.slider("YOLO confidence threshold", 0.1, 0.9, 0.35, 0.05)
    with col_ctrl3:
        reset_tracker = st.button("Reset tracker (new scene)")
        if reset_tracker:
            st.session_state.pop("tracker", None)
            st.success("Tracker reset.")

    run_pipeline = st.button("Run pipeline on current frame", type="primary")

    if run_pipeline:
        with st.spinner("Running YOLO + ByteTrack + OOD …"):
            result_img, tracks, anomaly_results = run_yolo_on_sample(
                nusc, sample_tok, t2_cam, conf_thr
            )
        if result_img:
            st.image(result_img, caption=f"{t2_cam} — YOLO + Tracks + Anomaly flags",
                     use_container_width=True)

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.subheader("Active Tracks")
                if tracks:
                    track_data = [{
                        "ID":       t.track_id,
                        "x1": int(t.bbox[0]), "y1": int(t.bbox[1]),
                        "x2": int(t.bbox[2]), "y2": int(t.bbox[3]),
                        "score":    round(t.score, 3),
                        "class_id": t.class_id,
                        "hits":     t.hits,
                    } for t in tracks]
                    st.dataframe(pd.DataFrame(track_data), use_container_width=True)
                    st.metric("Total tracks", len(tracks))
                else:
                    st.info("No tracks active in this frame.")

            with t_col2:
                st.subheader("Anomaly Report")
                if anomaly_results:
                    ood_count = sum(1 for r in anomaly_results if r.is_ood)
                    normal_count = len(anomaly_results) - ood_count
                    st.metric("OOD / Anomalous detections", ood_count, delta=f"-{normal_count} normal")
                    ood_data = [{
                        "Det #":      r.detection_idx,
                        "OOD":        r.is_ood,
                        "Confidence": round(r.confidence, 3),
                        "Reason":     r.reason,
                    } for r in anomaly_results]
                    st.dataframe(pd.DataFrame(ood_data), use_container_width=True)
                else:
                    st.info("No detections to analyse.")
        else:
            st.warning(f"No image found for {t2_cam} in this sample.")

    else:
        st.info("Press **Run pipeline** above to execute detection + tracking + anomaly analysis on the selected frame.")


# ════════════════════════ TAB 3: BENCHMARK RESULTS ════════════════════════════
with tab3:
    st.header("📊 Benchmark Results")

    results_path = os.path.join(ROOT, "experiments", "benchmark_results.json")

    if os.path.exists(results_path):
        import json
        with open(results_path) as f:
            bm = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision",   f"{bm['precision']:.3f}")
        m2.metric("Recall",      f"{bm['recall']:.3f}")
        m3.metric("F1 Score",    f"{bm['f1']:.3f}")
        m4.metric("FPS",         bm["fps"])

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Per-scene breakdown")
            if bm.get("per_scene"):
                df_ps = pd.DataFrame(bm["per_scene"])
                st.dataframe(df_ps, use_container_width=True)

                fig3, ax3 = plt.subplots(figsize=(5, 3))
                scenes_l = [r["scene"] for r in bm["per_scene"]]
                f1s      = [r["f1"]    for r in bm["per_scene"]]
                ax3.bar(range(len(scenes_l)), f1s, color="#30a5d4")
                ax3.set_xticks(range(len(scenes_l)))
                ax3.set_xticklabels([s.replace("scene-", "") for s in scenes_l],
                                    rotation=45, fontsize=7)
                ax3.set_ylabel("F1 @ IoU=0.5")
                ax3.set_title("Per-scene F1", fontsize=9)
                ax3.set_ylim(0, 1)
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

        with col_r:
            st.subheader("Anomaly counts per scene")
            if bm.get("anomaly_counts"):
                ac = bm["anomaly_counts"]
                fig4, ax4 = plt.subplots(figsize=(5, 3))
                ax4.bar(range(len(ac)), list(ac.values()), color="#e07b39")
                ax4.set_xticks(range(len(ac)))
                ax4.set_xticklabels([k.replace("scene-", "") for k in ac.keys()],
                                    rotation=45, fontsize=7)
                ax4.set_ylabel("Anomalous detections")
                ax4.set_title("Anomaly counts by scene", fontsize=9)
                fig4.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)
            else:
                st.info("No anomaly data in results file.")

        with st.expander("Full JSON results"):
            st.json(bm)

    else:
        st.info("No benchmark results found yet.")
        st.markdown("""
Run the benchmark from your terminal to generate results:

```bash
cd C:\\Users\\saira\\Downloads\\urbansense
venv\\Scripts\\python.exe experiments\\benchmark.py --conf 0.35
```

Results will be saved to `experiments/benchmark_results.json` and displayed here automatically.
        """)
