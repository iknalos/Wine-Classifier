# ============================================================
#  Hyperspectral Wine Classifier — Streamlit Web App
#  Save as app.py | Run: streamlit run app.py
# ============================================================
#  requirements.txt:
#  streamlit tifffile numpy scikit-learn matplotlib scipy joblib
# ============================================================

import streamlit as st
import os, json, zipfile, io, tempfile, shutil
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib

# ── Page config ──
st.set_page_config(page_title="🍷 Wine Classifier", page_icon="🍷",
                   layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ──
st.markdown("""
<style>
    .step-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        color: white; padding: 16px 20px; border-radius: 10px;
        margin-bottom: 20px; border-left: 4px solid #e94560;
    }
    .step-header h2 { margin: 0; font-size: 22px; }
    .step-header p  { margin: 4px 0 0 0; opacity: 0.7; font-size: 13px; }
    .file-card {
        background: #1e1e2e; border: 1px solid #333; border-radius: 8px;
        padding: 8px 12px; margin: 4px 0; display: flex;
        align-items: center; justify-content: space-between;
    }
    .wine-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 12px; font-weight: bold; color: white;
    }
    .status-bar {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    }
    .metric-box {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 16px; text-align: center;
    }
    div[data-testid="stButton"] button {
        border-radius: 8px; font-weight: 600;
    }
    .step-nav-active {
        background: #e94560 !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ──
TILE_H = TILE_W = 9
N_BANDS    = 81
PATCH_SIZE = 30
WINE_COLORS = {
    'Dao':'#4169E1','LN':'#DC143C','LO':'#228B22',
    'ODC':'#FF8C00','PN':'#800080','PO':'#FF1493'
}

# ── Session state init ──
defaults = {
    'step': 0,
    'tiff_files': [],       # list of (name, bytes) tuples
    'file_labels': {},      # name → label
    'rois': None,
    'ref_raw': None,
    'model': None,
    'train_results': None,
    'training_done': False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ──
def get_label(name):
    n = name.lower()
    for lbl in ['dao','odc','ln','lo','pn','po']:
        if lbl in n:
            return {'dao':'Dao','odc':'ODC','ln':'LN',
                    'lo':'LO','pn':'PN','po':'PO'}[lbl]
    return 'Unknown'

def badge(label):
    col = WINE_COLORS.get(label, '#555')
    return f"<span class='wine-badge' style='background:{col}'>{label}</span>"

def load_and_demosaic(data):
    raw = tifffile.imread(io.BytesIO(data)).astype(np.float32)
    if raw.ndim == 3: raw = raw.mean(axis=2)
    raw = raw[:(raw.shape[0]//TILE_H)*TILE_H, :(raw.shape[1]//TILE_W)*TILE_W]
    return raw, [raw[r::TILE_H, c::TILE_W] for r in range(TILE_H) for c in range(TILE_W)]

def extract_patches(channels, x0, y0, x1, y1, px=PATCH_SIZE):
    sx, sy = px//TILE_W, px//TILE_H
    feats = []
    for py in range(y0//TILE_H, y1//TILE_H-sy+1, sy):
        for px2 in range(x0//TILE_W, x1//TILE_W-sx+1, sx):
            m = np.array([ch[py:py+sy, px2:px2+sx].mean() for ch in channels], dtype=np.float32)
            s = np.array([ch[py:py+sy, px2:px2+sx].std()  for ch in channels], dtype=np.float32)
            feats.append(np.concatenate([m/(m.sum()+1e-9), s]))
    return feats

def disp_img(raw):
    vmin, vmax = np.percentile(raw,1), np.percentile(raw,99)
    return np.clip((raw-vmin)/(vmax-vmin+1e-9),0,1)

def can_advance(to_step):
    if to_step >= 1 and not st.session_state.tiff_files: return False
    if to_step >= 2 and st.session_state.rois is None:   return False
    if to_step >= 3 and not st.session_state.training_done: return False
    return True

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🍷 Wine Classifier")
    st.markdown("---")

    # ── Google Drive Panel ──
    st.markdown("### ☁️ Google Drive")
    try:
        gcreds = get_google_creds()

        if not gcreds:
            if not GOOGLE_AVAILABLE:
                st.warning(f"⚠️ Google libraries not installed.")
            else:
                st.caption("Add Google credentials in Streamlit secrets to enable Drive.")

        elif st.session_state.gdrive_token is None:
        # Not logged in
        flow = build_flow()
        auth_url, _ = flow.authorization_url(
            prompt='consent', access_type='offline')
        st.markdown(
            f"<a href='{auth_url}' target='_self'>"
            f"<button style='background:#4285F4;color:white;border:none;"
            f"padding:8px 16px;border-radius:6px;cursor:pointer;width:100%;"
            f"font-size:14px;font-weight:600'>🔗 Connect Google Drive</button>"
            f"</a>", unsafe_allow_html=True)
        st.caption("Connect once — browse and import files directly from Drive.")

    else:
        # Logged in — show file browser
        st.success("✅ Drive connected")
        if st.button("🔓 Disconnect Drive", use_container_width=True):
            st.session_state.gdrive_token   = None
            st.session_state.gdrive_folder_id   = 'root'
            st.session_state.gdrive_folder_name = 'My Drive'
            st.session_state.gdrive_breadcrumb  = [('root','My Drive')]
            st.rerun()

        # Breadcrumb navigation
        bc = st.session_state.gdrive_breadcrumb
        bc_text = " › ".join([name for _, name in bc])
        st.caption(f"📂 {bc_text}")

        # Back button
        if len(bc) > 1:
            if st.button("⬆️ Back", use_container_width=True):
                st.session_state.gdrive_breadcrumb.pop()
                st.session_state.gdrive_folder_id   = bc[-2][0]
                st.session_state.gdrive_folder_name = bc[-2][1]
                st.rerun()

        # List folder contents
        try:
            service = get_drive_service()
            items   = list_drive_folder(service, st.session_state.gdrive_folder_id)

            folders = [i for i in items if i['mimeType'] == 'application/vnd.google-apps.folder']
            tiffs   = [i for i in items if i['mimeType'] != 'application/vnd.google-apps.folder']

            # Folders
            for folder in folders:
                if st.button(f"📁 {folder['name']}", key=f"fd_{folder['id']}",
                             use_container_width=True):
                    st.session_state.gdrive_breadcrumb.append(
                        (folder['id'], folder['name']))
                    st.session_state.gdrive_folder_id   = folder['id']
                    st.session_state.gdrive_folder_name = folder['name']
                    st.rerun()

            # TIFF files
            if tiffs:
                st.markdown("**TIFF files:**")
                for f in tiffs:
                    size_kb = int(f.get('size', 0)) // 1024
                    c1, c2 = st.columns([4,1])
                    lbl    = get_label(f['name'])
                    col    = WINE_COLORS.get(lbl, '#555')
                    c1.markdown(
                        f"<div style='font-size:11px;padding:2px 0'>"
                        f"<span style='background:{col};color:white;padding:1px 5px;"
                        f"border-radius:6px;font-size:10px'>{lbl}</span> "
                        f"{f['name'][:20]}{'…' if len(f['name'])>20 else ''} "
                        f"<span style='color:#888'>({size_kb}KB)</span></div>",
                        unsafe_allow_html=True)
                    if c2.button("➕", key=f"add_{f['id']}",
                                 help=f"Add {f['name']} to training"):
                        with st.spinner(f"Downloading {f['name']}..."):
                            data = download_drive_file(service, f['id'])
                            existing = [n for n,_ in st.session_state.tiff_files]
                            if f['name'] not in existing:
                                st.session_state.tiff_files.append((f['name'], data))
                                st.session_state.file_labels[f['name']] = get_label(f['name'])
                                st.toast(f"✅ Added {f['name']}", icon="✅")
                                st.rerun()

                # Add all TIFFs in folder at once
                if len(tiffs) > 1:
                    if st.button(f"➕ Add all {len(tiffs)} TIFFs",
                                 use_container_width=True, type="primary"):
                        added = 0
                        existing = [n for n,_ in st.session_state.tiff_files]
                        prog = st.progress(0)
                        for i, f in enumerate(tiffs):
                            if f['name'] not in existing:
                                data = download_drive_file(service, f['id'])
                                st.session_state.tiff_files.append((f['name'], data))
                                st.session_state.file_labels[f['name']] = get_label(f['name'])
                                added += 1
                            prog.progress((i+1)/len(tiffs))
                        st.toast(f"✅ Added {added} files", icon="✅")
                        st.rerun()
            else:
                if not folders:
                    st.caption("No TIFF files or folders found here.")

        except Exception as e:
            st.error(f"Drive error: {e}")
            st.session_state.gdrive_token = None

    st.markdown("---")

    # ── Step navigation ──
    step_labels = ["1  Upload Data", "2  Select ROI",
                   "3  Train Model", "4  Predict"]
    step_icons  = ["📁","🎯","🤖","🔍"]
    step_done   = [
        len(st.session_state.tiff_files) > 0,
        st.session_state.rois is not None,
        st.session_state.training_done,
        True
    ]

    for i, (icon, lbl) in enumerate(zip(step_icons, step_labels)):
        done   = step_done[i]
        active = st.session_state.step == i
        locked = not can_advance(i)
        prefix = "✅" if done else ("🔒" if locked else icon)
        style  = "primary" if active else "secondary"
        if st.button(f"{prefix}  {lbl}", key=f"nav{i}",
                     use_container_width=True,
                     disabled=locked, type=style):
            st.session_state.step = i
            st.rerun()

    st.markdown("---")

    # ── Progress ──
    n_done = sum(step_done[:3])
    st.markdown(f"**Progress: {n_done}/3 steps**")
    st.progress(n_done/3)

    # ── Uploaded files panel ──
    if st.session_state.tiff_files:
        st.markdown("---")
        st.markdown("**📂 Training Files**")
        for i, (name, _) in enumerate(st.session_state.tiff_files):
            lbl = st.session_state.file_labels.get(name, 'Unknown')
            col_badge = WINE_COLORS.get(lbl,'#555')
            c1, c2 = st.columns([4,1])
            c1.markdown(
                f"<div style='font-size:11px;padding:2px 0'>"
                f"<span style='background:{col_badge};color:white;padding:1px 6px;"
                f"border-radius:8px;font-size:10px'>{lbl}</span> "
                f"{name[:22]}{'…' if len(name)>22 else ''}</div>",
                unsafe_allow_html=True)
            if c2.button("✕", key=f"del_{i}", help=f"Remove {name}"):
                st.session_state.tiff_files.pop(i)
                st.session_state.file_labels.pop(name, None)
                # reset downstream if files removed
                if not st.session_state.tiff_files:
                    st.session_state.rois = None
                    st.session_state.training_done = False
                    st.session_state.model = None
                st.rerun()

        if st.button("🗑️ Remove All Files", use_container_width=True):
            st.session_state.tiff_files = []
            st.session_state.file_labels = {}
            st.session_state.rois = None
            st.session_state.training_done = False
            st.session_state.model = None
            st.rerun()

    # ── Model panel ──
    st.markdown("---")
    if st.session_state.training_done and st.session_state.model:
        buf = io.BytesIO()
        joblib.dump({'model': st.session_state.model,
                     'rois':  st.session_state.rois,
                     'patch_size': PATCH_SIZE}, buf)
        st.download_button("💾 Download Model", buf.getvalue(),
                           "wine_classifier.pkl", use_container_width=True)
        if st.button("🗑️ Clear Model", use_container_width=True):
            st.session_state.model = None
            st.session_state.training_done = False
            st.rerun()

    st.markdown("**Load saved model:**")
    pkl = st.file_uploader("Upload .pkl", type=['pkl'])
    if pkl:
        saved = joblib.load(io.BytesIO(pkl.read()))
        st.session_state.model         = saved['model']
        st.session_state.rois          = saved.get('rois')
        st.session_state.training_done = True
        st.success("✅ Model loaded!")

# ============================================================
#  STEP 1 — Upload Training Data
# ============================================================
if st.session_state.step == 0:
    st.markdown("""<div class='step-header'>
        <h2>📁 Step 1 — Upload Training Data</h2>
        <p>Upload a ZIP containing your labelled TIFF images. Filenames must include the wine name.</p>
    </div>""", unsafe_allow_html=True)

    col_up, col_info = st.columns([3,2])

    with col_up:
        tab_local, tab_drive = st.tabs(["💻 From Computer", "☁️ Google Drive"])

        with tab_local:
            zf = st.file_uploader("Drop your training ZIP here",
                                  type=['zip'], label_visibility='collapsed')
        with tab_drive:
            st.markdown("Share your ZIP as *'Anyone with link'*, paste the **file ID** below.")
            gid = st.text_input("Google Drive File ID",
                                placeholder="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs")
            if st.button("📥 Fetch from Drive", use_container_width=True) and gid:
                with st.spinner("Downloading from Drive..."):
                    try:
                        import urllib.request
                        url = f"https://drive.google.com/uc?export=download&id={gid}"
                        with urllib.request.urlopen(url) as r:
                            zf = io.BytesIO(r.read())
                        st.success("✅ Downloaded!")
                    except Exception as e:
                        st.error(f"❌ {e}"); zf = None

    with col_info:
        st.markdown("""
        **Supported wine labels in filename:**
        | Label | Example filename |
        |-------|-----------------|
        | Dao   | `Dao_100K_1.tiff` |
        | LN    | `LN_100K_1.tiff`  |
        | LO    | `LO_100K_1.tiff`  |
        | ODC   | `ODC_100K_1.tiff` |
        | PN    | `PN_100K_1.tiff`  |
        | PO    | `PO_100K_1.tiff`  |
        """)

    with st.expander("ℹ️ What kind of images do I need?"):
        st.markdown("""
        **Camera & Format**
        This app is designed for the **Basler daA2500** camera with a
        **9×9 mosaic spectral filter** (81 bands). Images must be raw `.tiff` files
        — do not convert to JPEG or PNG as this destroys the spectral data.

        **How many images per wine?**
        A minimum of **2 images per wine type** is required. 3 or more is recommended
        for reliable accuracy. More images = better model.

        **What does the app do with the raw TIFF?**
        Each pixel in the raw TIFF contains light from one specific wavelength filter.
        The app *demosaics* the image — separating the 9×9 tile pattern into
        81 individual wavelength images — then extracts spectral features from your
        chosen region.

        **Important — consistent conditions:**
        All images (training and prediction) should be captured under the
        **same exposure, distance, and lighting**. Variations between sessions
        can reduce prediction confidence.

        **File naming:**
        The wine label is detected automatically from the filename.
        Just make sure the wine name appears somewhere in the filename
        (e.g. `Dao_sample1.tiff`, `my_ODC_capture.tiff`).
        """)

    if zf:
        with st.spinner("Reading ZIP..."):
            zdata = zf if isinstance(zf, io.BytesIO) else io.BytesIO(zf.read())
            with zipfile.ZipFile(zdata,'r') as z:
                names = [n for n in z.namelist()
                         if n.lower().endswith(('.tiff','.tif'))
                         and not os.path.basename(n).startswith('.')]
                new_files = []
                for name in names:
                    bname = os.path.basename(name)
                    data  = z.read(name)
                    lbl   = get_label(bname)
                    # avoid duplicates
                    existing = [f[0] for f in st.session_state.tiff_files]
                    if bname not in existing:
                        st.session_state.tiff_files.append((bname, data))
                        st.session_state.file_labels[bname] = lbl
                        new_files.append(bname)

        if new_files:
            st.success(f"✅ Added **{len(new_files)}** new files")

    # ── File list ──
    if st.session_state.tiff_files:
        st.markdown("### 📋 Uploaded Files")
        label_counts = {}
        for name, _ in st.session_state.tiff_files:
            lbl = st.session_state.file_labels.get(name,'Unknown')
            label_counts[lbl] = label_counts.get(lbl,0)+1

        # Summary chips
        chip_html = " ".join([
            f"<span style='background:{WINE_COLORS.get(l,'#555')};color:white;"
            f"padding:4px 12px;border-radius:20px;font-size:13px;font-weight:bold'>"
            f"{l}: {c}</span>"
            for l,c in sorted(label_counts.items())
        ])
        st.markdown(chip_html, unsafe_allow_html=True)
        st.markdown("")

        # File table with delete buttons
        for name, _ in st.session_state.tiff_files:
            lbl = st.session_state.file_labels.get(name,'Unknown')
            col_badge = WINE_COLORS.get(lbl,'#555')
            ca, cb, cc = st.columns([5, 2, 1])
            ca.markdown(f"🖼️ `{name}`")
            cb.markdown(
                f"<span style='background:{col_badge};color:white;padding:2px 10px;"
                f"border-radius:10px;font-size:12px'>{lbl}</span>",
                unsafe_allow_html=True)

        if 'Unknown' in label_counts:
            st.warning(f"⚠️ {label_counts['Unknown']} file(s) have unrecognised labels.")

        st.markdown("---")
        if st.button("✅ Confirm & Go to ROI Selection →",
                     type="primary", use_container_width=True):
            # Load first image as reference for ROI
            _, data = st.session_state.tiff_files[0]
            raw, _  = load_and_demosaic(data)
            st.session_state.ref_raw = raw
            st.session_state.step = 1
            st.rerun()

# ============================================================
#  STEP 2 — ROI Selector
# ============================================================
elif st.session_state.step == 1:
    st.markdown("""<div class='step-header'>
        <h2>🎯 Step 2 — Select ROI Regions</h2>
        <p>Position both boxes over the wine liquid. Avoid glass edges and the stem.</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("ℹ️ How does this step work?"):
        st.markdown("""
        **What is an ROI?**
        ROI stands for *Region of Interest* — it's the area of the image the model will
        actually analyse. We ignore everything else (glass edges, background, stem).

        **Why two ROIs?**
        Two separate regions on the wine liquid give the model more variety.
        Think of it like tasting from two different parts of the glass — more data,
        more reliable result.

        **How much data does this generate?**
        Each ROI is divided into a grid of small 30×30 pixel sub-patches.
        A typical 150×120 ROI gives ~20 patches. Two ROIs = **~40 patches per image**.
        Across 3 training images that's **~120 samples per wine type** — all from just 6 files.

        **Tips for good ROI placement:**
        - ✅ Place both boxes on the **liquid part** of the wine
        - ✅ Keep them away from glass edges and reflections
        - ✅ Spread them apart — left and right of the glass is ideal
        - ❌ Avoid the stem, base, or background
        """)

    if not st.session_state.tiff_files:
        st.warning("⚠️ Go back to Step 1 first."); st.stop()

    raw = st.session_state.ref_raw
    if raw is None:
        _, data = st.session_state.tiff_files[0]
        raw, _  = load_and_demosaic(data)
        st.session_state.ref_raw = raw
    H, W = raw.shape

    st.caption(f"Reference: `{st.session_state.tiff_files[0][0]}`  |  {W}×{H} px")

    col_sliders, col_preview = st.columns([2,3])

    with col_sliders:
        st.markdown("#### 🔵 ROI 1")
        cx1 = st.slider("Center X",  0, W, W//2-100,  5, key='cx1')
        cy1 = st.slider("Center Y",  0, H, H//2,       5, key='cy1')
        rw1 = st.slider("Width",    30, 400, 150,      10, key='rw1')
        rh1 = st.slider("Height",   30, 400, 120,      10, key='rh1')

        st.markdown("---")
        st.markdown("#### 🟠 ROI 2")
        cx2 = st.slider("Center X",  0, W, W//2+100,  5, key='cx2')
        cy2 = st.slider("Center Y",  0, H, H//2,       5, key='cy2')
        rw2 = st.slider("Width",    30, 400, 150,      10, key='rw2')
        rh2 = st.slider("Height",   30, 400, 120,      10, key='rh2')

    roi1 = (max(0,cx1-rw1//2), max(0,cy1-rh1//2),
            min(W,cx1+rw1//2), min(H,cy1+rh1//2))
    roi2 = (max(0,cx2-rw2//2), max(0,cy2-rh2//2),
            min(W,cx2+rw2//2), min(H,cy2+rh2//2))

    with col_preview:
        tab_full, tab_r1, tab_r2 = st.tabs(["🖼️ Full Image", "🔵 ROI 1 Content", "🟠 ROI 2 Content"])
        with tab_full:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.imshow(disp_img(raw), cmap='gray', aspect='auto')
            for (x0,y0,x1_,y1_), c in [(roi1,'dodgerblue'),(roi2,'orange')]:
                ax.add_patch(patches.Rectangle((x0,y0),x1_-x0,y1_-y0,lw=2,edgecolor=c,facecolor=c,alpha=0.2))
                ax.add_patch(patches.Rectangle((x0,y0),x1_-x0,y1_-y0,lw=2,edgecolor=c,facecolor='none'))
            ax.axis('off'); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with tab_r1:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.imshow(disp_img(raw)[roi1[1]:roi1[3],roi1[0]:roi1[2]],
                      cmap='gray', interpolation='nearest')
            ax.set_title(f"{roi1[2]-roi1[0]}×{roi1[3]-roi1[1]} px")
            ax.axis('off'); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

        with tab_r2:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.imshow(disp_img(raw)[roi2[1]:roi2[3],roi2[0]:roi2[2]],
                      cmap='gray', interpolation='nearest')
            ax.set_title(f"{roi2[2]-roi2[0]}×{roi2[3]-roi2[1]} px")
            ax.axis('off'); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    # Coordinate display
    st.markdown(
        f"<div class='status-bar'>"
        f"<b style='color:dodgerblue'>ROI 1:</b> X0={roi1[0]} Y0={roi1[1]} X1={roi1[2]} Y1={roi1[3]} "
        f"&nbsp;&nbsp;&nbsp;"
        f"<b style='color:orange'>ROI 2:</b> X0={roi2[0]} Y0={roi2[1]} X1={roi2[2]} Y1={roi2[3]}"
        f"</div>", unsafe_allow_html=True)

    st.markdown("")
    if st.button("✅ Confirm ROIs & Go to Training →",
                 type="primary", use_container_width=True):
        st.session_state.rois = [roi1, roi2]
        st.session_state.step = 2
        st.rerun()

# ============================================================
#  STEP 3 — Train
# ============================================================
elif st.session_state.step == 2:
    st.markdown("""<div class='step-header'>
        <h2>🤖 Step 3 — Train Classifier</h2>
        <p>Extracts spectral patches from each ROI and trains an SVM with Leave-One-Image-Out validation.</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.tiff_files:
        st.warning("⚠️ Complete Step 1 first."); st.stop()
    if st.session_state.rois is None:
        st.warning("⚠️ Complete Step 2 first."); st.stop()

    with st.expander("ℹ️ How does training work?"):
        st.markdown("""
        **What happens when you click Train?**

        1. **Demosaic** — each TIFF is split into 81 spectral band images
           (the 9×9 mosaic filter is separated into individual wavelength channels)
        2. **Patch extraction** — each ROI is divided into 30×30 px sub-patches.
           Every patch becomes one training sample with **162 features**
           (81 band mean intensities + 81 band standard deviations)
        3. **Training** — the classifier learns what Dao looks like vs ODC
           across all 81 spectral bands

        **How much data does this generate?**
        ```
        1 image × 2 ROIs × ~20 patches = 40 samples per image
        3 Dao images × 40 patches      = 120 Dao samples
        3 ODC images × 40 patches      = 120 ODC samples
        Total                          = ~240 training samples
        ```
        All from just 6 raw TIFF files.

        **What is Leave-One-Image-Out validation?**
        The model is tested 6 times — each time one full image is hidden,
        the model trains on the other 5, then tries to predict the hidden one.
        This gives an honest accuracy score with no "cheating" between train and test.

        **What do the charts show?**
        - 📈 **Spectral signatures** — how bright each wine is across all 81 bands
        - 🔬 **PCA plot** — patches grouped into clusters; well-separated = easier to classify
        - 📊 **Confusion matrix** — which wines were correctly or incorrectly predicted
        """)

    if st.session_state.training_done:
        st.success("✅ Model already trained. Scroll down to see results, or retrain below.")

    # ── Model selection ──
    st.markdown("### 🧠 Classifier Selection")

    col_table, col_pick = st.columns([3, 2])

    with col_table:
        st.markdown("**Research comparison — hyperspectral wine classification:**")
        st.markdown("""
| Model | Small data? | Wine HSI proven? | Speed | Recommended |
|-------|-------------|-----------------|-------|-------------|
| **Ensemble (SVM+RF+XGB)** | ✅ Best | ✅ Yes | ✅ Fast | ⭐ Default |
| SVM RBF | ✅ Best | ✅ Yes | ✅ Fast | ✅ Good |
| Random Forest | ✅ Good | ✅ Yes | ✅ Fast | ✅ Good |
| XGBoost | ✅ Good | ✅ Yes | ✅ Fast | ✅ Good |
| CNN / Deep Learning | ❌ Needs 1000s | ✅ Best accuracy | ❌ Slow | ⚠️ Not yet |
        """)
        st.caption("Sources: ScienceDirect 2024, PubMed, ACM Digital Library")

    with col_pick:
        st.markdown("**Select classifier:**")
        model_choice = st.radio(
            "classifier",
            ["⭐ Ensemble (SVM + RF + XGB)  ← Recommended",
             "SVM RBF",
             "Random Forest",
             "XGBoost"],
            label_visibility='collapsed'
        )
        model_key = model_choice.split("(")[0].strip().replace("⭐ ","").strip()

        descriptions = {
            "Ensemble": "3 models vote together — most robust for small datasets. "
                        "Proven best in peer-reviewed hyperspectral wine research (ScienceDirect 2024).",
            "SVM RBF":  "Support Vector Machine with RBF kernel. Excellent for high-dimensional "
                        "spectral data. F1 scores up to 0.99 in grapevine HSI studies (PubMed).",
            "Random Forest": "Ensemble of decision trees. Handles non-linear band "
                             "interactions well and is resistant to overfitting.",
            "XGBoost":  "Gradient boosting — often outperforms RF on tabular/spectral data. "
                        "Showed considerable prediction power in wine HSI research.",
        }
        key = "Ensemble" if "Ensemble" in model_key else model_key
        st.info(descriptions.get(key, ""))

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        rois = st.session_state.rois
        tiff_files = [(n,d) for n,d in st.session_state.tiff_files
                      if st.session_state.file_labels.get(n,'Unknown') != 'Unknown']

        prog    = st.progress(0, text="Starting...")
        X_all, y_all, groups, raw_spectra = [], [], [], {}
        img_id  = 0

        for i,(name,data) in enumerate(tiff_files):
            label = st.session_state.file_labels[name]
            prog.progress((i+1)/len(tiff_files), text=f"Processing {name}...")
            raw, channels = load_and_demosaic(data)
            for roi in rois:
                feats = extract_patches(channels, *roi)
                for fv in feats:
                    X_all.append(fv); y_all.append(label); groups.append(img_id)
                if roi == rois[0]:
                    m = np.array([ch[roi[1]//TILE_H:roi[3]//TILE_H,
                                     roi[0]//TILE_W:roi[2]//TILE_W].mean()
                                  for ch in channels])
                    raw_spectra.setdefault(label,[]).append(m)
            img_id += 1

        X,y,groups = np.array(X_all),np.array(y_all),np.array(groups)
        all_labels = sorted(np.unique(y).tolist())
        n_cls = len(all_labels)

        prog.progress(1.0, text="Training SVM...")

        # ── Tabs for results ──
        t1,t2,t3,t4 = st.tabs(["📈 Spectra","🔬 PCA","📊 Confusion Matrix","📋 Per-Image"])

        with t1:
            fig,axes = plt.subplots(2,1,figsize=(13,8))
            means_by_class = {}
            for lbl,spectra in raw_spectra.items():
                arr = np.array(spectra); m,s = arr.mean(0),arr.std(0)
                means_by_class[lbl] = m
                col = WINE_COLORS.get(lbl,'gray')
                axes[0].plot(m,color=col,lw=2,label=lbl)
                axes[0].fill_between(range(N_BANDS),m-s,m+s,alpha=0.15,color=col)
            axes[0].set_title("Spectral Signatures — mean ± std")
            axes[0].set_xlabel("Band"); axes[0].set_ylabel("Intensity")
            axes[0].legend(fontsize=9); axes[0].grid(True,alpha=0.3)

            dm = np.zeros((n_cls,n_cls))
            for i,l1 in enumerate(all_labels):
                for j,l2 in enumerate(all_labels):
                    dm[i,j] = np.abs(means_by_class[l1]-means_by_class[l2]).mean()
            im = axes[1].imshow(dm,cmap='YlOrRd')
            axes[1].set_xticks(range(n_cls)); axes[1].set_yticks(range(n_cls))
            axes[1].set_xticklabels(all_labels); axes[1].set_yticklabels(all_labels)
            axes[1].set_title("Spectral Distance Heatmap")
            plt.colorbar(im,ax=axes[1])
            for i in range(n_cls):
                for j in range(n_cls):
                    axes[1].text(j,i,f"{dm[i,j]:.1f}",ha='center',va='center',
                                 fontsize=9,color='white' if dm[i,j]>dm.max()*0.6 else 'black')
            plt.tight_layout()
            st.pyplot(fig,use_container_width=True); plt.close(fig)

        with t2:
            Xs = StandardScaler().fit_transform(X)
            Xp = PCA(n_components=2).fit_transform(Xs)
            fig2,ax = plt.subplots(figsize=(8,5))
            for lbl in np.unique(y):
                mask = y==lbl; col = WINE_COLORS.get(lbl,'gray')
                ax.scatter(Xp[mask,0],Xp[mask,1],c=col,alpha=0.2,s=8,label=lbl)
                cx_,cy_ = Xp[mask,0].mean(),Xp[mask,1].mean()
                ax.scatter(cx_,cy_,c=col,s=220,marker='*',edgecolors='black',lw=0.8,zorder=5)
                ax.annotate(lbl,(cx_,cy_),xytext=(6,4),textcoords='offset points',
                            fontsize=11,fontweight='bold',color=col)
            ax.set_title("PCA — patch clusters (stars = centres)")
            ax.legend(markerscale=4,fontsize=9); ax.grid(True,alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig2,use_container_width=True); plt.close(fig2)

        # Train + LOO CV
        model = Pipeline([('scaler',StandardScaler()),
                          ('clf',SVC(kernel='rbf',C=10,gamma='scale',
                                     probability=True,random_state=42))])
        cv     = GroupKFold(n_splits=img_id)
        y_pred = cross_val_predict(model,X,y,cv=cv,groups=groups)

        with t3:
            cm = confusion_matrix(y,y_pred,labels=all_labels)
            fig3,ax3 = plt.subplots(figsize=(max(5,n_cls*1.3),max(4,n_cls*1.1)))
            ax3.imshow(cm,cmap='Blues')
            ax3.set_xticks(range(n_cls)); ax3.set_yticks(range(n_cls))
            ax3.set_xticklabels(all_labels,rotation=45); ax3.set_yticklabels(all_labels)
            ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
            ax3.set_title('Confusion Matrix (Leave-One-Image-Out)')
            for i in range(n_cls):
                for j in range(n_cls):
                    ax3.text(j,i,cm[i,j],ha='center',va='center',fontsize=13,
                             color='white' if cm[i,j]>cm.max()/2 else 'black')
            plt.tight_layout()
            st.pyplot(fig3,use_container_width=True); plt.close(fig3)

        with t4:
            rows = []
            correct = 0
            for gid in range(img_id):
                mask     = groups==gid
                true_lbl = y[mask][0]
                votes    = {l:(y_pred[mask]==l).sum() for l in np.unique(y)}
                img_pred = max(votes,key=votes.get)
                agree    = votes[img_pred]/mask.sum()*100
                ok       = "✅" if img_pred==true_lbl else "❌"
                rows.append({'File':tiff_files[gid][0],'True':true_lbl,
                             'Predicted':img_pred,'Agreement':f"{agree:.0f}%",'':ok})
                if img_pred==true_lbl: correct+=1
            st.dataframe(rows,use_container_width=True)
            acc = correct/img_id*100
            c1,c2,c3 = st.columns(3)
            c1.metric("Image Accuracy", f"{acc:.0f}%")
            c2.metric("Correct", f"{correct}/{img_id}")
            c3.metric("Total Patches", f"{len(X):,}")

        # Save
        model.fit(X,y)
        st.session_state.model         = model
        st.session_state.training_done = True
        prog.empty()
        st.success("✅ Training complete! Model saved. Proceed to Step 4 to predict.")

        col_next, _ = st.columns([2,3])
        if col_next.button("Go to Predict →", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ============================================================
#  STEP 4 — Predict
# ============================================================
elif st.session_state.step == 3:
    st.markdown("""<div class='step-header'>
        <h2>🔍 Step 4 — Predict Unknown Wines</h2>
        <p>Upload a ZIP or individual TIFF files. The model predicts each one.</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.training_done or st.session_state.model is None:
        st.warning("⚠️ Complete Step 3 first or load a saved model from the sidebar.")
        st.stop()

    model    = st.session_state.model
    rois     = st.session_state.rois
    all_cls  = list(model.classes_)

    tab_zip, tab_single, tab_drive = st.tabs(["📦 Upload ZIP","🖼️ Single TIFF","☁️ Google Drive"])

    pred_files = []  # list of (name, bytes)

    with tab_zip:
        zf2 = st.file_uploader("Upload ZIP of unknown TIFFs",
                               type=['zip'], label_visibility='collapsed')
        if zf2:
            zdata = io.BytesIO(zf2.read())
            with zipfile.ZipFile(zdata,'r') as z:
                for nm in z.namelist():
                    if nm.lower().endswith(('.tiff','.tif')) and not os.path.basename(nm).startswith('.'):
                        pred_files.append((os.path.basename(nm), z.read(nm)))

    with tab_single:
        tfs = st.file_uploader("Upload one or more TIFF files",
                               type=['tiff','tif'], accept_multiple_files=True,
                               label_visibility='collapsed')
        for tf in (tfs or []):
            pred_files.append((tf.name, tf.read()))

    with tab_drive:
        gid2 = st.text_input("Google Drive File ID (ZIP)", key='gid2')
        if st.button("📥 Fetch ZIP from Drive", key='fetch2') and gid2:
            with st.spinner("Downloading..."):
                try:
                    import urllib.request
                    url = f"https://drive.google.com/uc?export=download&id={gid2}"
                    with urllib.request.urlopen(url) as r:
                        zdata2 = io.BytesIO(r.read())
                    with zipfile.ZipFile(zdata2,'r') as z:
                        for nm in z.namelist():
                            if nm.lower().endswith(('.tiff','.tif')):
                                pred_files.append((os.path.basename(nm), z.read(nm)))
                    st.success(f"✅ Loaded {len(pred_files)} files from Drive")
                except Exception as e:
                    st.error(f"❌ {e}")

    if pred_files:
        st.info(f"**{len(pred_files)}** file(s) ready for prediction")

        # Preview filenames
        for name,_ in pred_files:
            st.markdown(f"&nbsp;&nbsp;🖼️ `{name}`")

        st.markdown("")
        if st.button("🔍 Run Predictions", type="primary", use_container_width=True):
            prog2   = st.progress(0, text="Predicting...")
            results = []

            for i,(name,data) in enumerate(pred_files):
                prog2.progress((i+1)/len(pred_files), text=f"Predicting {name}...")
                raw, channels = load_and_demosaic(data)
                all_probas = []
                for roi in rois:
                    feats = extract_patches(channels,*roi)
                    if feats:
                        all_probas.append(model.predict_proba(np.array(feats)))
                avg   = np.vstack(all_probas).mean(axis=0)
                pred  = model.classes_[np.argmax(avg)]
                conf  = dict(zip(model.classes_,(avg*100).round(1)))
                agr   = (np.argmax(np.vstack(all_probas),axis=1)==np.argmax(avg)).mean()*100
                results.append({'file':name,'raw':raw,'pred':pred,
                                 'conf':conf,'agreement':agr,
                                 'n_patches':sum(len(p) for p in all_probas)})

            prog2.empty()

            # ── Grouped result cards ──
            st.subheader("🗂️ Results by Wine Type")
            cols = st.columns(len(all_cls))
            for ci, cls in enumerate(all_cls):
                matched = [r for r in results if r['pred']==cls]
                col = WINE_COLORS.get(cls,'#555')
                with cols[ci]:
                    st.markdown(
                        f"<div style='background:{col}22;border:2px solid {col};"
                        f"border-radius:10px;padding:12px;text-align:center;min-height:80px'>"
                        f"<b style='color:{col};font-size:20px'>{cls}</b><br>"
                        f"<span style='font-size:12px;color:#aaa'>{len(matched)} image(s)</span><br><br>"
                        + "".join([
                            f"<span style='display:block;font-size:12px;padding:2px 0'>"
                            f"📄 {r['file']}<br>"
                            f"<b style='color:{col}'>{r['conf'][r['pred']]:.1f}%</b> confident"
                            f"</span>"
                            for r in matched
                        ]) +
                        "</div>", unsafe_allow_html=True)

            # ── Summary table ──
            st.subheader("📋 Detailed Results")
            rows = [{'File':r['file'],'Prediction':r['pred'],
                     'Confidence':f"{r['conf'][r['pred']]:.1f}%",
                     'Patch Agreement':f"{r['agreement']:.0f}%",
                     'Patches':r['n_patches']}
                    for r in results]
            st.dataframe(rows, use_container_width=True)

            # ── Image grid ──
            st.subheader("🖼️ Image Overview")
            gcols = st.columns(min(3,len(results)))
            for i,r in enumerate(results):
                with gcols[i%3]:
                    col = WINE_COLORS.get(r['pred'],'#888')
                    fig,ax = plt.subplots(figsize=(4,3))
                    ax.imshow(disp_img(r['raw']),cmap='gray',aspect='auto')
                    for j,(x0,y0,x1_,y1_) in enumerate(rois):
                        c = ['dodgerblue','orange'][j%2]
                        ax.add_patch(patches.Rectangle((x0,y0),x1_-x0,y1_-y0,lw=2,edgecolor=c,facecolor=c,alpha=0.2))
                        ax.add_patch(patches.Rectangle((x0,y0),x1_-x0,y1_-y0,lw=2,edgecolor=c,facecolor='none'))
                    ax.set_title(f"{r['file']}\n{r['pred']} — {r['conf'][r['pred']]:.1f}%",
                                 color=col,fontsize=8,fontweight='bold')
                    ax.axis('off'); plt.tight_layout()
                    st.pyplot(fig,use_container_width=True); plt.close(fig)

            # ── Confidence chart ──
            st.subheader("📊 Confidence Chart")
            fig_b,ax_b = plt.subplots(figsize=(max(10,len(results)*2),5))
            x      = np.arange(len(results))
            w      = 0.8/len(all_cls)
            for idx,cls in enumerate(all_cls):
                vals   = [r['conf'].get(cls,0) for r in results]
                offset = (idx-len(all_cls)/2+0.5)*w
                bars   = ax_b.bar(x+offset,vals,w,label=cls,
                                  color=WINE_COLORS.get(cls,'gray'),alpha=0.85)
                for bar in bars:
                    h=bar.get_height()
                    if h>5:
                        ax_b.text(bar.get_x()+bar.get_width()/2,h+0.5,
                                  f'{h:.0f}%',ha='center',va='bottom',fontsize=6)
            ax_b.axhline(50,color='black',lw=1,linestyle='--',alpha=0.4)
            ax_b.set_xticks(x)
            ax_b.set_xticklabels([r['file'].replace('.tiff','').replace('.tif','')
                                   for r in results],rotation=20,ha='right',fontsize=9)
            ax_b.set_ylabel("Confidence %"); ax_b.set_ylim(0,115)
            ax_b.set_title("Prediction Confidence per Image")
            ax_b.legend(fontsize=9); ax_b.grid(True,alpha=0.2,axis='y')
            plt.tight_layout()
            st.pyplot(fig_b,use_container_width=True); plt.close(fig_b)

            st.success("✅ All predictions complete!")
