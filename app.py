# ============================================================
#  Hyperspectral Wine Classifier — Streamlit Web App
#  Save as app.py | Run: streamlit run app.py
# ============================================================

import streamlit as st
import os, zipfile, io
import urllib.parse
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
import requests as req_lib

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# ── Page config ──
st.set_page_config(page_title="🔬 Image Classifier", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
a[href*="github.com"] {display: none !important;}
[data-testid="stToolbarActions"] {display: none !important;}
[data-testid="stHeader"] {background: transparent;}

/* Main background */
.stApp { background: #0a0a0f; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #12121f 100%);
    border-right: 1px solid #1e1e30;
}

/* Step header */
.step-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white; padding: 24px 28px; border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    position: relative; overflow: hidden;
}
.step-header::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, #e94560, #c23152);
}
.step-header h2 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }
.step-header p  { margin: 6px 0 0 0; opacity: 0.6; font-size: 13px; font-weight: 400; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #13131f, #1a1a2e);
    border: 1px solid #1e1e35;
    border-radius: 12px; padding: 16px 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: #e94560; }

/* Buttons */
[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(233,69,96,0.3) !important;
}

/* Primary buttons */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #e94560, #c23152) !important;
    border: none !important;
    color: white !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #13131f !important;
    border: 2px dashed #2a2a40 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #e94560 !important;
}

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 500 !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #13131f !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 10px !important;
}

/* Progress bar */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #e94560, #c23152) !important;
    border-radius: 4px !important;
}

/* Sidebar logo area */
.sidebar-logo {
    text-align: center;
    padding: 8px 0 16px 0;
}
.sidebar-logo h1 {
    font-size: 20px; font-weight: 700;
    background: linear-gradient(135deg, #e94560, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0 2px 0;
}
.sidebar-logo p {
    font-size: 11px; color: #555; margin: 0;
    letter-spacing: 0.5px; text-transform: uppercase;
}

/* Badge */
.wine-badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: 11px;
    font-weight: 600; color: white;
    letter-spacing: 0.3px;
}

/* Info box */
.info-box {
    background: linear-gradient(135deg, #13131f, #1a1a2e);
    border: 1px solid #1e1e35; border-left: 3px solid #e94560;
    border-radius: 10px; padding: 14px 16px; margin: 8px 0;
    font-size: 13px;
}

/* Status badge */
.status-connected {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0d2b1f; border: 1px solid #1a5c3a;
    color: #4caf7d; border-radius: 20px;
    padding: 4px 12px; font-size: 12px; font-weight: 600;
}

/* Remove top padding */
.stApp > header { height: 0px !important; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 1100px !important;
}
[data-testid="stSidebarContent"] {
    padding-top: 1.5rem !important;
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

# ── Session state ──
defaults = {
    'step': 0, 'tiff_files': [], 'file_labels': {},
    'rois': None, 'ref_raw': None, 'model': None,
    'training_done': False, 'gdrive_token': None,
    'gdrive_folder_id': 'root', 'gdrive_folder_name': 'My Drive',
    'gdrive_breadcrumb': [('root', 'My Drive')],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Google helpers ──
def get_google_creds():
    if not GOOGLE_AVAILABLE:
        return None
    try:
        # Support [google_oauth] (preferred) or legacy [auth] without cookie_secret
        if "google_oauth" in st.secrets:
            section = st.secrets.google_oauth
        elif "auth" in st.secrets:
            section = st.secrets.auth
        else:
            return None
        return {
            "client_id":     section.client_id,
            "client_secret": section.client_secret,
            "redirect_uri":  section.redirect_uri,
        }
    except Exception:
        return None

def get_auth_url():
    c = get_google_creds()
    if not c:
        return None
    params = {
        'client_id':     c['client_id'],
        'redirect_uri':  c['redirect_uri'],
        'response_type': 'code',
        'scope':         'https://www.googleapis.com/auth/drive.readonly',
        'access_type':   'offline',
        'prompt':        'consent',
    }
    return 'https://accounts.google.com/o/oauth2/auth?' + urllib.parse.urlencode(params)

def fetch_token(code):
    c = get_google_creds()
    if not c:
        return None
    resp = req_lib.post('https://oauth2.googleapis.com/token', data={
        'code':          code,
        'client_id':     c['client_id'],
        'client_secret': c['client_secret'],
        'redirect_uri':  c['redirect_uri'],
        'grant_type':    'authorization_code',
    })
    return resp.json()

def get_drive_service():
    token = st.session_state.gdrive_token
    if not token or not GOOGLE_AVAILABLE:
        return None
    c = get_google_creds()
    creds = Credentials(
        token=token['access_token'],
        refresh_token=token.get('refresh_token'),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=c['client_id'],
        client_secret=c['client_secret'],
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=creds)

def list_drive_folder(service, folder_id='root'):
    q = (f"'{folder_id}' in parents and trashed=false and "
         f"(mimeType='application/vnd.google-apps.folder' "
         f"or name contains '.tiff' or name contains '.tif')")
    res = service.files().list(
        q=q, fields="files(id,name,mimeType,size)",
        orderBy="folder,name").execute()
    return res.get('files', [])

def download_drive_file(service, file_id):
    req = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    dl  = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    buf.seek(0)
    return buf.read()

# ── Handle OAuth callback ──
try:
    params = st.query_params
    if 'code' in params and st.session_state.gdrive_token is None:
        token_data = fetch_token(params['code'])
        if token_data and 'access_token' in token_data:
            st.session_state.gdrive_token = {
                'access_token':  token_data['access_token'],
                'refresh_token': token_data.get('refresh_token'),
            }
            st.query_params.clear()
            st.rerun()
        else:
            err = token_data.get('error_description', str(token_data)) if token_data else 'Unknown'
            st.sidebar.error(f"Token error: {err}")
except Exception as e:
    st.sidebar.error(f"OAuth error: {e}")

# ── Core helpers ──
def get_label(name):
    n = name.lower()
    for lbl in ['dao','odc','ln','lo','pn','po']:
        if lbl in n:
            return {'dao':'Dao','odc':'ODC','ln':'LN',
                    'lo':'LO','pn':'PN','po':'PO'}[lbl]
    return 'Unknown'

def load_and_demosaic(data):
    raw = tifffile.imread(io.BytesIO(data)).astype(np.float32)
    if raw.ndim == 3: raw = raw.mean(axis=2)
    raw = raw[:(raw.shape[0]//TILE_H)*TILE_H,
               :(raw.shape[1]//TILE_W)*TILE_W]
    return raw, [raw[r::TILE_H, c::TILE_W]
                 for r in range(TILE_H) for c in range(TILE_W)]

def extract_patches(channels, x0, y0, x1, y1, px=PATCH_SIZE):
    sx, sy = px//TILE_W, px//TILE_H
    feats  = []
    for py in range(y0//TILE_H, y1//TILE_H-sy+1, sy):
        for px2 in range(x0//TILE_W, x1//TILE_W-sx+1, sx):
            m = np.array([ch[py:py+sy, px2:px2+sx].mean()
                          for ch in channels], dtype=np.float32)
            s = np.array([ch[py:py+sy, px2:px2+sx].std()
                          for ch in channels], dtype=np.float32)
            feats.append(np.concatenate([m/(m.sum()+1e-9), s]))
    return feats

def disp_img(raw):
    vmin, vmax = np.percentile(raw, 1), np.percentile(raw, 99)
    return np.clip((raw-vmin)/(vmax-vmin+1e-9), 0, 1)

def can_advance(to_step):
    if to_step >= 1 and not st.session_state.tiff_files: return False
    if to_step >= 2 and st.session_state.rois is None:   return False
    if to_step >= 3 and not st.session_state.training_done: return False
    return True

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>
        <div style='font-size:72px;line-height:1;margin-bottom:10px;
             filter:drop-shadow(0 4px 16px rgba(233,69,96,0.5))'>🔬</div>
        <h1>Image Classifier</h1>
        <p>Hyperspectral Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    step_icons  = ["📁","🎯","🤖","🔍"]
    step_labels = ["1  Upload Data","2  Select ROI","3  Train Model","4  Predict"]
    step_done   = [len(st.session_state.tiff_files)>0,
                   st.session_state.rois is not None,
                   st.session_state.training_done, True]
    for i,(icon,lbl) in enumerate(zip(step_icons,step_labels)):
        prefix = "✅" if step_done[i] else ("🔒" if not can_advance(i) else icon)
        style  = "primary" if st.session_state.step==i else "secondary"
        if st.button(f"{prefix}  {lbl}", key=f"nav{i}",
                     use_container_width=True,
                     disabled=not can_advance(i), type=style):
            st.session_state.step = i
            st.rerun()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    n_done = sum(step_done[:3])
    st.markdown(f"""
    <div style='padding:4px 4px 12px'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
            <span style='font-size:10px;color:#3a3a5c;text-transform:uppercase;letter-spacing:1px;font-weight:600'>Progress</span>
            <span style='font-size:11px;color:#e94560;font-weight:700'>{n_done} / 3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.progress(n_done/3)
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    if st.session_state.tiff_files:
        st.markdown("---")
        st.markdown("**📂 Training Files**")
        for i,(name,_) in enumerate(st.session_state.tiff_files):
            lbl = st.session_state.file_labels.get(name,'Unknown')
            col = WINE_COLORS.get(lbl,'#555')
            c1,c2 = st.columns([4,1])
            c1.markdown(
                f"<div style='font-size:11px;padding:2px 0'>"
                f"<span style='background:{col};color:white;padding:1px 6px;"
                f"border-radius:8px;font-size:10px'>{lbl}</span> "
                f"{name[:22]}{'…' if len(name)>22 else ''}</div>",
                unsafe_allow_html=True)
            if c2.button("✕", key=f"del_{i}", help=f"Remove {name}"):
                st.session_state.tiff_files.pop(i)
                st.session_state.file_labels.pop(name, None)
                if not st.session_state.tiff_files:
                    st.session_state.rois = None
                    st.session_state.training_done = False
                    st.session_state.model = None
                st.rerun()
        if st.button("🗑️ Remove All", use_container_width=True):
            st.session_state.tiff_files    = []
            st.session_state.file_labels   = {}
            st.session_state.rois          = None
            st.session_state.training_done = False
            st.session_state.model         = None
            st.rerun()

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

    # ── Google Drive panel ──
    st.markdown("---")
    st.markdown("### ☁️ Google Drive")
    gcreds = get_google_creds()

    if not GOOGLE_AVAILABLE:
        st.caption("⚠️ Google libraries not installed.")
    elif gcreds is None:
        st.caption("Add Google credentials in Streamlit secrets to enable Drive.")
    elif st.session_state.gdrive_token is None:
        auth_url = get_auth_url()
        if auth_url:
            st.link_button("🔗 Connect Google Drive",
                           auth_url, use_container_width=True)
            st.caption("Connect once — browse and import files directly.")
    else:
        st.success("✅ Drive connected")
        if st.button("🔓 Disconnect", use_container_width=True):
            st.session_state.gdrive_token       = None
            st.session_state.gdrive_folder_id   = 'root'
            st.session_state.gdrive_folder_name = 'My Drive'
            st.session_state.gdrive_breadcrumb  = [('root','My Drive')]
            st.rerun()

    st.markdown("---")

# ============================================================
#  STEP 1 — Upload Training Data
# ============================================================
if st.session_state.step == 0:
    st.markdown("""<div class='step-header'>
        <h2>📁 Step 1 — Upload Training Data</h2>
        <p>Upload individual TIFF images or a ZIP containing labelled TIFFs.</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("ℹ️ What kind of images do I need?"):
        st.markdown("""
        **Camera & Format**
        Designed for the **Basler daA2500** with a 9×9 mosaic spectral filter (81 bands).
        Images must be raw `.tiff` files — do not convert to JPEG or PNG.

        **How many images per class?**
        Minimum **2 per class**, 3 or more recommended.

        **File naming**
        The class label is detected from the filename automatically.
        Just make sure the label appears somewhere in the filename:
        `Dao_100K_1.tiff`, `ODC_sample2.tiff` etc.

        **Important:** All images should be captured under the same exposure,
        distance and lighting conditions.
        """)

    col_up, col_info = st.columns([3,2])

    # Track new files added this run (for ZIP and Drive)
    zf = None

    with col_up:
        tab_local, tab_drive = st.tabs(["💻 From Computer", "☁️ Google Drive"])

        # ── LOCAL UPLOAD TAB ──
        with tab_local:
            col_tiff, col_zip = st.columns(2)

            # ── LEFT: Individual TIFFs ──
            with col_tiff:
                st.markdown("""
                <div style='background:#13131f;border:1px solid #2a2a40;border-radius:12px;
                     padding:14px 16px 8px;margin-bottom:4px'>
                    <div style='font-size:13px;font-weight:600;color:#e94560;margin-bottom:10px'>
                        🖼️ Individual TIFFs
                    </div>
                """, unsafe_allow_html=True)
                uploaded_tiffs = st.file_uploader(
                    "Select .tiff files",
                    type=["tiff", "tif"],
                    accept_multiple_files=True,
                    key="step1_tiff_uploader",
                )
                st.markdown("</div>", unsafe_allow_html=True)
                if uploaded_tiffs:
                    existing_names = [n for n, _ in st.session_state.tiff_files]
                    added = []
                    for f in uploaded_tiffs:
                        if f.name not in existing_names:
                            data = f.read()
                            st.session_state.tiff_files.append((f.name, data))
                            st.session_state.file_labels[f.name] = get_label(f.name)
                            added.append(f.name)
                    if added:
                        st.success(f"✅ Added {len(added)} TIFF(s)")
                        st.rerun()

            # ── RIGHT: ZIP ──
            with col_zip:
                st.markdown("""
                <div style='background:#13131f;border:1px solid #2a2a40;border-radius:12px;
                     padding:14px 16px 8px;margin-bottom:4px'>
                    <div style='font-size:13px;font-weight:600;color:#e94560;margin-bottom:10px'>
                        📦 ZIP File
                    </div>
                """, unsafe_allow_html=True)
                zf = st.file_uploader(
                    "Select a ZIP",
                    type=["zip"],
                    key="step1_zip_uploader",
                )
                st.markdown("</div>", unsafe_allow_html=True)

        # ── GOOGLE DRIVE TAB ──
        with tab_drive:
            if st.session_state.gdrive_token is None:
                st.info("🔗 Connect Google Drive from the sidebar first, then browse your folders here.")
            else:
                try:
                    service = get_drive_service()

                    bc = st.session_state.gdrive_breadcrumb
                    bc_html = " › ".join(
                        [f"<b>{n}</b>" if i==len(bc)-1 else n for i,(_, n) in enumerate(bc)])
                    st.markdown(f"<small>📂 {bc_html}</small>", unsafe_allow_html=True)

                    nav1, nav2, nav3 = st.columns([1,1,3])
                    if len(bc) > 1:
                        if nav1.button("⬆️ Up", key="s1_up", use_container_width=True):
                            st.session_state.gdrive_breadcrumb.pop()
                            st.session_state.gdrive_folder_id = bc[-2][0]
                            st.rerun()
                        if nav2.button("🏠", key="s1_home", use_container_width=True):
                            st.session_state.gdrive_breadcrumb = [('root','My Drive')]
                            st.session_state.gdrive_folder_id  = 'root'
                            st.rerun()

                    search = nav3.text_input("🔍 Search files",
                                             placeholder="type to filter...",
                                             key="drive_search",
                                             label_visibility='collapsed')

                    items   = list_drive_folder(service, st.session_state.gdrive_folder_id)
                    folders = [i for i in items
                               if i['mimeType']=='application/vnd.google-apps.folder']
                    tiffs   = [i for i in items
                               if i['mimeType']!='application/vnd.google-apps.folder']

                    if search:
                        folders = [f for f in folders if search.lower() in f['name'].lower()]
                        tiffs   = [f for f in tiffs   if search.lower() in f['name'].lower()]

                    if folders:
                        st.markdown("<small><b>📁 Folders</b></small>", unsafe_allow_html=True)
                        FCOLS = 3
                        for row_start in range(0, len(folders), FCOLS):
                            row_folders = folders[row_start:row_start+FCOLS]
                            cols = st.columns(FCOLS)
                            for ci, folder in enumerate(row_folders):
                                name_short = folder['name'][:14] + ('…' if len(folder['name'])>14 else '')
                                if cols[ci].button(f"📁 {name_short}",
                                                   key=f"s1_fd_{folder['id']}",
                                                   use_container_width=True,
                                                   help=folder['name']):
                                    st.session_state.gdrive_breadcrumb.append(
                                        (folder['id'], folder['name']))
                                    st.session_state.gdrive_folder_id = folder['id']
                                    st.rerun()

                    if tiffs:
                        PAGE_SIZE = 15
                        total_pages = max(1, (len(tiffs) + PAGE_SIZE - 1) // PAGE_SIZE)
                        page_key = 'drive_page'
                        if page_key not in st.session_state:
                            st.session_state[page_key] = 0
                        if search:
                            st.session_state[page_key] = 0
                        page     = st.session_state[page_key]
                        page_tiffs = tiffs[page*PAGE_SIZE:(page+1)*PAGE_SIZE]
                        existing = [n for n,_ in st.session_state.tiff_files]

                        st.markdown(
                            f"<small><b>🖼️ TIFF Files</b> — "
                            f"{len(tiffs)} total, showing {page*PAGE_SIZE+1}–"
                            f"{min((page+1)*PAGE_SIZE, len(tiffs))}</small>",
                            unsafe_allow_html=True)

                        for f in page_tiffs:
                            size_kb = int(f.get('size',0))//1024
                            lbl     = get_label(f['name'])
                            col     = WINE_COLORS.get(lbl,'#555')
                            already = f['name'] in existing
                            ca, cb  = st.columns([6,1])
                            ca.markdown(
                                f"<div style='padding:2px 0;font-size:12px'>"
                                f"<span style='background:{col};color:white;"
                                f"padding:1px 6px;border-radius:6px;font-size:10px'>{lbl}</span> "
                                f"{f['name'][:32]}{'…' if len(f['name'])>32 else ''} "
                                f"<span style='color:#666'>({size_kb}KB)</span>"
                                f"{'  ✅' if already else ''}</div>",
                                unsafe_allow_html=True)
                            if not already:
                                if cb.button("➕", key=f"s1_add_{f['id']}",
                                             help=f"Add {f['name']}"):
                                    with st.spinner(f"Downloading..."):
                                        data = download_drive_file(service, f['id'])
                                        st.session_state.tiff_files.append((f['name'], data))
                                        st.session_state.file_labels[f['name']] = lbl
                                        st.toast(f"✅ Added {f['name']}")
                                        st.rerun()

                        if total_pages > 1:
                            pc1, pc2, pc3 = st.columns([1,2,1])
                            if pc1.button("◀ Prev", disabled=page==0,
                                          key="s1_prev", use_container_width=True):
                                st.session_state[page_key] -= 1; st.rerun()
                            pc2.markdown(
                                f"<div style='text-align:center;padding:6px;font-size:12px'>"
                                f"Page {page+1} / {total_pages}</div>",
                                unsafe_allow_html=True)
                            if pc3.button("Next ▶", disabled=page>=total_pages-1,
                                          key="s1_next", use_container_width=True):
                                st.session_state[page_key] += 1; st.rerun()

                        not_added = [f for f in tiffs if f['name'] not in existing]
                        if not_added:
                            if st.button(f"➕ Add all {len(not_added)} remaining TIFFs",
                                         type="primary", use_container_width=True,
                                         key="s1_add_all"):
                                prog = st.progress(0, text="Downloading...")
                                for i,f in enumerate(not_added):
                                    data = download_drive_file(service, f['id'])
                                    st.session_state.tiff_files.append((f['name'], data))
                                    st.session_state.file_labels[f['name']] = get_label(f['name'])
                                    prog.progress((i+1)/len(not_added),
                                                  text=f"Downloaded {f['name']}")
                                st.toast(f"✅ Added {len(not_added)} files")
                                st.rerun()
                        else:
                            st.success("✅ All files in this folder already added!")

                    elif not folders and not search:
                        st.caption("No TIFF files or folders found here.")
                    elif not folders and not tiffs:
                        st.caption(f"No results for '{search}'")

                except Exception as e:
                    st.error(f"Drive error: {e}")

    with col_info:
        st.markdown("""
        **Supported labels:**
        | Label | Example |
        |-------|---------|
        | Dao   | `Dao_100K_1.tiff` |
        | LN    | `LN_100K_1.tiff`  |
        | LO    | `LO_100K_1.tiff`  |
        | ODC   | `ODC_100K_1.tiff` |
        | PN    | `PN_100K_1.tiff`  |
        | PO    | `PO_100K_1.tiff`  |
        """)
        st.markdown("""
        **Upload options:**
        - 🖼️ **Individual TIFFs** — select one or more `.tiff` files directly
        - 📦 **ZIP** — a ZIP containing multiple labelled TIFFs
        - ☁️ **Google Drive** — browse and import from your Drive
        """)

    # ── Process ZIP if uploaded ──
    if zf:
        with st.spinner("Reading ZIP..."):
            zdata = zf if isinstance(zf, io.BytesIO) else io.BytesIO(zf.read())
            with zipfile.ZipFile(zdata,'r') as z:
                names = [n for n in z.namelist()
                         if n.lower().endswith(('.tiff','.tif'))
                         and not os.path.basename(n).startswith('.')]
                existing = [f[0] for f in st.session_state.tiff_files]
                added = []
                for name in names:
                    bname = os.path.basename(name)
                    if bname not in existing:
                        st.session_state.tiff_files.append((bname, z.read(name)))
                        st.session_state.file_labels[bname] = get_label(bname)
                        added.append(bname)
        if added:
            st.success(f"✅ Added {len(added)} files from ZIP")

    # ── File list summary ──
    if st.session_state.tiff_files:
        st.markdown("### 📋 Uploaded Files")
        label_counts = {}
        for name,_ in st.session_state.tiff_files:
            lbl = st.session_state.file_labels.get(name,'Unknown')
            label_counts[lbl] = label_counts.get(lbl,0)+1

        chips = []
        for l, c in sorted(label_counts.items()):
            bg = WINE_COLORS.get(l, '#555')
            chips.append(
                f"<span style='background:{bg};color:white;"
                f"padding:4px 12px;border-radius:20px;"
                f"font-size:13px;font-weight:bold'>{l}: {c}</span>")
        st.markdown(" ".join(chips), unsafe_allow_html=True)
        st.markdown("")

        for name,_ in st.session_state.tiff_files:
            lbl = st.session_state.file_labels.get(name,'Unknown')
            col = WINE_COLORS.get(lbl,'#555')
            ca,cb = st.columns([5,2])
            ca.markdown(f"🖼️ `{name}`")
            cb.markdown(
                f"<span style='background:{col};color:white;padding:2px 10px;"
                f"border-radius:10px;font-size:12px'>{lbl}</span>",
                unsafe_allow_html=True)

        if 'Unknown' in label_counts:
            st.warning(f"⚠️ {label_counts['Unknown']} file(s) have unrecognised labels.")

        st.markdown("---")
        if st.button("✅ Confirm & Go to ROI Selection →",
                     type="primary", use_container_width=True):
            _,data = st.session_state.tiff_files[0]
            raw,_  = load_and_demosaic(data)
            st.session_state.ref_raw = raw
            st.session_state.step = 1
            st.rerun()

# ============================================================
#  STEP 2 — ROI Selector  (canvas drag + PIL zoom previews)
# ============================================================
elif st.session_state.step == 1:
    from PIL import Image as PILImage
    import base64

    st.markdown("""<div class='step-header'>
        <h2>🎯 Step 2 — Select ROI Regions</h2>
        <p>Drag the boxes on the image to position them, then fine-tune with sliders.</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("ℹ️ How does this step work?"):
        st.markdown("""
        **Drag** the coloured boxes anywhere on the image.
        Use the **sliders** below to fine-tune size and position.
        The **zoom panels** on the right show the actual pixels inside each ROI —
        use them to check you're covering the sample, not the glass edge.
        """)

    if not st.session_state.tiff_files:
        st.warning("⚠️ Go back to Step 1 first."); st.stop()

    raw = st.session_state.ref_raw
    if raw is None:
        _, data = st.session_state.tiff_files[0]
        raw, _ = load_and_demosaic(data)
        st.session_state.ref_raw = raw
    H_orig, W_orig = raw.shape
    ref_name = st.session_state.tiff_files[0][0]

    # ── Cache uint8 RGB array from full-res raw (computed once) ──
    if ('roi_base_img' not in st.session_state
            or st.session_state.get('roi_base_src') != ref_name):
        disp = (disp_img(raw) * 255).astype(np.uint8)
        st.session_state.roi_base_img = np.stack([disp, disp, disp], axis=2)
        st.session_state.roi_base_src = ref_name

    base_rgb = st.session_state.roi_base_img   # full-res H×W×3 uint8

    # ── Cache downscaled JPEG base64 for the canvas (once) ──
    CANVAS_MAX_W = 820
    if ('roi_canvas_b64' not in st.session_state
            or st.session_state.get('roi_canvas_src') != ref_name):
        _scale  = min(1.0, CANVAS_MAX_W / W_orig)
        _cw, _ch = int(W_orig * _scale), int(H_orig * _scale)
        _cimg = PILImage.fromarray(base_rgb).resize((_cw, _ch), PILImage.LANCZOS)
        _buf  = io.BytesIO()
        _cimg.save(_buf, format='JPEG', quality=85)
        st.session_state.roi_canvas_b64   = base64.b64encode(_buf.getvalue()).decode()
        st.session_state.roi_canvas_src   = ref_name
        st.session_state.roi_canvas_w     = _cw
        st.session_state.roi_canvas_h     = _ch
        st.session_state.roi_canvas_scale = _scale

    b64      = st.session_state.roi_canvas_b64
    canvas_w = st.session_state.roi_canvas_w
    canvas_h = st.session_state.roi_canvas_h
    c_scale  = st.session_state.roi_canvas_scale
    inv_x    = W_orig / canvas_w
    inv_y    = H_orig / canvas_h

    # ── Read dragged positions from URL query params ──
    qp = st.query_params
    has_drag = 'drag_roi1' in qp and 'drag_roi2' in qp

    if has_drag:
        st.info("📍 Boxes moved — click **Apply** to confirm positions.")
        if st.button("📍 Apply dragged positions", type="primary", use_container_width=True):
            try:
                r1 = list(map(int, qp['drag_roi1'].split(',')))
                r2 = list(map(int, qp['drag_roi2'].split(',')))
                def _cl(v, lo, hi): return max(lo, min(hi, v))
                st.session_state.cx1 = _cl((r1[0]+r1[2])//2, 0, W_orig)
                st.session_state.cy1 = _cl((r1[1]+r1[3])//2, 0, H_orig)
                st.session_state.rw1 = _cl(r1[2]-r1[0], 30, min(400,W_orig))
                st.session_state.rh1 = _cl(r1[3]-r1[1], 30, min(400,H_orig))
                st.session_state.cx2 = _cl((r2[0]+r2[2])//2, 0, W_orig)
                st.session_state.cy2 = _cl((r2[1]+r2[3])//2, 0, H_orig)
                st.session_state.rw2 = _cl(r2[2]-r2[0], 30, min(400,W_orig))
                st.session_state.rh2 = _cl(r2[3]-r2[1], 30, min(400,H_orig))
                st.query_params.clear()
            except Exception as ex:
                st.error(f"Could not apply: {ex}")
            st.rerun()

    st.caption(f"Reference: `{ref_name}`  |  {W_orig}×{H_orig} px")

    # ── Layout: canvas left, zoom previews right ──
    col_canvas, col_zoom = st.columns([3, 2])

    # ── Sliders ──
    with st.container():
        cs1, cs2 = st.columns(2)
        with cs1:
            st.markdown("##### 🔵 ROI 1")
            cx1 = st.slider("Center X ",  0, W_orig, st.session_state.get('cx1', W_orig//2-100), 5,  key='cx1')
            cy1 = st.slider("Center Y ",  0, H_orig, st.session_state.get('cy1', H_orig//2),     5,  key='cy1')
            rw1 = st.slider("Width  ",   30, min(400,W_orig), st.session_state.get('rw1', 150),   10, key='rw1')
            rh1 = st.slider("Height  ",  30, min(400,H_orig), st.session_state.get('rh1', 120),   10, key='rh1')
        with cs2:
            st.markdown("##### 🟠 ROI 2")
            cx2 = st.slider("Center X  ", 0, W_orig, st.session_state.get('cx2', W_orig//2+100), 5,  key='cx2')
            cy2 = st.slider("Center Y  ", 0, H_orig, st.session_state.get('cy2', H_orig//2),     5,  key='cy2')
            rw2 = st.slider("Width   ",  30, min(400,W_orig), st.session_state.get('rw2', 150),   10, key='rw2')
            rh2 = st.slider("Height   ", 30, min(400,H_orig), st.session_state.get('rh2', 120),   10, key='rh2')

    roi1 = (max(0,cx1-rw1//2), max(0,cy1-rh1//2), min(W_orig,cx1+rw1//2), min(H_orig,cy1+rh1//2))
    roi2 = (max(0,cx2-rw2//2), max(0,cy2-rh2//2), min(W_orig,cx2+rw2//2), min(H_orig,cy2+rh2//2))

    # Scale ROI to canvas coords for JS
    def _sc(v, s): return int(round(v * s))
    sr1 = [_sc(roi1[0],c_scale), _sc(roi1[1],c_scale), _sc(roi1[2],c_scale), _sc(roi1[3],c_scale)]
    sr2 = [_sc(roi2[0],c_scale), _sc(roi2[1],c_scale), _sc(roi2[2],c_scale), _sc(roi2[3],c_scale)]

    # ── Canvas HTML (move only, no resize handles) ──
    with col_canvas:
        canvas_html = f"""<!DOCTYPE html><html><head>
<style>
  body{{margin:0;padding:0;background:#0a0a0f;overflow:hidden;}}
  #c{{display:block;width:100%;cursor:default;}}
  #tip{{font-family:monospace;font-size:11px;color:#666;padding:3px 6px;
        background:#111;border-top:1px solid #1e1e30;white-space:nowrap;overflow:hidden;}}
</style></head><body>
<canvas id="c" width="{canvas_w}" height="{canvas_h}"></canvas>
<div id="tip">Drag a box to reposition it — then click Apply above</div>
<script>
const canvas=document.getElementById('c'),ctx=canvas.getContext('2d');
const INV_X={inv_x},INV_Y={inv_y};

let rois=[
  {{x0:{sr1[0]},y0:{sr1[1]},x1:{sr1[2]},y1:{sr1[3]},color:'#1E90FF',fill:'rgba(30,144,255,0.15)',name:'ROI 1'}},
  {{x0:{sr2[0]},y0:{sr2[1]},x1:{sr2[2]},y1:{sr2[3]},color:'#FFA500',fill:'rgba(255,165,0,0.15)',name:'ROI 2'}}
];

const img=new Image();
img.onload=()=>draw();
img.src='data:image/jpeg;base64,{b64}';

function draw(){{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(img.complete&&img.naturalWidth) ctx.drawImage(img,0,0,canvas.width,canvas.height);
  rois.forEach(roi=>{{
    const {{x0,y0,x1,y1,color,fill,name}}=roi;
    // Subtle fill
    ctx.fillStyle=fill;
    ctx.fillRect(x0,y0,x1-x0,y1-y0);
    // Clean solid border, 2px
    ctx.strokeStyle=color;
    ctx.lineWidth=2;
    ctx.strokeRect(x0+1,y0+1,x1-x0-2,y1-y0-2);
    // Small label pill
    ctx.font='bold 11px sans-serif';
    const tw=ctx.measureText(name).width;
    ctx.fillStyle=color;
    ctx.globalAlpha=0.9;
    ctx.beginPath();
    ctx.roundRect?ctx.roundRect(x0+5,y0+5,tw+12,18,4):ctx.rect(x0+5,y0+5,tw+12,18);
    ctx.fill();
    ctx.globalAlpha=1;
    ctx.fillStyle='#fff';
    ctx.fillText(name,x0+11,y0+17);
  }});
}}

function getPos(e){{
  const r=canvas.getBoundingClientRect();
  const sx=canvas.width/r.width,sy=canvas.height/r.height;
  const src=e.touches?e.touches[0]:e;
  return {{x:(src.clientX-r.left)*sx,y:(src.clientY-r.top)*sy}};
}}

let drag=null;

function hitTest(x,y){{
  for(let i=rois.length-1;i>=0;i--){{
    const {{x0,y0,x1,y1}}=rois[i];
    if(x>x0&&x<x1&&y>y0&&y<y1) return i;
  }}
  return -1;
}}

function onDown(e){{
  const p=getPos(e),i=hitTest(p.x,p.y);
  if(i<0) return;
  const roi=rois[i];
  drag={{i,sx:p.x,sy:p.y,ox0:roi.x0,oy0:roi.y0,ox1:roi.x1,oy1:roi.y1}};
  canvas.style.cursor='grabbing';
  e.preventDefault();
}}

function onMove(e){{
  const p=getPos(e);
  if(!drag){{
    canvas.style.cursor=hitTest(p.x,p.y)>=0?'grab':'default';
    return;
  }}
  const dx=p.x-drag.sx,dy=p.y-drag.sy;
  const roi=rois[drag.i];
  const W=canvas.width,H=canvas.height;
  const w=drag.ox1-drag.ox0,h=drag.oy1-drag.oy0;
  roi.x0=Math.round(Math.max(0,Math.min(W-w,drag.ox0+dx)));
  roi.y0=Math.round(Math.max(0,Math.min(H-h,drag.oy0+dy)));
  roi.x1=roi.x0+w; roi.y1=roi.y0+h;
  draw(); updateTip();
  e.preventDefault();
}}

function onUp(){{
  if(!drag) return;
  drag=null;
  canvas.style.cursor='default';
  pushToParent();
}}

function toOrig(v,inv){{return Math.round(v*inv);}}

function updateTip(){{
  const r1=rois[0],r2=rois[1];
  document.getElementById('tip').innerText=
    `🔵(${{toOrig(r1.x0,INV_X)}},${{toOrig(r1.y0,INV_Y)}})`+
    `→(${{toOrig(r1.x1,INV_X)}},${{toOrig(r1.y1,INV_Y)}})  `+
    `🟠(${{toOrig(r2.x0,INV_X)}},${{toOrig(r2.y0,INV_Y)}})`+
    `→(${{toOrig(r2.x1,INV_X)}},${{toOrig(r2.y1,INV_Y)}})  ← Apply ↑`;
}}

function pushToParent(){{
  try{{
    const url=new URL(window.parent.location.href);
    const r1=rois[0],r2=rois[1];
    url.searchParams.set('drag_roi1',
      [toOrig(r1.x0,INV_X),toOrig(r1.y0,INV_Y),toOrig(r1.x1,INV_X),toOrig(r1.y1,INV_Y)].join(','));
    url.searchParams.set('drag_roi2',
      [toOrig(r2.x0,INV_X),toOrig(r2.y0,INV_Y),toOrig(r2.x1,INV_X),toOrig(r2.y1,INV_Y)].join(','));
    window.parent.history.replaceState({{}},'',url.toString());
  }}catch(err){{console.warn(err);}}
}}

canvas.addEventListener('mousedown', onDown);
canvas.addEventListener('mousemove', onMove);
canvas.addEventListener('mouseup',   onUp);
canvas.addEventListener('mouseleave',onUp);
canvas.addEventListener('touchstart',onDown,{{passive:false}});
canvas.addEventListener('touchmove', onMove,{{passive:false}});
canvas.addEventListener('touchend',  onUp);
</script></body></html>"""
        st.components.v1.html(canvas_html, height=canvas_h + 26, scrolling=False)

    # ── ROI zoom previews from FULL-RES array (sharp, real pixels) ──
    with col_zoom:
        st.markdown("##### 🔍 ROI Zoom (actual pixels)")

        def zoom_roi(arr, roi, target_px=320):
            """Crop ROI from full-res array, zoom with NEAREST to show real pixels."""
            x0,y0,x1,y1 = roi
            crop = arr[y0:y1, x0:x1]
            if crop.size == 0:
                return PILImage.fromarray(np.zeros((10,10,3),dtype=np.uint8))
            ch, cw = crop.shape[:2]
            # Zoom so the larger dimension fills target_px
            zoom_factor = max(1, target_px // max(ch, cw))
            new_w = cw * zoom_factor
            new_h = ch * zoom_factor
            # Cap at 2x target to avoid huge images
            if new_w > target_px * 2:
                new_w = target_px * 2
                new_h = int(ch * new_w / cw)
            pil = PILImage.fromarray(crop)
            # NEAREST keeps pixel boundaries crisp — best for scientific inspection
            return pil.resize((new_w, new_h), PILImage.NEAREST)

        z1 = zoom_roi(base_rgb, roi1)
        z2 = zoom_roi(base_rgb, roi2)

        tab1, tab2 = st.tabs(["🔵 ROI 1", "🟠 ROI 2"])
        with tab1:
            st.image(z1, use_container_width=True)
            st.caption(f"Source: {roi1[2]-roi1[0]}×{roi1[3]-roi1[1]} px  |  "
                       f"Zoom ×{max(1, 320//(max(roi1[3]-roi1[1],roi1[2]-roi1[0]) or 1))}")
        with tab2:
            st.image(z2, use_container_width=True)
            st.caption(f"Source: {roi2[2]-roi2[0]}×{roi2[3]-roi2[1]} px  |  "
                       f"Zoom ×{max(1, 320//(max(roi2[3]-roi2[1],roi2[2]-roi2[0]) or 1))}")

    # ── Coordinate readout ──
    st.markdown(
        f"<div style='background:#0d1117;border:1px solid #30363d;border-radius:8px;"
        f"padding:10px 16px;margin:8px 0;font-size:13px'>"
        f"<b style='color:#1E90FF'>ROI 1:</b> "
        f"({roi1[0]}, {roi1[1]}) → ({roi1[2]}, {roi1[3]}) &nbsp;|&nbsp; "
        f"{roi1[2]-roi1[0]}×{roi1[3]-roi1[1]} px"
        f"&emsp;<b style='color:#FFA500'>ROI 2:</b> "
        f"({roi2[0]}, {roi2[1]}) → ({roi2[2]}, {roi2[3]}) &nbsp;|&nbsp; "
        f"{roi2[2]-roi2[0]}×{roi2[3]-roi2[1]} px"
        f"</div>", unsafe_allow_html=True)

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
        <p>Extracts spectral patches and trains an ensemble classifier.</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.tiff_files:
        st.warning("⚠️ Complete Step 1 first."); st.stop()
    if st.session_state.rois is None:
        st.warning("⚠️ Complete Step 2 first."); st.stop()

    with st.expander("ℹ️ How does training work?"):
        st.markdown("""
        **Pipeline:**
        1. Each TIFF is demosaiced into 81 spectral band images
        2. Each ROI is divided into 30×30 px patches → each patch = one training sample (162 features)
        3. Ensemble classifier (SVM + RF + XGB) learns the spectral signature of each class

        **Leave-One-Image-Out validation:** trained on N-1 images, tested on 1, repeated N times.
        """)

    st.markdown("### 🧠 Classifier Selection")
    col_table, col_pick = st.columns([3,2])

    with col_table:
        st.markdown("**Research comparison — hyperspectral classification:**")
        st.markdown("""
| Model | Small data? | Speed | Recommended |
|-------|-------------|-------|-------------|
| **Ensemble (SVM+RF+XGB)** | ✅ Best | ✅ Fast | ⭐ Default |
| SVM RBF | ✅ Best | ✅ Fast | ✅ Good |
| Random Forest | ✅ Good | ✅ Fast | ✅ Good |
| XGBoost | ✅ Good | ✅ Fast | ✅ Good |
| CNN / Deep Learning | ❌ Needs 1000s | ❌ Slow | ⚠️ Not yet |
        """)

    with col_pick:
        model_choice = st.radio("Select classifier:", [
            "⭐ Ensemble (SVM + RF + XGB)",
            "SVM RBF", "Random Forest", "XGBoost"])
        descs = {
            "Ensemble":      "3 models vote together — most robust for small datasets.",
            "SVM RBF":       "Excellent for high-dimensional spectral data.",
            "Random Forest": "Handles non-linear band interactions.",
            "XGBoost":       "Gradient boosting — catches patterns others may miss.",
        }
        key = "Ensemble" if "Ensemble" in model_choice else model_choice
        st.info(descs.get(key,""))

    if st.session_state.training_done:
        st.success("✅ Model already trained. Retrain below or go to Step 4.")

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        rois     = st.session_state.rois
        labelled = [(n,d) for n,d in st.session_state.tiff_files
                    if st.session_state.file_labels.get(n,'Unknown') != 'Unknown']

        prog = st.progress(0, text="Starting...")
        X_all, y_all, groups, raw_spectra = [], [], [], {}
        img_id = 0

        for i,(name,data) in enumerate(labelled):
            label = st.session_state.file_labels[name]
            prog.progress((i+1)/len(labelled), text=f"Processing {name}...")
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

        X, y, groups = np.array(X_all), np.array(y_all), np.array(groups)
        all_labels = sorted(np.unique(y).tolist())
        n_cls = len(all_labels)

        t1,t2,t3,t4 = st.tabs(["📈 Spectra","🔬 PCA","📊 Confusion","📋 Per-Image"])

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
            Xp = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
            fig2,ax = plt.subplots(figsize=(8,5))
            for lbl in np.unique(y):
                mask = y==lbl; col = WINE_COLORS.get(lbl,'gray')
                ax.scatter(Xp[mask,0],Xp[mask,1],c=col,alpha=0.2,s=8,label=lbl)
                cx_,cy_ = Xp[mask,0].mean(),Xp[mask,1].mean()
                ax.scatter(cx_,cy_,c=col,s=220,marker='*',
                           edgecolors='black',lw=0.8,zorder=5)
                ax.annotate(lbl,(cx_,cy_),xytext=(6,4),
                            textcoords='offset points',fontsize=11,
                            fontweight='bold',color=col)
            ax.set_title("PCA clusters (stars = centres)")
            ax.legend(markerscale=4,fontsize=9); ax.grid(True,alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig2,use_container_width=True); plt.close(fig2)

        try:
            from xgboost import XGBClassifier
            has_xgb = True
        except ImportError:
            has_xgb = False

        sel = "Ensemble" if "Ensemble" in model_choice else model_choice
        if sel == "Ensemble":
            ests = [('svm', SVC(kernel='rbf',C=10,gamma='scale',
                                probability=True,random_state=42)),
                    ('rf',  RandomForestClassifier(n_estimators=200,random_state=42))]
            if has_xgb:
                ests.append(('xgb', XGBClassifier(n_estimators=100,random_state=42,
                                                   eval_metric='mlogloss',verbosity=0)))
            clf      = VotingClassifier(estimators=ests, voting='soft')
            clf_name = f"Ensemble (SVM + RF{' + XGB' if has_xgb else ''})"
        elif sel == "SVM RBF":
            clf      = SVC(kernel='rbf',C=10,gamma='scale',probability=True,random_state=42)
            clf_name = "SVM with RBF Kernel"
        elif sel == "Random Forest":
            clf      = RandomForestClassifier(n_estimators=200,random_state=42)
            clf_name = "Random Forest (200 trees)"
        else:
            if has_xgb:
                clf      = XGBClassifier(n_estimators=100,random_state=42,
                                         eval_metric='mlogloss',verbosity=0)
                clf_name = "XGBoost"
            else:
                clf      = SVC(kernel='rbf',C=10,gamma='scale',
                               probability=True,random_state=42)
                clf_name = "SVM RBF (XGBoost unavailable)"

        model = Pipeline([('scaler',StandardScaler()),('clf',clf)])
        prog.progress(1.0, text=f"Training {clf_name}...")
        st.markdown(f"**Using:** `{clf_name}`")

        cv     = GroupKFold(n_splits=img_id)
        y_pred = cross_val_predict(model, X, y, cv=cv, groups=groups)

        with t3:
            cm = confusion_matrix(y,y_pred,labels=all_labels)
            fig3,ax3 = plt.subplots(figsize=(max(5,n_cls*1.3),max(4,n_cls*1.1)))
            ax3.imshow(cm,cmap='Blues')
            ax3.set_xticks(range(n_cls)); ax3.set_yticks(range(n_cls))
            ax3.set_xticklabels(all_labels,rotation=45)
            ax3.set_yticklabels(all_labels)
            ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
            ax3.set_title('Confusion Matrix (Leave-One-Image-Out)')
            for i in range(n_cls):
                for j in range(n_cls):
                    ax3.text(j,i,cm[i,j],ha='center',va='center',fontsize=13,
                             color='white' if cm[i,j]>cm.max()/2 else 'black')
            plt.tight_layout()
            st.pyplot(fig3,use_container_width=True); plt.close(fig3)

        with t4:
            rows = []; correct = 0
            for gid in range(img_id):
                mask     = groups==gid
                true_lbl = y[mask][0]
                votes    = {l:(y_pred[mask]==l).sum() for l in np.unique(y)}
                img_pred = max(votes,key=votes.get)
                agree    = votes[img_pred]/mask.sum()*100
                ok       = "✅" if img_pred==true_lbl else "❌"
                rows.append({'File':labelled[gid][0],'True':true_lbl,
                             'Predicted':img_pred,'Agreement':f"{agree:.0f}%",'':ok})
                if img_pred==true_lbl: correct+=1
            st.dataframe(rows,use_container_width=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Image Accuracy", f"{correct/img_id*100:.0f}%")
            c2.metric("Correct", f"{correct}/{img_id}")
            c3.metric("Total Patches", f"{len(X):,}")

        model.fit(X,y)
        st.session_state.model         = model
        st.session_state.training_done = True
        prog.empty()
        st.success(f"✅ Training complete using {clf_name}!")

        if st.button("Go to Predict →", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()

# ============================================================
#  STEP 4 — Predict
# ============================================================
elif st.session_state.step == 3:
    st.markdown("""<div class='step-header'>
        <h2>🔍 Step 4 — Predict Unknown Samples</h2>
        <p>Upload a ZIP or individual TIFFs. The model predicts each one.</p>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.training_done or st.session_state.model is None:
        st.warning("⚠️ Complete Step 3 first or load a saved model from the sidebar.")
        st.stop()

    model   = st.session_state.model
    rois    = st.session_state.rois
    all_cls = list(model.classes_)

    with st.expander("ℹ️ How does prediction work?"):
        st.markdown("""
        **Process:**
        1. Each TIFF is demosaiced and the same ROI positions are applied automatically
        2. ~40 patches are extracted and each gets its own prediction
        3. All patch confidence scores are averaged into one final answer

        **Confidence guide:**
        - **>90%** — very confident  |  **70–90%** — confident  |  **50–70%** — uncertain
        """)

    tab_single, tab_zip, tab_drive = st.tabs(["🖼️ Individual TIFFs", "📦 ZIP", "☁️ Google Drive"])
    pred_files = []

    with tab_single:
        tfs = st.file_uploader("Upload one or more TIFFs",
                               type=['tiff','tif'],
                               accept_multiple_files=True,
                               label_visibility='collapsed',
                               key="pred_tiff_uploader")
        for tf in (tfs or []):
            pred_files.append((tf.name, tf.read()))
        if not tfs:
            st.markdown("""
            <div style='text-align:center;padding:20px;
                 border:1px dashed rgba(255,255,255,0.07);border-radius:12px'>
                <div style='font-size:28px;margin-bottom:6px'>🖼️</div>
                <div style='font-size:13px;color:#aaa'>Upload one or more TIFF files to predict</div>
            </div>""", unsafe_allow_html=True)

    with tab_zip:
        zf2 = st.file_uploader("Upload ZIP", type=['zip'], label_visibility='collapsed')
        if zf2:
            zdata = io.BytesIO(zf2.read())
            with zipfile.ZipFile(zdata,'r') as z:
                for nm in z.namelist():
                    if nm.lower().endswith(('.tiff','.tif')) and \
                       not os.path.basename(nm).startswith('.'):
                        pred_files.append((os.path.basename(nm), z.read(nm)))

    with tab_drive:
        gid2 = st.text_input("Google Drive ZIP File ID", key='gid2')
        if st.button("📥 Fetch from Drive", key='fetch2') and gid2:
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
                    st.success(f"✅ Loaded {len(pred_files)} files")
                except Exception as e:
                    st.error(f"❌ {e}")

    if pred_files:
        st.info(f"**{len(pred_files)}** file(s) ready to predict")
        for name,_ in pred_files:
            st.markdown(f"&nbsp;&nbsp;🖼️ `{name}`")

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

            st.subheader("🗂️ Results by Class")
            cols = st.columns(max(1,len(all_cls)))
            for ci,cls in enumerate(all_cls):
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
                            f"<b style='color:{col}'>{r['conf'][r['pred']]:.1f}%</b></span>"
                            for r in matched
                        ]) + "</div>", unsafe_allow_html=True)

            st.subheader("📋 Detailed Results")
            st.dataframe([{
                'File':r['file'],'Prediction':r['pred'],
                'Confidence':f"{r['conf'][r['pred']]:.1f}%",
                'Patch Agreement':f"{r['agreement']:.0f}%",
                'Patches':r['n_patches']
            } for r in results], use_container_width=True)

            st.subheader("🖼️ Image Overview")
            gcols = st.columns(min(3,len(results)))
            for i,r in enumerate(results):
                with gcols[i%3]:
                    col = WINE_COLORS.get(r['pred'],'#888')
                    fig,ax = plt.subplots(figsize=(4,3))
                    ax.imshow(disp_img(r['raw']),cmap='gray',aspect='auto')
                    for j,(x0,y0,x1_,y1_) in enumerate(rois):
                        c = ['dodgerblue','orange'][j%2]
                        ax.add_patch(patches.Rectangle((x0,y0),x1_-x0,y1_-y0,
                            lw=2,edgecolor=c,facecolor=c,alpha=0.2))
                        ax.add_patch(patches.Rectangle((x0,y0),x1_-x0,y1_-y0,
                            lw=2,edgecolor=c,facecolor='none'))
                    ax.set_title(f"{r['file']}\n{r['pred']} — {r['conf'][r['pred']]:.1f}%",
                                 color=col,fontsize=8,fontweight='bold')
                    ax.axis('off'); plt.tight_layout()
                    st.pyplot(fig,use_container_width=True); plt.close(fig)

            st.subheader("📊 Confidence Chart")
            fig_b,ax_b = plt.subplots(figsize=(max(10,len(results)*2),5))
            x = np.arange(len(results))
            w = 0.8/len(all_cls)
            for idx,cls in enumerate(all_cls):
                vals   = [r['conf'].get(cls,0) for r in results]
                offset = (idx-len(all_cls)/2+0.5)*w
                bars   = ax_b.bar(x+offset,vals,w,label=cls,
                                  color=WINE_COLORS.get(cls,'gray'),alpha=0.85)
                for bar in bars:
                    h = bar.get_height()
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
