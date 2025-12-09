# CS436 Structure from Motion Project — Group 17

**Muhammad Shazil Nadeem (27100183)**  
**Muhammad Taimur Jahanzeb (27100028)**  
**Group 17 — LUMS CS436**

This repository contains our full implementation of a complete multi-stage **Structure from Motion (SfM)** pipeline, including:

- **Week 1 (Phase 1):** Feature Detection & Matching  
- **Week 2 (Phase 2):** Two-View Geometry & 3D Reconstruction  
- **Week 3 (Phase 2):** Incremental Multi-View SfM with drift refinement  
- **Week 4 (Phase 3):** Interactive *Photosynth-style* Virtual Tour Viewer (Three.js)

The final application includes a global sparse point cloud, interlinked camera poses, and a browser-based navigation viewer with smooth transitions and image cross-fading.

---

## **Project Structure**

```
cs436-sfm-project-group17/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── images/
│   │   ├── *.HEIC
│   │   └── images_jpg/
│   │
│   ├── metashape/
│   │   ├── corridor_project.psx
│   │   ├── exports/
│   │   │   ├── cameras_corridor.xml
│   │   │   └── dense_corridor.ply
│   │   └── thumbnails/
│   │
│   └── results/
│       ├── week1/
│       ├── week2/
│       ├── week3/
│       └── final/
│           ├── cameras_corridor.json
│           ├── dense_corridor.ply
│           │
│           └── viewer/
│               ├── index.html
│               ├── app.js
│               ├── style.css
│               └── libs/
│                   ├── three.module.js
│                   ├── PLYLoader.js
│                   └── OrbitControls.js
│
├── notebooks/
│   ├── week3_g17.ipynb
│   └── outputs/
│       ├── *.ply
│       ├── matches_*.png
│       └── point_cloud.ply
│
├── scripts/
│   ├── run_week1_feature_matching.py
│   ├── run_week2_two_view.py
│   ├── run_week3_multiview.py
│   ├── run_convert_metashape.py
│   └── convert_heic_to_jpg.py
│
└── src/
    ├── features.py
    ├── two_view.py
    ├── multiview_sfm.py
    ├── io_utils.py
    ├── vis_open3d.py
    ├── convert_metashape_xml.py
    └── __init__.py
```

---

## ** Installation**

### **1. Create environment**
```bash
conda create -n cs436_sfm python=3.10
conda activate cs436_sfm
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. If using HEIC images**
```bash
pip install pillow-heif
python scripts/convert_heic_to_jpg.py
```

---

## **Running Each Phase**

Run all scripts from the **project root directory**.

---

## **Week 1 — Feature Matching**
```bash
python -m scripts.run_week1_feature_matching
```
Outputs → `data/results/week1/`

---

## **Week 2 — Two-View Reconstruction**
```bash
python -m scripts.run_week2_two_view
```

Outputs include:

- **Essential matrix**  
- **Relative camera pose**  
- **Triangulated 3D point cloud**  
- **Visualizations**

Saved in `data/results/week2/`.

---

## **Week 3 — Incremental Multi-View SfM**
```bash
python -m scripts.run_week3_multiview
```

Outputs:

- **Registered camera poses**  
- **Incremental triangulation**  
- **Refined multi-view point cloud**  
- **3D + XY visualizations**

Saved in `data/results/week3/`.

---

## **Phase 3 — Interactive Virtual Tour (Three.js Viewer)**

This is the final visualization stage: a complete virtual tour that uses the recovered camera poses and sparse 3D geometry.

### **Features**
- Smooth **camera navigation**  
- **Lerp** (position) + **Slerp** (rotation) interpolation  
- **Cross-fade** between reference images  
- Full **point cloud rendering**  
- **OrbitControls** (zoom/pan/rotate)  
- Multi-wall scene alignment using transformation matrices  

---

## **Running the Virtual Tour (Local Web Server)**

Navigate to:

```
data/results/final/viewer/
```

Run:

```bash
python -m http.server 8000
```

Open in browser:

```
http://localhost:8000
```

---

## Multi-Wall Scene Alignment

We implemented:

- **Transformation matrix merging of partial clouds using Agisoft Metashape**  
- **Camera pose transformation into global frame**  
- **Three.js Group hierarchy for synchronized geometry**  

Ensures cameras, images, and points remain aligned in the final viewer.

---

## Work Completed:

 **Week 1:** Feature extraction & matching  
 **Week 2:** Essential matrix + pose + triangulation  
 **Week 3:** Incremental SfM, PnP, map expansion  
 **Phase 3:** Full Photosynth-style interactive viewer  
 **Global merged scene + transformed cameras**

---

## Demonstration Video
[Click here](https://drive.google.com/file/d/1-WGQwF8KxVIHH3iXtd2YSuCKYrp6yr95/view?usp=sharing)

