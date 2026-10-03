import os
import sys
import pptx
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_10_slide_presentation(output_path="Hybrid_XAI_Lung_Nodule_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette Definitions (Clinical AI Dark Theme)
    COLOR_BG_DARK = RGBColor(11, 19, 43)        # #0B132B Deep Slate Navy
    COLOR_CARD_DARK = RGBColor(28, 37, 65)      # #1C2541 Slate Card
    COLOR_CARD_ALT = RGBColor(38, 51, 88)       # Slightly lighter card
    COLOR_BORDER = RGBColor(58, 80, 107)        # #3A506B Subdued Border
    COLOR_WHITE = RGBColor(255, 255, 255)       # White
    COLOR_TEXT_MUTED = RGBColor(148, 163, 184)  # #94A3B8 Slate Gray
    COLOR_CYAN = RGBColor(0, 245, 212)          # #00F5D4 Bright Cyan / Mint
    COLOR_TEAL = RGBColor(72, 202, 229)         # #48CAE4 Bright Teal
    COLOR_GOLD = RGBColor(247, 184, 1)          # #F7B801 Golden Alert
    COLOR_RED = RGBColor(239, 71, 111)          # #EF476F Soft Crimson / Alert
    COLOR_GREEN = RGBColor(6, 214, 160)         # #06D6A0 Forest Teal / Benign

    def set_slide_background(slide, color=COLOR_BG_DARK):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background()
        return bg

    def add_header(slide, category_text, title_text, subtitle_text=""):
        # Category pill
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.4), Inches(3.0), Inches(0.35))
        pill.fill.solid()
        pill.fill.fore_color.rgb = RGBColor(15, 30, 65)
        pill.line.color.rgb = COLOR_TEAL
        pill.line.width = Pt(1)
        tf = pill.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.text = category_text.upper()
        p.alignment = PP_ALIGN.CENTER
        p.font.name = "Arial"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = COLOR_CYAN

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.7), Inches(0.65))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Arial"
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_WHITE

        # Subtitle
        if subtitle_text:
            sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.45), Inches(11.7), Inches(0.4))
            tf_sub = sub_box.text_frame
            tf_sub.word_wrap = True
            tf_sub.margin_left = tf_sub.margin_right = tf_sub.margin_top = tf_sub.margin_bottom = 0
            p_sub = tf_sub.paragraphs[0]
            p_sub.text = subtitle_text
            p_sub.font.name = "Arial"
            p_sub.font.size = Pt(11.5)
            p_sub.font.color.rgb = COLOR_TEXT_MUTED

    def add_footer(slide, current_slide, total_slides=10):
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(11.7), Inches(0.35))
        tf = footer_box.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = f"Hybrid XAI Lung Nodule Framework  |  LIDC-IDRI Deep Learning Pipeline  |  Slide {current_slide} of {total_slides}"
        p.font.name = "Arial"
        p.font.size = Pt(9)
        p.font.color.rgb = COLOR_BORDER

    def add_card(slide, left, top, width, height, title="", border_color=COLOR_BORDER, bg_color=COLOR_CARD_DARK):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.2)
        
        if title:
            tbox = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.15), width - Inches(0.4), Inches(0.4))
            tf = tbox.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = title
            p.font.name = "Arial"
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = COLOR_CYAN
        return card

    # ==========================================
    # SLIDE 1: TITLE & KEY HIGHLIGHTS
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)
    
    hero = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(11.733), Inches(5.8))
    hero.fill.solid()
    hero.fill.fore_color.rgb = COLOR_CARD_DARK
    hero.line.color.rgb = COLOR_TEAL
    hero.line.width = Pt(1.5)

    pill1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.3), Inches(1.2), Inches(3.8), Inches(0.4))
    pill1.fill.solid()
    pill1.fill.fore_color.rgb = RGBColor(15, 30, 65)
    pill1.line.color.rgb = COLOR_CYAN
    tf1 = pill1.text_frame
    p1 = tf1.paragraphs[0]
    p1.text = "RESEARCH & ENGINEERING DEFENSE"
    p1.alignment = PP_ALIGN.CENTER
    p1.font.bold = True
    p1.font.size = Pt(10)
    p1.font.color.rgb = COLOR_CYAN

    tbox = s1.shapes.add_textbox(Inches(1.3), Inches(1.8), Inches(10.7), Inches(1.4))
    tf = tbox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Hybrid Explainable AI Framework for\nPulmonary Nodule Malignancy Detection"
    p.font.name = "Arial"
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = COLOR_WHITE

    subbox = s1.shapes.add_textbox(Inches(1.3), Inches(3.3), Inches(10.7), Inches(0.8))
    tf_sub = subbox.text_frame
    tf_sub.word_wrap = True
    p_sub = tf_sub.paragraphs[0]
    p_sub.text = "An End-to-End Multimodal Deep Learning System Merging 3D CT Volumetric Tensors with Handcrafted Clinical Radiomics via Dynamic Gated Fusion and 3D Grad-CAM Explainability"
    p_sub.font.name = "Arial"
    p_sub.font.size = Pt(13)
    p_sub.font.color.rgb = COLOR_TEXT_MUTED

    stat_data = [
        ("0.9042", "Ensemble ROC-AUC", COLOR_CYAN),
        ("86.00%", "Overall Accuracy", COLOR_GREEN),
        ("0.5783", "Youden's J Cutoff", COLOR_GOLD),
        ("1,242", "Consensus Nodules", COLOR_TEAL)
    ]
    for i, (val, lbl, col) in enumerate(stat_data):
        bx = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.3 + i*2.7), Inches(4.3), Inches(2.5), Inches(1.2))
        bx.fill.solid()
        bx.fill.fore_color.rgb = RGBColor(18, 26, 48)
        bx.line.color.rgb = col
        bx.line.width = Pt(1.2)
        btf = bx.text_frame
        btf.word_wrap = True
        bp1 = btf.paragraphs[0]
        bp1.text = val
        bp1.alignment = PP_ALIGN.CENTER
        bp1.font.bold = True
        bp1.font.size = Pt(22)
        bp1.font.color.rgb = col
        bp2 = btf.add_paragraph()
        bp2.text = lbl
        bp2.alignment = PP_ALIGN.CENTER
        bp2.font.size = Pt(10)
        bp2.font.color.rgb = COLOR_TEXT_MUTED

    add_footer(s1, 1, 10)

    # ==========================================
    # SLIDE 2: CLINICAL PROBLEM & MOTIVATION
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "Clinical Context", "Clinical Motivation & The Healthcare Dilemma", 
               "Lung cancer causes 1.8M annual deaths worldwide; diagnostic uncertainty and opaque AI models create clinical bottlenecks.")
    
    add_card(s2, Inches(0.8), Inches(2.0), Inches(3.6), Inches(4.7), "1. The High Clinical Stakes")
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(3.2), Inches(3.9))
    tf = tb.text_frame
    tf.word_wrap = True
    bullets = [
        "Leading Cancer Killer: Lung cancer causes more deaths than breast, colon, and prostate cancers combined.",
        "Early Detection Saves Lives: 5-year survival jumps from <10% (Late Stage) to >60% if detected as an early Stage I nodule.",
        "Diagnostic Urgency: CT screening catches thousands of subtle (3–30mm) nodules that require immediate, accurate risk triaging."
    ]
    for b in bullets:
        p = tf.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(12)

    add_card(s2, Inches(4.8), Inches(2.0), Inches(3.6), Inches(4.7), "2. Radiologist Uncertainty")
    tb2 = s2.shapes.add_textbox(Inches(5.0), Inches(2.6), Inches(3.2), Inches(3.9))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    bullets2 = [
        "Inter-Observer Variability: Even veteran radiologists frequently disagree on borderline, indeterminate lesions.",
        "Unnecessary Biopsy Trauma: Over 20–30% of invasive lung biopsies turn out to be benign, causing patient trauma and high pneumothorax risks.",
        "Cognitive Overload: Reviewing hundreds of thin-slice CT images per patient creates perceptual misses in high-volume wards."
    ]
    for b in bullets2:
        p = tf2.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(12)

    add_card(s2, Inches(8.8), Inches(2.0), Inches(3.7), Inches(4.7), "3. The AI 'Black-Box' Trap")
    tb3 = s2.shapes.add_textbox(Inches(9.0), Inches(2.6), Inches(3.3), Inches(3.9))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    bullets3 = [
        "Opaque Predictions: Traditional 3D CNNs output raw diagnostic probabilities without anatomical evidence or visual reasoning.",
        "Clinician Distrust: Doctors will not schedule aggressive surgery based solely on an unexplainable statistical number.",
        "Our Solution: A hybrid multimodal architecture combining 3D image voxels with radiomic features + 3D Grad-CAM heatmaps."
    ]
    for b in bullets3:
        p = tf3.add_paragraph()
        p.text = "• " + b
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_CYAN if "Our Solution" in b else COLOR_WHITE
        p.space_after = Pt(12)

    add_footer(s2, 2, 10)

    # ==========================================
    # SLIDE 3: DATASET ARCHITECTURE & CONSENSUS
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Dataset & Ground Truth", "LIDC-IDRI Dataset: 4-Radiologist Consensus Ground Truth",
               "Establishing a rigorous binary classification benchmark across 1,010 thoracic CT scan subjects.")

    add_card(s3, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7), "The LIDC-IDRI Cohort")
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(5.2), Inches(3.9))
    tf = tb.text_frame
    tf.word_wrap = True
    pts = [
        "Repository: The Cancer Imaging Archive (TCIA) National Biomedical Imaging Archive.",
        "Total Patients & Series: 1,010 thoracic CT patients spanning 1,018 series.",
        "Raw Data Footprint: ~133 Gigabytes of high-resolution DICOM slices.",
        "Multi-Reader Annotation: Up to 4 experienced thoracic radiologists evaluated every nodule independently on a 1 (low) to 5 (high) malignancy scale.",
        "Total Expert Nodules: 1,608 unique nodules annotated across the full dataset."
    ]
    for pt in pts:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(10)

    add_card(s3, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.7), "Consensus Ground Truth Partitioning")
    tb2 = s3.shapes.add_textbox(Inches(7.0), Inches(2.6), Inches(5.3), Inches(3.9))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    pts2 = [
        "Consensus Score: Average rating across all reviewing radiologists for that nodule.",
        "Class 0 (Benign): Consensus score < 3.0 (709 nodules).",
        "Class 1 (Malignant): Consensus score > 3.0 (533 nodules).",
        "Class -1 (Indeterminate / Dropped): Exactly 366 nodules had a score == 3.0 (even 50/50 radiologist split).",
        "The Key Data Hygiene Decision: Dropping ambiguous Class -1 nodules prevents label noise from corrupting neural gradients.",
        "Total Valid Cohort: Exactly 1,242 consensus-labeled nodules for training and evaluation."
    ]
    for pt in pts2:
        p = tf2.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_CYAN if "1,242" in pt else COLOR_WHITE
        p.space_after = Pt(8)

    add_footer(s3, 3, 10)

    # ==========================================
    # SLIDE 4: PHASES 0 & 1 - INGESTION & PREPROCESSING
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Phases 0 & 1 Pipeline", "Automated Cloud Ingestion & Medical 3D Preprocessing",
               "Translating 133 GB of heterogeneous clinical DICOM scans into standardized, AI-ready 3D tensors.")

    p1_cards = [
        ("Phase 0: Cloud Streaming", "downloader.ipynb (tcia_utils)",
         "• Queries TCIA REST API for collection='LIDC-IDRI' and modality='CT'.\n• Direct cloud-to-cloud byte streaming into Google Drive (zero local storage footprint).\n• Resume-safe checkpoint logic skips already-downloaded series upon Colab timeout.",
         COLOR_TEAL, Inches(0.8), Inches(2.0)),
        ("Isotropic 1.0mm Resampling", "SimpleITK Volume Processing",
         "• Scanners have varying slice thicknesses (0.6mm to 2.5mm).\n• Resamples all CT volumes to universal 1.0 x 1.0 x 1.0 mm isotropic spacing.\n• Ensures spherical geometry is preserved regardless of hospital machine.",
         COLOR_CYAN, Inches(6.8), Inches(2.0)),
        ("HU Windowing & 64³ Patches", "Intensity & Subvolume Cropping",
         "• Lung Windowing [-1000, 400] HU isolates lung soft tissue; normalized to [0, 1].\n• Extracts uniform 64 x 64 x 64 voxel 3D subvolume around each nodule centroid.\n• Zero-padding handles peripheral nodules near chest walls.",
         COLOR_GOLD, Inches(0.8), Inches(4.4)),
        ("5 Radiomic Features & Manifest", "pylidc Tabular Extraction",
         "• Extracts 5 expert semantic attributes: Subtlety, Sphericity, Margin, Spiculation, and Texture (1 to 5 scale).\n• Patches serialized as individual PyTorch .pt tensors linked via master manifest.csv.",
         COLOR_GREEN, Inches(6.8), Inches(4.4)),
    ]

    for title, sub, content, col, l, t in p1_cards:
        add_card(s4, l, t, Inches(5.7), Inches(2.2), title, border_color=col)
        tb = s4.shapes.add_textbox(l + Inches(0.2), t + Inches(0.55), Inches(5.3), Inches(1.5))
        tf = tb.text_frame
        tf.word_wrap = True
        for line in content.split("\n"):
            p = tf.add_paragraph()
            p.text = line
            p.font.size = Pt(10)
            p.font.color.rgb = COLOR_WHITE
            p.space_after = Pt(3)

    add_footer(s4, 4, 10)

    # ==========================================
    # SLIDE 5: PHASE 2 - HYBRID GATED FUSION ARCHITECTURE
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Model Architecture", "Phase 2: Dual-Branch Network & Dynamic Gated Fusion",
               "A dual-stream architecture adaptively modulating spatial 3D convolutions with clinical radiomics.")

    add_card(s5, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7), "Dual-Branch Network Structure", border_color=COLOR_CYAN)
    tb = s5.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(5.2), Inches(3.9))
    tf = tb.text_frame
    tf.word_wrap = True
    c1 = [
        "Branch 1: 3D CNN (Visual Morphology):",
        "  • Input: (B, 1, 64, 64, 64) 3D voxel subvolumes.",
        "  • 4 Cascaded Blocks: Conv3D(1->16->32->64->128) + ReLU + MaxPool3D(2).",
        "  • Dense Projection: Flatten(8,192) -> Linear(256) -> Linear(128).",
        "  • Output: 128-dimensional spatial feature embedding.",
        "",
        "Branch 2: Tabular MLP (Clinical Semantics):",
        "  • Input: (B, 5) vector [subtlety, sphericity, margin, spiculation, texture].",
        "  • Layers: Linear(5->16) -> ReLU -> Linear(16->32) -> ReLU.",
        "  • Output: 32-dimensional radiomic embedding."
    ]
    for line in c1:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_CYAN if ("Branch" in line) else COLOR_WHITE
        p.space_after = Pt(2)

    add_card(s5, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.7), "Dynamic Sigmoid Gating Mechanism", border_color=COLOR_GOLD)
    tb2 = s5.shapes.add_textbox(Inches(7.0), Inches(2.6), Inches(5.3), Inches(3.9))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    c2 = [
        "Concatenation: z = [ f_visual (128)  ||  f_tabular (32) ]  ∈  R^160",
        "",
        "Sigmoid Gating Vector: g = σ( W_gate · z + b_gate ),   g ∈ (0, 1)^160",
        "",
        "Hadamard Fusion: z_gated = z ⊙ g",
        "",
        "Classification Head: Linear(160, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 2)",
        "",
        "Why Gated Fusion Beats Simple Concat?",
        "  • Noise Suppression: If the CT scan is noisy or motion-blurred, the gate suppresses visual channels and routes decisions through radiomics.",
        "  • Adaptive Balancing: Prevents the 128 image features from overwhelming the 5 clinical features during gradient updates."
    ]
    for line in c2:
        p = tf2.add_paragraph()
        p.text = line
        p.font.name = "Courier New" if ("=" in line or "∈" in line) else "Arial"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_GOLD if ("=" in line or "∈" in line) else COLOR_WHITE
        p.space_after = Pt(2)

    add_footer(s5, 5, 10)

    # ==========================================
    # SLIDE 6: PHASES 2 & 4 - LEAKAGE-FREE TRAINING & SCALING
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Training & Optimization", "Leakage-Free Validation & Full-Scale GPU Training",
               "Ensuring rigorous patient-level generalizability and efficient mixed-precision GPU optimization.")

    v_cards = [
        ("Patient-Level GroupKFold", "Zero Data Leakage Protocol",
         "• The Threat: Multiple nodules often belong to the same patient. Random splits leak patient-specific scanner artifacts into test sets.\n• The Solution: GroupKFold(n_splits=5) grouped strictly by patient_id.\n• Guarantee: Zero patient identity overlap between training and validation folds.",
         COLOR_RED, Inches(0.8)),
        ("Data Hygiene & Imbalance", "Dynamic Scaling & Class Weights",
         "• Fold Isolation: StandardScaler fit strictly on training fold indices before transforming validation fold.\n• Inverse Class Weights: Loss weighted inversely by class frequencies: w = 1 / [N_benign, N_malignant], preventing majority benign collapse.",
         COLOR_CYAN, Inches(4.8)),
        ("Automatic Mixed Precision (AMP)", "phase4_full_scale.ipynb",
         "• torch.cuda.amp.autocast() executes convolutions in FP16; GradScaler() prevents underflow.\n• Halves GPU VRAM and doubles throughput.\n• 3D spatial augmentations (RandomAffine, RandomFlip) prevent slice memorization.",
         COLOR_GOLD, Inches(8.8))
    ]

    for title, sub, content, col, left in v_cards:
        add_card(s6, left, Inches(2.0), Inches(3.7), Inches(4.7), title, border_color=col)
        tb = s6.shapes.add_textbox(left + Inches(0.2), Inches(2.6), Inches(3.3), Inches(3.9))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p_sub = tf.paragraphs[0]
        p_sub.text = sub.upper()
        p_sub.font.bold = True
        p_sub.font.size = Pt(10)
        p_sub.font.color.rgb = col
        p_sub.space_after = Pt(10)

        for line in content.split("\n"):
            p = tf.add_paragraph()
            p.text = line
            p.font.size = Pt(10.5)
            p.font.color.rgb = COLOR_WHITE
            p.space_after = Pt(6)

    add_footer(s6, 6, 10)

    # ==========================================
    # SLIDE 7: PHASE 3 - ENSEMBLE & DECISION CALIBRATION
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "Evaluation & Calibration", "Phase 3: Ensemble Soft Voting & Decision Calibration",
               "Aggregating 5 cross-validation fold models with mathematically calibrated threshold optimization.")

    add_card(s7, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7), "5-Fold Committee Consensus (Ensemble)")
    tb = s7.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(5.2), Inches(3.9))
    tf = tb.text_frame
    tf.word_wrap = True
    ens_pts = [
        "Committee of 5 Experts: Instead of relying on a single checkpoint, inference executes across all 5 saved fold models simultaneously.",
        "Soft Probability Averaging: P_consensus = (1/5) ∑ Softmax(Model_k(x)).",
        "Variance Reduction: Cancels individual fold variance and eliminates single-split overfitting.",
        "Proven Diagnostic Power: Pushes the full-cohort discrimination to 0.9042 ROC-AUC."
    ]
    for pt in ens_pts:
        p = tf.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(10)

    add_card(s7, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.7), "Decision Cutoff Calibration (Youden's J)")
    tb2 = s7.shapes.add_textbox(Inches(7.0), Inches(2.6), Inches(5.3), Inches(3.9))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    youden_pts = [
        "The Default 0.50 Flaw: The default 50% probability cutoff assumes equal class costs, causing excessive false-positive alarms in clinical screening.",
        "Youden's J Index Formulation: J = Sensitivity + Specificity - 1 = TPR - FPR.",
        "Optimal Calibrated Threshold: 0.5783 (57.83%).",
        "Clinical Impact: Maximizes cancer sensitivity (76%) while providing 90% benign specificity, eliminating unnecessary surgical biopsies."
    ]
    for pt in youden_pts:
        p = tf2.add_paragraph()
        p.text = "• " + pt
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_CYAN if "0.5783" in pt else COLOR_WHITE
        p.space_after = Pt(10)

    add_footer(s7, 7, 10)

    # ==========================================
    # SLIDE 8: PHASE 3 - EXPLAINABLE AI (3D GRAD-CAM)
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "Explainable AI (XAI)", "Phase 3: 3D Grad-CAM Voxel-Level Transparency",
               "Visualizing the network's internal diagnostic reasoning across Axial, Coronal, and Sagittal planes.")

    add_card(s8, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7), "3D Grad-CAM Mathematical Engine")
    tb = s8.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(5.2), Inches(3.9))
    tf = tb.text_frame
    tf.word_wrap = True
    xai_steps = [
        "Hook Registration: Forward and backward hooks capture activations and gradients from conv_blocks[9] (final 3D convolutional layer).",
        "Volumetric Channel Weights: α_k^c = (1/D·H·W) ∑∑∑ ∂y^c / ∂A^k.",
        "ReLU Heatmap Generation: L = ReLU( ∑ α_k^c A^k ) retains features positively contributing to the malignancy score.",
        "Trilinear Upsampling: Scipy zoom resamples the 4x4x4 activation volume back to 64x64x64 voxel matrix."
    ]
    for s in xai_steps:
        p = tf.add_paragraph()
        p.text = "• " + s
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_WHITE
        p.space_after = Pt(10)

    add_card(s8, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.7), "Tri-Planar Clinical Verification")
    tb2 = s8.shapes.add_textbox(Inches(7.0), Inches(2.6), Inches(5.3), Inches(3.9))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    xai_clin = [
        "Synchronized 3-Plane Plot:",
        "  1. Axial Slice (Transverse cross-section of nodule)",
        "  2. Coronal Slice (Frontal anatomical perspective)",
        "  3. Sagittal Slice (Lateral depth view)",
        "Clinical Verification: Proves the network focuses directly on irregular, spiculated nodule margins rather than ribs or scanner noise.",
        "Regulatory Auditability: Provides the interpretable evidence trail required by the EU AI Act and FDA SaMD clinical guidelines."
    ]
    for s in xai_clin:
        p = tf2.add_paragraph()
        p.text = s if "Synchronized" in s else "• " + s
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_CYAN if "Axial" in s else COLOR_WHITE
        p.space_after = Pt(8)

    add_footer(s8, 8, 10)

    # ==========================================
    # SLIDE 9: RESULTS, ABLATION & BENCHMARKS
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9)
    add_header(s9, "Experimental Results", "Model Performance, Ablation Study & Benchmarks",
               "Comprehensive validation across 1,242 consensus nodules under 5-fold cross-validation.")

    # Left: Results Table
    add_card(s9, Inches(0.8), Inches(2.0), Inches(6.0), Inches(4.7), "Performance Metrics (1,242 Nodules)")
    rows = 6
    cols = 3
    table_shape = s9.shapes.add_table(rows, cols, Inches(1.0), Inches(2.6), Inches(5.6), Inches(3.8))
    table = table_shape.table
    table.columns[0].width = Inches(2.6)
    table.columns[1].width = Inches(1.3)
    table.columns[2].width = Inches(1.7)

    table_data = [
        ["Metric", "Result", "Threshold / Note"],
        ["Overall Accuracy", "86.00%", "Cutoff: 0.5783"],
        ["Ensemble ROC-AUC", "0.9042", "All Thresholds"],
        ["Calibrated Cutoff", "0.5783", "Youden's J Index"],
        ["Benign (P / R / F1)", "0.89 / 0.90 / 0.90", "High Specificity"],
        ["Malignant (P / R / F1)", "0.77 / 0.76 / 0.77", "High Sensitivity"]
    ]

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.font.name = "Arial"
            p.font.size = Pt(9.5) if r_idx > 0 else Pt(10)
            p.font.bold = True if (r_idx == 0 or c_idx == 1) else False
            if r_idx == 0:
                p.font.color.rgb = COLOR_CYAN
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(15, 25, 50)
            else:
                p.font.color.rgb = COLOR_GOLD if c_idx == 1 else COLOR_WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_CARD_DARK if r_idx % 2 == 1 else COLOR_CARD_ALT

    # Right: Ablation & Benchmarks Card
    add_card(s9, Inches(7.0), Inches(2.0), Inches(5.5), Inches(4.7), "Ablation Study & Benchmarks")
    tb = s9.shapes.add_textbox(Inches(7.2), Inches(2.6), Inches(5.1), Inches(3.9))
    tf = tb.text_frame
    tf.word_wrap = True
    ablation_text = [
        "Multimodal Ablation Study:",
        "  • 3D CNN Only (Tabular zeroed): 0.6901 ROC-AUC (Drop of -0.2141)",
        "  • Tabular MLP Only (Images zeroed): 0.8985 ROC-AUC",
        "  • Hybrid Gated Fusion: 0.9042 ROC-AUC (Best Overall)",
        "  -> Proves multimodal synergy statistically outperforms single branches.",
        "",
        "Literature Benchmarks on LIDC-IDRI:",
        "  • Traditional SVM (Classical ML): 0.850 AUC",
        "  • Standard 3D CNN (Image Only): 0.930 AUC (Black-box)",
        "  • Our Hybrid Model: 0.9042 AUC (With 3D XAI & Gating)",
        "  • NoduleX (SOTA Literature): 0.971 AUC"
    ]
    for line in ablation_text:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(10)
        p.font.bold = True if ("Ablation" in line or "Literature" in line) else False
        p.font.color.rgb = COLOR_CYAN if ("Ablation" in line or "Literature" in line) else (COLOR_GREEN if "0.9042" in line else COLOR_WHITE)
        p.space_after = Pt(2)

    add_footer(s9, 9, 10)

    # ==========================================
    # SLIDE 10: DASHBOARD, CLINICAL IMPACT & CONCLUSION
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Deployment & Conclusion", "Live Gradio Dashboard, Clinical Impact & Conclusion",
               "Translating deep learning research into a point-of-care clinical tool (final_inference_pipeline.ipynb).")

    add_card(s10, Inches(0.8), Inches(2.0), Inches(5.6), Inches(4.7), "Live Production Web Dashboard", border_color=COLOR_TEAL)
    tb = s10.shapes.add_textbox(Inches(1.0), Inches(2.6), Inches(5.2), Inches(3.9))
    tf = tb.text_frame
    tf.word_wrap = True
    d_pts = [
        "Point-of-Care Gradio Web App:",
        "  • Dropdown Nodule Selector: Instant loading of preprocessed 3D CT scans.",
        "  • 5 Interactive Radiomic Sliders: Allows doctors to adjust subtlety, sphericity, margin, spiculation, and texture in real time.",
        "  • 5-Model Ensemble Engine: Sub-2-second inference runtime.",
        "  • Diagnostic Readout: Status ('MALIGNANT' vs 'BENIGN') + consensus probability + calibrated 57.83% threshold comparison.",
        "  • Multi-Planar Heatmap Plot: Generates synchronized Axial, Coronal, and Sagittal Grad-CAM overlays."
    ]
    for pt in d_pts:
        p = tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_CYAN if "Point-of-Care" in pt else COLOR_WHITE
        p.space_after = Pt(4)

    add_card(s10, Inches(6.8), Inches(2.0), Inches(5.7), Inches(4.7), "Clinical Impact & Core Conclusions", border_color=COLOR_GREEN)
    tb2 = s10.shapes.add_textbox(Inches(7.0), Inches(2.6), Inches(5.3), Inches(3.9))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    c_pts = [
        "Clinical Impact: 89% benign precision and 90% specificity drastically reduce unnecessary diagnostic lung biopsies.",
        "Human-in-the-Loop CADx: Operates as an interactive second reader validating human judgment, not replacing clinicians.",
        "Core Project Achievements:",
        "  ✔ Cloud-to-cloud automated pipeline processing 133 GB DICOMs.",
        "  ✔ Dynamic Gated Fusion surpassing standalone modalities (0.9042 AUC).",
        "  ✔ Zero patient leakage via strict 5-Fold GroupKFold validation.",
        "  ✔ Youden's J calibration (0.5783) and 3D Grad-CAM explainability."
    ]
    for pt in c_pts:
        p = tf2.add_paragraph()
        p.text = pt
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_GREEN if ("✔" in pt) else COLOR_WHITE
        p.space_after = Pt(6)

    # Q&A prompt
    qa_p = tf2.add_paragraph()
    qa_p.text = "Thank you. Open for Questions & Defense."
    qa_p.alignment = PP_ALIGN.CENTER
    qa_p.font.bold = True
    qa_p.font.size = Pt(14)
    qa_p.font.color.rgb = COLOR_GOLD

    add_footer(s10, 10, 10)

    # Save presentation
    prs.save(output_path)
    print(f"Successfully generated 10-slide PowerPoint presentation at: {output_path}")

if __name__ == "__main__":
    out_file = "Hybrid_XAI_Lung_Nodule_Presentation.pptx"
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    create_10_slide_presentation(out_file)
