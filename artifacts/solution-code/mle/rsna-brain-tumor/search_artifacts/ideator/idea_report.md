# Idea Report

## Geometric Affinity Pre-training Curriculum

**Score**: 7.85

**Abstract**:
The RSNA-MICCAI Brain Tumor Radiogenomic Classification challenge requires accurately predicting MGMT promoter methylation from highly dimensional 3D multiparametric MRI data. While it is biologically established that explicit geometric properties of a tumor correlate with MGMT status, standard 3D deep learning models often fail to capture these priors, and state-of-the-art hybrid models that fuse explicit radiomic features with deep network outputs suffer from dimensionality mismatch, training instability, and complex inference pipelines. 

To solve this, we propose a "Geometric Affinity Pre-training Curriculum". Instead of extracting handcrafted radiomic features during inference, we shift the integration of explicit geometric radiomics to the pre-training phase. We procedurally generate a large-scale dataset of synthetic 3D geometric tumor-like shapes (ranging from simple ellipsoids to complex lobulated and nested structures representing distinct multi-habitat niches). Using task affinity metrics to ensure the synthetic source distribution optimally aligns with the target BraTS multi-parametric MRI distribution, we sequentially pre-train a 3D CNN to classify and regress these geometric properties. The network is subsequently fine-tuned end-to-end on the masked BraTS MRI ROIs.

We expect this curriculum to match or exceed the AUC performance of complex hybrid deep radiomic baselines like C0 (Explainable Deep Radiogenomic Molecular Imaging) and static mask methods like C1 (Domain Knowledge Augmented Mask Fusion). Importantly, it achieves this while maintaining a lightweight, zero-inference-overhead single-branch 3D CNN, heavily mitigating overfitting and simplifying the final radiogenomic pipeline.


**Related Work**:
Standard 2D and 3D CNN architectures establish the baseline for MGMT promoter methylation prediction but frequently suffer from severe overfitting on small datasets like BraTS, while 2D ensembles lose critical volumetric spatial context (`[2024][IEEE CSDGAIS][DL-Radiogenomic]Deep_Learning_for_Brain_Tumor_Radiogenomic_Classif.md`, `[2024][arXiv][ResNet-Glioma]Comparative_Analysis_of_2D_and_3D_ResNet_Architect.md`). To address this, the literature highlights the integration of explicit geometric radiomics (e.g., tumor sphericity, surface area) via hybrid deep radiomics and feature weighting architectures. However, these methods often rely heavily on handcrafted feature extraction (`[2024][Diagnostics][MGMT_ProFWise]Radiomics_and_AI_Based_Prediction_of_MGMT_Methylat.md`) or attempt complex late-fusion with deep features, leading to dimensionality mismatches (`[2026][arXiv][MGMT-XAI]Explainable_Deep_Radiogenomic_Molecular_Imaging_fo.md`, `[2025][arXiv][ReFRM3D]ReFRM3D__A_Radiomics_enhanced_Fused_Residual_Multi.md`).

Recent domain knowledge augmented mask fusion techniques successfully isolate regions of interest (ROIs) to reduce healthy tissue noise (`[2025][Scientific Reports][MGMT-MaskFusion]Deep_learning_classification_of_MGMT_status_of_gli.md`), yet they still lack explicit geometric priors without adding multi-branch complexity. Concurrently, advancements in sequential transfer learning demonstrate that task affinity metrics can successfully guide source selection and pre-training paths for medical image analysis (`[2024][IEEE Transactions on Medical Imaging (TMI)][GGST]Graph_guided_Source_Selection_with_Sequential_Tran.md`, `[2024][MICCAI 2024][SSTP]Selecting_the_Best_Sequential_Transfer_Path_for_Me.md`). We bridge these domains by replacing the late-fusion of handcrafted geometric features with a task-affinity-guided pre-training phase on synthetic geometric shapes, injecting spatial priors directly into a single-branch architecture.


**Hypothesis**:
Using task affinity metrics to pre-train a 3D CNN on synthetic, purely geometric tumor shapes before fine-tuning on MRIs embeds spatial radiomic priors in the network weights, circumventing the dimensionality mismatch of hybrid dual-branch architectures.

**Proposed Experiments**:
- **Exp**: Sequential pre-training on synthetic 3D tumor shapes significantly improves the fine-tuned MGMT classification AUC compared to training from scratch and matches or exceeds the hybrid fusion baseline C0.
  - Step 1: Procedurally generate a synthetic 3D dataset containing thousands of variable geometric shapes (spheres, ellipsoids, irregular lobules) mapping to known tumor multi-habitat spatial distributions.
  - Step 2: Pre-train a standard 3D ResNet-18 architecture (matching the C0 baseline feature extractor) on the synthetic dataset to predict explicitly defined geometric descriptors (e.g., sphericity, surface-to-volume ratio).
  - Step 3: Fine-tune the pre-trained 3D network on the BraTS 2021 multi-parametric MRI dataset, utilizing domain knowledge augmented cropped ROIs.
  - Step 4: Evaluate the model on the hidden validation set using AUC, comparing inference latency and predictive performance against baseline implementations of C0 and C1.
- **Exp**: The performance gain is fundamentally derived from the alignment of explicit geometric spatial priors with target MRI data, and filtering the pre-training dataset using task affinity metrics yields higher downstream AUC than unfiltered pre-training.
  - Step 1: Compute task affinity metrics between various subclasses of the synthetic geometric dataset and the BraTS 2021 dataset.
  - Step 2: Filter the synthetic dataset to retain only the shape configurations that maximize the task affinity score.
  - Step 3: Train identical 3D ResNet-18 models under three conditions: (A) Random initialization, (B) Unfiltered synthetic pre-training, and (C) Affinity-filtered synthetic pre-training.
  - Step 4: Compare fine-tuning convergence rates and final AUC scores across the three configurations to isolate the effect of the affinity-guided curriculum.

**Risk Factors and Limitations**:
1. Gap between synthetic geometry and real MRI texture.
   Diagnostic: High validation AUC on the synthetic pre-training task but rapid plateau and low performance during BraTS fine-tuning.
   Fallback: Inject synthetic Gaussian and Perlin noise into the generated shapes during pre-training to closer simulate the textural noise inherent in MRI scans.
2. Catastrophic forgetting of geometric priors during fine-tuning.
   Diagnostic: Downstream model performance initially spikes but slowly degrades back to the random-initialization baseline performance across extended fine-tuning epochs.
   Fallback: Freeze the early 3D feature-extraction blocks (which encode the spatial priors) and strictly fine-tune the deeper, high-level abstraction layers, or apply a heavily decayed learning rate.
3. Pre-training overhead and tuning complexity.
   Diagnostic: Time required to generate and validate task affinity scores for millions of synthetic shapes exceeds competition timeline constraints.
   Fallback: Reduce the complexity of the synthetic shapes to basic primitives (e.g., basic ellipsoids) and bypass the strict affinity filtering step, relying purely on the sheer volume of shape variability.

## Critic justifications

- **novelty**: Circumvents the dimensionality mismatch of hybrid fusion by shifting geometric radiomics from inference-time features to a pre-training curriculum using synthetic shapes.
- **feasibility**: Generating synthetic 3D geometric shapes and performing sequential pre-training on standard CNN architectures is computationally straightforward and well-supported.
- **excitement**: A zero-inference-overhead method to inject explicit geometric priors into deep learning models would be highly attractive to the broader medical imaging community.
- **impact**: Could match or exceed the performance of complex hybrid models while significantly simplifying the inference pipeline and improving generalization.

## Candidate ranking

| Rank | Mode | Title | Overall | Nov | Feas | Exc | Imp | Viable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | unconventional | Geometric Affinity Pre-training Curriculum | 7.85 | 8 | 8 | 8 | 7 | True |
| 2 | unconventional | Inter-Sequence Latent Model Diffing for Isotopic Tumor Niche Extraction | 7.85 | 8 | 8 | 8 | 7 | True |
| 3 | unconventional | Task-Affinity Modulated Target Smoothing for Segmentation-Brittle Samples | 7.75 | 8 | 9 | 7 | 6 | True |
| 4 | unconventional | Soft-Mask Latent Diffing for Error-Resilient Classification | 7.7 | 7 | 8 | 8 | 8 | True |
| 5 | unconventional | Explicit Topological Skeletons as Hypergraph Scaffolding | 7.7 | 8 | 7 | 8 | 8 | True |
| 6 | unconventional | MGMT Prediction via Multi-Contrast Synthesis Reconstruction Residuals | 7.65 | 8 | 6 | 9 | 8 | True |
| 7 | unconventional | Spatially Antagonistic Contrastive Learning Across Tumor Sub-Habitats | 7.55 | 8 | 7 | 8 | 7 | True |
| 8 | unconventional | Spatially Broadcasted Geometric Radiomics | 7.3 | 7 | 8 | 7 | 7 | True |
| 9 | unconventional | Macroscopic Radiomic Modulation of Multi-Dilated Receptive Fields | 7.25 | 8 | 6 | 8 | 7 | True |
| 10 | unconventional | Distance-to-Metabolic-State Optimization via Inverse Latent Mapping | 7.25 | 8 | 6 | 8 | 7 | True |
| 11 | conservative | Uncertainty-Aware Mask Fusion via Bayesian Deep Supervision for MGMT Prediction | 7.1 | 7 | 6 | 8 | 8 | True |
| 12 | unconventional | Information Geometric Test-Time Adaptation for Missing Scans | 6.95 | 7 | 6 | 8 | 7 | True |
| 13 | unconventional | Contrast-Stripped Latent Tumor Topologies via Disentangled Representation Learning | 6.8 | 8 | 5 | 8 | 6 | True |
| 14 | unconventional | Shapley-Guided Dynamic Modality Dropout | 6.7 | 7 | 6 | 7 | 7 | True |
| 15 | unconventional | Hypothesis-Driven Niche Subtyping as Proxy | 6.4 | 8 | 4 | 7 | 7 | True |
| 16 | unconventional | XAI Heatmaps as Active Contrastive Regularizers | 6.4 | 7 | 5 | 7 | 7 | True |
| 17 | unconventional | Adversarial Latent Translation of Noisy Masks using Self-Attention GANs | 5.95 | 7 | 4 | 7 | 6 | True |
