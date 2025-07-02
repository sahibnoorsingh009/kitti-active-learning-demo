# KITTI PointNet Ensemble Active Learning Demo

🚀 **Live Demo**: https://sahibnoorsingh009.github.io/kitti-active-learning-demo

## Overview
Interactive demonstration of uncertainty-driven active learning using 5-model PointNet ensemble for 3D object detection on KITTI dataset.

## Key Features
- **Real-time uncertainty estimation** using ensemble model disagreement
- **Intelligent frame selection** with detailed explanations
- **Interactive frame analysis** showing why each frame was selected
- **Cost optimization** demonstrating 60-80% annotation savings
- **Educational AI** that explains every decision process

## How to Use
1. Visit the [live demo](https://sahibnoorsingh009.github.io/kitti-active-learning-demo)
2. Click "Start Processing" to begin simulation
3. Watch real-time frame selection based on uncertainty
4. Click red frames in timeline for detailed analysis
5. Explore the Selected Frames Gallery

## Technical Details
- **Framework**: React with interactive visualizations
- **Dataset**: KITTI autonomous driving simulation
- **Models**: 5 PointNet variants with different training strategies
- **Uncertainty Methods**: Ensemble disagreement, predictive entropy, mutual information
- **Selection Strategy**: Multi-factor uncertainty quantification

## Research Contributions
- Novel ensemble uncertainty framework for 3D object detection
- Practical active learning methodology with real-world cost analysis
- Educational visualization of AI decision-making process
- Interactive demonstration of uncertainty-driven sample selection

## Local Development
```bash
git clone https://github.com/sahibnoorsingh009/kitti-active-learning-demo.git
cd kitti-active-learning-demo
npm install
npm start
```
Flowchart
```mermaid
flowchart TD
    A[Unlabeled Dataset] --> B{Pre-trained Model?}
    B -->|Yes| C[Load Pre-trained Model]
    B -->|No| D[Bootstrap Labeled Set]
    D --> E[Train Initial Model]
    E --> F[Active Learning Loop]
    C --> F
    
    F --> G[Inference on Unlabeled Data]
    G --> H[Calculate Uncertainties]
    
    H --> H1[Classification Uncertainty<br/>Entropy: H = -Σ p*log p]
    H --> H2[Localization Uncertainty<br/>Bbox Variance]
    H --> H3[Confidence Uncertainty<br/>1 - max confidence]
    
    G --> I[Optional: Consistency Analysis]
    I --> I1[Apply Augmentations]
    I1 --> I2[Compare Predictions]
    I2 --> I3[Calculate Inconsistency]
    
    H1 --> J[Combine Scores]
    H2 --> J
    H3 --> J
    I3 --> K[Final Score = α×Uncertainty + β×Inconsistency]
    J --> L{Include Consistency?}
    L -->|Yes| K
    L -->|No| M[Score = Uncertainty Only]
    
    K --> N[Select Top-K Samples]
    M --> N
    N --> O[Human Annotation]
    O --> P[Update Dataset]
    P --> Q[Retrain/Fine-tune Model]
    
    Q --> R{Stopping Criteria?}
    R -->|Continue| F
    R -->|Stop| S[Final Model]
    
    style A fill:#e1f5fe
    style S fill:#c8e6c9
    style O fill:#fff3e0
    style Q fill:#f3e5f5
```


