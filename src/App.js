import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Brain, Target, Zap, Eye, Database, Cpu, BarChart3, AlertTriangle, CheckCircle, Info } from 'lucide-react';

const KITTIEnsembleDemo = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [processedFrames, setProcessedFrames] = useState(0);
  const [selectedFrames, setSelectedFrames] = useState([]);
  const [frameUncertainties, setFrameUncertainties] = useState({});
  const [ensembleResults, setEnsembleResults] = useState([]);
  const [frameAnalysis, setFrameAnalysis] = useState({});
  const [modelPerformance, setModelPerformance] = useState(68.5);
  const [labelingBudget, setLabelingBudget] = useState(100);
  const [usedBudget, setUsedBudget] = useState(0);
  const [uncertaintyThreshold, setUncertaintyThreshold] = useState(0.65);
  const [processingSpeed, setProcessingSpeed] = useState(500);
  const [showEnsembleDetails, setShowEnsembleDetails] = useState(false);
  const [selectedFrameForAnalysis, setSelectedFrameForAnalysis] = useState(null);
  
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  
  // Simulated KITTI dataset
  const totalFrames = 100;
  const kittiFrames = Array.from({ length: totalFrames }, (_, i) => ({
    id: String(i).padStart(6, '0'),
    pointCount: Math.floor(Math.random() * 120000) + 80000,
    objects: Math.floor(Math.random() * 15) + 3,
    scene: ['Urban', 'Highway', 'Residential', 'Country'][Math.floor(Math.random() * 4)],
    weather: ['Clear', 'Overcast', 'Rain'][Math.floor(Math.random() * 3)],
    timeOfDay: ['Morning', 'Noon', 'Evening'][Math.floor(Math.random() * 3)],
    difficulty: ['Easy', 'Moderate', 'Hard'][Math.floor(Math.random() * 3)],
  }));
  
  // Ensemble model information
  const ensembleModels = [
    { id: 1, name: 'PointNet-Base', accuracy: 0.852, training: 'KITTI-Full' },
    { id: 2, name: 'PointNet-Aug', accuracy: 0.847, training: 'Data-Augmented' },
    { id: 3, name: 'PointNet-Deep', accuracy: 0.861, training: 'Deep-Architecture' },
    { id: 4, name: 'PointNet-Ensemble', accuracy: 0.839, training: 'Multi-Scale' },
    { id: 5, name: 'PointNet-Refined', accuracy: 0.855, training: 'Fine-Tuned' }
  ];
  
  // Processing effect
  useEffect(() => {
    if (isProcessing) {
      intervalRef.current = setInterval(() => {
        setCurrentFrame(prev => {
          const nextFrame = (prev + 1) % totalFrames;
          
          // Simulate ensemble processing
          const ensembleOutput = simulateEnsembleProcessing(kittiFrames[nextFrame]);
          
          setFrameUncertainties(prevUncertainties => ({
            ...prevUncertainties,
            [nextFrame]: ensembleOutput.frameUncertainty
          }));
          
          setFrameAnalysis(prevAnalysis => ({
            ...prevAnalysis,
            [nextFrame]: ensembleOutput.analysis
          }));
          
          setEnsembleResults(prevResults => [
            ...prevResults.slice(-9),
            ensembleOutput
          ]);
          
          setProcessedFrames(prev => prev + 1);
          
          // Auto-select high uncertainty frames
          if (ensembleOutput.frameUncertainty > uncertaintyThreshold && usedBudget < labelingBudget) {
            setSelectedFrames(prev => {
              if (!prev.includes(nextFrame)) {
                setUsedBudget(budget => budget + 1);
                return [...prev, nextFrame];
              }
              return prev;
            });
          }
          
          // Draw visualization
          drawKITTIVisualization(kittiFrames[nextFrame], ensembleOutput);
          
          return nextFrame;
        });
      }, processingSpeed);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isProcessing, uncertaintyThreshold, usedBudget, labelingBudget, processingSpeed]);
  
  const simulateEnsembleProcessing = (frame) => {
    // Simulate 5 PointNet models processing the frame
    const modelPredictions = ensembleModels.map(model => {
      const baseUncertainty = Math.random() * 0.4 + 0.1;
      
      // Add scene-specific uncertainty factors
      let sceneUncertainty = 0;
      if (frame.scene === 'Urban') sceneUncertainty += 0.1;
      if (frame.weather === 'Rain') sceneUncertainty += 0.15;
      if (frame.difficulty === 'Hard') sceneUncertainty += 0.2;
      if (frame.timeOfDay === 'Evening') sceneUncertainty += 0.1;
      
      const totalUncertainty = Math.min(baseUncertainty + sceneUncertainty, 0.95);
      
      return {
        modelId: model.id,
        modelName: model.name,
        classProbs: generateRandomProbs(10),
        bboxPred: generateRandomBbox(),
        uncertainty: totalUncertainty,
        confidence: 1 - totalUncertainty
      };
    });
    
    // Calculate ensemble metrics
    const avgUncertainty = modelPredictions.reduce((sum, pred) => sum + pred.uncertainty, 0) / 5;
    const modelDisagreement = calculateModelDisagreement(modelPredictions);
    const predictiveEntropy = calculatePredictiveEntropy(modelPredictions);
    const mutualInformation = calculateMutualInformation(modelPredictions);
    
    // Object-level uncertainties
    const objectUncertainties = Array.from({ length: frame.objects }, () => 
      Math.random() * 0.6 + 0.2
    );
    
    const frameUncertainty = objectUncertainties.reduce((sum, u) => sum + u, 0) / frame.objects;
    
    // Detailed analysis of why this frame is uncertain
    const analysis = analyzeFrameUncertainty(frame, modelPredictions, frameUncertainty);
    
    return {
      frameId: frame.id,
      frameUncertainty,
      objectUncertainties,
      modelPredictions,
      analysis,
      ensembleMetrics: {
        avgUncertainty,
        modelDisagreement,
        predictiveEntropy,
        mutualInformation
      }
    };
  };
  
  // Analyze why a frame has high/low uncertainty
  const analyzeFrameUncertainty = (frame, modelPredictions, frameUncertainty) => {
    const uncertaintyFactors = [];
    const strengths = [];
    const concerns = [];
    
    // Weather analysis
    if (frame.weather === 'Rain') {
      uncertaintyFactors.push({
        factor: 'Adverse Weather',
        impact: 'High',
        description: 'Rain reduces LiDAR point quality and creates noise',
        contribution: 0.15
      });
      concerns.push('Rain affects point cloud density and accuracy');
    } else if (frame.weather === 'Clear') {
      strengths.push('Clear weather provides optimal LiDAR conditions');
    }
    
    // Time of day analysis
    if (frame.timeOfDay === 'Evening') {
      uncertaintyFactors.push({
        factor: 'Low Light Conditions',
        impact: 'Medium',
        description: 'Evening conditions may affect object visibility',
        contribution: 0.1
      });
      concerns.push('Reduced visibility in evening conditions');
    } else if (frame.timeOfDay === 'Noon') {
      strengths.push('Optimal lighting conditions for detection');
    }
    
    // Scene complexity analysis
    if (frame.scene === 'Urban') {
      uncertaintyFactors.push({
        factor: 'Complex Urban Environment',
        impact: 'Medium',
        description: 'Dense urban scenes with multiple objects increase uncertainty',
        contribution: 0.1
      });
      concerns.push('High object density and occlusion in urban environment');
    } else if (frame.scene === 'Highway') {
      strengths.push('Simple highway scene with clear object separation');
    }
    
    // Object count analysis
    if (frame.objects > 10) {
      uncertaintyFactors.push({
        factor: 'High Object Density',
        impact: 'Medium',
        description: `${frame.objects} objects detected - increases scene complexity`,
        contribution: 0.08
      });
      concerns.push(`Scene contains ${frame.objects} objects, increasing complexity`);
    } else if (frame.objects < 5) {
      strengths.push(`Simple scene with only ${frame.objects} objects`);
    }
    
    // Point cloud density analysis
    if (frame.pointCount < 90000) {
      uncertaintyFactors.push({
        factor: 'Sparse Point Cloud',
        impact: 'Medium',
        description: 'Lower point density may affect detection accuracy',
        contribution: 0.12
      });
      concerns.push('Below average point cloud density');
    } else if (frame.pointCount > 110000) {
      strengths.push('High-density point cloud provides detailed information');
    }
    
    // Difficulty assessment
    if (frame.difficulty === 'Hard') {
      uncertaintyFactors.push({
        factor: 'Challenging Scenario',
        impact: 'High',
        description: 'Frame marked as difficult due to multiple factors',
        contribution: 0.2
      });
      concerns.push('Multiple challenging factors present in this frame');
    } else if (frame.difficulty === 'Easy') {
      strengths.push('Straightforward detection scenario');
    }
    
    // Model disagreement analysis
    const disagreement = calculateModelDisagreement(modelPredictions);
    if (disagreement > 0.3) {
      uncertaintyFactors.push({
        factor: 'High Model Disagreement',
        impact: 'High',
        description: 'Ensemble models show significant disagreement',
        contribution: disagreement
      });
      concerns.push('Models disagree significantly on this frame');
    } else if (disagreement < 0.1) {
      strengths.push('Strong model consensus on predictions');
    }
    
    // Selection recommendation
    const shouldSelect = frameUncertainty > 0.65;
    const confidence = shouldSelect ? 'High' : frameUncertainty > 0.4 ? 'Medium' : 'Low';
    
    return {
      frameId: frame.id,
      uncertaintyScore: frameUncertainty,
      shouldSelect,
      confidence,
      uncertaintyFactors,
      strengths,
      concerns,
      summary: generateFrameSummary(frame, frameUncertainty, shouldSelect, uncertaintyFactors.length),
      recommendation: generateRecommendation(frameUncertainty, uncertaintyFactors)
    };
  };
  
  const generateFrameSummary = (frame, uncertainty, shouldSelect, factorCount) => {
    if (shouldSelect) {
      return `Frame ${frame.id} selected for labeling due to high uncertainty (${(uncertainty * 100).toFixed(1)}%). ${factorCount} uncertainty factors identified in this ${frame.scene.toLowerCase()} ${frame.weather.toLowerCase()} scene.`;
    } else if (uncertainty > 0.4) {
      return `Frame ${frame.id} shows moderate uncertainty (${(uncertainty * 100).toFixed(1)}%). Consider for labeling if budget allows.`;
    } else {
      return `Frame ${frame.id} has low uncertainty (${(uncertainty * 100).toFixed(1)}%). Model is confident - skip labeling.`;
    }
  };
  
  const generateRecommendation = (uncertainty, factors) => {
    if (uncertainty > 0.7) {
      return 'PRIORITY: Label immediately - high learning value expected';
    } else if (uncertainty > 0.5) {
      return 'RECOMMENDED: Good candidate for labeling';
    } else if (uncertainty > 0.3) {
      return 'OPTIONAL: Label if budget permits';
    } else {
      return 'SKIP: Model is confident, labeling not needed';
    }
  };
  
  const generateRandomProbs = (numClasses) => {
    const probs = Array.from({ length: numClasses }, () => Math.random());
    const sum = probs.reduce((a, b) => a + b, 0);
    return probs.map(p => p / sum);
  };
  
  const generateRandomBbox = () => {
    return {
      x: Math.random() * 50 - 25,
      y: Math.random() * 50 - 25,
      z: Math.random() * 3 - 1.5,
      l: Math.random() * 5 + 2,
      w: Math.random() * 3 + 1,
      h: Math.random() * 2 + 1,
      rotation: Math.random() * Math.PI * 2
    };
  };
  
  const calculateModelDisagreement = (predictions) => {
    const uncertainties = predictions.map(p => p.uncertainty);
    const mean = uncertainties.reduce((a, b) => a + b, 0) / uncertainties.length;
    const variance = uncertainties.reduce((sum, u) => sum + Math.pow(u - mean, 2), 0) / uncertainties.length;
    return Math.sqrt(variance);
  };
  
  const calculatePredictiveEntropy = (predictions) => {
    const avgUncertainty = predictions.reduce((sum, pred) => sum + pred.uncertainty, 0) / predictions.length;
    return -avgUncertainty * Math.log(avgUncertainty + 1e-8) - (1 - avgUncertainty) * Math.log(1 - avgUncertainty + 1e-8);
  };
  
  const calculateMutualInformation = (predictions) => {
    const disagreement = calculateModelDisagreement(predictions);
    const entropy = calculatePredictiveEntropy(predictions);
    return Math.min(disagreement * entropy * 2, 1.0);
  };
  
  const drawKITTIVisualization = (frame, ensembleOutput) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, width, height);
    
    // Draw point cloud
    const pointCount = Math.floor(frame.pointCount / 500);
    const uncertainty = ensembleOutput.frameUncertainty;
    const isSelected = selectedFrames.includes(currentFrame);
    
    // Draw ground grid
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 1;
    for (let i = 0; i < width; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, height);
      ctx.stroke();
    }
    for (let i = 0; i < height; i += 40) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(width, i);
      ctx.stroke();
    }
    
    // Draw LiDAR points
    for (let i = 0; i < pointCount; i++) {
      const x = Math.random() * width;
      const y = Math.random() * height;
      const distance = Math.sqrt(Math.pow(x - width/2, 2) + Math.pow(y - height/2, 2));
      const maxDistance = Math.sqrt(Math.pow(width/2, 2) + Math.pow(height/2, 2));
      const intensity = 1 - (distance / maxDistance);
      
      if (isSelected) {
        ctx.fillStyle = `rgba(255, 100, 100, ${intensity * 0.8})`;
      } else if (uncertainty > uncertaintyThreshold) {
        ctx.fillStyle = `rgba(255, 200, 100, ${intensity * 0.6})`;
      } else {
        ctx.fillStyle = `rgba(100, 150, 255, ${intensity * 0.4})`;
      }
      
      ctx.fillRect(x - 1, y - 1, 2, 2);
    }
    
    // Draw detected objects
    ctx.lineWidth = 2;
    for (let i = 0; i < frame.objects; i++) {
      const centerX = (Math.random() * 0.6 + 0.2) * width;
      const centerY = (Math.random() * 0.6 + 0.2) * height;
      const objWidth = 20 + Math.random() * 40;
      const objHeight = 15 + Math.random() * 25;
      
      const objUncertainty = ensembleOutput.objectUncertainties[i] || 0.5;
      if (objUncertainty > 0.7) {
        ctx.strokeStyle = '#ff4444';
      } else if (objUncertainty > 0.4) {
        ctx.strokeStyle = '#ffaa44';
      } else {
        ctx.strokeStyle = '#44ff44';
      }
      
      ctx.strokeRect(centerX - objWidth/2, centerY - objHeight/2, objWidth, objHeight);
      
      ctx.fillStyle = '#ffffff';
      ctx.font = '10px monospace';
      ctx.fillText(`Obj${i+1}`, centerX - objWidth/2, centerY - objHeight/2 - 5);
      
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fillRect(centerX + objWidth/2 - 10, centerY - objHeight/2, 8, objUncertainty * 20);
    }
    
    // Frame uncertainty overlay
    if (uncertainty > uncertaintyThreshold) {
      ctx.fillStyle = `rgba(255, 255, 0, ${(uncertainty - uncertaintyThreshold) * 0.2})`;
      ctx.fillRect(0, 0, width, height);
    }
  };
  
  const resetDemo = () => {
    setIsProcessing(false);
    setCurrentFrame(0);
    setProcessedFrames(0);
    setSelectedFrames([]);
    setFrameUncertainties({});
    setFrameAnalysis({});
    setEnsembleResults([]);
    setUsedBudget(0);
    setModelPerformance(68.5);
    setSelectedFrameForAnalysis(null);
  };
  
  const toggleProcessing = () => {
    setIsProcessing(!isProcessing);
  };
  
  // Update performance based on selected frames
  useEffect(() => {
    if (selectedFrames.length > 0) {
      const improvement = selectedFrames.length * 0.3;
      setModelPerformance(prev => Math.min(92, 68.5 + improvement));
    }
  }, [selectedFrames]);
  
  const currentFrameData = kittiFrames[currentFrame];
  const currentUncertainty = frameUncertainties[currentFrame] || 0;
  const isCurrentSelected = selectedFrames.includes(currentFrame);
  const currentEnsembleResult = ensembleResults[ensembleResults.length - 1];
  const currentAnalysis = frameAnalysis[currentFrame];
  
  // Get analysis for selected frame
  const selectedAnalysis = selectedFrameForAnalysis !== null ? frameAnalysis[selectedFrameForAnalysis] : null;
  const selectedFrameData = selectedFrameForAnalysis !== null ? kittiFrames[selectedFrameForAnalysis] : null;
  
  return (
    <div style={{
      minHeight: '100vh',
      backgroundColor: '#111827',
      color: 'white',
      padding: '1rem'
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <h1 style={{
            fontSize: '2rem',
            fontWeight: 'bold',
            marginBottom: '0.5rem',
            background: 'linear-gradient(to right, #60a5fa, #a855f7)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            KITTI PointNet Ensemble Active Learning
          </h1>
          <p style={{ color: '#d1d5db' }}>
            Real-time uncertainty estimation with intelligent frame selection analysis
          </p>
        </div>
        
        {/* Control Panel */}
        <div style={{
          backgroundColor: '#1f2937',
          borderRadius: '0.5rem',
          padding: '1rem',
          marginBottom: '1.5rem'
        }}>
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <button
                onClick={toggleProcessing}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.5rem 1rem',
                  borderRadius: '0.5rem',
                  fontWeight: 'bold',
                  border: 'none',
                  cursor: 'pointer',
                  backgroundColor: isProcessing ? '#dc2626' : '#16a34a',
                  color: 'white'
                }}
              >
                {isProcessing ? <Pause size={18} /> : <Play size={18} />}
                {isProcessing ? 'Pause' : 'Start'} Processing
              </button>
              
              <button
                onClick={resetDemo}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.5rem 1rem',
                  backgroundColor: '#4b5563',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0.5rem',
                  cursor: 'pointer'
                }}
              >
                <RotateCcw size={18} />
                Reset
              </button>
              
              <button
                onClick={() => setShowEnsembleDetails(!showEnsembleDetails)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.5rem 1rem',
                  backgroundColor: '#7c3aed',
                  color: 'white',
                  border: 'none',
                  borderRadius: '0.5rem',
                  cursor: 'pointer'
                }}
              >
                <Brain size={18} />
                Ensemble Details
              </button>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.875rem' }}>
              <div>
                <label style={{ color: '#9ca3af' }}>Speed (ms):</label>
                <input
                  type="range"
                  min="100"
                  max="1000"
                  value={processingSpeed}
                  onChange={(e) => setProcessingSpeed(parseInt(e.target.value))}
                  style={{ marginLeft: '0.5rem', width: '80px' }}
                />
                <span style={{ marginLeft: '0.5rem' }}>{processingSpeed}</span>
              </div>
              <div>
                <label style={{ color: '#9ca3af' }}>Threshold:</label>
                <input
                  type="range"
                  min="0.3"
                  max="0.9"
                  step="0.05"
                  value={uncertaintyThreshold}
                  onChange={(e) => setUncertaintyThreshold(parseFloat(e.target.value))}
                  style={{ marginLeft: '0.5rem', width: '80px' }}
                />
                <span style={{ marginLeft: '0.5rem' }}>{uncertaintyThreshold.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Main Content Grid */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: selectedAnalysis ? '1fr 1fr 1fr' : '1fr 1fr', 
          gap: '1.5rem', 
          marginBottom: '1.5rem' 
        }}>
          {/* Point Cloud Visualization */}
          <div style={{
            backgroundColor: '#1f2937',
            borderRadius: '0.5rem',
            padding: '1rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h2 style={{ fontSize: '1.125rem', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Database style={{ color: '#60a5fa' }} size={20} />
                KITTI Frame Visualization
              </h2>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.875rem' }}>
                <span style={{
                  padding: '0.25rem 0.5rem',
                  borderRadius: '0.25rem',
                  fontSize: '0.75rem',
                  fontWeight: '500',
                  backgroundColor: isCurrentSelected ? '#7f1d1d' : currentUncertainty > uncertaintyThreshold ? '#78350f' : '#14532d',
                  color: isCurrentSelected ? '#fecaca' : currentUncertainty > uncertaintyThreshold ? '#fed7aa' : '#bbf7d0'
                }}>
                  {isCurrentSelected ? 'SELECTED' : currentUncertainty > uncertaintyThreshold ? 'HIGH UNCERTAINTY' : 'PROCESSED'}
                </span>
                <span style={{ color: '#9ca3af' }}>Frame {currentFrameData.id}</span>
              </div>
            </div>
            
            <div style={{ position: 'relative', backgroundColor: 'black', borderRadius: '0.5rem', overflow: 'hidden' }}>
              <canvas 
                ref={canvasRef}
                width={500}
                height={350}
                style={{ width: '100%', height: 'auto' }}
              />
              
              {/* Frame Info Overlay */}
              <div style={{
                position: 'absolute',
                top: '0.75rem',
                left: '0.75rem',
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                borderRadius: '0.375rem',
                padding: '0.75rem',
                fontSize: '0.75rem'
              }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
                  <div>
                    <div style={{ color: '#9ca3af' }}>Points:</div>
                    <div style={{ fontFamily: 'monospace', color: '#60a5fa' }}>{currentFrameData.pointCount.toLocaleString()}</div>
                  </div>
                  <div>
                    <div style={{ color: '#9ca3af' }}>Objects:</div>
                    <div style={{ fontFamily: 'monospace', color: '#10b981' }}>{currentFrameData.objects}</div>
                  </div>
                  <div>
                    <div style={{ color: '#9ca3af' }}>Scene:</div>
                    <div style={{ fontFamily: 'monospace', color: '#a855f7' }}>{currentFrameData.scene}</div>
                  </div>
                  <div>
                    <div style={{ color: '#9ca3af' }}>Weather:</div>
                    <div style={{ fontFamily: 'monospace', color: '#f97316' }}>{currentFrameData.weather}</div>
                  </div>
                </div>
              </div>
              
              {/* Uncertainty Meter */}
              <div style={{
                position: 'absolute',
                bottom: '0.75rem',
                left: '0.75rem',
                right: '0.75rem'
              }}>
                <div style={{ backgroundColor: 'rgba(31, 41, 55, 0.9)', borderRadius: '0.375rem', padding: '0.5rem' }}>
                  <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginBottom: '0.25rem' }}>
                    Frame Uncertainty: {(currentUncertainty * 100).toFixed(1)}%
                  </div>
                  <div style={{ width: '100%', backgroundColor: '#374151', borderRadius: '9999px', height: '0.5rem' }}>
                    <div 
                      style={{
                        height: '0.5rem',
                        borderRadius: '9999px',
                        transition: 'all 0.3s',
                        backgroundColor: currentUncertainty > 0.7 ? '#ef4444' :
                          currentUncertainty > 0.5 ? '#f97316' :
                          currentUncertainty > 0.3 ? '#eab308' : '#10b981',
                        width: `${currentUncertainty * 100}%`
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
            
            {/* Current Frame Analysis */}
            {currentAnalysis && (
              <div style={{
                marginTop: '1rem',
                backgroundColor: '#374151',
                borderRadius: '0.5rem',
                padding: '0.75rem'
              }}>
                <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Info size={16} style={{ color: '#60a5fa' }} />
                  Current Frame Analysis
                </h4>
                <p style={{ fontSize: '0.75rem', color: '#d1d5db', marginBottom: '0.5rem' }}>
                  {currentAnalysis.summary}
                </p>
                <div style={{ fontSize: '0.75rem', fontWeight: '500', color: currentAnalysis.shouldSelect ? '#ef4444' : '#10b981' }}>
                  {currentAnalysis.recommendation}
                </div>
                {currentAnalysis.shouldSelect && (
                  <button
                    onClick={() => setSelectedFrameForAnalysis(currentFrame)}
                    style={{
                      marginTop: '0.5rem',
                      padding: '0.25rem 0.5rem',
                      backgroundColor: '#3b82f6',
                      color: 'white',
                      border: 'none',
                      borderRadius: '0.25rem',
                      fontSize: '0.75rem',
                      cursor: 'pointer'
                    }}
                  >
                    Analyze Why Selected
                  </button>
                )}
              </div>
            )}
          </div>
          
          {/* Ensemble Models Panel */}
          <div style={{
            backgroundColor: '#1f2937',
            borderRadius: '0.5rem',
            padding: '1rem'
          }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Cpu style={{ color: '#a855f7' }} size={20} />
              Ensemble Model Status
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {ensembleModels.map((model) => {
                const modelPrediction = currentEnsembleResult?.modelPredictions?.find(p => p.modelId === model.id);
                const uncertainty = modelPrediction?.uncertainty || 0;
                const confidence = modelPrediction?.confidence || model.accuracy;
                
                return (
                  <div key={model.id} style={{ backgroundColor: '#374151', borderRadius: '0.375rem', padding: '0.75rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div style={{
                          width: '0.75rem',
                          height: '0.75rem',
                          borderRadius: '50%',
                          backgroundColor: isProcessing ? '#10b981' : '#6b7280',
                          animation: isProcessing ? 'pulse 2s infinite' : 'none'
                        }} />
                        <span style={{ fontWeight: '500' }}>{model.name}</span>
                      </div>
                      <span style={{ fontSize: '0.75rem', color: '#9ca3af' }}>{model.training}</span>
                    </div>
                    
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <div>
                          <span style={{ color: '#9ca3af' }}>Confidence:</span>
                          <span style={{ marginLeft: '0.25rem', fontFamily: 'monospace', color: '#10b981' }}>
                            {(confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div>
                          <span style={{ color: '#9ca3af' }}>Uncertainty:</span>
                          <span style={{
                            marginLeft: '0.25rem',
                            fontFamily: 'monospace',
                            color: uncertainty > 0.7 ? '#ef4444' : uncertainty > 0.5 ? '#f97316' : uncertainty > 0.3 ? '#eab308' : '#10b981'
                          }}>
                            {(uncertainty * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    <div style={{ marginTop: '0.5rem', width: '100%', backgroundColor: '#4b5563', borderRadius: '9999px', height: '0.375rem' }}>
                      <div 
                        style={{
                          backgroundColor: '#10b981',
                          height: '0.375rem',
                          borderRadius: '9999px',
                          transition: 'all 0.3s',
                          width: `${confidence * 100}%`
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
            
            {showEnsembleDetails && currentEnsembleResult && (
              <div style={{ marginTop: '1rem', backgroundColor: '#374151', borderRadius: '0.375rem', padding: '0.75rem' }}>
                <h4 style={{ fontWeight: '500', marginBottom: '0.5rem' }}>Ensemble Metrics</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.75rem' }}>
                  <div>
                    <span style={{ color: '#9ca3af' }}>Model Disagreement:</span>
                    <span style={{ marginLeft: '0.25rem', fontFamily: 'monospace', color: '#f97316' }}>
                      {(currentEnsembleResult.ensembleMetrics.modelDisagreement * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span style={{ color: '#9ca3af' }}>Mutual Information:</span>
                    <span style={{ marginLeft: '0.25rem', fontFamily: 'monospace', color: '#60a5fa' }}>
                      {(currentEnsembleResult.ensembleMetrics.mutualInformation * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span style={{ color: '#9ca3af' }}>Predictive Entropy:</span>
                    <span style={{ marginLeft: '0.25rem', fontFamily: 'monospace', color: '#a855f7' }}>
                      {currentEnsembleResult.ensembleMetrics.predictiveEntropy.toFixed(3)}
                    </span>
                  </div>
                  <div>
                    <span style={{ color: '#9ca3af' }}>Avg Uncertainty:</span>
                    <span style={{ marginLeft: '0.25rem', fontFamily: 'monospace', color: '#ef4444' }}>
                      {(currentEnsembleResult.ensembleMetrics.avgUncertainty * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Detailed Frame Analysis Panel */}
          {selectedAnalysis && (
            <div style={{
              backgroundColor: '#1f2937',
              borderRadius: '0.5rem',
              padding: '1rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <h3 style={{ fontSize: '1.125rem', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertTriangle style={{ color: '#f59e0b' }} size={20} />
                  Frame {selectedFrameData.id} Analysis
                </h3>
                <button
                  onClick={() => setSelectedFrameForAnalysis(null)}
                  style={{
                    padding: '0.25rem 0.5rem',
                    backgroundColor: '#4b5563',
                    color: 'white',
                    border: 'none',
                    borderRadius: '0.25rem',
                    fontSize: '0.75rem',
                    cursor: 'pointer'
                  }}
                >
                  Close
                </button>
              </div>
              
              {/* Selection Summary */}
              <div style={{
                backgroundColor: selectedAnalysis.shouldSelect ? '#7f1d1d' : '#14532d',
                borderRadius: '0.375rem',
                padding: '0.75rem',
                marginBottom: '1rem'
              }}>
                <div style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.25rem' }}>
                  {selectedAnalysis.shouldSelect ? '🎯 SELECTED FOR LABELING' : '✅ SKIP LABELING'}
                </div>
                <div style={{ fontSize: '0.75rem', opacity: 0.9 }}>
                  Uncertainty: {(selectedAnalysis.uncertaintyScore * 100).toFixed(1)}% 
                  ({selectedAnalysis.confidence} confidence)
                </div>
              </div>
              
              {/* Uncertainty Factors */}
              {selectedAnalysis.uncertaintyFactors.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem', color: '#ef4444' }}>
                    🚨 Uncertainty Factors
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '150px', overflowY: 'auto' }}>
                    {selectedAnalysis.uncertaintyFactors.map((factor, index) => (
                      <div key={index} style={{
                        backgroundColor: '#374151',
                        borderRadius: '0.25rem',
                        padding: '0.5rem',
                        borderLeft: `3px solid ${factor.impact === 'High' ? '#ef4444' : factor.impact === 'Medium' ? '#f59e0b' : '#84cc16'}`
                      }}>
                        <div style={{ fontSize: '0.75rem', fontWeight: '500' }}>
                          {factor.factor} ({factor.impact} Impact)
                        </div>
                        <div style={{ fontSize: '0.6875rem', color: '#d1d5db', marginTop: '0.25rem' }}>
                          {factor.description}
                        </div>
                        <div style={{ fontSize: '0.6875rem', color: '#9ca3af', marginTop: '0.25rem' }}>
                          Contribution: +{(factor.contribution * 100).toFixed(1)}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Strengths */}
              {selectedAnalysis.strengths.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem', color: '#10b981' }}>
                    ✅ Strengths
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', maxHeight: '100px', overflowY: 'auto' }}>
                    {selectedAnalysis.strengths.map((strength, index) => (
                      <div key={index} style={{
                        fontSize: '0.75rem',
                        color: '#bbf7d0',
                        backgroundColor: '#14532d',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '0.25rem'
                      }}>
                        • {strength}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Concerns */}
              {selectedAnalysis.concerns.length > 0 && (
                <div style={{ marginBottom: '1rem' }}>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem', color: '#f59e0b' }}>
                    ⚠️ Concerns
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', maxHeight: '100px', overflowY: 'auto' }}>
                    {selectedAnalysis.concerns.map((concern, index) => (
                      <div key={index} style={{
                        fontSize: '0.75rem',
                        color: '#fed7aa',
                        backgroundColor: '#78350f',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '0.25rem'
                      }}>
                        • {concern}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {/* Final Recommendation */}
              <div style={{
                backgroundColor: '#4b5563',
                borderRadius: '0.375rem',
                padding: '0.75rem'
              }}>
                <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                  📝 Recommendation
                </h4>
                <div style={{ fontSize: '0.75rem', color: '#d1d5db' }}>
                  {selectedAnalysis.recommendation}
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Selected Frames Timeline */}
        <div style={{
          backgroundColor: '#1f2937',
          borderRadius: '0.5rem',
          padding: '1rem',
          marginBottom: '1.5rem'
        }}>
          <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Eye style={{ color: '#f59e0b' }} size={20} />
            Selected Frames for Manual Annotation
          </h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gap: '0.5rem', marginBottom: '1rem' }}>
            {Array.from({ length: Math.min(totalFrames, 60) }, (_, i) => {
              const isSelected = selectedFrames.includes(i);
              const uncertainty = frameUncertainties[i] || 0;
              const isProcessed = i <= currentFrame;
              const isCurrent = i === currentFrame;
              const isAnalyzed = selectedFrameForAnalysis === i;
              
              let bgColor = '#374151';
              if (isAnalyzed) {
                bgColor = '#3b82f6';
              } else if (isCurrent) {
                bgColor = '#06b6d4';
              } else if (isSelected) {
                bgColor = '#ef4444';
              } else if (isProcessed) {
                bgColor = uncertainty > uncertaintyThreshold ? '#f59e0b' : '#4b5563';
              }
              
              return (
                <button
                  key={i}
                  onClick={() => isSelected ? setSelectedFrameForAnalysis(i) : null}
                  style={{
                    height: '2rem',
                    borderRadius: '0.25rem',
                    fontSize: '0.75rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontFamily: 'monospace',
                    transition: 'all 0.3s',
                    backgroundColor: bgColor,
                    color: 'white',
                    border: 'none',
                    cursor: isSelected ? 'pointer' : 'default',
                    opacity: isSelected ? 1 : 0.8
                  }}
                  title={isSelected ? `Click to analyze Frame ${String(i).padStart(6, '0')} - Uncertainty: ${(uncertainty * 100).toFixed(1)}%` : `Frame ${String(i).padStart(6, '0')}`}
                >
                  {i < 10 ? i : ''}
                </button>
              );
            })}
          </div>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', fontSize: '0.875rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '1rem', height: '1rem', backgroundColor: '#3b82f6', borderRadius: '0.25rem' }}></div>
              <span style={{ color: '#9ca3af' }}>Analyzing</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '1rem', height: '1rem', backgroundColor: '#06b6d4', borderRadius: '0.25rem' }}></div>
              <span style={{ color: '#9ca3af' }}>Current</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '1rem', height: '1rem', backgroundColor: '#ef4444', borderRadius: '0.25rem' }}></div>
              <span style={{ color: '#9ca3af' }}>Selected (click to analyze)</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '1rem', height: '1rem', backgroundColor: '#f59e0b', borderRadius: '0.25rem' }}></div>
              <span style={{ color: '#9ca3af' }}>High uncertainty</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ width: '1rem', height: '1rem', backgroundColor: '#4b5563', borderRadius: '0.25rem' }}></div>
              <span style={{ color: '#9ca3af' }}>Processed</span>
            </div>
          </div>
        </div>
        
        {/* Statistics Panel */}
        <div style={{
          backgroundColor: '#1f2937',
          borderRadius: '0.5rem',
          padding: '1rem',
          marginBottom: '1.5rem'
        }}>
          <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '1rem' }}>Active Learning Statistics</h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#60a5fa' }}>{processedFrames}</div>
              <div style={{ fontSize: '0.875rem', color: '#9ca3af' }}>Frames Processed</div>
            </div>
            
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#ef4444' }}>{selectedFrames.length}</div>
              <div style={{ fontSize: '0.875rem', color: '#9ca3af' }}>Selected for Labeling</div>
            </div>
            
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#10b981' }}>
                {processedFrames > 0 ? ((selectedFrames.length / processedFrames) * 100).toFixed(1) : 0}%
              </div>
              <div style={{ fontSize: '0.875rem', color: '#9ca3af' }}>Selection Rate</div>
            </div>
            
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#a855f7' }}>
                {processedFrames > 0 ? (((processedFrames - selectedFrames.length) / processedFrames) * 100).toFixed(1) : 0}%
              </div>
              <div style={{ fontSize: '0.875rem', color: '#9ca3af' }}>Cost Savings</div>
            </div>
          </div>
          
          <div style={{ marginTop: '1.5rem', backgroundColor: '#374151', borderRadius: '0.375rem', padding: '1rem' }}>
            <h4 style={{ fontWeight: '500', marginBottom: '0.5rem' }}>Expected Outcomes</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', fontSize: '0.875rem' }}>
              <div>
                <div style={{ color: '#9ca3af' }}>Full Dataset Cost:</div>
                <div style={{ fontFamily: 'monospace', color: '#ef4444' }}>${(totalFrames * 25).toLocaleString()}</div>
              </div>
              <div>
                <div style={{ color: '#9ca3af' }}>Active Learning Cost:</div>
                <div style={{ fontFamily: 'monospace', color: '#10b981' }}>${(selectedFrames.length * 25).toLocaleString()}</div>
              </div>
              <div>
                <div style={{ color: '#9ca3af' }}>Potential Savings:</div>
                <div style={{ fontFamily: 'monospace', color: '#60a5fa' }}>
                  ${((totalFrames - selectedFrames.length) * 25).toLocaleString()}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Key Features */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
          <div style={{
            background: 'linear-gradient(to bottom right, #1e3a8a, #1e40af)',
            borderRadius: '0.5rem',
            padding: '1rem'
          }}>
            <h4 style={{ fontWeight: '600', marginBottom: '0.5rem' }}>5-Model Ensemble</h4>
            <p style={{ color: '#bfdbfe', fontSize: '0.875rem' }}>
              Combines predictions from 5 different PointNet models to calculate robust uncertainty estimates.
            </p>
          </div>
          
          <div style={{
            background: 'linear-gradient(to bottom right, #14532d, #166534)',
            borderRadius: '0.5rem',
            padding: '1rem'
          }}>
            <h4 style={{ fontWeight: '600', marginBottom: '0.5rem' }}>Intelligent Analysis</h4>
            <p style={{ color: '#bbf7d0', fontSize: '0.875rem' }}>
              Explains why each frame is selected based on weather, complexity, and model disagreement.
            </p>
          </div>
          
          <div style={{
            background: 'linear-gradient(to bottom right, #581c87, #6b21a8)',
            borderRadius: '0.5rem',
            padding: '1rem'
          }}>
            <h4 style={{ fontWeight: '600', marginBottom: '0.5rem' }}>Cost Optimization</h4>
            <p style={{ color: '#e9d5ff', fontSize: '0.875rem' }}>
              Reduces annotation costs by 60-80% while maintaining optimal model performance.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <KITTIEnsembleDemo />
    </div>
  );
}

export default App;