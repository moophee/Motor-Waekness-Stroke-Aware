import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import * as handPoseDetection from '@tensorflow-models/hand-pose-detection';
import * as poseDetection from '@tensorflow-models/pose-detection';
import Webcam from 'react-webcam';

const MotorWeaknessAssessment = ({ onComplete }) => {
  // Refs
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  
  // State
  const [holdCountdown, setHoldCountdown] = useState(null);
  const [isComplete, setIsComplete] = useState(false);
  const [handDetector, setHandDetector] = useState(null);
  const [poseDetector, setPoseDetector] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [handStates, setHandStates] = useState({
    left: { detected: false, correctAngle: false, shoulderTouching: false },
    right: { detected: false, correctAngle: false, shoulderTouching: false }
  });
  const [showStartModal, setShowStartModal] = useState(true);
  const [timeLeft, setTimeLeft] = useState(60);
  const [challengeStarted, setChallengeStarted] = useState(false);
  const [showCompletionModal, setShowCompletionModal] = useState(false);
  const [holdDuration] = useState(10); // Need to hold for 10 seconds
  const [currentSide, setCurrentSide] = useState('right'); // Start with right side

  // Configuration
  const detectionLineY = 0.7; // Shoulder line position (70% down the screen)
  const targetAngle = 45; // 45 degrees from shoulder
  const angleTolerance = 15; // ±15 degrees tolerance
  const colors = {
    right: '#4CAF50',
    left: '#FF5252',
    line: '#3B82F6',
    target: 'rgba(251, 191, 36, 0.7)'
  };

  // Load models
  useEffect(() => {
    const loadModels = async () => {
      await tf.ready();
      
      // Load hand detector
      const handModel = handPoseDetection.SupportedModels.MediaPipeHands;
      const handDetectorConfig = {
        runtime: 'tfjs',
        modelType: 'full',
        maxHands: 2
      };
      const handDetector = await handPoseDetection.createDetector(handModel, handDetectorConfig);
      
      // Load pose detector
      const poseModel = poseDetection.SupportedModels.MoveNet;
      const poseDetectorConfig = {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
        enableSmoothing: true
      };
      const poseDetector = await poseDetection.createDetector(poseModel, poseDetectorConfig);
      
      setHandDetector(handDetector);
      setPoseDetector(poseDetector);
      setIsLoading(false);
    };

    loadModels();

    return () => {
      if (handDetector) handDetector.dispose();
      if (poseDetector) poseDetector.dispose();
    };
  }, []);

  // Timer effect
  useEffect(() => {
    if (!challengeStarted || timeLeft <= 0) return;

    const timer = setInterval(() => {
      setTimeLeft(prev => prev - 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [challengeStarted, timeLeft]);

  // Handle timer completion
  useEffect(() => {
    if (timeLeft <= 0 && challengeStarted) {
      setChallengeStarted(false);
      if (isComplete) {
        setShowCompletionModal(true);
      }
    }
  }, [timeLeft, challengeStarted, isComplete]);

  // Detection loop
  useEffect(() => {
    if (!handDetector || !poseDetector || isLoading || !challengeStarted) return;

    const detect = async () => {
      if (webcamRef.current?.video?.readyState === 4) {
        const video = webcamRef.current.video;
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        // Set dimensions
        webcamRef.current.video.width = videoWidth;
        webcamRef.current.video.height = videoHeight;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
        
        // Get canvas context
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, videoWidth, videoHeight);

        // Draw detection line for shoulders
        const shoulderLineY = videoHeight * detectionLineY;
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 3;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(0, shoulderLineY);
        ctx.lineTo(videoWidth, shoulderLineY);
        ctx.stroke();
        ctx.setLineDash([]);

        // Detect pose (for shoulders)
        const posePredictions = await poseDetector.estimatePoses(video);
        let shoulders = { left: null, right: null };
        
        if (posePredictions.length > 0) {
          const keypoints = posePredictions[0].keypoints;
          // MoveNet keypoints: 5 - left shoulder, 6 - right shoulder
          shoulders.left = keypoints[5];
          shoulders.right = keypoints[6];
          
          // Draw shoulders
          [shoulders.left, shoulders.right].forEach((shoulder, index) => {
            if (shoulder.score > 0.3) {
              const isLeft = index === 0;
              ctx.fillStyle = isLeft ? colors.left : colors.right;
              ctx.beginPath();
              ctx.arc(shoulder.x, shoulder.y, 8, 0, 2 * Math.PI);
              ctx.fill();
              
              // Draw line from shoulder to hand target position
              const targetX = isLeft 
                ? shoulder.x - (videoHeight * 0.3) // Left side: negative X
                : shoulder.x + (videoHeight * 0.3); // Right side: positive X
              const targetY = shoulder.y - (videoHeight * 0.3); // Upwards
              
              // Draw target circle
              ctx.fillStyle = colors.target;
              ctx.beginPath();
              ctx.arc(targetX, targetY, 15, 0, 2 * Math.PI);
              ctx.fill();
              
              // Draw line from shoulder to target
              ctx.strokeStyle = isLeft ? colors.left : colors.right;
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(shoulder.x, shoulder.y);
              ctx.lineTo(targetX, targetY);
              ctx.stroke();
            }
          });
        }

        // Detect hands
        const handPredictions = await handDetector.estimateHands(video);
        
        // Reset hand states
        const newHandStates = {
          left: { detected: false, correctAngle: false, shoulderTouching: false },
          right: { detected: false, correctAngle: false, shoulderTouching: false }
        };

        // Process each hand
        handPredictions.forEach((prediction) => {
          const landmarks = prediction.keypoints;
          const wrist = landmarks[0]; // Wrist is keypoint 0
          const handeness = prediction.handedness.toLowerCase();
          const isLeft = handeness === 'left';
          const color = isLeft ? colors.left : colors.right;
          const side = isLeft ? 'left' : 'right';
          const shoulder = isLeft ? shoulders.left : shoulders.right;

          // Update detection state
          newHandStates[side].detected = true;

          // Draw hand landmarks
          ctx.fillStyle = color;
          landmarks.forEach((keypoint) => {
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
            ctx.fill();
          });

          // Check if shoulder is touching the line
          if (shoulder && shoulder.score > 0.3) {
            const shoulderOnLine = Math.abs(shoulder.y - shoulderLineY) < 20;
            newHandStates[side].shoulderTouching = shoulderOnLine;
            
            // Calculate angle between shoulder and wrist
            const dx = wrist.x - shoulder.x;
            const dy = shoulder.y - wrist.y; // Inverted because y increases downward
            const angle = Math.atan2(dy, dx) * (180 / Math.PI);
            
            // Normalize angle to be positive (0-180)
            const normalizedAngle = angle < 0 ? angle + 360 : angle;
            
            // For left hand, we want ~135 degrees (180 - 45)
            // For right hand, we want ~45 degrees
            const targetAngleForSide = isLeft ? 135 : 45;
            const angleDiff = Math.abs(normalizedAngle - targetAngleForSide);
            
            newHandStates[side].correctAngle = angleDiff <= angleTolerance;
            
            // Draw angle text
            ctx.fillStyle = 'black';
            ctx.font = '16px Arial';
            ctx.fillText(
              `${Math.round(normalizedAngle)}°`, 
              wrist.x + 10, 
              wrist.y - 10
            );
          }
        });

        setHandStates(newHandStates);

        // Only check for current side
        const currentSideState = newHandStates[currentSide];
        
        // Only start countdown if BOTH conditions are met for the current side
        if (currentSideState.detected && 
            currentSideState.correctAngle && 
            currentSideState.shoulderTouching) {
          if (holdCountdown === null) {
            setHoldCountdown(holdDuration);
          } else if (holdCountdown > 0) {
            setTimeout(() => setHoldCountdown(prev => prev - 1), 1000);
          } else {
            // Switch sides or complete if both done
            if (currentSide === 'right') {
              setCurrentSide('left');
              setHoldCountdown(null);
            } else {
              setIsComplete(true);
            }
          }
        } else {
          setHoldCountdown(null);
        }
      }
    };

    const interval = setInterval(detect, 100);
    return () => clearInterval(interval);
  }, [handDetector, poseDetector, isLoading, holdCountdown, challengeStarted, holdDuration, currentSide]);

  // Start the challenge
  const startChallenge = () => {
    setShowStartModal(false);
    setChallengeStarted(true);
    setTimeLeft(60);
    setHoldCountdown(null);
    setIsComplete(false);
    setCurrentSide('right'); // Start with right side
  };

  // Reset Challenge
  const resetChallenge = () => {
    setIsComplete(false);
    setHoldCountdown(null);
    setChallengeStarted(false);
    setShowStartModal(true);
    setTimeLeft(60);
    setShowCompletionModal(false);
    setCurrentSide('right');
  };

  // Calculate detection status message
  const getStatusMessage = () => {
    if (!challengeStarted) return 'Click "Start Assessment" to begin';
    
    if (isComplete) return 'Assessment complete!';
    
    if (timeLeft <= 0) return "Time's up! You didn't complete the assessment in time.";
    
    const sideName = currentSide === 'right' ? 'Right' : 'Left';
    const currentState = handStates[currentSide];
    
    if (!currentState.detected) return `Show your ${sideName} arm to the camera`;
    if (!currentState.shoulderTouching) return `Position your ${sideName} shoulder on the line`;
    if (!currentState.correctAngle) return `Extend your ${sideName} arm at 45° angle`;
    
    return `Hold your ${sideName} arm position! ${holdDuration - (holdCountdown || holdDuration)}s`;
  };

  return (
    <div style={{
      maxWidth: '800px',
      margin: '0 auto',
      padding: '20px',
      fontFamily: "'Inter', sans-serif",
      color: '#2d3748'
    }}>
      {/* Start Modal */}
      {showStartModal && !isLoading && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '12px',
            textAlign: 'center',
            maxWidth: '500px'
          }}>
            <h2 style={{
              fontSize: '1.8rem',
              fontWeight: '700',
              marginBottom: '20px',
              color: '#2d3748'
            }}>
              Motor Weakness Assessment
            </h2>
            <p style={{ marginBottom: '20px', fontSize: '1.1rem' }}>
              This test assesses arm strength by having you hold each arm at a 45-degree angle.
            </p>
            <p style={{ marginBottom: '20px', fontSize: '1.1rem' }}>
              You'll need to hold each arm position for {holdDuration} seconds within 60 seconds total.
            </p>
            <button
              onClick={startChallenge}
              style={{
                backgroundColor: '#6366f1',
                color: 'white',
                border: 'none',
                padding: '12px 24px',
                borderRadius: '8px',
                fontSize: '1.1rem',
                fontWeight: '600',
                cursor: 'pointer',
                transition: 'background-color 0.3s'
              }}
            >
              Start Assessment
            </button>
          </div>
        </div>
      )}

      {/* Completion Modal */}
      {showCompletionModal && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.7)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '12px',
            textAlign: 'center',
            maxWidth: '500px'
          }}>
            <h2 style={{
              fontSize: '1.8rem',
              fontWeight: '700',
              marginBottom: '20px',
              color: '#10B981'
            }}>
              Assessment Complete!
            </h2>
            <p style={{ marginBottom: '30px', fontSize: '1.1rem' }}>
              You successfully completed the motor weakness assessment.
            </p>
            <div style={{ display: 'flex', gap: '15px', justifyContent: 'center' }}>
              <button
                onClick={resetChallenge}
                style={{
                  backgroundColor: '#E5E7EB',
                  color: '#374151',
                  border: 'none',
                  padding: '12px 24px',
                  borderRadius: '8px',
                  fontSize: '1.1rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'background-color 0.3s'
                }}
              >
                Retake Assessment
              </button>
              <button
                onClick={onComplete}
                style={{
                  backgroundColor: '#6366f1',
                  color: 'white',
                  border: 'none',
                  padding: '12px 24px',
                  borderRadius: '8px',
                  fontSize: '1.1rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'background-color 0.3s'
                }}
              >
                Continue
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div style={{
        textAlign: 'center',
        marginBottom: '30px'
      }}>
        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: '700',
          background: 'linear-gradient(90deg, #6366f1, #8b5cf6)',
          WebkitBackgroundClip: 'text',
          backgroundClip: 'text',
          color: 'transparent',
          marginBottom: '10px'
        }}>
          Motor Weakness Assessment
        </h1>
        <p style={{
          fontSize: '1.1rem',
          color: '#4a5568'
        }}>
          Hold each arm at 45° for {holdDuration} seconds within 60 seconds total
        </p>
      </div>

      {/* Timer Display */}
      {challengeStarted && (
        <div style={{
          textAlign: 'center',
          marginBottom: '20px',
          fontSize: '1.5rem',
          fontWeight: 'bold',
          color: timeLeft <= 10 ? '#ef4444' : '#2d3748'
        }}>
          Time Remaining: {timeLeft}s | 
          Current Side: {currentSide === 'right' ? 'Right Arm' : 'Left Arm'} | 
          Hold Time: {holdCountdown !== null ? `${holdDuration - holdCountdown}/${holdDuration}s` : '0s'}
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '480px',
          background: '#f8fafc',
          borderRadius: '12px',
          marginBottom: '20px'
        }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '4px solid #e2e8f0',
            borderTopColor: '#6366f1',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite',
            marginBottom: '20px'
          }}></div>
          <p style={{
            fontSize: '1.2rem',
            color: '#4a5568'
          }}>Loading pose detection models...</p>
        </div>
      )}

      {/* Camera Feed */}
      {!isLoading && (
        <div style={{
          position: 'relative',
          borderRadius: '12px',
          overflow: 'hidden',
          boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)',
          marginBottom: '20px'
        }}>
          <Webcam
            ref={webcamRef}
            mirrored={true}
            style={{
              display: 'block',
              width: '100%',
              height: 'auto',
              aspectRatio: '4/3'
            }}
          />
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%'
            }}
          />
          
          {/* Side Indicators */}
          <div style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            display: 'flex',
            gap: '10px'
          }}>
            <div style={{
              background: currentSide === 'left' ? 'rgba(0, 0, 0, 0.7)' : 'rgba(0, 0, 0, 0.4)',
              color: 'white',
              padding: '8px 12px',
              borderRadius: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'all 0.3s ease'
            }}>
              <div style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: handStates.left.detected ? colors.left : '#ccc',
                transition: 'all 0.3s ease'
              }}></div>
              Left
            </div>
            <div style={{
              background: currentSide === 'right' ? 'rgba(0, 0, 0, 0.7)' : 'rgba(0, 0, 0, 0.4)',
              color: 'white',
              padding: '8px 12px',
              borderRadius: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'all 0.3s ease'
            }}>
              <div style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                background: handStates.right.detected ? colors.right : '#ccc',
                transition: 'all 0.3s ease'
              }}></div>
              Right
            </div>
          </div>
          
          {/* Countdown - Only shows when current side is in correct position */}
          {holdCountdown !== null && (
            <div style={{
              position: 'absolute',
              bottom: '20px',
              left: '50%',
              transform: 'translateX(-50%)',
              background: 'rgba(0, 0, 0, 0.7)',
              color: 'white',
              padding: '12px 24px',
              borderRadius: '30px',
              fontSize: '1.5rem',
              fontWeight: 'bold'
            }}>
              {holdCountdown > 0 ? holdCountdown : 'Side Complete!'}
            </div>
          )}
        </div>
      )}

      {/* Status Message */}
      <div style={{
        background: '#f8fafc',
        padding: '16px',
        borderRadius: '8px',
        marginBottom: '20px',
        textAlign: 'center',
        fontWeight: '500'
      }}>
        {getStatusMessage()}
      </div>

      {/* Instructions */}
      <div style={{
        background: '#f0f9ff',
        padding: '16px',
        borderRadius: '8px',
        borderLeft: '4px solid #3b82f6'
      }}>
        <h3 style={{
          marginTop: '0',
          marginBottom: '8px',
          color: '#1e40af'
        }}>How to perform the assessment:</h3>
        <ol style={{
          paddingLeft: '20px',
          margin: '0',
          color: '#4b5563'
        }}>
          <li>Stand facing the camera with your side to the camera</li>
          <li>Position your shoulder on the blue line</li>
          <li>Extend your arm at a 45-degree angle from your shoulder</li>
          <li>Hold the position for {holdDuration} seconds</li>
          <li>The system will guide you through both arms</li>
          <li>You have 60 seconds total to complete both sides</li>
        </ol>
      </div>
    </div>
  );
};

export default MotorWeaknessAssessment;
