import React, { useEffect } from 'react';

const PipelineVisualization = ({ pipeline, progress }) => {
  useEffect(() => {
    const arrowSize = Math.max(10, 50 - pipeline.length);
    document.documentElement.style.setProperty('--pipeline-step-size', `${Math.max(20, 50 - pipeline.length)}px`);
    document.documentElement.style.setProperty('--pipeline-arrow-width', `${arrowSize}px`);
  }, [pipeline.length]);

  if (pipeline.length === 0) {
    return (
      <div className="flex-1 p-4 bg-white">
        <h2 className="text-2xl font-bold mb-8 text-center">Pipeline Visualization</h2>
        <p className="mt-4">Drag and drop to create pipeline!</p>
      </div>
    );
  }

  const calculateStepProgress = (index) => {
    return ((index + 1) / pipeline.length) * 100;
  };

  return (
    <div className="flex-1 p-4 bg-white">
      <h2 className="text-2xl font-bold mb-8 text-center">Pipeline Visualization</h2>
      <div className="flex items-center justify-center pipeline-container mb-8">
        {pipeline.map((_, index) => (
          <div key={index} className="flex items-center pipeline-step">
            <div className={`relative flex items-center justify-center rounded-full border-2 ${progress >= calculateStepProgress(index) ? 'bg-green-500 border-green-500' : 'bg-blue-500 border-blue-500'}`} style={{ width: 'var(--pipeline-step-size)', height: 'var(--pipeline-step-size)' }}>
              {progress >= calculateStepProgress(index) ? (
                <svg className="text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ width: '60%', height: '60%' }}>
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                </svg>
              ) : (
                <span className="text-white">{index + 1}</span>
              )}
            </div>
            {index < pipeline.length - 1 && (
              <div className="flex-grow arrow mx-2" style={{ width: 'var(--pipeline-arrow-width)', height: 'var(--pipeline-arrow-size)' }}></div>
            )}
          </div>
        ))}
      </div>
      <div className="mt-8">
        <p className="text-xl font-bold text-center mb-2">Progress: {progress}%</p>
        <div className="w-full bg-gray-200 rounded-full h-4">
          <div className="bg-blue-500 h-4 rounded-full" style={{ width: `${progress}%` }}></div>
        </div>
      </div>
    </div>
  );
};

export default PipelineVisualization;
