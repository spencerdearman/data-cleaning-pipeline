import React from 'react';

const PipelineVisualization = ({ pipeline, progress }) => {
  if (pipeline.length === 0) {
    return (
      <div className="flex-1 p-4 bg-white">
        <h2 className="text-2xl font-bold mb-4">Pipeline Visualization</h2>
        <p>No steps in the pipeline</p>
      </div>
    );
  }

  const calculateStepProgress = (index) => {
    return (index / pipeline.length) * 100;
  };

  return (
    <div className="flex-1 p-4 bg-white">
      <h2 className="text-2xl font-bold mb-4">Pipeline Visualization</h2>
      <div className="flex items-center justify-center pipeline-container">
        {pipeline.map((step, index) => (
          <div key={index} className="flex flex-col items-center pipeline-step">
            <div className="flex items-center">
              <div className={`relative flex items-center justify-center w-12 h-12 rounded-full border-2 ${progress >= calculateStepProgress(index) ? 'bg-green-500 border-green-500' : 'bg-blue-500 border-blue-500'}`}>
                {progress >= calculateStepProgress(index) ? (
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                ) : (
                  <span className="text-white">{index + 1}</span>
                )}
              </div>
              {index < pipeline.length - 1 && (
                <div className="flex-grow border-t-4 border-gray-400 mx-4"></div>
              )}
            </div>
            <div className="mt-2 text-center">{step}</div>
          </div>
        ))}
      </div>
      <div className="mt-4">
        <p className="text-xl font-bold">Progress: {progress}%</p>
        <div className="w-full bg-gray-200 rounded-full h-4">
          <div className="bg-blue-500 h-4 rounded-full" style={{ width: `${progress}%` }}></div>
        </div>
      </div>
    </div>
  );
};

export default PipelineVisualization;
