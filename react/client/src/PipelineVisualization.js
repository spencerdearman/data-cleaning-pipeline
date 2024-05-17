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

  return (
    <div className="flex-1 p-4 bg-white">
      <h2 className="text-2xl font-bold mb-4">Pipeline Visualization</h2>
      <div className="flex items-center">
        {pipeline.map((step, index) => (
          <div key={index} className="flex items-center">
            <div className={`bg-blue-500 text-white p-4 rounded ${progress > (index / pipeline.length) * 100 ? 'bg-green-500' : ''}`}>
              {step}
            </div>
            {index < pipeline.length - 1 && (
              <div className="flex-grow border-t-4 border-gray-400 mx-4"></div>
            )}
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
