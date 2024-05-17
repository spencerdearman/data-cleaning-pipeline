import React from 'react';

const PipelineVisualization = ({ pipeline }) => {
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
      <div className="flex flex-col items-center">
        {pipeline.map((step, index) => (
          <div key={index} className="flex items-center mb-4">
            <div className="bg-blue-500 text-white p-4 rounded">
              {step}
            </div>
            {index < pipeline.length - 1 && (
              <div className="flex-grow border-t-2 border-gray-400 mx-4"></div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default PipelineVisualization;
