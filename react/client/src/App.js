import React, { useState } from 'react';
import Sidebar from './Sidebar';
import PipelineVisualization from './PipelineVisualization';
import axios from 'axios';

const App = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [pipeline, setPipeline] = useState([]);
  const [progress, setProgress] = useState(0);
  const [cleanedFile, setCleanedFile] = useState('');

  // Handling file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
  };

  // Handling file uploading
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setMessage('Please select a file to upload');
      console.error("No file selected");
      return;
    }

    // Preparing the file data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('options', JSON.stringify(pipeline));
    console.log("Form data prepared:", formData);

    // Uploading the file
    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('Response:', response.data);
      setMessage(response.data.message);
      setCleanedFile(response.data.cleaned_file);
      // Updating the progress bar to 100% if the cleaning is complete
      setProgress(100);
    } catch (error) {
      console.error('Error uploading file:', error);
      setMessage(error.response?.data?.message || 'Error uploading file');
      // Resetting the progress bar
      setProgress(0);
    }
  };

  // Function that will reset the pipeline
  const resetPipeline = () => {
    setPipeline([]);
    setProgress(0);
    setMessage('');
    setCleanedFile('');
  };

  return (
    <div className="flex h-screen">
      <Sidebar
        handleFileChange={handleFileChange}
        handleSubmit={handleSubmit}
        pipeline={pipeline}
        setPipeline={setPipeline}
        message={message}
        cleanedFile={cleanedFile}
        resetPipeline={resetPipeline}
      />
      <PipelineVisualization pipeline={pipeline} progress={progress} />
    </div>
  );
};

export default App;
