import React, { useState } from 'react';
import Sidebar from './Sidebar';
import PipelineVisualization from './PipelineVisualization';
import axios from 'axios';

const App = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [pipeline, setPipeline] = useState([]);
  const [cleanedFile, setCleanedFile] = useState('');
  const [progress, setProgress] = useState(0); // New state for progress

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setMessage('Please select a file to upload');
      console.error("No file selected");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('options', JSON.stringify(pipeline));
    console.log("Form data prepared:", formData);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted); // Update progress
        }
      });
      console.log('Response:', response.data);
      setMessage(response.data.message);
      setCleanedFile(response.data.cleaned_file);
      setProgress(100); // Set progress to 100 when done
    } catch (error) {
      console.error('Error uploading file:', error);
      setMessage(error.response?.data?.message || 'Error uploading file');
      setProgress(0); // Reset progress on error
    }
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
      />
      <PipelineVisualization pipeline={pipeline} progress={progress} /> {/* Pass progress */}
    </div>
  );
};

export default App;
