import React, { useState } from 'react';
import axios from 'axios';
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

const cleaningOptions = [
  'Lower Case Columns',
  'Remove Duplicates',
  'Encode Categorical Columns',
  'Fix Missing Values',
  'Clean Uniform Prefixes',
  'Clean Uniform Postfixes',
  'Clean Uniform Substrings',
  'Detect And Remove Outliers',
  'Standardize Dates',
  'Remove Highly Missing Columns',
  'Anonymize Columns'
];

const ItemTypes = {
  TILE: 'tile',
};

const Tile = ({ option, index, moveTile, removeTile }) => {
  const [{ isDragging }, drag] = useDrag({
    type: ItemTypes.TILE,
    item: { option, index },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  return (
    <div
      ref={drag}
      className={`p-2 mb-2 rounded cursor-pointer shadow-md ${isDragging ? 'bg-gray-300' : 'bg-blue-200'}`}
      style={{ opacity: isDragging ? 0.5 : 1 }}
      onClick={() => removeTile(index)}
    >
      {option}
    </div>
  );
};

const Pipeline = ({ pipeline, setPipeline }) => {
  const [{ isOver }, drop] = useDrop({
    accept: ItemTypes.TILE,
    drop: (item) => {
      if (!pipeline.includes(item.option)) {
        setPipeline([...pipeline, item.option]);
      }
    },
    collect: (monitor) => ({
      isOver: monitor.isOver(),
    }),
  });

  return (
    <div
      ref={drop}
      className={`p-4 mb-4 rounded ${isOver ? 'bg-green-200' : 'bg-gray-200'}`}
      style={{ minHeight: '100px' }}
    >
      {pipeline.length === 0 && <p className="text-center text-gray-500">Drop cleaning options here</p>}
      {pipeline.map((option, index) => (
        <Tile
          key={index}
          index={index}
          option={option}
          moveTile={(dragIndex, hoverIndex) => {
            const newPipeline = Array.from(pipeline);
            const [movedTile] = newPipeline.splice(dragIndex, 1);
            newPipeline.splice(hoverIndex, 0, movedTile);
            setPipeline(newPipeline);
          }}
          removeTile={(index) => setPipeline(pipeline.filter((_, i) => i !== index))}
        />
      ))}
    </div>
  );
};

const Upload = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [pipeline, setPipeline] = useState([]);
  const [cleanedFile, setCleanedFile] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    console.log("File selected:", e.target.files[0]);
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
    formData.append('pipeline', JSON.stringify(pipeline));
    console.log("Form data prepared:", formData);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      console.log('Response:', response.data);
      setMessage(response.data.message);
      setCleanedFile(response.data.cleaned_file);
    } catch (error) {
      console.error('Error uploading file:', error);
      setMessage(error.response?.data?.message || 'Error uploading file');
    }
  };

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
        <h1 className="text-4xl font-bold mb-4 text-center">Upload and Clean Data</h1>
        <form onSubmit={handleSubmit} className="bg-white p-6 rounded shadow-md w-full max-w-sm">
          <input
            type="file"
            onChange={handleFileChange}
            className="mb-4 w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <Pipeline pipeline={pipeline} setPipeline={setPipeline} />
          <div className="mb-4">
            {cleaningOptions.filter(option => !pipeline.includes(option)).map((option, index) => (
              <Tile key={option} index={index} option={option} moveTile={() => {}} removeTile={() => {}} />
            ))}
          </div>
          <button
            type="submit"
            className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-700 w-full"
          >
            Upload and Clean
          </button>
        </form>
        {message && <p className="mt-4 text-xl text-center">{message}</p>}
        {cleanedFile && (
          <a href={`http://127.0.0.1:5000/${cleanedFile}`} className="mt-4 text-xl text-blue-500">
            Download Cleaned File
          </a>
        )}
      </div>
    </DndProvider>
  );
};

export default Upload;
