import React, { useState } from 'react';
import axios from 'axios';

const cleaningOptions = [
  'lower_case_columns',
  'remove_duplicates',
  'encode_categorical_columns',
  'fix_missing_values',
  'clean_uniform_prefixes',
  'clean_uniform_postfixes',
  'clean_uniform_substrings'
];

const Upload = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [selectedOptions, setSelectedOptions] = useState([]);
  const [cleanedFile, setCleanedFile] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    console.log("File selected:", e.target.files[0]);
  };

  const handleOptionChange = (option) => {
    setSelectedOptions(prev =>
      prev.includes(option) ? prev.filter(o => o !== option) : [...prev, option]
    );
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
    formData.append('options', JSON.stringify(selectedOptions));
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
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-4xl font-bold mb-4 text-center">Upload and Clean Data</h1>
      <form onSubmit={handleSubmit} className="bg-white p-6 rounded shadow-md w-full max-w-sm">
        <input
          type="file"
          onChange={handleFileChange}
          className="mb-4 w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <div className="mb-4">
          {cleaningOptions.map(option => (
            <label key={option} className="block">
              <input
                type="checkbox"
                checked={selectedOptions.includes(option)}
                onChange={() => handleOptionChange(option)}
                className="mr-2"
              />
              {option}
            </label>
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
  );
};

export default Upload;
