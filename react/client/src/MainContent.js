import React from 'react';

const MainContent = ({ data, headers }) => {
  if (!data || data.length === 0) {
    return (
      <div className="flex-1 p-4 bg-white">
        <h2 className="text-2xl font-bold mb-4">Data Preview</h2>
        <p>No data available</p>
      </div>
    );
  }

  return (
    <div className="flex-1 p-4 bg-white overflow-auto">
      <h2 className="text-2xl font-bold mb-4">Data Preview</h2>
      <table className="min-w-full border-collapse border border-gray-200">
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header} className="border border-gray-200 px-4 py-2">{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {headers.map((header, cellIndex) => (
                <td key={cellIndex} className="border border-gray-200 px-4 py-2">{row[header]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default MainContent;
