import React from 'react';
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';

const cleaningOptions = [
  'lower_case_columns',
  'remove_duplicates',
  'encode_categorical_columns',
  'fix_missing_values',
  'clean_uniform_prefixes',
  'clean_uniform_postfixes',
  'clean_uniform_substrings',
  'remove_outliers',
  'normalize_data',
  'standardize_data',
  'create_features',
  'process_text',
  'balance_data',
  'reduce_dimensions'
];

const ItemTypes = {
  TILE: 'tile',
};

const Tile = ({ option, index, moveTile, removeTile, numbered }) => {
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
      style={{ opacity: isDragging ? 0.5 : 1, color: 'black' }}  // Set text color to black
      onClick={() => removeTile(index)}
    >
      {numbered ? `${index + 1}. ${option}` : option}
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
          numbered
        />
      ))}
    </div>
  );
};

const Sidebar = ({ handleFileChange, handleSubmit, pipeline, setPipeline, message, cleanedFile }) => {
  return (
    <DndProvider backend={HTML5Backend}>
      <div className="bg-gray-800 text-white p-4 flex flex-col sidebar">
        <h2 className="text-xl font-bold mb-4">Data Cleaning</h2>
        <form onSubmit={handleSubmit} className="flex flex-col">
          <input
            type="file"
            onChange={handleFileChange}
            className="mb-4 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <Pipeline pipeline={pipeline} setPipeline={setPipeline} />
          <div className="mb-4 overflow-y-auto">
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

export default Sidebar;
