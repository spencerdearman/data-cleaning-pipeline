import React from 'react';
import { DndProvider, useDrag, useDrop } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import { cleaningOptionsMap } from './utils';

const ItemTypes = {
  TILE: 'tile',
};

const displayToOptionMap = Object.entries(cleaningOptionsMap).reduce((acc, [key, value]) => {
  acc[value] = key;
  return acc;
}, {});

const Tile = ({ displayName, option, index, moveTile, removeTile }) => {
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
      style={{ opacity: isDragging ? 0.5 : 1, color: 'black' }}
      onClick={() => removeTile(index)}
    >
      {displayName}
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
          displayName={displayToOptionMap[option]}
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

const Sidebar = ({ handleFileChange, handleSubmit, pipeline, setPipeline, message, cleanedFile, resetPipeline }) => {
  const availableOptions = Object.keys(cleaningOptionsMap).filter(option => !pipeline.includes(cleaningOptionsMap[option]));

  const handleAddAllOptions = () => {
    const allOptions = Object.values(cleaningOptionsMap);
    setPipeline(allOptions);
  };

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="bg-gray-800 text-white w-64 p-4 flex flex-col sidebar">
        <h2 className="text-xl font-bold mb-4">Data Cleaning</h2>
        <form onSubmit={handleSubmit} className="flex flex-col">
          <input
            type="file"
            onChange={handleFileChange}
            className="mb-4 p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="button"
            onClick={handleAddAllOptions}
            className="bg-purple-500 text-white py-2 px-4 rounded hover:bg-purple-700 w-full mb-4"
          >
            Automatic Clean
          </button>
          <Pipeline pipeline={pipeline} setPipeline={setPipeline} />
          <div className="mb-4">
            {availableOptions.map((option, index) => (
              <Tile key={option} index={index} option={cleaningOptionsMap[option]} displayName={option} moveTile={() => {}} removeTile={() => {}} />
            ))}
          </div>
          <button
            type="submit"
            className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-700 w-full"
          >
            Upload and Clean
          </button>
          <button
            type="button"
            onClick={resetPipeline}
            className="bg-red-500 text-white py-2 px-4 rounded hover:bg-red-700 w-full mt-2"
          >
            Reset
          </button>
        </form>
        {message && <p className="mt-4 text-xl text-center">{message}</p>}
        {cleanedFile && (
          <a href={`http://127.0.0.1:5000/${cleanedFile}`} className="bg-green-500 text-white py-2 px-4 rounded hover:bg-blue-700 text-center">
            Download Cleaned File
          </a>
        )}
      </div>
    </DndProvider>
  );
};

export default Sidebar;
