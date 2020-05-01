import React, { useRef, useState } from "react";
import './App.css';
import CanvasDraw from "react-canvas-draw";

export default function App() {
  const cors = require('cors')
  
  const axios = require('axios').default;
  const [color, setColor] = useState('#000');
  const [width, setWidth] = useState(480);
  const [height, setHeight] = useState(480);
  const [brushRadius, setBrushRadius] = useState(6);
  const [lazyRadius, setLazyRadius] = useState(6);

  const canvasRef = useRef(null);

  const updateColor = (colorStr) => {
    setColor(colorStr);
  }

  const saveFile = () => {
    if (canvasRef.current) {
      let asdf = canvasRef.current.canvasContainer.children[1].toDataURL();
      var params = {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json;charset=UTF-8'
        },
        data: asdf
      }
      if (asdf !== "") {
        axios.post("/query", params)
        .then(response => {
          alert(JSON.stringify(response));
        })
        .catch(error => {
          alert("some error");
        });
      } else {
        alert("Image data is empty for some reason");
      }
      console.log(asdf);
    }
  }
  return (
    <div className="bigdaddy">
      <h1>CIFAR-10 Sketch Completion Demo</h1>
      <div className="drawboard">
        <CanvasDraw ref={canvasRef} hideGrid={true} brushRadius={brushRadius} brushColor={color} canvasWidth={width} canvasHeight={height} />
      </div>
      <div className="buttons">
        <button onClick={() => updateColor('#ff0000')}>Red</button>
        <button onClick={() => updateColor('#ffff00')}>Yellow</button>
        <button onClick={() => updateColor('#00ff00')}>Green</button>
        <button onClick={() => updateColor('#0000ff')}>Blue</button>
        <button onClick={() => updateColor('#00ffff')}>Teal</button>
        <button onClick={() => updateColor('#000000')}>Black</button>
      </div>
      <div className="submit">
        <button onClick={() => saveFile()}>Complete my image!</button>
      </div>
      <p>Note: Your sketch will be downsampled to 32x32, and an upsampled version will be displayed.</p>
    </div>
  );
}