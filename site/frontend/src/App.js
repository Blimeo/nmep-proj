import React, { useRef, useState } from "react";
import './App.css';
import CanvasDraw from 'react-canvas-draw';


export default function App() {
  const axios = require('axios').default;
  const [color, setColor] = useState('#000');
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
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
            let img = new Image();
            img.src = response.data;
            var x = document.getElementsByClassName('asdf')[0];
            if (x.hasChildNodes()) {
              x.removeChild(x.childNodes[0]);
            }
            x.appendChild(img);
          })
          .catch(error => {
            alert(error);
          });
      } else {
        alert("Image data is empty for some reason");
      }
    }
  }

  const resetCanvas = () => {
    if (canvasRef.current) {
      var banvas = canvasRef.current.canvasContainer.children[1];
      var context = banvas.getContext('2d');
      context.clearRect(0, 0, banvas.width, banvas.height);
    }
  }
  return (
    <body>

      <h1>Sketch Completion Demo</h1>
      <p>Note: Your sketch will be downsampled for performance reasons.</p>
      <div className="left">
        <div className="drawboard">
          <CanvasDraw ref={canvasRef} hideGrid={true} brushRadius={brushRadius} brushColor={color} canvasWidth={width} canvasHeight={height} />
        </div>
        <div className="buttons">
          <button onClick={() => updateColor('#ff0000')}>Red</button>
          <button onClick={() => updateColor('#228B22')}>Green</button>
          <button onClick={() => updateColor('#0000ff')}>Blue</button>
          <button onClick={() => updateColor('#ffff00')}>Yellow</button>
          <button onClick={() => updateColor('#00ffff')}>Teal</button>
          <button onClick={() => updateColor('#964b00')}>Brown</button>
          <button onClick={() => updateColor('#000000')}>Black</button>
        </div>
        <div className="submit">
        <button onClick={() => resetCanvas()}>Reset canvas</button>
          <button onClick={() => saveFile()}>Complete my image!</button>
        </div>
        
      </div>
      <div className="right">
        <div className="asdf"></div>
      </div>
      
    </body>
  );
}