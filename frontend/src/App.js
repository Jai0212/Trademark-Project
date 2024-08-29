import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

function App() {
  const BACKEND_URL = 'http://127.0.0.1:5000';

  const [uploadedFile, setUploadedFile] = useState(null);
  const [similarityScore, setSimilarityScore] = useState(null);
  const [similarImage, setSimilarImage] = useState(null); // State to hold the similar image
  const [generatedLogo, setGeneratedLogo] = useState(null);
  const [textInput, setTextInput] = useState("");

  // For drag and drop file upload
  const { getRootProps, getInputProps } = useDropzone({
    onDrop: (acceptedFiles) => {
      setUploadedFile(acceptedFiles[0]);
    },
  });

  const handleFileUpload = async () => {
    const formData = new FormData();
    formData.append('file', uploadedFile);

    await axios.post(`${BACKEND_URL}/compare-logo`, formData)
      .then(response => {
        setSimilarityScore(response.data.similarity_score);
        setSimilarImage(`data:image/png;base64,${response.data.image_base64}`); // Set the similar image
      })
      .catch(error => {
        console.error('Error uploading file:', error);
      });
  };

  const handleGenerateLogo = async () => {
    if (!textInput) {
      alert('Please enter text for logo generation');
      return;
    }

    await axios.post(`${BACKEND_URL}/generate-logo`, { prompt: textInput })
      .then(response => {
        setGeneratedLogo(response.data.image_url);
      })
      .catch(error => {
        console.error('Error generating logo:', error);
      });
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Logo Similarity and Generation</h1>

        <div {...getRootProps({ className: 'dropzone' })}>
          <input {...getInputProps()} />
          <p>Drag 'n' drop a logo file here, or click to select a file</p>
        </div>
        {uploadedFile && (
          <div>
            <p>Selected file: {uploadedFile.name}</p>
            <button onClick={handleFileUpload}>Upload</button>
          </div>
        )}

        {similarityScore && (
          <div>
            <h3>Similarity Score: {similarityScore}</h3>
          </div>
        )}

        {similarImage && (
          <div>
            <img className="generated-image" src={similarImage} alt="Most Similar Logo" />
            <h3 className='most-similar-image'>Most Similar Image</h3>
          </div>
        )}

        <div className='generate-logo-container'>
          <h1>Generate a Logo from Text</h1>
          <div className='generate-logo'>
            <input className='text-input'
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Enter text for logo"
            />
            <button onClick={handleGenerateLogo}>Generate Logo</button>
          </div>
        </div>

        {generatedLogo && (
          <div>
            <img className="generated-image" src={generatedLogo} alt="Generated Logo" />
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
