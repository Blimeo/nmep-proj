# Sketch Completion Application

Contains a Flask application that will process a base64-encoded PNG file, used in conjunction with [this frontend.](https://github.com/Blimeo/sketch-completion-web/tree/master) To host this backend locally, run the following with sudo privileges (so that the flask binary is in your PATH):
```
sudo pip install -r requirements.txt
flask run
```

Make a post request to localhost:5000/query with base64-encoded PNG data. Make sure the header {`data:image/png;base64,`} is included. The 32x32 output image will be in `converted.png`.
Todo: make CLI tool to output completed sketch with just a filename.
