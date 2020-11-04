from flask import Flask, render_template, Response
from flask_cors import CORS

import config
import cv2

app = Flask(__name__)
CORS(app)

@app.route('/crowd_count_video')
def live_output_video_feed():
    return Response(crowd_count_video_gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def crowd_count_video_gen():

    ret, crowd_count_img = config.crowd_count_pipe.pull()
    if not ret:
        print("crowd_count image", crowd_count_img)
        yield (b'--frame\r\n'
             b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', crowd_count_img)[1].tostring() + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)

