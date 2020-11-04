from json import dumps
from threading import Thread

import tensorflow as tf
import numpy as np
import cv2
from influxdb_client import Point, InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
import config
import logging
from flask import Flask, render_template, Response
from flask_cors import CORS

import config
import cv2
import time

app = Flask(__name__)
CORS(app)

log = logging.getLogger("crowd_count")
counter = 1

@app.route('/crowd_count_video')
def live_output_video_feed():
    return Response(crowd_count(config.crowd_count_video), mimetype='multipart/x-mixed-replace; boundary=frame')

def crowd_count(video_src):
    font = cv2.FONT_HERSHEY_SIMPLEX
    with tf.Session() as sess:
        feed_vid = cv2.VideoCapture(video_src)
        success = True

        new_saver = tf.train.import_meta_graph("data/model/model.ckpt.meta")
        new_saver.restore(sess, tf.train.latest_checkpoint("data/model/"))
        graph = tf.get_default_graph()
        op_to_restore = graph.get_tensor_by_name("add_12:0")
        x = graph.get_tensor_by_name('Placeholder:0')
        fps = feed_vid.get(cv2.CAP_PROP_FPS)
        fps = np.int32(fps)
        print("Frames Per Second:", fps, "\n")

        while True:
            success, im = feed_vid.read()
            if not success:
                break
            img = np.copy(im)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.array(img)
            img = (img - 127.5) / 128
            x_in = np.reshape(img, (1, img.shape[0], img.shape[1], 1))
            x_in = np.float32(x_in)
            y_pred = sess.run(op_to_restore, feed_dict={x: x_in})
            sum = np.absolute(np.int32(np.sum(y_pred)))
            print("current crowd count",sum)
            write_to_influxdb(video_src,sum)
            global counter
            counter+=1

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', im)[1].tostring() + b'\r\n')

        feed_vid.release()

def write_to_influxdb(video_src,sum):

    print(counter)
    log.info("writing to db for video src..", video_src)
    client = InfluxDBClient(config.influxdb_url, config.influxdb_token)
    point =  Point("crowd_count_camera")\
             .field("current_crowd_count", int(sum))\
             .field("video_src", video_src)

    # point = Point.field("location", "bangalore").field("cameraId", "cam1").measurement("crowd-count").field("value",5)

    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(config.infuxdb_bucket, config.influxdb_org, point)

    log.info("crowd_count", sum, "written to db..")

if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True, use_reloader=False)

    influxdb_token = "UzWGrd7DzTahaedkk1PwblSOZlz28wlT-HIoLUnlzEOI0AAu1qtTT9EVnVgLH_Ti6ToYERDCojKLvE3z5p608w=="
    influxdb_org = "infosys"
    infuxdb_bucket = "surveillance"
    influxdb_url = "http://localhost:8086"

    client = InfluxDBClient(config.influxdb_url, config.influxdb_token)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    for i in range(10):
        point1 = Point("crowd-count").field("count", i).tag("location","bangalore").tag("camid", "cam1")
        point2 = Point("crowd-count").field("count", 10).tag("location", "chennai").tag("camid", "cam2")
        time.sleep(0.01)
        write_api.write(config.infuxdb_bucket, config.influxdb_org, point1)
        write_api.write(config.infuxdb_bucket, config.influxdb_org, point2)
        # print("hello")


    # query = f'from(bucket: "surveillance") |> range(start: 2020-10-30T11:59:46+05:30, stop: 2020-10-30T12:07:48+05:30)  |> filter(fn: (r) => r["location"] == "bangalore") \
    # |> filter(fn: (r) => r["camid"] == "cam1") \
    # |> filter(fn: (r) => r["_measurement"] == "crowd-count") \
    # |> filter(fn: (r) => r["_field"] == "count") \
    # |> sort() \
    # |> yield (name: "sort")'
    # tables = client.query_api().query(query, org=config.influxdb_org)
    # print(tables.pop())
