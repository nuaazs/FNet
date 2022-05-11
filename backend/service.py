from logging.handlers import RotatingFileHandler
import json
import time
import logging

from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import scipy as sp
from sqlalchemy import null
import flask
import torch
import numpy as np
import pickle
import os
# SpeechBrain
from speechbrain.pretrained import SpeakerRecognition

# utils
from utils.save import save_dcm_from_url,save_dcm_from_file
from utils.preprocess import preprocess_dcm
# from utils.scores import get_scores

# config file
import cfg

# cosine
similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

# embedding model
spkreg = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="/mnt/zhaosheng/brain/notebooks/pretrained_ecapa")

# log
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "[%(asctime)s]  %(levelname)s  [%(filename)s]  #%(lineno)d <%(process)d:%(thread)d>  %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
handler = RotatingFileHandler(
    "./vvs_server.log", maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
)
handler.setFormatter(formatter)
handler.namer = lambda x: "vvs_server." + x.split(".")[-1]
logger.addHandler(handler)

# app
app = Flask(__name__)
CORS(app, supports_credentials=True,
        origins="*", methods="*", allow_headers="*")

logger.info("\tLoad blacklist files ... ")
with open(cfg.BLACK_LIST, 'rb') as base:
    black_database = pickle.load(base)

@app.route("/", methods=["GET"])
def index():
    spks = list(black_database.keys())
    
    spks_num = len(spks)
    kwargs = {
        "spks_num": spks_num,
        "spks":spks
    }

    return render_template('index.html',**kwargs)

@app.route("/tt", methods=["GET"])
def test2(aa):
    print(aa)
    return index()

@app.route("/ttt", methods=["GET"])
def result():
    return test2("测试")

@app.route("/test/<test_type>", methods=["POST","GET"])
def test(test_type):
    if request.method == "GET":
        if test_type == "file":
            return render_template('score_from_file.html')
        elif test_type == "url":
            return render_template('score_from_url.html')
    if request.method == "POST":
        names_inbase = black_database.keys()
        logger.info("@ -> Test")
        logger.info(f"\tBlack spk number: {len(names_inbase)}")
        

        # get request.files
        new_spkid = flask.request.form.get("spkid")


        if test_type == "file":
            new_file = request.files["wav_file"]
            filepath,speech_number = save_wav_from_file(new_file,new_spkid,os.path.join(cfg.SAVE_PATH,"raw"))
        elif test_type == "url":
            new_url =request.form.get("wav_url")
            filepath,speech_number = save_wav_from_url(new_url,new_spkid,os.path.join(cfg.SAVE_PATH,"raw"))
        start_time = time.time()
        # Preprocess: vad + upsample to 16k + self test
        wav = vad_and_upsample(filepath,savepath=os.path.join(cfg.SAVE_PATH,"preprocessed"),spkid=new_spkid)
        pass_test, msg = self_test(wav, spkreg,similarity, sr=16000, split_num=cfg.TEST_SPLIT_NUM, min_length=cfg.MIN_LENGTH, similarity_limit=cfg.SELF_TEST_TH)
        if not pass_test:
            response = {
                "code": 2000,
                "status": "error",
                "err_msg": msg
            }
            end_time = time.time()
            time_used = end_time - start_time
            logger.info(f"\t# Time using: {np.round(time_used, 1)}s")
            logger.info(f"\t# Error: {msg}s")
            return json.dumps(response, ensure_ascii=False)

        embedding = spkreg.encode_batch(wav)[0][0]

        scores = get_scores(black_database,embedding,cfg.BLACK_TH,similarity,top_num=10)

        end_time = time.time()
        time_used = end_time - start_time
        logger.info(f"\t# Success: {msg}")
        logger.info(f"\t# Time using: {np.round(time_used, 1)}s")
        response = {
            "code": 2000,
            "status": "success",
            "scores": scores,
            "err_msg": "null"
        }
        print(response)
        return json.dumps(response, ensure_ascii=False)


@app.route("/namelist", methods=["GET"])
def namelist():
    if request.method == "GET":
        start_time = time.time()
        names_inbase = list(black_database.keys())
        logger.info("@ -> NameList")
        logger.info(f"\tBlack spk number: {len(names_inbase)}")
        end_time = time.time()
        time_used = end_time - start_time
        logger.info(f"\t# Time using: {np.round(time_used, 1)}s")
        response = {
            "code": 2000,
            "status": "success",
            "names": names_inbase,
            "err_msg": "null"
        }
        print(response)
        return json.dumps(response, ensure_ascii=False)

@app.route("/register/<register_type>", methods=["POST","GET"])
def register(register_type):
    if request.method == "GET":
        if register_type == "file":
            return render_template('register_from_file.html')
        elif register_type == "url":
            return render_template('register_from_url.html')

    
    if request.method == "POST":  
        names_inbase = black_database.keys()
        logger.info("# => Register")
        logger.info(f"\tBlack spk number: {len(names_inbase)}")
        

        # get request.files
        new_spkid = request.form.get("spkid")
        
        
        if register_type == "file":
            new_file = request.files["wav_file"]
            filepath,speech_number = save_wav_from_file(new_file,new_spkid,os.path.join(cfg.BASE_WAV_PATH,"raw"))
        elif register_type == "url":
            new_url =request.form.get("wav_url")
            filepath,speech_number = save_wav_from_url(new_url,new_spkid,os.path.join(cfg.BASE_WAV_PATH,"raw"))
        start_time = time.time()
        # Preprocess: vad + upsample to 16k + self test
        wav = vad_and_upsample(filepath,savepath=os.path.join(cfg.BASE_WAV_PATH,"preprocessed"),spkid=new_spkid)
        pass_test, msg = self_test(wav, spkreg,similarity, sr=16000, split_num=cfg.TEST_SPLIT_NUM, min_length=cfg.MIN_LENGTH, similarity_limit=cfg.SELF_TEST_TH)
        if not pass_test:
            response = {
                "code": 2000,
                "status": "error",
                "err_msg": msg
            }
            end_time = time.time()
            time_used = end_time - start_time
            logger.info(f"\t# Time using: {np.round(time_used, 1)}s")
            logger.info(f"\t# Error: {msg}")
            return json.dumps(response, ensure_ascii=False)

        embedding = spkreg.encode_batch(wav)[0][0]
        add_success = add_to_database(database=black_database,embedding=embedding,spkid=new_spkid,wav_file_path=filepath,database_filepath=cfg.BLACK_LIST)
        if not add_success:
            response = {
                "code": 2000,
                "status": "error",
                "err_msg": "Already in database!"
            }
            end_time = time.time()
            time_used = end_time - start_time
            logger.info(f"\t# Time using: {np.round(time_used, 1)}s")
            logger.info(f"\t# Error: Already in database!")
            return json.dumps(response, ensure_ascii=False)
        else:
            logger.info(f"\t# Msg: Save to dabase success.")
        end_time = time.time()
        time_used = end_time - start_time
        logger.info(f"\t# Success: {msg}")
        logger.info(f"\t# Time using: {np.round(time_used, 1)}s")
        response = {
            "code": 2000,
            "status": "success",
            "err_msg": "null"
        }
        #return redirect(url_for('login'))

        return json.dumps(response, ensure_ascii=False)

if __name__ == "__main__":
    app.run(host='127.0.0.1', threaded=True, port=8170, debug=True,)
    # host="0.0.0.0"
