import os
from os.path import join, split, exists
from shutil import copyfile
from glob import glob
from io import open

from flask import Flask
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Custom code to handle memory allocation problem
config = tf.ConfigProto()
# dynamically grow GPU memory
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from src.keras_utils import load_model
from gen_outputs import generate_outputs
from stdout_capture import Capturing
from license_plate_ocr import lp_ocr
from license_plate_detection import detect

# Necessary for some thread safety stuff, dont really looked into it
global graph
graph = tf.get_default_graph()
WPOD_NET = load_model("./data/lp-detector/wpod-net_update1.h5")

app = Flask(__name__)

# RUN_SCRIPT_FOLDER = "/alpr-unconstrained"
CASE_FOLDER_NAMES = ["0_case", "1_case", "2_case"]
MAIN_FOLDER_PATH = "/mnt"
# MAIN_FOLDER_PATH = "/home/atoth/temp/generali/prod_test/test"


@app.route('/lpr/<session_id>')
def run_lpr(session_id):
    # base_wd = os.getcwd()

    processed_images_folder = join(MAIN_FOLDER_PATH, session_id, "processed_images")

    # create bulk images folder
    bulk_images_folder = join(processed_images_folder, "bulk_images")
    os.mkdir(bulk_images_folder)

    result_file_name = join(bulk_images_folder, "result.csv")

    # copy images to bulk folder
    for case_folder_name in CASE_FOLDER_NAMES:
        if case_folder_name == "2_case":
            case_folder_path = join(processed_images_folder, case_folder_name)
            folders_from_images = os.listdir(case_folder_path)
            for folder_from_img in folders_from_images:
                actual_img_folder = join(case_folder_path, folder_from_img)
                img_names_in_act_folder = os.listdir(actual_img_folder)
                for img_name in img_names_in_act_folder:
                    src = join(actual_img_folder, img_name)
                    dst_img_name = "2case_" + folder_from_img + "_" + img_name
                    dst = join(bulk_images_folder, dst_img_name)
                    copyfile(src, dst)
        else:
            case_folder_path = join(processed_images_folder, case_folder_name)
            images_name = os.listdir(case_folder_path)
            for img_name in images_name:
                src = join(case_folder_path, img_name)
                dst = join(bulk_images_folder, img_name)
                copyfile(src, dst)
    with graph.as_default():
        detect(WPOD_NET, bulk_images_folder, bulk_images_folder)

    # change working directory
    # os.chdir(RUN_SCRIPT_FOLDER)

    lp_ocr(bulk_images_folder)

    with Capturing() as output:
        generate_outputs(bulk_images_folder, bulk_images_folder)

    # from io library
    with open(result_file_name, "w", encoding="utf-8") as f:
        f.writelines(output)

    # change working directory to base
    # os.chdir(base_wd)

    # lpr_txt_paths = glob('%s/*_lp_str.txt' % bulk_images_folder)
    imgs_path = glob('%s/*car.png' % bulk_images_folder)

    results = {}
    for img_p in imgs_path:
        path, file_name = split(img_p)

        current_key = None
        if file_name[0:6] == "2case_":
            img_name = file_name.split("_")[1]
            current_key = img_name

            if img_name not in results.keys():
                results[img_name] = []
        else:
            img_name = file_name[:-7]
            current_key = img_name

            if img_name not in results.keys():
                results[img_name] = []

        possible_txt = file_name[:-4] + "_lp_str.txt"
        possible_path = join(path, possible_txt)
        if exists(possible_path):
            with open(possible_path) as f:
                lp = f.readline()
                probs = f.readline()
            results[current_key].append({"plate": lp.rstrip('\n'), "probs": probs.rstrip('\n')})

    return results


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
