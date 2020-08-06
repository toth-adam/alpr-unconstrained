import os
import subprocess
from os.path import splitext, basename
from os.path import join, split, exists
from shutil import copyfile
from glob import glob

import cv2
from flask import Flask
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes

# Custom code to handle memory allocation problem
config = tf.ConfigProto()
# dynamically grow GPU memory
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

wpod_net = load_model("data/lp-detector/wpod-net_update1.h5")

app = Flask(__name__)

RUN_SCRIPT_FOLDER = "./alpr-unconstrained"
CASE_FOLDER_NAMES = ["0_case", "1_case", "2_case"]
MAIN_FOLDER_PATH = "/mnt"
# MAIN_FOLDER_PATH = "/home/istvanmo/Lexunit_related/generali_adatok/main_folder_mock"


@app.route('/lpr/<session_id>')
def run_lpr(session_id):
    base_wd = os.getcwd()

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

    detect(bulk_images_folder, bulk_images_folder)

    # change working directory
    os.chdir(RUN_SCRIPT_FOLDER)

    # run licence plate recognition
    subprocess.call(["./run_wo_car_detection.sh", "-i", bulk_images_folder, "-o", bulk_images_folder, "-c", result_file_name])

    # change working directory to base
    os.chdir(base_wd)

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


def detect(input_dir, output_dir):
    lp_threshold = .5

    imgs_paths = glob('%s/*car.png' % input_dir)

    print 'Searching for license plates using WPOD-NET'

    for i, img_path in enumerate(imgs_paths):

        print '\t Processing %s' % img_path

        bname = splitext(basename(img_path))[0]
        Ivehicle = cv2.imread(img_path)

        ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
        side = int(ratio * 288.)
        bound_dim = min(side + (side % (2 ** 4)), 608)
        print "\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio)

        Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2 ** 4, (240, 80), lp_threshold)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            s = Shape(Llp[0].pts)

            cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp * 255.)
            writeShapes('%s/%s_lp.txt' % (output_dir, bname), [s])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
