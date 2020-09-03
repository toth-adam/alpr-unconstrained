import sys
import traceback

import darknet.python.darknet as dn

from os.path import splitext, basename
from glob import glob
from darknet.python.darknet import detect
from src.label import dknet_label_conversion
from src.utils import nms


def lp_ocr(input_dir):

    output_dir = input_dir

    try:
        ocr_threshold = .4

        ocr_weights = 'data/ocr/ocr-net.weights'
        ocr_netcfg = 'data/ocr/ocr-net.cfg'
        ocr_dataset = 'data/ocr/ocr-net.data'

        ocr_net = dn.load_net(ocr_netcfg, ocr_weights, 0)
        ocr_meta = dn.load_meta(ocr_dataset)

        imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

        print 'Performing OCR...'

        for i, img_path in enumerate(imgs_paths):

            print '\tScanning %s' % img_path

            bname = basename(splitext(img_path)[0])

            R, (width, height) = detect(ocr_net, ocr_meta, img_path, thresh=ocr_threshold, nms=None)

            if len(R):

                L = dknet_label_conversion(R, width, height)
                L = nms(L, .45)

                L.sort(key=lambda x: x.tl()[0])
                lp_str = ''.join([chr(l.cl()) for l in L])

                # print "AFTER NMS AND SORT"
                # for label in L:
                #     print label.letter(), label.prob(), label
                # print "=========================================================================="

                prob_str = ''.join([str(l.prob()) + ',' for l in L])

                with open('%s/%s_str.txt' % (output_dir, bname), 'w') as f:
                    f.write(lp_str + '\n')
                    f.write(prob_str + '\n')

                print '\t\tLP: %s' % lp_str

            else:

                print 'No characters found'

    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    input_dir = sys.argv[1]
    lp_ocr(input_dir)