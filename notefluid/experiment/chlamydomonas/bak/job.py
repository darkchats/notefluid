import logging

import cv2

from notefluid.experiment.chlamydomonas.bak.config import DetectConfig
from notefluid.experiment.chlamydomonas.bak.detect_contain import FindContain
from notefluid.utils.log import logger

logger.setLevel(logging.INFO)
path_root = '/Users/chen/data/experiment'
path = f"{path_root}/11-5-01013.avi"
# path = f"{path_root}/150_11-3-01007.avi"
dir_name = f'{path_root}/output'

conf = DetectConfig(path, dir_name)
video = FindContain(conf=conf)
video.detect_contain_video()
video.detect_particle_video(debug=True)
cv2.waitKey()
cv2.destroyAllWindows()
