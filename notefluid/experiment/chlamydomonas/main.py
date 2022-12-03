import logging
import os

import cv2

from notefluid.experiment.chlamydomonas.progress.particle import ParticleWithoutBackgroundList
from notefluid.utils.log import logger

logger.setLevel(logging.INFO)


class Main:
    def __init__(self, path_root):
        self.path_root = path_root
        self.videos_dir = f'{self.path_root}/videos'
        self.results_dir = f'{self.path_root}/results'

    def run_video(self, video_path):
        if not video_path.endswith('.avi'):
            return
        cache_dir = os.path.dirname(video_path.replace(self.videos_dir, self.results_dir))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # back_ground = BackGroundList(path)
        # back_ground.process()
        # back_ground.save()

        # back_contain = BackContainList(path)
        # back_contain.process()
        # back_contain.save()

        # particle_detect = ParticleList(path)
        # particle_detect.process(overwrite=True, debug=True)
        # particle_detect.save(overwrite=True)

        particle_detect = ParticleWithoutBackgroundList(video_path, cache_dir=cache_dir)

        particle_detect.process_background_video()

        particle_detect.process_particle_video(debug=False)

    def run(self):
        for root, directories, files in os.walk(self.videos_dir):
            if root.endswith('useless'):
                continue
            for file in files:
                if file in ('11.23005.avi', '11.23006.avi'):
                    # continue
                    pass
                self.run_video(os.path.join(root, file))
        logger.info("all is done")
        cv2.waitKey()
        cv2.destroyAllWindows()


Main(path_root='/Volumes/ChenDisk/experiment').run()
# Main(path_root='/Users/chen/data/experiment').run()
