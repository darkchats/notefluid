import json
import logging
import os

from notefluid.experiment.chlamydomonas.progress.particle import ParticleWithoutBackgroundList
from notefluid.utils.log import logger

logger.setLevel(logging.INFO)


class Main:
    def __init__(self, path_root):
        self.path_root = path_root
        self.videos_dir = f'{self.path_root}/videos'
        self.results_dir = f'{self.path_root}/results'

        self.results_json = f'{self.path_root}/result.json'

    def run_video(self, video_path, ext_json={}) -> dict:
        if not video_path.endswith('.avi'):
            return {}
        cache_dir = os.path.dirname(video_path.replace(self.videos_dir, self.results_dir))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        particle_detect = ParticleWithoutBackgroundList(video_path, cache_dir=cache_dir)

        particle_detect.process_background_video()
        particle_detect.process_contain_video(overwrite=True)
        particle_detect.process_particle_video(debug=True)
        ext_json.update({
            "particles": len(particle_detect.particle_list),
            "background_size": len(particle_detect.background_list),
            "backcontain_size": len(particle_detect.backcontain_list),
        })
        return ext_json

    def run(self):
        result = []
        for root, directories, files in os.walk(self.videos_dir):
            if root.endswith('useless'):
                continue

            for file in files:
                ext = {
                    "file": file,
                }
                result.append(self.run_video(os.path.join(root, file), ext_json=ext))
        with open(self.results_json, 'w') as fr:
            fr.write(json.dumps(result))
        logger.info("all is done")


Main(path_root='/Volumes/ChenDisk/experiment').run()
# Main(path_root='/Users/chen/data/experiment').run()
