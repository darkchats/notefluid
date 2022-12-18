import json
import logging
import os

from notefluid.experiment.chlamydomonas.progress.analyse import AnalyseParticle
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
            return ext_json
        cache_dir = os.path.dirname(video_path.replace(self.videos_dir, self.results_dir))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        particle_detect = AnalyseParticle(video_path, cache_dir=cache_dir)

        particle_detect.process()
        particle_detect.process_background_video()
        particle_detect.process_contain_video()
        particle_detect.process_particle_video()
        particle_detect.analyse_track(overwrite=True)
        ext_json.update({
            "video_height": particle_detect.video_height,
            "video_width": particle_detect.video_width,
            "frame_count": particle_detect.frame_count,
            "particles": len(particle_detect.particle_list),
            "tracks": len(particle_detect.particle_track),
            "background_size": len(particle_detect.background_list),
            "backcontain_size": len(particle_detect.backcontain_list),
            "track_path": particle_detect.particle_track_path
        })
        return ext_json

    def run(self):
        result = []
        for root, directories, files in os.walk(self.videos_dir):
            if root.endswith('useless'):
                continue

            for file in files:
                if not file.endswith('.avi'):
                    continue
                ext = {
                    "file": file,
                }
                if file not in ('11.23005.avi', '11.23006.avi'):
                    # continue
                    pass
                result.append(self.run_video(os.path.join(root, file), ext_json=ext))
        with open(self.results_json, 'w') as fr:
            fr.write(json.dumps(result))
        logger.info("all is done")


Main(path_root='/Volumes/ChenDisk/experiment').run()
# Main(path_root='/Users/chen/data/experiment').run()
