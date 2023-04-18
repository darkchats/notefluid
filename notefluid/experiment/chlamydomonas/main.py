import json
import logging
import os

from notefluid.experiment.chlamydomonas.progress.analyse import TrackAnalyse, MSDCalculate
from notefluid.experiment.chlamydomonas.progress.background import BackGroundDetect
from notefluid.experiment.chlamydomonas.progress.base import VideoBase
from notefluid.experiment.chlamydomonas.progress.config import Config
from notefluid.experiment.chlamydomonas.progress.contain import ContainDetect
from notefluid.experiment.chlamydomonas.progress.particle import ParticleWithoutBackgroundList
from notefluid.utils.log import logger

logger.setLevel(logging.INFO)


class VideoProgress:
    def __init__(self, video_path, cache_dir):
        self.base_video = VideoBase(video_path, cache_dir=cache_dir)
        self.detect_background = BackGroundDetect(config=self.base_video)
        self.detect_particle = ParticleWithoutBackgroundList(config=self.base_video)
        self.detect_contain = ContainDetect(config=self.base_video)

        self.analyse_track = TrackAnalyse(config=self.base_video)
        self.analyse_msd = MSDCalculate(config=self.base_video)

    def execute(self, ext_json=None):
        ext_json = ext_json or {}
        self.base_video.read()
        self.detect_background.read()
        self.detect_background.read()
        self.detect_particle.read()

        self.analyse_track.read(contain=self.detect_contain, particle=self.detect_particle)
        self.analyse_msd.read()
        try:
            ext_json.update({
                "video_path": self.base_video.video_path,
                "cache_dir": self.base_video.cache_dir,
                "video_height": self.base_video.video_height,
                "video_width": self.base_video.video_width,
                "frame_count": self.base_video.frame_count,
                "particles": len(self.detect_particle.particle_list),
                "tracks": len(self.analyse_track.df),
                "background_size": len(self.detect_background.background_list),
                "backcontain_size": len(self.detect_contain.contain_list),
                "track_path": self.analyse_track.filepath,
                "msd_path": self.analyse_msd.filepath
            })
        except Exception as e:
            print(f"main_error {self.base_video.video_path}")
        return ext_json


class Main:
    def __init__(self, config):
        self.config = config

    def run(self):
        result = []
        for root, directories, files in os.walk(self.config.videos_dir):
            if root.endswith('useless'):
                continue

            for file in files:
                if not file.endswith('.avi'):
                    continue
                ext_json = {
                    "file": file,
                }
                if file not in ('11.23005.avi', '11.23006.avi', '150_11-3-01015', '150_11-3-01003'):
                    # continue
                    pass

                video_path = os.path.join(root, file)
                if not video_path.endswith('.avi'):
                    continue

                cache_dir = os.path.dirname(video_path.replace(self.config.videos_dir, self.config.results_dir))
                video_progress = VideoProgress(video_path=video_path, cache_dir=cache_dir)
                result.append(video_progress.execute(ext_json=ext_json))

        with open(self.config.results_json, 'w') as fr:
            fr.write(json.dumps(result))
        logger.info("all is done")


config = Config(path_root='/Volumes/ChenDisk/experiment')
Main(config).run()
# Main(path_root='/Users/chen/data/experiment').run()
