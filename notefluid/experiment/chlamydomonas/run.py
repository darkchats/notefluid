import json
import logging
import os

from notefluid.experiment.chlamydomonas.base.globalconfig import GlobalConfig
from notefluid.experiment.chlamydomonas.progress.video_progress import VideoProgress
from notefluid.utils.log import logger

logger.setLevel(logging.INFO)


class MainProgress:
    def __init__(self, config: GlobalConfig):
        self.config = config

    def run(self, debug=False):
        result = []
        for root, directories, files in os.walk(self.config.videos_dir):
            if root.endswith('useless'):
                continue

            for file in files:
                if not file.endswith('.avi'):
                    continue
                ext_json = {"file": file}
                if file not in ('11.23005.avi', '11.23006.avi', '150_11-3-01015', '150_11-3-01003'):
                    # continue
                    pass

                video_split = global_config.get_result_path(video_path=os.path.join(root, file))
                if video_split is None:
                    continue
                if not video_split.video_name in ('150_11-3-01001', ''):
                    continue
                video_progress = VideoProgress(video_split=video_split)
                result.append(video_progress.execute(ext_json=ext_json, debug=debug))

        with open(self.config.results_json, 'w') as fr:
            fr.write(json.dumps(result))
        logger.info("all is done")


global_config = GlobalConfig(path_root='/Volumes/ChenDisk/experiment')
MainProgress(global_config).run(debug=False)
# Main(path_root='/Users/chen/data/experiment').run()
