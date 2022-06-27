import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(process)d-%(processName)s - %(filename)s-%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s')

logger = logging.getLogger("notefluid")
logger.setLevel(logging.DEBUG)
