import logging
 
# 配置日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if not logger.handlers:
    fhandler = logging.FileHandler('livetalking.log')
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.INFO)
    logger.addHandler(fhandler)

    shandler = logging.StreamHandler()
    shandler.setFormatter(formatter)
    shandler.setLevel(logging.INFO)
    logger.addHandler(shandler)

logger.propagate = False
