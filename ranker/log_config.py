import logging

# create logger
log = logging.getLogger("ranker_fine_tuner")
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# create file handler which logs even debug messages
error_handler = logging.FileHandler("errors.log")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)
log.addHandler(error_handler)

# create console handler with a higher log level
info_handler = logging.StreamHandler()
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)
log.addHandler(info_handler)
