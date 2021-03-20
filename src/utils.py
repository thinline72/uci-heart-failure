from loguru import logger


def log_multiple_string_obj(obj: object) -> None:
    for row in str(obj).split("\n"):
        logger.info(row)
