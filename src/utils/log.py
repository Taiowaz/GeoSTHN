import logging
from datetime import datetime, timezone


class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, timezone.utc)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime("%Y-%m-%d %H:%M:%S")


def setup_logger(log_file, log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=log_file,
        filemode="a",
    )

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        formatter = UTCFormatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

    return root_logger
