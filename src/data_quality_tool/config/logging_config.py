import logging


def get_logger(name: str = "dq_logger", log_file: str = "logs/dq_checks.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if already configured
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

        # Optional: prevent propagation to root logger
        logger.propagate = False

    return logger