import logging


def log_line(color_name, effect_name=None):
    Colors = {
        "VIOLET": "\033[95m",
        "RED": "\033[91m",
        "GREEN": "\033[92m",
        "BLUE": "\033[94m",
        "CYAN": "\033[96m",
        "YELLOW": "\033[93m",
    }
    Effects = {"BOLD": "\033[1m", "UNDERLINE": "\033[4m"}
    ENDC = "\033[0m"

    def color(txt):
        effect = Effects[effect_name] if effect_name else ""
        return Colors[color_name] + effect + txt + ENDC

    return (
        f'{color("[%(levelname)s]")} %(message)s\t'
        f'({color("%(filename)s")} | line: %(lineno)d)'
    )


class CustomFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: log_line("VIOLET"),
        logging.INFO: log_line("GREEN"),
        logging.WARNING: log_line("YELLOW"),
        logging.ERROR: log_line("RED"),
        logging.CRITICAL: log_line("RED", "BOLD"),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(level=logging.INFO) -> "logging.Logger":
    # create logger with 'My_app'
    logger = logging.getLogger()
    logger.setLevel(level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    return logger
