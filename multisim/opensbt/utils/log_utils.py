import logging as log
import logging.config
from importlib import reload


def setup_logging(log_to):
    # Disable messages from Matplotlib to reduce noise
    log.getLogger('matplotlib.font_manager').disabled = True
    
    # Configure logging
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard'
            },
        },
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'loggers': {
            '': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': True
            }
        }
    })

    if log_to is not None:
        try:
            file_handler = log.FileHandler(log_to, 'a', 'utf-8')
            file_handler.setLevel(log.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            log.getLogger('').addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file handler: {e}")

    # Log initial message
    start_msg = "Logging setup."
    if log_to:
        start_msg += f" Writing to file: {log_to}"
    log.info(start_msg)

def disable_pymoo_warnings():
    from pymoo.config import Config

    Config.warnings['not_compiled'] = False
