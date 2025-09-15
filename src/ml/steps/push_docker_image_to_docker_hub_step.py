import logging
from pathlib import Path
import subprocess

from utils.config import get_root_dir_path

# Configure Logger:
# ANSI Escape Code for white letters
WHITE = "\033[37m"
RESET = "\033[0m"  # reset of color

# Logger configure
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console-Handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Formatter with ANSI Escape Code for white letters
formatter = logging.Formatter(f'{WHITE}%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s{RESET}')
handler.setFormatter(formatter)

# Handler for Logger added
logger.addHandler(handler)


def push_docker_image_to_docker_hub() -> bool:
    """
    Push this microservice as docker image to docker hub

    :return: Gives True when it could be successfully uploaded if not False
    """
    script_path = get_root_dir_path() / "images_push_to_dockerhub.sh"

    if not script_path.exists():
        logger.error(f"Bash script not found: {script_path}")
        return False

    try:
        # Make script executable
        subprocess.run(["chmod", "+x", str(script_path)], check=True, capture_output=True, text=True)
        logger.info(f"Made script executable: {script_path}")

        # Execute the script
        result = subprocess.run([str(script_path)], check=True, capture_output=True, text=True)
        logger.info(f"Script executed successfully:\n{result.stdout.strip()}")

        return True
    except subprocess.CalledProcessError as ce:
        logger.error(f"Script execution failed:\n{ce.stderr.strip()}")
        return False
    except Exception as ex:
        logger.error(f"Unexpected exception:\n{ex}")
        return False
