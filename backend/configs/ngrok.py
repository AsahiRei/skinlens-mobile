from dotenv import load_dotenv
from pyngrok import conf
import os
import ssl
import urllib.request

def ngrok_config():
    load_dotenv()
    token = os.getenv("NGROK_AUTH_TOKEN")
    if not token:
        raise ValueError("NGROK_AUTH_TOKEN not found in environment variables.")
    
    pyngrok_config = conf.get_default()
    pyngrok_config.auth_token = token
    
    pyngrok_config.request_header_timeout = 30
    
    ngrok_paths = [
        os.path.expanduser("~/.ngrok2/ngrok.exe"),  # Default ngrok location
        os.path.join(os.getenv("LOCALAPPDATA"), "Packages", "PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0", "LocalCache", "Local", "ngrok", "ngrok.exe"),
        os.path.join(os.getenv("LOCALAPPDATA"), "Microsoft", "WinGet", "Packages", "Ngrok.Ngrok_Microsoft.Winget.Source_8wekyb3d8bbwe", "ngrok.exe"),
    ]
    
    for path in ngrok_paths:
        if os.path.exists(path):
            pyngrok_config.ngrok_path = path
            print(f"Using ngrok binary at: {path}")
            break

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    urllib.request.ssl._create_default_https_context = lambda: ssl_context